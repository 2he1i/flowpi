"""Dependency-light DOMINO client adapter for the FlowPi policy server.

This module is imported by DOMINO's simulation process. It deliberately does not import OpenPI,
JAX, Torch, or the FlowPi checkpoint; all model work happens in the separate policy server.
"""

from __future__ import annotations

import atexit
import base64
from collections import deque
from contextlib import suppress
import importlib
import json
import multiprocessing as mp
import os
import pathlib
from queue import Empty
from queue import Full
import socket
import threading
import time
from typing import Any

import numpy as np

_METRICS_PATH = os.environ.get("FLOWPI_DOMINO_METRICS_PATH")
_METRICS_FLUSH_EVERY = max(1, int(os.environ.get("FLOWPI_METRICS_FLUSH_EVERY", "10")))
_METRICS_START = time.perf_counter()
_METRICS_EVENTS: list[dict[str, Any]] = []
_METRICS_QUERY_EVENTS: list[dict[str, Any]] = []
_METRICS_RPC_LATENCIES: list[float] = []
_METRICS_CONTROL_TIMES: list[float] = []
_METRICS_ACTION_EXECUTE: list[float] = []
_METRICS_LOCK = threading.RLock()
_METRICS_RUNTIME_CONFIG: dict[str, Any] = {
    "control_target_hz": None,
    "direct_control": False,
    "control_mode": "official",
    "sim_steps_per_control": None,
    "copy_observations": False,
    "drain_action_chunk": True,
    "hold_last_burst": 3,
    "state_process": True,
    "fast_state_target_hz": 25.0,
    "state_idle_timeout_s": 5.0,
    "observation_throttle": True,
    "image_target_hz": 10.0,
    "wall_clock_catchup": True,
    "max_catchup_actions": 8,
    "compress_rgb": True,
    "jpeg_quality": 95,
    "max_pending_observations": 16,
    "worker_join_timeout_s": 10.0,
    "worker_join_timeouts": 0,
    "pending_observation_drops": 0,
    "pending_state_drops": 0,
}
_JPEG_IMAGE_KEY = "__flowpi_jpeg_rgb__"
_CONTROL_TICK_KEY = "__flowpi_control_tick__"


def _series_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected {name} to be boolean, got {value!r}")


def _write_metrics() -> None:
    if not _METRICS_PATH:
        return
    path = pathlib.Path(_METRICS_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _METRICS_LOCK:
        events = list(_METRICS_EVENTS)
        query_events = list(_METRICS_QUERY_EVENTS)
        latencies = list(_METRICS_RPC_LATENCIES)
        control_times = list(_METRICS_CONTROL_TIMES)
        action_execute = list(_METRICS_ACTION_EXECUTE)
        target_hz = _METRICS_RUNTIME_CONFIG["control_target_hz"]
        direct_control = _METRICS_RUNTIME_CONFIG["direct_control"]
        sim_steps = _METRICS_RUNTIME_CONFIG["sim_steps_per_control"]
    timestamps = np.asarray(control_times, dtype=np.float64)
    intervals = np.diff(timestamps) * 1000 if timestamps.size > 1 else np.asarray([], dtype=np.float64)
    intervals = intervals[intervals > 0]
    target_hz = float(target_hz) if target_hz else 50.0
    # A synchronous first policy query may include one-time JAX compilation. Keep the raw
    # all-sample metric, but expose a steady-state metric that removes startup-sized gaps so a
    # 70-second compile cannot masquerade as a 2-Hz control loop.
    startup_gap_ms = max(1000.0, 10_000.0 / target_hz)
    steady_intervals = intervals[intervals <= startup_gap_ms]
    query_timestamps = np.asarray(
        [event["timestamp_s"] for event in query_events], dtype=np.float64
    )
    query_intervals = np.diff(query_timestamps) * 1000 if query_timestamps.size > 1 else np.asarray([], dtype=np.float64)
    query_intervals = query_intervals[query_intervals > 0]
    steady_query_intervals = query_intervals[query_intervals <= startup_gap_ms]
    state_query_events = [event for event in query_events if event.get("source") == "fast_state"]
    image_update_events = [event for event in query_events if event.get("source") == "image_update"]

    def _event_rate(events: list[dict[str, Any]]) -> float | None:
        if len(events) < 2:
            return None
        event_times = np.asarray([event["timestamp_s"] for event in events], dtype=np.float64)
        event_intervals = np.diff(event_times)
        event_intervals = event_intervals[event_intervals > 0]
        return float(1.0 / np.mean(event_intervals)) if event_intervals.size else None

    steady_loop_hz = float(1000.0 / np.mean(steady_intervals)) if steady_intervals.size else None
    steady_query_hz = float(1000.0 / np.mean(steady_query_intervals)) if steady_query_intervals.size else None
    payload = {
        "schema_version": 2,
        "client": {
            "request_count": len(query_events),
            "control_step_count": len(events),
            "control_target_hz": target_hz,
            "direct_control": direct_control,
            "control_mode": _METRICS_RUNTIME_CONFIG["control_mode"],
            "sim_steps_per_control": sim_steps,
            "pending_observation_drops": _METRICS_RUNTIME_CONFIG["pending_observation_drops"],
            "pending_state_drops": _METRICS_RUNTIME_CONFIG["pending_state_drops"],
            "max_pending_observations": _METRICS_RUNTIME_CONFIG["max_pending_observations"],
            "worker_join_timeout_s": _METRICS_RUNTIME_CONFIG["worker_join_timeout_s"],
            "worker_join_timeouts": _METRICS_RUNTIME_CONFIG["worker_join_timeouts"],
            "copy_observations": _METRICS_RUNTIME_CONFIG["copy_observations"],
            "drain_action_chunk": _METRICS_RUNTIME_CONFIG["drain_action_chunk"],
            "hold_last_burst": _METRICS_RUNTIME_CONFIG["hold_last_burst"],
            "state_process": _METRICS_RUNTIME_CONFIG["state_process"],
            "fast_state_target_hz": _METRICS_RUNTIME_CONFIG["fast_state_target_hz"],
            "state_idle_timeout_s": _METRICS_RUNTIME_CONFIG["state_idle_timeout_s"],
            "observation_throttle": _METRICS_RUNTIME_CONFIG["observation_throttle"],
            "image_target_hz": _METRICS_RUNTIME_CONFIG["image_target_hz"],
            "wall_clock_catchup": _METRICS_RUNTIME_CONFIG["wall_clock_catchup"],
            "max_catchup_actions": _METRICS_RUNTIME_CONFIG["max_catchup_actions"],
            "compress_rgb": _METRICS_RUNTIME_CONFIG["compress_rgb"],
            "jpeg_quality": _METRICS_RUNTIME_CONFIG["jpeg_quality"],
            "rpc_roundtrip_ms": _series_summary(latencies),
            "action_execute_ms": _series_summary(action_execute),
            "policy_query_rate": (
                float(
                    1.0
                    / np.mean(
                        np.diff(np.asarray([event["timestamp_s"] for event in query_events], dtype=np.float64))
                    )
                )
                if len(query_events) > 1
                else None
            ),
            "loop_hz": float(1000.0 / np.mean(intervals)) if intervals.size else None,
            "loop_interval_ms": _series_summary(intervals.tolist()),
            "steady_loop_hz": steady_loop_hz,
            "steady_policy_query_rate": steady_query_hz,
            "fast_state_query_count": len(state_query_events),
            "fast_state_query_rate": _event_rate(state_query_events),
            "image_update_count": len(image_update_events),
            "image_update_rate": _event_rate(image_update_events),
            "fast_state_rpc_ms": _series_summary(
                [event["rpc_roundtrip_ms"] for event in state_query_events]
            ),
            "image_update_rpc_ms": _series_summary(
                [event["rpc_roundtrip_ms"] for event in image_update_events]
            ),
            "steady_loop_interval_ms": _series_summary(steady_intervals.tolist()),
            "startup_gap_threshold_ms": startup_gap_ms,
            "events": events,
            "query_events": query_events,
        },
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    os.replace(temporary_path, path)


atexit.register(_write_metrics)


def _encode_rgb_for_rpc(image: Any, *, jpeg_quality: int) -> dict[str, str]:
    """JPEG-encode one RGB frame for DOMINO's JSON/base64 transport.

    DOMINO's stock ModelClient serializes numpy arrays as base64 inside JSON. Three raw 480x640
    frames therefore dominate every RPC. The marker stays JSON-native, while the policy adapter
    decodes it back to the original RGB geometry before any resize or model transform.
    """
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - DOMINO normally ships OpenCV
        raise RuntimeError(
            "FLOWPI_DOMINO_COMPRESS_RGB=1 requires OpenCV in the DOMINO environment; "
            "set FLOWPI_DOMINO_COMPRESS_RGB=0 to use raw RGB transport"
        ) from exc
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB frame for RPC compression, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    # OpenCV's JPEG path is BGR-oriented. Convert explicitly so the policy receives RGB again.
    frame_bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to JPEG-encode a DOMINO RGB frame")
    return {_JPEG_IMAGE_KEY: base64.b64encode(encoded).decode("ascii")}


def _slim_observation(
    observation: dict[str, Any],
    *,
    copy_arrays: bool = False,
    compress_rgb: bool = False,
    jpeg_quality: int = 95,
    control_tick: int | None = None,
) -> dict[str, Any]:
    """Keep only the three RGB frames and qpos needed by FlowPi.

    DOMINO may attach point clouds, depth, segmentation, or dense-flow ground truth to the same
    observation. Sending those fields over the RPC link would add latency without changing the
    policy output.
    """
    try:
        camera_data = observation["observation"]
        joint_data = observation["joint_action"]
        slim_cameras = {
            camera_name: {
                "rgb": (
                    _encode_rgb_for_rpc(camera_data[camera_name]["rgb"], jpeg_quality=jpeg_quality)
                    if compress_rgb
                    else (
                        np.asarray(camera_data[camera_name]["rgb"]).copy()
                        if copy_arrays
                        else camera_data[camera_name]["rgb"]
                    )
                )
            }
            for camera_name in ("head_camera", "left_camera", "right_camera")
        }
        state = np.asarray(joint_data["vector"]).copy() if copy_arrays else joint_data["vector"]
    except (KeyError, TypeError) as exc:
        raise KeyError("DOMINO observation must contain RGB head/left/right cameras and joint_action.vector") from exc

    slim = {
        "observation": slim_cameras,
        "joint_action": {"vector": state},
    }
    if control_tick is not None:
        try:
            control_tick = int(control_tick)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"control_tick must be an integer, got {control_tick!r}") from exc
        if control_tick < 0:
            raise ValueError(f"control_tick must be non-negative, got {control_tick}")
        slim[_CONTROL_TICK_KEY] = control_tick
    return slim


def _parse_policy_response(response: Any) -> tuple[np.ndarray, int]:
    """Validate and detach one action chunk returned by the policy server."""
    if not isinstance(response, dict) or "actions" not in response:
        raise RuntimeError(f"FlowPi server returned an invalid response: {response!r}")
    actions = np.asarray(response["actions"], dtype=np.float32)
    action_count = int(response.get("pi0_step", actions.shape[0]))
    if actions.ndim != 2 or actions.shape[-1] < 14:
        raise ValueError(f"FlowPi server returned invalid actions with shape {actions.shape}")
    if not 0 < action_count <= actions.shape[0]:
        raise ValueError(f"FlowPi server returned invalid pi0_step={action_count} for {actions.shape}")
    actions = np.ascontiguousarray(actions[:action_count, :14], dtype=np.float32)
    if not np.all(np.isfinite(actions)):
        raise FloatingPointError("FlowPi server returned non-finite actions")
    return actions, action_count


def _state_rpc_process_main(
    client_module: str,
    client_class: str,
    host: str,
    port: int,
    timeout: float | None,
    target_hz: float,
    idle_timeout_s: float,
    request_queue: Any,
    result_queue: Any,
) -> None:
    """Run the fast-state socket in a process isolated from DOMINO's image/GIL workload.

    The simulator process still owns action execution and the latest-state mailbox. This child
    only performs the tiny state RPC and returns an action chunk. Keeping the socket and JSON
    request/response loop out of the image worker's process removes long JPEG/base64 critical
    sections from the 25 Hz fast channel without changing the policy server protocol.
    """
    # The module-level metrics atexit hook is inherited under ``fork``. The child must never
    # write a partial copy of the parent's metrics file when it exits.
    with suppress(Exception):
        atexit.unregister(_write_metrics)
    connection = None
    try:
        module = importlib.import_module(client_module)
        client_type = getattr(module, client_class)
        try:
            connection = client_type(host=host, port=port, timeout=timeout)
        except TypeError:
            connection = client_type(host, port, timeout)
        _configure_low_latency_socket(connection)
        latest_item = request_queue.get()
        if latest_item is None:
            return
        period_s = 1.0 / max(float(target_hz), 1.0)
        next_due_s = time.perf_counter()
        last_input_s = time.perf_counter()
        fresh_state = True
        while True:
            # Drain to the newest simulator state before every NFE. The parent queue is bounded
            # to one item, so this is O(1) in the normal latest-state path.
            while True:
                try:
                    item = request_queue.get_nowait()
                except Empty:
                    break
                if item is None:
                    return
                latest_item = item
                last_input_s = time.perf_counter()
                fresh_state = True

            if time.perf_counter() - last_input_s >= idle_timeout_s:
                # DOMINO can stop calling the policy immediately after an out-of-bounds or
                # failed episode. Do not keep extrapolating a cached proprioception forever;
                # the parent will reap this child during normal process shutdown.
                return

            wait_s = next_due_s - time.perf_counter()
            if wait_s > 0:
                try:
                    item = request_queue.get(timeout=min(wait_s, idle_timeout_s))
                except Empty:
                    continue
                if item is None:
                    return
                latest_item = item
                last_input_s = time.perf_counter()
                fresh_state = True
                continue

            state, instruction, control_tick = latest_item
            # When no newer simulator state arrived, let the runtime advance its internal NFE
            # clock from the cached state. A repeated explicit DOMINO tick would freeze prefix /
            # flow age, while extrapolating a physical tick here would invent simulator time.
            request_control_tick = control_tick if fresh_state else None
            fresh_state = False
            request_t0 = time.perf_counter()
            response = connection.call(
                func_name="infer_fast_state",
                obs={
                    "state": np.asarray(state, dtype=np.float32),
                    "instruction": instruction,
                    _CONTROL_TICK_KEY: request_control_tick,
                },
            )
            roundtrip_ms = (time.perf_counter() - request_t0) * 1000
            actions, action_count = _parse_policy_response(response)
            result_queue.put(
                {
                    "kind": "result",
                    "started_at": request_t0,
                    "roundtrip_ms": roundtrip_ms,
                    "actions": actions,
                    "action_count": action_count,
                }
            )
            next_due_s += period_s
            now_s = time.perf_counter()
            if next_due_s < now_s - period_s:
                next_due_s = now_s
    except BaseException as exc:
        with suppress(Exception):
            result_queue.put(
                {
                    "kind": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    finally:
        if connection is not None:
            with suppress(Exception):
                connection.close()
        with suppress(Exception):
            result_queue.put(None)


def _record_policy_query(*, started_at: float, roundtrip_ms: float, action_count: int, source: str) -> None:
    with _METRICS_LOCK:
        _METRICS_RPC_LATENCIES.append(float(roundtrip_ms))
        _METRICS_QUERY_EVENTS.append(
            {
                "query_index": len(_METRICS_QUERY_EVENTS),
                "timestamp_s": started_at - _METRICS_START,
                "rpc_roundtrip_ms": float(roundtrip_ms),
                "actions_received": int(action_count),
                "source": source,
            }
        )


def _install_model_call_lock(model: Any) -> None:
    """Serialize the background infer worker with DOMINO's out-of-band reset RPC.

    ``ModelClient`` owns one TCP stream. The evaluator sends ``reset_model`` directly on that
    object between episodes, so a worker must not write an infer request concurrently with reset.
    Installing this instance-level wrapper keeps the policy-side worker and evaluator calls on
    the same lock without importing or modifying DOMINO.
    """
    if getattr(model, "_flowpi_domino_call_lock", None) is not None:
        return
    lock = threading.RLock()
    original_call = model.call

    def synchronized_call(*args: Any, **kwargs: Any) -> Any:
        func_name = kwargs.get("func_name")
        if func_name is None and args:
            func_name = args[0]
        with lock:
            if func_name == "reset_model":
                # The worker uses this same wrapped call. Stop it only after acquiring the lock;
                # stopping before the lock could join a worker that is itself waiting on the
                # lock and deadlock the DOMINO evaluator at episode boundaries.
                controller = getattr(model, "_flowpi_domino_controller", None)
                if controller is not None:
                    controller.stop_worker(clear_buffers=True)
            return original_call(*args, **kwargs)

    model.call = synchronized_call
    model._flowpi_domino_call_lock = lock  # noqa: SLF001


def _configure_low_latency_socket(model: Any) -> None:
    """Disable Nagle for the small state-only request/response path."""
    sock = getattr(model, "sock", None)
    if sock is None:
        return
    with suppress(OSError):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _open_parallel_model_connection(model: Any, *, role: str) -> Any:
    """Open an independent TCP connection for a long-running policy channel.

    DOMINO's ``ModelClient`` owns one socket, so sharing it between the image-update and
    fast-state workers would serialize the very RPCs that are meant to overlap.  The policy
    server already accepts one handler thread per socket; use two client instances so the
    image/flow stream cannot hold the state/control stream behind a 400 ms JPEG/update call.
    """
    client_type = type(model)
    host = getattr(model, "host", "localhost")
    port = getattr(model, "port", None)
    timeout = getattr(model, "timeout", None)
    try:
        connection = client_type(host=host, port=port, timeout=timeout)
        _configure_low_latency_socket(connection)
        return connection
    except TypeError:
        try:
            connection = client_type(host, port, timeout)
            _configure_low_latency_socket(connection)
            return connection
        except Exception as exc:  # pragma: no cover - only custom DOMINO clients use this path
            raise RuntimeError(f"Could not open {role} policy connection") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not open {role} policy connection") from exc


def _close_model_connection(model: Any) -> None:
    """Close an auxiliary ModelClient without masking an evaluation error."""
    close = getattr(model, "close", None)
    if close is not None:
        with suppress(Exception):
            close()


class _ContinuousPolicyController:
    """DOMINO-side high-rate loop with an asynchronous latest-observation query worker.

    DOMINO's stock ``take_action`` performs TOPP and advances many SAPIEN physics steps before
    returning. Calling it once per RPC serialized the whole benchmark at roughly 2 Hz. This
    controller executes one or more fixed simulation control intervals per evaluator callback
    (one action chunk when available) and lets two independent workers query FlowPi in the
    background. The image worker consumes accumulated physical-frame history in bounded batches;
    the state worker publishes action chunks; the main thread holds the last action only when a
    fresh chunk has not arrived yet.
    """

    def __init__(self, model: Any):
        self.model = model
        _configure_low_latency_socket(model)
        self.control_hz = float(os.environ.get("FLOWPI_DOMINO_CONTROL_HZ", "50"))
        if self.control_hz <= 0:
            raise ValueError("FLOWPI_DOMINO_CONTROL_HZ must be positive")
        configured_mode = os.environ.get("FLOWPI_DOMINO_CONTROL_MODE")
        if configured_mode is None:
            # The legacy boolean remains supported. Without either setting, use DOMINO's
            # official TOPP/take_action semantics so importing this adapter cannot silently
            # change benchmark behavior. run_flowpi_domino.sh explicitly selects direct mode for
            # FlowPi's high-rate deployment path.
            legacy_direct = os.environ.get("FLOWPI_DOMINO_DIRECT_CONTROL")
            configured_mode = (
                "direct"
                if legacy_direct is not None
                and legacy_direct.strip().lower() not in {"0", "false", "no", "off"}
                else "official"
            )
        control_mode = configured_mode.strip().lower()
        if control_mode in {"topp", "take_action", "stock", "official"}:
            control_mode = "official"
        elif control_mode in {"direct", "fixed", "fixed_step"}:
            control_mode = "direct"
        else:
            raise ValueError(
                "FLOWPI_DOMINO_CONTROL_MODE must be 'direct' or 'official', "
                f"got {configured_mode!r}"
            )
        self.control_mode = control_mode
        self.direct_control = control_mode == "direct"
        self.max_ready_chunks = max(1, int(os.environ.get("FLOWPI_DOMINO_MAX_READY_CHUNKS", "1")))
        self.max_pending_observations = max(1, int(os.environ.get("FLOWPI_DOMINO_MAX_PENDING_OBS", "16")))
        self.worker_join_timeout_s = float(os.environ.get("FLOWPI_DOMINO_WORKER_JOIN_TIMEOUT", "10"))
        if self.worker_join_timeout_s <= 0:
            raise ValueError("FLOWPI_DOMINO_WORKER_JOIN_TIMEOUT must be positive")
        # RoboTwin's get_obs() creates fresh uint8 arrays for every camera call. Avoid copying
        # those arrays once more on the main control thread; set FLOWPI_DOMINO_COPY_OBS=1 for a
        # defensive copy when integrating a camera backend that reuses its buffers.
        self.copy_observations = _env_bool("FLOWPI_DOMINO_COPY_OBS", default=False)
        self.compress_rgb = _env_bool("FLOWPI_DOMINO_COMPRESS_RGB", default=True)
        self.jpeg_quality = int(os.environ.get("FLOWPI_DOMINO_JPEG_QUALITY", "95"))
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("FLOWPI_DOMINO_JPEG_QUALITY must be in [1, 100]")
        # A policy RPC returns an action chunk. Consume the current chunk in the same evaluator
        # callback so DOMINO does not render a full three-camera observation between every action
        # in that chunk. Each consumed action still advances the exact configured simulator
        # interval, so this preserves the training control clock while removing redundant camera
        # callbacks. Set FLOWPI_DOMINO_DRAIN_ACTION_CHUNK=0 to restore one action per callback.
        self.drain_action_chunk = _env_bool("FLOWPI_DOMINO_DRAIN_ACTION_CHUNK", default=True)
        # If the latest-state worker has not finished a replacement chunk, advance a bounded
        # number of physical control intervals with the existing safe hold-last target in the
        # same callback. This compresses repeated fallback actions without changing their
        # simulated duration or the trained d support; the next callback still submits the newest
        # proprioception and consumes a fresh chunk as soon as one is ready.
        self.hold_last_burst = max(1, int(os.environ.get("FLOWPI_DOMINO_HOLD_LAST_BURST", "3")))
        # A slow multi-camera get_obs() can represent several physical 50 Hz periods. Catch up
        # by executing that many exact simulator intervals in this callback instead of letting
        # the physical control clock fall behind the wall clock.
        self.wall_clock_catchup = _env_bool("FLOWPI_DOMINO_WALL_CLOCK_CATCHUP", default=True)
        self.max_catchup_actions = max(1, int(os.environ.get("FLOWPI_DOMINO_MAX_CATCHUP_ACTIONS", "8")))
        self.state_process_enabled = _env_bool("FLOWPI_DOMINO_STATE_PROCESS", default=True)
        self.fast_state_target_hz = float(os.environ.get("FLOWPI_DOMINO_FAST_STATE_HZ", "25"))
        if self.fast_state_target_hz <= 0:
            raise ValueError("FLOWPI_DOMINO_FAST_STATE_HZ must be positive")
        self.state_idle_timeout_s = float(os.environ.get("FLOWPI_DOMINO_STATE_IDLE_TIMEOUT", "5"))
        if self.state_idle_timeout_s <= 0:
            raise ValueError("FLOWPI_DOMINO_STATE_IDLE_TIMEOUT must be positive")
        self.observation_throttle = _env_bool("FLOWPI_DOMINO_THROTTLE_OBS", default=True)
        self.image_target_hz = float(os.environ.get("FLOWPI_DOMINO_IMAGE_HZ", "10"))
        if self.image_target_hz <= 0:
            raise ValueError("FLOWPI_DOMINO_IMAGE_HZ must be positive")
        sim_steps = os.environ.get("FLOWPI_DOMINO_SIM_STEPS", "")
        self.sim_steps_override = int(sim_steps) if sim_steps else None
        if self.sim_steps_override is not None and self.sim_steps_override <= 0:
            raise ValueError("FLOWPI_DOMINO_SIM_STEPS must be positive when provided")

        with _METRICS_LOCK:
            _METRICS_RUNTIME_CONFIG.update(
                {
                    "control_target_hz": self.control_hz,
                    "direct_control": self.direct_control,
                    "control_mode": self.control_mode,
                    "sim_steps_per_control": self.sim_steps_override,
                    "copy_observations": self.copy_observations,
                    "drain_action_chunk": self.drain_action_chunk,
                    "hold_last_burst": self.hold_last_burst,
                    "state_process": self.state_process_enabled,
                    "fast_state_target_hz": self.fast_state_target_hz,
                    "state_idle_timeout_s": self.state_idle_timeout_s,
                    "observation_throttle": self.observation_throttle,
                    "image_target_hz": self.image_target_hz,
                    "wall_clock_catchup": self.wall_clock_catchup,
                    "max_catchup_actions": self.max_catchup_actions,
                    "compress_rgb": self.compress_rgb,
                    "jpeg_quality": self.jpeg_quality,
                    "max_pending_observations": self.max_pending_observations,
                    "worker_join_timeout_s": self.worker_join_timeout_s,
                    "worker_join_timeouts": 0,
                    "pending_observation_drops": 0,
                    "pending_state_drops": 0,
                }
            )

        _install_model_call_lock(model)
        # A ModelClient socket is request/response serialized. Keep the evaluator's main socket
        # for the synchronous warm-start/reset path and open two independent sockets for the
        # long-running image and state workers. The state worker is optionally moved into a
        # process below so DOMINO-side image JPEG/JSON work cannot starve it under the GIL.
        self._image_model = _open_parallel_model_connection(model, role="image-update")
        if self.state_process_enabled:
            self._state_model = None
            self._state_client_module = type(model).__module__
            self._state_client_class = type(model).__name__
            self._state_client_host = getattr(model, "host", "localhost")
            self._state_client_port = int(model.port)
            self._state_client_timeout = getattr(model, "timeout", None)
            try:
                self._state_mp_context = mp.get_context("fork")
            except ValueError:  # pragma: no cover - Linux DOMINO uses fork
                self._state_mp_context = mp.get_context()
        else:
            self._state_model = _open_parallel_model_connection(model, role="fast-state")
        model._flowpi_domino_controller = self  # noqa: SLF001
        self._condition = threading.Condition()
        self._pending_observations: list[dict[str, Any]] = []
        self._pending_instruction: str | None = None
        self._pending_observation_drops = 0
        self._pending_state: np.ndarray | None = None
        self._pending_state_instruction: str | None = None
        self._pending_state_control_tick: int | None = None
        self._pending_state_drops = 0
        self._ready_chunks: deque[tuple[np.ndarray, int]] = deque()
        self._current_chunk: np.ndarray | None = None
        self._current_index = 0
        self._last_action: np.ndarray | None = None
        self._worker: threading.Thread | None = None
        self._image_worker: threading.Thread | None = None
        self._state_process: mp.Process | None = None
        self._state_result_worker: threading.Thread | None = None
        self._state_request_queue: Any | None = None
        self._state_result_queue: Any | None = None
        self._stop_requested = False
        self._worker_error: BaseException | None = None
        self._episode_marker: int | None = None
        self._episode_query_count = 0
        self._max_ready_depth = 0
        self._last_step_start_s: float | None = None
        self._control_budget = 0.0
        self._last_image_observation: dict[str, Any] | None = None
        self._closed = False
        atexit.register(self.close)

    def _raise_worker_error(self) -> None:
        with self._condition:
            error = self._worker_error
            self._worker_error = None
        if error is not None:
            raise RuntimeError("FlowPi background policy query failed") from error

    def _infer(
        self,
        observation_history: list[dict[str, Any]],
        instruction: str | None,
        *,
        source: str,
    ) -> np.ndarray:
        request_t0 = time.perf_counter()
        response = self.model.call(
            func_name="infer",
            obs={
                "observation": observation_history[-1],
                "observation_history": observation_history,
                "instruction": instruction,
            },
        )
        roundtrip_ms = (time.perf_counter() - request_t0) * 1000
        actions, action_count = _parse_policy_response(response)
        _record_policy_query(
            started_at=request_t0,
            roundtrip_ms=roundtrip_ms,
            action_count=action_count,
            source=source,
        )
        return actions

    def _update_observation(
        self,
        observation_history: list[dict[str, Any]],
        instruction: str | None,
        *,
        call_model: Any | None = None,
    ) -> None:
        """Push full RGB frames to the policy without consuming a fast NFE."""
        request_t0 = time.perf_counter()
        connection = self._image_model if call_model is None else call_model
        response = connection.call(
            func_name="update_observation",
            obs={
                "observation": observation_history[-1],
                "observation_history": observation_history,
                "instruction": instruction,
            },
        )
        roundtrip_ms = (time.perf_counter() - request_t0) * 1000
        if not isinstance(response, dict) or not response.get("updated", False):
            raise RuntimeError(f"FlowPi server returned an invalid image-update response: {response!r}")
        _record_policy_query(
            started_at=request_t0,
            roundtrip_ms=roundtrip_ms,
            action_count=0,
            source="image_update",
        )

    def _infer_fast_state(
        self,
        state: np.ndarray,
        instruction: str | None,
        control_tick: int | None = None,
        *,
        call_model: Any | None = None,
    ) -> np.ndarray:
        """Push only qpos/proprioception and receive one fresh action chunk."""
        request_t0 = time.perf_counter()
        connection = self._state_model if call_model is None else call_model
        response = connection.call(
            func_name="infer_fast_state",
            obs={
                "state": np.asarray(state, dtype=np.float32),
                "instruction": instruction,
                _CONTROL_TICK_KEY: control_tick,
            },
        )
        roundtrip_ms = (time.perf_counter() - request_t0) * 1000
        actions, action_count = _parse_policy_response(response)
        _record_policy_query(
            started_at=request_t0,
            roundtrip_ms=roundtrip_ms,
            action_count=action_count,
            source="fast_state",
        )
        return actions

    def _set_worker_error(self, error: BaseException) -> None:
        with self._condition:
            if self._worker_error is None:
                self._worker_error = error
            self._stop_requested = True
            self._condition.notify_all()

    def _start_state_process(self) -> None:
        """Start the process-isolated fast-state RPC and its parent-side result pump."""
        if not self.state_process_enabled:
            return
        request_queue = self._state_mp_context.Queue(maxsize=1)
        result_queue = self._state_mp_context.Queue(maxsize=16)
        process = self._state_mp_context.Process(
            target=_state_rpc_process_main,
            args=(
                self._state_client_module,
                self._state_client_class,
                self._state_client_host,
                self._state_client_port,
                self._state_client_timeout,
                self.fast_state_target_hz,
                self.state_idle_timeout_s,
                request_queue,
                result_queue,
            ),
            name="flowpi-domino-fast-state-process",
            daemon=True,
        )
        self._state_request_queue = request_queue
        self._state_result_queue = result_queue
        self._state_process = process
        process.start()
        result_worker = threading.Thread(
            target=self._state_result_loop,
            name="flowpi-domino-fast-state-results",
            daemon=True,
        )
        self._state_result_worker = result_worker
        result_worker.start()

    def _state_result_loop(self) -> None:
        """Move process results into the same bounded action-chunk mailbox as the old thread."""
        result_queue = self._state_result_queue
        if result_queue is None:
            return
        while True:
            try:
                result = result_queue.get()
            except (EOFError, OSError):
                return
            if result is None:
                return
            if result.get("kind") == "error":
                self._set_worker_error(RuntimeError(result.get("error", "fast-state process failed")))
                return
            try:
                actions = np.asarray(result["actions"], dtype=np.float32)
                _record_policy_query(
                    started_at=float(result["started_at"]),
                    roundtrip_ms=float(result["roundtrip_ms"]),
                    action_count=int(result.get("action_count", actions.shape[0])),
                    source="fast_state",
                )
                with self._condition:
                    if self._stop_requested:
                        return
                    while len(self._ready_chunks) >= self.max_ready_chunks:
                        self._ready_chunks.popleft()
                    self._ready_chunks.append((actions, int(actions.shape[0])))
                    self._max_ready_depth = max(self._max_ready_depth, len(self._ready_chunks))
                    self._condition.notify_all()
            except BaseException as exc:
                self._set_worker_error(exc)
                return

    def _state_worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._stop_requested:
                    # The state worker is a latest-state stream, not a blocking action queue. It
                    # must keep issuing NFEs while the evaluator is consuming the current chunk;
                    # when a future chunk is already buffered, a newer result replaces it below.
                    # Waiting for the mailbox to become empty artificially capped the fast loop
                    # at one NFE per d-action chunk and produced long hold-last stretches.
                    has_fast_state = self._pending_state is not None
                    if has_fast_state:
                        break
                    self._condition.wait()
                if self._stop_requested:
                    return
                state = self._pending_state
                self._pending_state = None
                instruction = self._pending_state_instruction
                self._pending_state_instruction = None
                control_tick = self._pending_state_control_tick
                self._pending_state_control_tick = None
            try:
                actions = self._infer_fast_state(
                    state,
                    instruction,
                    control_tick,
                    call_model=self._state_model,
                )
            except BaseException as exc:  # propagate on the next main-thread control step
                self._set_worker_error(exc)
                return
            with self._condition:
                if self._stop_requested:
                    return
                if actions is not None:
                    # There is at most one future chunk worth preserving. If the worker finished
                    # another NFE before the current chunk was exhausted, keep the freshest
                    # state-conditioned result and discard the older future action chunk. Every
                    # emitted width remains within the training support; only stale deployment
                    # results are coalesced.
                    while len(self._ready_chunks) >= self.max_ready_chunks:
                        self._ready_chunks.popleft()
                    self._ready_chunks.append((actions, int(actions.shape[0])))
                    self._max_ready_depth = max(self._max_ready_depth, len(self._ready_chunks))
                self._condition.notify_all()

    def _image_worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._stop_requested and not self._pending_observations:
                    self._condition.wait()
                if self._stop_requested:
                    return
                raw_observation_history = self._pending_observations
                self._pending_observations = []
                instruction = self._pending_instruction
                self._pending_instruction = None
            try:
                # JPEG encoding/copying is intentionally off the simulator thread. This worker
                # has its own socket, so it can spend hundreds of ms moving RGB/SEA-RAFT data
                # while the state worker continues issuing fast NFEs.
                observation_history = [
                    _slim_observation(
                        observation,
                        copy_arrays=self.copy_observations,
                        compress_rgb=self.compress_rgb,
                        jpeg_quality=self.jpeg_quality,
                        control_tick=observation.get(_CONTROL_TICK_KEY),
                    )
                    for observation in raw_observation_history
                ]
                self._update_observation(
                    observation_history,
                    instruction,
                    call_model=self._image_model,
                )
            except BaseException as exc:  # propagate on the next main-thread control step
                self._set_worker_error(exc)
                return
            with self._condition:
                self._condition.notify_all()

    def _ensure_worker(self) -> None:
        with self._condition:
            if self.state_process_enabled:
                state_alive = self._state_process is not None and self._state_process.is_alive()
                result_alive = self._state_result_worker is not None and self._state_result_worker.is_alive()
            else:
                state_alive = self._worker is not None and self._worker.is_alive()
                result_alive = True
            image_alive = self._image_worker is not None and self._image_worker.is_alive()
            if not state_alive or not result_alive or not image_alive:
                self._stop_requested = False
                if self.state_process_enabled and (not state_alive or not result_alive):
                    self._start_state_process()
                elif not self.state_process_enabled and not state_alive:
                    self._worker = threading.Thread(
                        target=self._state_worker_loop,
                        name="flowpi-domino-fast-state",
                        daemon=True,
                    )
                    self._worker.start()
                if not image_alive:
                    self._image_worker = threading.Thread(
                        target=self._image_worker_loop,
                        name="flowpi-domino-image-update",
                        daemon=True,
                    )
                    self._image_worker.start()

    def _submit_observation(
        self,
        observation: dict[str, Any],
        instruction: str | None,
        control_tick: int | None = None,
    ) -> None:
        if observation is self._last_image_observation:
            return
        self._last_image_observation = observation
        # Drop optional depth/point-cloud/segmentation payloads on the evaluator thread, but do
        # not JPEG-encode yet. The worker performs the expensive transport preparation.
        observation_view = _slim_observation(
            observation,
            copy_arrays=self.copy_observations,
            compress_rgb=False,
            control_tick=control_tick,
        )
        with self._condition:
            # Keep only the three RGB/state references until the worker serializes them. This
            # removes JPEG encoding (and the default array copying) from the control thread.
            self._pending_observations.append(observation_view)
            self._pending_instruction = instruction
            overflow = len(self._pending_observations) - self.max_pending_observations
            if overflow > 0:
                del self._pending_observations[:overflow]
                self._pending_observation_drops += overflow
                with _METRICS_LOCK:
                    _METRICS_RUNTIME_CONFIG["pending_observation_drops"] = self._pending_observation_drops
            self._condition.notify_all()
        self._ensure_worker()

    def _submit_fast_state(self, task_env: Any, instruction: str | None) -> None:
        """Queue the newest simulator proprioception without rendering another RGB frame."""
        robot = getattr(task_env, "robot", None)
        if robot is None or not hasattr(robot, "get_left_arm_jointState") or not hasattr(robot, "get_right_arm_jointState"):
            raise AttributeError(
                "DOMINO robot must expose get_left_arm_jointState/get_right_arm_jointState for the fast state loop"
            )
        state = np.asarray(
            robot.get_left_arm_jointState() + robot.get_right_arm_jointState(),
            dtype=np.float32,
        )
        if state.shape != (14,):
            raise ValueError(f"DOMINO proprioception must have shape (14,), got {state.shape}")
        if self.state_process_enabled:
            # The process queue is deliberately one slot: the child is a latest-state stream and
            # should never build a backlog of stale proprioception requests. Replace the queued
            # item when the child is still serving the previous one.
            self._ensure_worker()
            request_queue = self._state_request_queue
            if request_queue is None:
                raise RuntimeError("FlowPi fast-state process queue is not running")
            item = (np.ascontiguousarray(state), instruction, int(task_env.take_action_cnt))
            try:
                request_queue.put_nowait(item)
            except Full:
                with suppress(Empty):
                    request_queue.get_nowait()
                try:
                    request_queue.put_nowait(item)
                except Full:
                    # A process-level queue can race with its feeder thread during shutdown.
                    # Keep the same observable semantics as the old latest-state mailbox.
                    with _METRICS_LOCK:
                        self._pending_state_drops += 1
                        _METRICS_RUNTIME_CONFIG["pending_state_drops"] = self._pending_state_drops
                    return
                with _METRICS_LOCK:
                    self._pending_state_drops += 1
                    _METRICS_RUNTIME_CONFIG["pending_state_drops"] = self._pending_state_drops
            return
        with self._condition:
            if self._pending_state is not None:
                self._pending_state_drops += 1
                with _METRICS_LOCK:
                    _METRICS_RUNTIME_CONFIG["pending_state_drops"] = self._pending_state_drops
            self._pending_state = np.ascontiguousarray(state)
            self._pending_state_instruction = instruction
            self._pending_state_control_tick = int(task_env.take_action_cnt)
            self._condition.notify_all()
        self._ensure_worker()

    def stop_worker(self, *, clear_buffers: bool) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
            worker = self._worker
            image_worker = self._image_worker
            state_process = self._state_process
            state_result_worker = self._state_result_worker
            state_request_queue = self._state_request_queue
            state_result_queue = self._state_result_queue
        if self.state_process_enabled and state_process is not None:
            # Wake a child that is waiting for the next latest-state item. If its one-slot queue
            # still contains a stale request, discard that item before placing the sentinel.
            if state_request_queue is not None:
                try:
                    state_request_queue.put_nowait(None)
                except Full:
                    with suppress(Empty):
                        state_request_queue.get_nowait()
                    with suppress(Full):
                        state_request_queue.put_nowait(None)
            state_process.join(timeout=self.worker_join_timeout_s)
            if state_process.is_alive():
                # The child owns only a client socket and no simulator/GPU state. Terminating it
                # is safe recovery for a hung RPC and prevents reset_model from deadlocking the
                # evaluator indefinitely.
                state_process.terminate()
                state_process.join(timeout=1.0)
                with _METRICS_LOCK:
                    _METRICS_RUNTIME_CONFIG["worker_join_timeouts"] += 1
            if state_result_worker is not None and state_result_worker is not threading.current_thread():
                if state_result_queue is not None:
                    with suppress(Exception):
                        state_result_queue.put_nowait(None)
                state_result_worker.join(timeout=1.0)
        for worker_name, worker_thread in (("fast-state", worker), ("image-update", image_worker)):
            if worker_thread is not None and worker_thread is not threading.current_thread():
                worker_thread.join(timeout=self.worker_join_timeout_s)
                if worker_thread.is_alive():
                    with _METRICS_LOCK:
                        _METRICS_RUNTIME_CONFIG["worker_join_timeouts"] += 1
                    raise TimeoutError(
                        f"FlowPi DOMINO {worker_name} worker did not stop within "
                        f"{self.worker_join_timeout_s:.1f}s. The policy RPC may be hung; "
                        "terminate the evaluator/server pair instead of blocking episode reset."
                    )
        with self._condition:
            self._worker = None
            self._image_worker = None
            self._state_process = None
            self._state_result_worker = None
            self._state_request_queue = None
            self._state_result_queue = None
            self._stop_requested = False
            self._pending_observations = []
            self._pending_instruction = None
            self._pending_state = None
            self._pending_state_instruction = None
            self._pending_state_control_tick = None
            self._worker_error = None
            if clear_buffers:
                self._ready_chunks.clear()
                self._current_chunk = None
                self._current_index = 0
                self._last_action = None
                self._episode_marker = None
                self._episode_query_count = 0
                self._last_step_start_s = None
                self._control_budget = 0.0
                self._last_image_observation = None

    def _install_observation_throttle(self, task_env: Any, initial_observation: dict[str, Any]) -> None:
        """Keep the simulator control loop at 50 Hz while sampling RGB at the training rate.

        RoboTwin's evaluator calls ``get_obs`` immediately before every policy callback.  A
        camera render costs roughly one 10-Hz period, so rendering on every callback makes the
        nominal 50-Hz control loop impossible.  Cache the last rendered observation between
        image periods; proprioception still comes directly from the robot in ``_submit_fast_state``.
        The policy receives each newly rendered RGB frame exactly once, preserving the 10-Hz
        training image clock rather than sending duplicate frames to SEA-RAFT.
        """
        if not self.observation_throttle:
            return
        task_env._flowpi_sensor_capture_enabled = True  # noqa: SLF001
        task_env._flowpi_sensor_period_s = 1.0 / self.image_target_hz  # noqa: SLF001
        task_env._flowpi_sensor_elapsed = 0.0  # noqa: SLF001
        task_env._flowpi_sensor_next_capture = 1.0 / self.image_target_hz  # noqa: SLF001
        task_env._flowpi_rendered_rgb = None  # noqa: SLF001
        cache = getattr(task_env, "_flowpi_observation_cache", None)
        if cache is None:
            original_get_obs = task_env.get_obs
            cache = {"observation": initial_observation, "timestamp_s": time.perf_counter()}
            period_s = 1.0 / self.image_target_hz

            def throttled_get_obs() -> dict[str, Any]:
                now_s = time.perf_counter()
                take_action_cnt = int(getattr(task_env, "take_action_cnt", 0))
                force_refresh = (
                    take_action_cnt == 0
                    or bool(getattr(task_env, "eval_success", False))
                    or take_action_cnt >= int(getattr(task_env, "step_lim", 2**31 - 1))
                )
                if (
                    not force_refresh
                    and cache["observation"] is not None
                    and now_s - cache["timestamp_s"] < period_s
                ):
                    return cache["observation"]
                # Measure the image period from render start to render start. Rendering itself is
                # close to one 10-Hz period on RoboTwin; recording the timestamp after the render
                # would accidentally produce ``period + render_time`` and halve the camera/flow
                # rate.
                cache["timestamp_s"] = now_s
                observation = self._observation_from_rendered_cameras(task_env)
                if observation is None:
                    observation = original_get_obs()
                cache["observation"] = observation
                return observation

            task_env.get_obs = throttled_get_obs
            task_env._flowpi_observation_cache = cache  # noqa: SLF001
        else:
            # The evaluator's first get_obs after setup_demo was already a real render. Refresh
            # the cache reference so a controller reused across episodes cannot return the prior
            # episode's image during the first callback.
            cache["observation"] = initial_observation
            cache["timestamp_s"] = time.perf_counter()

    def _begin_episode_if_needed(self, task_env: Any, initial_observation: dict[str, Any]) -> None:
        marker = id(task_env)
        is_new_episode = self._episode_marker is None or getattr(task_env, "take_action_cnt", 0) == 0
        if is_new_episode:
            self.stop_worker(clear_buffers=True)
            self._episode_marker = marker
            self._episode_query_count = 0
            task_env._flowpi_video_frame_ready = False  # noqa: SLF001
            task_env._flowpi_rendered_rgb = None  # noqa: SLF001
            self._install_observation_throttle(task_env, initial_observation)

    def _take_next_buffered_action(self) -> tuple[np.ndarray, str, int, int, bool]:
        with self._condition:
            if (self._current_chunk is None or self._current_index >= len(self._current_chunk)) and self._ready_chunks:
                self._current_chunk, _ = self._ready_chunks.popleft()
                self._current_index = 0
                self._condition.notify_all()
            if self._current_chunk is not None and self._current_index < len(self._current_chunk):
                action = self._current_chunk[self._current_index].copy()
                self._current_index += 1
                source = "fresh_chunk"
                chunk_d = len(self._current_chunk)
                chunk_has_remaining = self._current_index < len(self._current_chunk)
            elif self._last_action is not None:
                action = self._last_action.copy()
                source = "hold_last"
                chunk_d = 0
                chunk_has_remaining = False
            else:
                raise RuntimeError("FlowPi has not returned an initial action")
            self._last_action = action.copy()
            ready_depth = len(self._ready_chunks)
            return action, source, chunk_d, ready_depth, chunk_has_remaining

    def _actions_due_for_callback(self) -> int:
        """Return the bounded number of physical ticks this callback should catch up.

        RoboTwin calls the policy after ``get_obs()``. A multi-camera render can take longer than
        one 20 ms control period, so limiting a callback to one ``d``-chunk makes the physical
        clock fall behind even when direct SAPIEN stepping itself is fast. Accumulate elapsed wall
        time at the configured control rate and execute the corresponding number of exact physics
        intervals. The cap prevents a delayed callback from turning into an unbounded action burst.
        """
        if not self.drain_action_chunk or not self.wall_clock_catchup:
            return 1
        now = time.perf_counter()
        if self._last_step_start_s is None:
            self._last_step_start_s = now
            self._control_budget = 1.0
        else:
            elapsed_s = max(0.0, now - self._last_step_start_s)
            self._last_step_start_s = now
            self._control_budget = min(
                float(self.max_catchup_actions),
                self._control_budget + elapsed_s * self.control_hz,
            )
        # When RGB is throttled, get_obs returns immediately between camera frames. Sleep only
        # until the next physical control period instead of letting the evaluator spin at the
        # Python maximum and violate the 50-Hz simulator clock.
        while self._control_budget < 1.0:
            wait_s = (1.0 - self._control_budget) / self.control_hz
            if wait_s > 0:
                time.sleep(min(wait_s, 0.02))
            now = time.perf_counter()
            elapsed_s = max(0.0, now - self._last_step_start_s)
            self._last_step_start_s = now
            self._control_budget = min(
                float(self.max_catchup_actions),
                self._control_budget + elapsed_s * self.control_hz,
            )
        due = max(1, min(self.max_catchup_actions, int(self._control_budget)))
        self._control_budget -= due
        return due

    def _simulation_steps(self, task_env: Any) -> int:
        if self.sim_steps_override is not None:
            return self.sim_steps_override
        timestep = float(task_env.scene.get_timestep())
        if timestep <= 0:
            raise ValueError(f"DOMINO scene timestep must be positive, got {timestep}")
        return max(1, round(1.0 / (self.control_hz * timestep)))

    @staticmethod
    def _advance_video_capture(task_env: Any) -> None:
        """Render one shared sensor/video frame when the configured image sampler is due.

        The same RGB buffer is used by FlowPi and, when enabled, the video encoder.  This keeps
        video logging from causing a second three-camera render and also keeps non-video inference
        on the same 10-Hz observation clock.
        """
        if not getattr(task_env, "_flowpi_sensor_capture_enabled", False):
            return
        timestep = float(task_env.scene.get_timestep())
        if timestep <= 0:
            raise ValueError(f"DOMINO scene timestep must be positive, got {timestep}")
        task_env._flowpi_sensor_elapsed += timestep  # noqa: SLF001
        capture_period = float(task_env._flowpi_sensor_period_s)  # noqa: SLF001
        while task_env._flowpi_sensor_elapsed + 1e-9 >= task_env._flowpi_sensor_next_capture:  # noqa: SLF001
            task_env._update_render()  # noqa: SLF001
            task_env.cameras.update_picture()
            rendered_rgb = task_env.cameras.get_rgb()
            task_env._flowpi_rendered_rgb = rendered_rgb  # noqa: SLF001
            task_env._flowpi_video_frame_ready = True  # noqa: SLF001
            ffmpeg = getattr(task_env, "eval_video_ffmpeg", None)
            if ffmpeg is not None and ffmpeg.stdin is not None:
                head_rgb = rendered_rgb["head_camera"]["rgb"]
                ffmpeg.stdin.write(head_rgb.tobytes())
                task_env._eval_video_last_capture = task_env._flowpi_sensor_elapsed  # noqa: SLF001
            task_env._flowpi_sensor_next_capture += capture_period  # noqa: SLF001

    def _execute_direct(self, task_env: Any, action: np.ndarray) -> tuple[float, int]:
        """Apply one qpos target and advance exactly one configured control interval."""
        action_t0 = time.perf_counter()
        if task_env.take_action_cnt == task_env.step_lim or task_env.eval_success:
            return 0.0, 0
        task_env.take_action_cnt += 1
        print(f"step: \033[92m{task_env.take_action_cnt} / {task_env.step_lim}\033[0m", end="\r")

        action = np.asarray(action, dtype=np.float32)
        robot = task_env.robot
        robot.set_arm_joints(action[:6], np.zeros(6, dtype=np.float32), "left")
        robot.set_gripper(action[6], "left", gripper_eps=0.0)
        robot.set_arm_joints(action[7:13], np.zeros(6, dtype=np.float32), "right")
        robot.set_gripper(action[13], "right", gripper_eps=0.0)

        sim_steps = self._simulation_steps(task_env)
        for _ in range(sim_steps):
            task_env._update_kinematic_tasks()  # noqa: SLF001
            task_env.scene.step()
            if not getattr(task_env, "transient_event", False):
                task_env._update_transient_checks()  # noqa: SLF001
            # Rendering is deferred to the next get_obs() in the normal no-video path. For video,
            # render only at the configured capture cadence; calling DOMINO's stock helper on
            # every physics step needlessly serializes the 50 Hz control loop with ffmpeg.
            self._advance_video_capture(task_env)
            if task_env.check_success():
                task_env.eval_success = True
                task_env.get_obs()
                task_env._record_metrics_step()  # noqa: SLF001
                return (time.perf_counter() - action_t0) * 1000, sim_steps

        if getattr(task_env, "render_freq", 0) and task_env.take_action_cnt % task_env.render_freq == 0:
            task_env._update_render()  # noqa: SLF001
            task_env.viewer.render()
        task_env._record_metrics_step()  # noqa: SLF001
        return (time.perf_counter() - action_t0) * 1000, sim_steps

    def _execute(self, task_env: Any, action: np.ndarray) -> tuple[float, int]:
        if self.direct_control:
            return self._execute_direct(task_env, action)
        action_t0 = time.perf_counter()
        task_env.take_action(action, action_type="qpos")
        return (time.perf_counter() - action_t0) * 1000, 0

    @staticmethod
    def _observation_from_rendered_cameras(task_env: Any) -> dict[str, Any] | None:
        """Build the minimal FlowPi observation from an already-rendered camera buffer."""
        if not getattr(task_env, "_flowpi_video_frame_ready", False):
            return None
        try:
            rgb = getattr(task_env, "_flowpi_rendered_rgb", None)
            if rgb is None:
                rgb = task_env.cameras.get_rgb()
            camera_names = ("head_camera", "left_camera", "right_camera")
            if any(name not in rgb or "rgb" not in rgb[name] for name in camera_names):
                return None
            left_state = task_env.robot.get_left_arm_jointState()
            right_state = task_env.robot.get_right_arm_jointState()
            state = np.asarray(left_state + right_state, dtype=np.float32)
            if state.shape != (14,):
                return None
            observation = {
                "observation": {
                    name: {"rgb": rgb[name]["rgb"]}
                    for name in camera_names
                },
                "joint_action": {"vector": state},
            }
            task_env._flowpi_video_frame_ready = False  # noqa: SLF001
            task_env._flowpi_rendered_rgb = None  # noqa: SLF001
            task_env.now_obs = observation
            return observation
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def _publish_ready_video_observation(self, task_env: Any, instruction: str | None) -> None:
        """Submit a video-sampled RGB frame without waiting for the evaluator's next get_obs."""
        observation = self._observation_from_rendered_cameras(task_env)
        if observation is None:
            return
        cache = getattr(task_env, "_flowpi_observation_cache", None)
        if cache is not None:
            cache["observation"] = observation
            cache["timestamp_s"] = time.perf_counter()
        self._submit_observation(observation, instruction, int(task_env.take_action_cnt))

    def step(self, task_env: Any, observation: dict[str, Any]) -> None:
        self._raise_worker_error()
        self._begin_episode_if_needed(task_env, observation)
        instruction = task_env.get_instruction()
        inline_query = self._episode_query_count == 0
        if inline_query:
            # A synchronous first query establishes the initial action chunk and avoids a cold
            # start where the simulator has no safe action to hold.
            actions = self._infer(
                [
                    _slim_observation(
                        observation,
                        copy_arrays=self.copy_observations,
                        compress_rgb=self.compress_rgb,
                        jpeg_quality=self.jpeg_quality,
                        control_tick=int(task_env.take_action_cnt),
                    )
                ],
                instruction,
                source="warm_start",
            )
            with self._condition:
                self._current_chunk = actions
                self._current_index = 0
                self._max_ready_depth = max(self._max_ready_depth, 0)
            self._last_image_observation = observation
            self._episode_query_count = 1
        else:
            # Publish this physical frame before consuming the next action. The worker drains all
            # accumulated frames in one request, preserving the camera history while it runs.
            self._submit_observation(observation, instruction, int(task_env.take_action_cnt))

        # The evaluator invokes us after every get_obs(). Once a fresh chunk is available, drain
        # its remaining actions without forcing another render/RPC round trip between them. Each
        # action still advances the simulator and records its own control event, so success and
        # per-step metrics retain their original granularity.
        first_action = True
        hold_last_actions = 0
        actions_due = self._actions_due_for_callback()
        actions_run = 0
        while True:
            action_start = time.perf_counter()
            action, action_source, chunk_d, ready_depth, chunk_has_remaining = self._take_next_buffered_action()
            if action_source == "fresh_chunk":
                hold_last_actions = 0
            else:
                hold_last_actions += 1
            action_execute_ms, sim_steps = self._execute(task_env, action)
            action_end = time.perf_counter()
            event = {
                "control_index": len(_METRICS_EVENTS),
                "timestamp_s": action_start - _METRICS_START,
                "end_to_end_ms": (action_end - action_start) * 1000,
                "action_execute_ms": action_execute_ms,
                "actions_emitted": 1,
                "action_source": action_source,
                "action_chunk_d": chunk_d,
                "ready_chunks": ready_depth,
                "sim_steps": sim_steps,
                "policy_query_completed_inline": inline_query and first_action,
                "batched_in_callback": not first_action,
            }
            with _METRICS_LOCK:
                _METRICS_EVENTS.append(event)
                _METRICS_CONTROL_TIMES.append(event["timestamp_s"])
                _METRICS_ACTION_EXECUTE.append(action_execute_ms)
            actions_run += 1
            # In video mode _advance_video_capture has already rendered the next 10-Hz frame
            # inside this physical action. Publish it immediately; waiting for the evaluator's
            # subsequent get_obs call can shift the frame by one control period and lower the
            # measured image/flow rate without improving freshness.
            self._publish_ready_video_observation(task_env, instruction)
            # This is the actual fast closed-loop trigger: read the new proprioception after the
            # simulator interval and let the background worker run one state-only NFE. It is
            # intentionally independent of the RGB/image-update RPC above.
            if task_env.take_action_cnt > 0 and not task_env.eval_success:
                self._submit_fast_state(task_env, instruction)
            first_action = False

            terminal = task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success
            if self.wall_clock_catchup:
                continue_callback = actions_run < actions_due
            else:
                continue_hold_last = (
                    action_source == "hold_last"
                    and hold_last_actions < self.hold_last_burst
                    and not terminal
                )
                continue_callback = chunk_has_remaining or continue_hold_last
            if not (
                self.drain_action_chunk
                and continue_callback
                and not terminal
            ):
                break

        # DOMINO does not issue a second reset RPC after the final episode. Stop the periodic
        # latest-state worker here so it cannot keep consuming the cached state while the
        # evaluator is closing the video/server.
        if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success:
            self.stop_worker(clear_buffers=False)

        with _METRICS_LOCK:
            should_flush = len(_METRICS_EVENTS) % _METRICS_FLUSH_EVERY == 0
        if should_flush:
            _write_metrics()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.stop_worker(clear_buffers=True)
        finally:
            _close_model_connection(self._image_model)
            _close_model_connection(self._state_model)


def _get_controller(model: Any) -> _ContinuousPolicyController:
    controller = getattr(model, "_flowpi_domino_controller", None)
    if controller is None or getattr(controller, "_closed", False):
        controller = _ContinuousPolicyController(model)
    return controller


def eval(task_env: Any, model: Any, observation: dict[str, Any]) -> None:
    """Consume a ready action chunk and pipeline the next FlowPi query."""
    _get_controller(model).step(task_env, observation)


def reset_model(model: Any) -> None:
    """Stop any local query worker; the evaluator sends the actual reset RPC to the server."""
    controller = getattr(model, "_flowpi_domino_controller", None)
    if controller is not None:
        controller.stop_worker(clear_buffers=True)
