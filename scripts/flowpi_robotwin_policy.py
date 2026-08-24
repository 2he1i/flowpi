"""RoboTwin adapter for FlowPi π0.5 inference.

The RoboTwin evaluator imports a policy module and calls ``get_model``, ``eval`` and
``reset_model``. This adapter returns a latency-sized action chunk. The FlowPi runtime maintains
the frame history, computes online SEA-RAFT flow, and refreshes the slow prefix asynchronously;
the chunk width ``d`` is estimated from measured query latency in control ticks, as in the
original πR² deployment.

GPU roles in the default three-card layout:

* JAX slow prefix/VLM replica: ``gpu:0``
* JAX fast action-expert replica: ``gpu:1``
* Torch SEA-RAFT: ``cuda:2``

The fourth visible GPU is not selected by this module. Set ``FLOWPI_*`` environment variables or
the matching fields in ``flowpi_robotwin_deploy.yml`` to change paths/devices.
"""

from __future__ import annotations

import atexit
import base64
from contextlib import suppress
import dataclasses
import json
import os
import pathlib
from queue import Empty
from queue import Full
from queue import Queue
import sys
import threading
import time
from typing import Any

# JAX must see the memory policy before it is imported by openpi. RoboTwin already imports torch,
# but that does not initialize JAX, so setting this here still works when the adapter is imported
# by script/eval_policy.py.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import cv2
import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

import flowpi_checkpoint as _checkpoint_config  # noqa: E402

import openpi.models.model as _model  # noqa: E402
import openpi.policies.aloha_policy as _aloha_policy  # noqa: E402
import openpi.policies.flowpi_runtime as _flowpi_runtime  # noqa: E402
import openpi.training.checkpoints as _checkpoints  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.sea_raft as _sea_raft  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402

_DEFAULT_CONFIG_NAME = "flowpi_aloha"
_DEFAULT_SLOW_DEVICE = "gpu:0"
_DEFAULT_FAST_DEVICE = "gpu:1"
_DEFAULT_SEA_RAFT_DEVICE = "cuda:2"
_DEFAULT_SEA_RAFT_PRECISION = "fp16"
_DEFAULT_ASYNC_FLOW = True
_DEFAULT_FLOW_PROCESS = True
_DEFAULT_SLOW_EVERY_N = 10
_DEFAULT_CONTROL_HZ = 50.0
_DEFAULT_INITIAL_D = 1
_DEFAULT_D_WINDOW = 8
_DEFAULT_DEPLOY_D_MAX = 3
_DEFAULT_PREWARM_RUNTIME = True
_FLOW_CAMERAS = (
    ("head_camera", "cam_high"),
    ("left_camera", "cam_left_wrist"),
    ("right_camera", "cam_right_wrist"),
)
_RUNTIME_CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_JPEG_IMAGE_KEY = "__flowpi_jpeg_rgb__"
_CONTROL_TICK_KEY = "__flowpi_control_tick__"


def _setting(usr_args: dict[str, Any], key: str, env_key: str, default: Any = None) -> Any:
    value = usr_args.get(key)
    if value is None or value == "":
        value = os.environ.get(env_key, default)
    return value


def _extract_control_tick(observation: dict[str, Any] | None) -> int | None:
    """Read the optional simulator control clock attached by the DOMINO client."""
    if not isinstance(observation, dict):
        return None
    value = observation.get(_CONTROL_TICK_KEY)
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{_CONTROL_TICK_KEY} must be an integer, got {value!r}") from exc
    if value < 0:
        raise ValueError(f"{_CONTROL_TICK_KEY} must be non-negative, got {value}")
    return value


def _bool_setting(usr_args: dict[str, Any], key: str, env_key: str, *, default: bool) -> bool:
    value = _setting(usr_args, key, env_key, default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean for {key}/{env_key}, got {value!r}")


def _path_setting(usr_args: dict[str, Any], key: str, env_key: str, default: pathlib.Path | None = None) -> pathlib.Path:
    value = _setting(usr_args, key, env_key, default)
    if value is None:
        raise ValueError(
            f"Missing {key!r}. Pass it in the RoboTwin config/overrides or set {env_key}."
        )
    return pathlib.Path(value).expanduser().resolve()


def _resize_rgb(image: Any, image_size: tuple[int, int]) -> np.ndarray:
    """Convert one RoboTwin camera frame to full-resolution uint8 HWC RGB."""
    if isinstance(image, dict) and _JPEG_IMAGE_KEY in image:
        payload = base64.b64decode(image[_JPEG_IMAGE_KEY])
        encoded = np.frombuffer(payload, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode a FlowPi JPEG RGB observation")
        image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an RGB image with 3 dimensions, got shape {image.shape}")

    if image.shape[-1] == 3:
        hwc = image
    elif image.shape[0] == 3:
        hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Expected HWC or CHW RGB input, got shape {image.shape}")

    if np.issubdtype(hwc.dtype, np.floating):
        minimum = float(np.nanmin(hwc))
        maximum = float(np.nanmax(hwc))
        if minimum >= -1.0 and maximum <= 1.0:
            hwc = (hwc + 1.0) * 127.5 if minimum < 0 else hwc * 255.0
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)
    elif hwc.dtype != np.uint8:
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)

    height, width = image_size
    if hwc.shape[:2] != (height, width):
        # The training recipe and SEA-RAFT both use this exact geometry. RoboTwin camera configs
        # normally already produce 480x640; resizing here keeps the adapter explicit and makes a
        # different camera preset fail neither at the flow extractor nor at the model preprocessor.
        hwc = cv2.resize(hwc, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(hwc)


def _make_data_config(train_config: _config.TrainConfig, checkpoint: pathlib.Path):
    """Build only the robot/model transforms; never instantiate training flow-cache transforms."""
    data_factory = train_config.data
    flow_factory = getattr(data_factory, "flow", None)
    if flow_factory is None:
        raise ValueError(
            "The selected training config has no data.flow settings. Use the FlowPi training config "
            "that produced the checkpoint (normally flowpi_aloha)."
        )

    checkpoint_assets = _checkpoints.resolve_checkpoint_assets_dir(checkpoint)
    if checkpoint_assets is not None and hasattr(data_factory, "assets"):
        data_factory = dataclasses.replace(
            data_factory,
            assets=dataclasses.replace(data_factory.assets, assets_dir=str(checkpoint_assets)),
        )

    # The online runtime owns both flow and slow-delay scheduling. Setting flow=None here retains
    # AlohaInputs/DeltaActions and the prompt tokenizer but avoids loading an offline flow cache.
    data_config = dataclasses.replace(data_factory, flow=None).create(train_config.assets_dirs, train_config.model)
    return data_config, flow_factory


class FlowPiRoboTwinPolicy:
    """Stateful one-step policy wrapper consumed by RoboTwin's evaluator."""

    def __init__(self, usr_args: dict[str, Any]):
        checkpoint = _path_setting(usr_args, "flowpi_checkpoint", "FLOWPI_CHECKPOINT")
        if not checkpoint.exists():
            raise FileNotFoundError(f"FlowPi checkpoint does not exist: {checkpoint}")
        self._checkpoint = checkpoint

        config_name = _setting(usr_args, "flowpi_config_name", "FLOWPI_CONFIG_NAME", _DEFAULT_CONFIG_NAME)
        train_config = _checkpoint_config.align_train_config_with_checkpoint(
            _config.get_config(config_name), checkpoint
        )
        flow_config = train_config.model.flow
        if flow_config is None or not flow_config.enabled:
            raise ValueError(f"Training config {config_name!r} does not have model.flow.enabled=True")

        data_config, flow_factory = _make_data_config(train_config, checkpoint)
        sea_raft_checkpoint = _path_setting(
            usr_args,
            "flowpi_sea_raft_ckpt",
            "FLOWPI_SEA_RAFT_CKPT",
            _sea_raft.default_inference_checkpoint(),
        )
        if not sea_raft_checkpoint.is_file():
            raise FileNotFoundError(
                f"Inference SEA-RAFT checkpoint not found: {sea_raft_checkpoint}. "
                "Pass flowpi_sea_raft_ckpt or set FLOWPI_SEA_RAFT_CKPT."
            )

        self._slow_device = str(
            _setting(usr_args, "flowpi_slow_device", "FLOWPI_SLOW_DEVICE", _DEFAULT_SLOW_DEVICE)
        )
        self._fast_device = str(
            _setting(usr_args, "flowpi_fast_device", "FLOWPI_FAST_DEVICE", _DEFAULT_FAST_DEVICE)
        )
        self._sea_raft_device = str(
            _setting(usr_args, "flowpi_sea_raft_device", "FLOWPI_SEA_RAFT_DEVICE", _DEFAULT_SEA_RAFT_DEVICE)
        )
        self._sea_raft_precision = str(
            _setting(
                usr_args,
                "flowpi_sea_raft_precision",
                "FLOWPI_SEA_RAFT_PRECISION",
                _DEFAULT_SEA_RAFT_PRECISION,
            )
        )
        self._async_flow = _bool_setting(
            usr_args,
            "flowpi_async_flow",
            "FLOWPI_ASYNC_FLOW",
            default=_DEFAULT_ASYNC_FLOW,
        )
        self._flow_process = _bool_setting(
            usr_args,
            "flowpi_flow_process",
            "FLOWPI_FLOW_PROCESS",
            default=_DEFAULT_FLOW_PROCESS,
        )
        self._prewarm_runtime = _bool_setting(
            usr_args,
            "flowpi_prewarm_runtime",
            "FLOWPI_PREWARM_RUNTIME",
            default=_DEFAULT_PREWARM_RUNTIME,
        )
        self._slow_every_n = int(
            _setting(usr_args, "flowpi_slow_every_n", "FLOWPI_SLOW_EVERY_N", _DEFAULT_SLOW_EVERY_N)
        )
        if self._slow_every_n <= 0:
            raise ValueError("flowpi_slow_every_n/FLOWPI_SLOW_EVERY_N must be positive")
        self._control_hz = float(
            _setting(usr_args, "flowpi_control_hz", "FLOWPI_CONTROL_HZ", _DEFAULT_CONTROL_HZ)
        )
        if self._control_hz <= 0:
            raise ValueError("flowpi_control_hz/FLOWPI_CONTROL_HZ must be positive")
        self._d_window = int(
            _setting(usr_args, "flowpi_d_window", "FLOWPI_D_WINDOW", _DEFAULT_D_WINDOW)
        )
        if self._d_window <= 0:
            raise ValueError("flowpi_d_window/FLOWPI_D_WINDOW must be positive")
        self._initial_d = int(
            _setting(usr_args, "flowpi_initial_d", "FLOWPI_INITIAL_D", _DEFAULT_INITIAL_D)
        )
        self._deployment_d_max = int(
            _setting(usr_args, "flowpi_deploy_d_max", "FLOWPI_DEPLOY_D_MAX", _DEFAULT_DEPLOY_D_MAX)
        )
        if self._deployment_d_max < 1:
            raise ValueError("flowpi_deploy_d_max/FLOWPI_DEPLOY_D_MAX must be positive")
        # The checkpoint's trained support is the hard upper bound. Deployment intentionally uses
        # only the Pi-R2 widths 1/2/3 even when the training recipe was compiled with d_max=5.
        self._d_max = min(int(flow_config.d_max), self._deployment_d_max)
        if not 1 <= self._initial_d <= self._d_max:
            raise ValueError(f"flowpi_initial_d must be in [1, {self._d_max}], got {self._initial_d}")

        prompt = _setting(usr_args, "flowpi_prompt", "FLOWPI_PROMPT", None)
        self._default_prompt = None if prompt is None else str(prompt)
        # Tokenize the prompt once per episode. Re-running the full model-input transform from
        # the image worker competes with the fast-state RPC for the policy process GIL.
        self._prompt_observation_fields: dict[str, Any] | None = None

        # Load two independent JAX replicas with single-device Orbax sharding. Loading without
        # jax_device would replicate each checkpoint over all visible GPUs, defeating the three
        # channel layout and needlessly consuming the fourth card as well.
        print(f"[FlowPi/RoboTwin] loading slow VLM replica on {self._slow_device}")
        slow_model = _checkpoints.load_model_from_checkpoint(
            train_config.model,
            str(checkpoint),
            jax_device=self._slow_device,
        )
        print(f"[FlowPi/RoboTwin] loading fast action replica on {self._fast_device}")
        fast_model = _checkpoints.load_model_from_checkpoint(
            train_config.model,
            str(checkpoint),
            jax_device=self._fast_device,
        )

        self._data_config = data_config
        self._norm = _transforms.Normalize(
            data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
        self._data_input_transforms = tuple(data_config.data_transforms.inputs)
        # The runtime must receive full-resolution images for the ring buffer/SEA-RAFT. The model
        # preprocessor performs its own 224x224 resize, so drop only the data-pipeline resize.
        self._model_input_transforms = tuple(
            transform
            for transform in data_config.model_transforms.inputs
            if not isinstance(transform, _transforms.ResizeImages)
        )
        self._model_output_transforms = tuple(data_config.model_transforms.outputs)
        self._data_output_transforms = tuple(data_config.data_transforms.outputs)
        self._flow_image_size = tuple(flow_config.flow_image_size)

        # SEA-RAFT is an inference-only Torch worker in this adapter. Keep its host helper pools
        # bounded by default so they do not compete with JAX dispatch and the fast-state RPC;
        # callers can override the value for a different CPU topology.
        os.environ.setdefault("FLOWPI_SEA_RAFT_TORCH_THREADS", "1")
        self._runtime = _flowpi_runtime.FlowPiRuntime(
            fast_model,
            slow_model=slow_model,
            flow_config=flow_config,
            sea_raft_ckpt=str(sea_raft_checkpoint),
            sea_raft_variant=flow_factory.sea_raft_variant,
            sea_raft_iters=flow_factory.sea_raft_iters,
            sea_raft_device=self._sea_raft_device,
            sea_raft_precision=self._sea_raft_precision,
            jax_device=self._fast_device,
            slow_jax_device=self._slow_device,
            d=self._initial_d,
            precompile_d_max=self._d_max,
            async_flow=self._async_flow,
            flow_process=self._flow_process,
        )
        # DOMINO opens independent TCP connections for RGB/flow updates and state-only NFEs.
        # Fast ticks and episode transitions use this lock. Image-ring ingestion has a separate
        # lock below: it mutates only the frame ring/latest-image mailbox and submits immutable
        # SEA-RAFT snapshots, so it must not make a state-only NFE wait behind RGB copies.
        self._runtime_lock = threading.RLock()
        self._image_ingest_lock = threading.RLock()
        self._event_lock = threading.RLock()
        self._metrics_write_lock = threading.RLock()
        if self._prewarm_runtime:
            print(
                "[FlowPi/RoboTwin] prewarming streaming signatures before serving episodes "
                "(one-time JAX compile)"
            )
            self._runtime.prewarm(self._make_prewarm_observation(train_config.model, flow_config))
            print("[FlowPi/RoboTwin] runtime prewarm complete")
        metrics_path = _setting(usr_args, "flowpi_metrics_path", "FLOWPI_METRICS_PATH", None)
        self._metrics_path = pathlib.Path(metrics_path).expanduser() if metrics_path else None
        self._metrics_start = time.perf_counter()
        self._metrics_flush_every = int(
            _setting(usr_args, "flowpi_metrics_flush_every", "FLOWPI_METRICS_FLUSH_EVERY", 10)
        )
        if self._metrics_flush_every <= 0:
            raise ValueError("flowpi_metrics_flush_every/FLOWPI_METRICS_FLUSH_EVERY must be positive")
        self._request_events: list[dict[str, Any]] = []
        self._request_times_s: list[float] = []
        self._episode_count = 0
        self._episode_started = False
        self._control_ticks = 0
        # Image-frame clock is deliberately independent from DOMINO's physical control tick.
        # The latter can jump when the catch-up controller executes several simulator intervals
        # between RGB observations; using it as the flow ring index would mask nearly every
        # valid stride as a dropped frame.
        self._image_frame_tick = 0
        self._next_slow_refresh_tick = self._slow_every_n
        # Warm-start/JIT compilation is not a representative query latency. Only fast requests
        # participate in the deployment-time d estimator.
        self._fast_request_latencies_ms: list[float] = []
        self._fast_state_request_latencies_ms: list[float] = []
        self._last_latency_estimate_ms: float | None = None
        # Image RPCs are accepted quickly and processed in order by a bounded local worker. This
        # keeps the policy server's request handler free to return a fast-state response while
        # JPEG decode, model preprocessing, ring copies, and SEA-RAFT submission run elsewhere.
        self._image_update_queue: Queue[tuple[int, list[dict], str | None] | None] = Queue(maxsize=8)
        self._image_update_queue_lock = threading.RLock()
        self._image_update_execution_lock = threading.RLock()
        self._image_update_generation = 0
        self._image_update_stop = False
        self._image_update_error: BaseException | None = None
        self._image_update_drops = 0
        self._image_update_thread = threading.Thread(
            target=self._image_update_worker_loop,
            name="flowpi-policy-image-update",
            daemon=True,
        )
        self._image_update_thread.start()
        atexit.register(self.close)

    def _raise_image_update_error(self) -> None:
        with self._image_update_queue_lock:
            error = self._image_update_error
            self._image_update_error = None
        if error is not None:
            raise RuntimeError("FlowPi asynchronous image update failed") from error

    def _image_update_worker_loop(self) -> None:
        while True:
            item = self._image_update_queue.get()
            if item is None:
                return
            generation, history, prompt = item
            with self._image_update_execution_lock:
                with self._image_update_queue_lock:
                    if self._image_update_stop or generation != self._image_update_generation:
                        continue
                try:
                    self._update_with_prompt(history, prompt)
                except BaseException as exc:
                    with self._image_update_queue_lock:
                        if self._image_update_error is None:
                            self._image_update_error = exc

    def _stop_image_update_worker(self) -> None:
        worker = self._image_update_thread
        if worker is None:
            return
        with self._image_update_execution_lock, self._image_update_queue_lock:
            self._image_update_stop = True
            while True:
                try:
                    self._image_update_queue.get_nowait()
                except Empty:
                    break
            with suppress(Full):
                self._image_update_queue.put_nowait(None)
        if worker is not threading.current_thread():
            worker.join(timeout=10.0)
        self._image_update_thread = None

    def _metrics_payload(self) -> dict[str, Any]:
        with self._event_lock:
            request_events = list(self._request_events)
            request_times_s = list(self._request_times_s)
            last_latency_estimate_ms = self._last_latency_estimate_ms
        with self._runtime_lock:
            runtime_metrics = self._runtime.metrics_summary()
            action_chunk_d = self._runtime.action_chunk_d
            action_chunk_d_training_max = int(self._runtime.flow_config.d_max)
        request_latencies = [float(event["request_ms"]) for event in request_events]
        return {
            "schema_version": 1,
            "checkpoint": str(getattr(self, "_checkpoint", "")),
            "devices": {
                "slow_jax": self._slow_device,
                "fast_jax": self._fast_device,
                "sea_raft": self._sea_raft_device,
                "sea_raft_precision": self._sea_raft_precision,
            },
            "policy": {
                "request_count": len(request_events),
                "request_rate": _flowpi_runtime.summarize_rate(request_times_s),
                "latency_ms": _flowpi_runtime.summarize_series(request_latencies),
                "fast_state_latency_ms": _flowpi_runtime.summarize_series(
                    [event["request_ms"] for event in request_events if event.get("mode") == "fast_state"]
                ),
                "image_update_latency_ms": _flowpi_runtime.summarize_series(
                    [event["request_ms"] for event in request_events if event.get("mode") == "image_update"]
                ),
                "control_hz": self._control_hz,
                "action_chunk_d": action_chunk_d,
                "action_chunk_d_max": self._d_max,
                "action_chunk_d_training_max": action_chunk_d_training_max,
                "action_chunk_d_deployment_max": self._deployment_d_max,
                "action_chunk_d_history": list(
                    runtime_metrics["config"].get("action_chunk_d_history", [])
                ),
                "d_latency_window": self._d_window,
                "d_latency_estimate_ms": last_latency_estimate_ms,
                "fast_state_count": sum(event.get("mode") == "fast_state" for event in request_events),
                "image_update_count": sum(event.get("mode") == "image_update" for event in request_events),
                "fast_state_rate": _flowpi_runtime.summarize_rate(
                    [event["timestamp_s"] for event in request_events if event.get("mode") == "fast_state"]
                ),
                "image_update_rate": _flowpi_runtime.summarize_rate(
                    [event["timestamp_s"] for event in request_events if event.get("mode") == "image_update"]
                ),
                "events": request_events,
            },
            "runtime": runtime_metrics,
        }

    def _write_metrics(self, payload: dict[str, Any] | None = None) -> None:
        if self._metrics_path is None:
            return
        with self._metrics_write_lock:
            if payload is None:
                payload = self._metrics_payload()
            self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self._metrics_path.with_suffix(self._metrics_path.suffix + ".tmp")
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)
            os.replace(temporary_path, self._metrics_path)

    def _encode_observation(self, observation: dict, prompt: str | None) -> tuple[_model.Observation, np.ndarray]:
        images = {}
        for robotwin_key, model_key in _FLOW_CAMERAS:
            try:
                image = observation["observation"][robotwin_key]["rgb"]
            except KeyError as exc:
                raise KeyError(
                    f"RoboTwin observation is missing observation.{robotwin_key}.rgb; "
                    "enable rgb collection for head/left/right cameras."
                ) from exc
            rgb = _resize_rgb(image, self._flow_image_size)
            images[model_key] = np.transpose(rgb, (2, 0, 1))

        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"FlowPi Aloha preprocessing expects a 14-D RoboTwin qpos vector, got {state.shape}")
        prompt = prompt or self._default_prompt
        if not prompt:
            raise ValueError("RoboTwin returned no instruction; set flowpi_prompt/FLOWPI_PROMPT explicitly.")

        data: dict[str, Any] = {"images": images, "state": state, "prompt": str(prompt)}
        for transform in self._data_input_transforms:
            data = transform(data)

        for transform in (self._norm, *self._model_input_transforms):
            data = transform(data)
        # RoboTwin supplies one frame at a time, while the model/runtime contract is batched.
        # Keep an unbatched copy for AbsoluteActions during output decoding.
        model_state = np.asarray(data["state"], dtype=np.float32).copy()
        data = jax.tree.map(
            lambda value: np.asarray(value)[None, ...]
            if isinstance(value, (np.ndarray, np.generic))
            else value,
            data,
        )
        model_observation = _model.Observation.from_dict(data)
        # The lightweight image path reuses these immutable fields, so slow-prefix refreshes see
        # exactly the same language conditioning as the synchronous warm-start path.
        self._prompt_observation_fields = {
            field_name: getattr(model_observation, field_name)
            for field_name in (
                "tokenized_prompt",
                "tokenized_prompt_mask",
                "token_ar_mask",
                "token_loss_mask",
            )
        }
        return model_observation, model_state

    @staticmethod
    def _make_prewarm_observation(model_config: Any, flow_config: Any) -> _model.Observation:
        """Create a full-resolution, shape-compatible observation for server-side prewarm."""
        fake = model_config.fake_obs(batch_size=1)
        height, width = flow_config.flow_image_size
        images = {
            key: jnp.ones((1, height, width, 3), dtype=jnp.float32)
            for key in fake.images
        }
        return dataclasses.replace(fake, images=images)

    def _encode_runtime_observation(self, observation: dict) -> _model.Observation:
        """Encode an intermediate physical frame without model/tokenizer preprocessing.

        Intermediate history frames are consumed only by the runtime ring and SEA-RAFT. Running
        Aloha input transforms, prompt tokenization, padding, and model-side preparation on every
        one inflated the RPC encode time when a query accumulated several frames. The current
        episode's prompt fields are attached below so this same lightweight observation can also
        be the slow-prefix snapshot.
        """
        images = {}
        for (robotwin_key, _data_key), runtime_key in zip(_FLOW_CAMERAS, _RUNTIME_CAMERA_KEYS, strict=True):
            image = observation["observation"][robotwin_key]["rgb"]
            rgb = _resize_rgb(image, self._flow_image_size)
            # ``_encode_observation`` first uses the data-pipeline camera names (cam_high, ...)
            # and then maps them to the model contract. This lightweight path bypasses those
            # transforms, so it must emit the final model names directly.
            images[runtime_key] = np.asarray(rgb, dtype=np.float32)[None, ...] / 127.5 - 1.0
        # Keep the cached latest observation's state shape identical to the full warm-start
        # observation. State-only requests still pass through ``_encode_state`` independently,
        # while this small CPU-only transform avoids a (1, 14) versus (1, action_dim) mismatch
        # when the runtime replaces the cached state during a fast NFE.
        state = self._encode_state(observation["joint_action"]["vector"])[None, ...]
        mask = {key: np.ones((1,), dtype=bool) for key in images}
        prompt_fields = self._prompt_observation_fields
        if prompt_fields is None:
            raise RuntimeError("lightweight runtime image encoding requires a warm-start prompt")
        return _model.Observation(
            images=images,
            image_masks=mask,
            state=state,
            **prompt_fields,
        )

    def _ingest_image_frame(
        self,
        observation: _model.Observation,
        *,
        update_latest: bool,
        submit_flow: bool,
        source_control_tick: int | None,
        place_observation: bool = True,
    ) -> None:
        """Ingest one received RGB frame on the independent image clock."""
        # Do not take _runtime_lock here. The image path can copy three 480x640 frames and enqueue
        # SEA-RAFT work while the fast state path is running on the action-expert replica. The
        # runtime's ring, flow mailbox, and latest-observation mailbox have their own ownership
        # locks; this lock only preserves frame order between image RPC handler calls.
        with self._image_ingest_lock:
            source_frame_tick = self._image_frame_tick
            self._runtime.ingest_observation(
                observation,
                update_latest=update_latest,
                submit_flow=submit_flow,
                source_control_tick=source_control_tick,
                source_frame_tick=source_frame_tick,
                place_observation=place_observation,
            )
            self._image_frame_tick += 1

    def _encode_state(self, state: Any) -> np.ndarray:
        """Apply the same Aloha adaptation/normalization as the full observation path.

        The state-only RPC deliberately does not carry images. FlowPi's RoboTwin recipe uses a
        continuous state input, so tokenization and image transforms are unnecessary here; the
        latest full observation already owns those cached model fields inside ``FlowPiRuntime``.
        """
        state_array = np.asarray(state, dtype=np.float32).copy()
        if state_array.shape != (14,):
            raise ValueError(f"FlowPi state-only inference expects a 14-D qpos vector, got {state_array.shape}")
        aloha_inputs = next(
            (transform for transform in self._data_input_transforms if isinstance(transform, _aloha_policy.AlohaInputs)),
            None,
        )
        if aloha_inputs is not None:
            state_array = _aloha_policy._decode_state(  # noqa: SLF001
                state_array,
                adapt_to_pi=aloha_inputs.adapt_to_pi,
            )
        data: dict[str, Any] = {"state": state_array}
        data = self._norm(data)
        for transform in self._model_input_transforms:
            if isinstance(transform, _transforms.PadStatesAndActions):
                data = transform(data)
        return np.asarray(data["state"], dtype=np.float32)

    def _monotonic_control_tick(self, control_tick: int | None) -> int | None:
        """Clamp an out-of-order RPC timestamp to the runtime's current physical clock.

        The image and state sockets are intentionally independent. A stale full-image request
        can therefore arrive after a newer state-only NFE; allowing it to move the runtime clock
        backwards would fail the next tick. Clamping preserves the latest visual snapshot while
        keeping the authoritative control clock monotonic.
        """
        if control_tick is None:
            return None
        return max(int(control_tick), int(getattr(self._runtime, "_control_tick", 0)))

    def _estimate_action_chunk_d(self) -> int:
        """Convert recent measured query latency into a bounded πR² chunk width."""
        with self._event_lock:
            latency_samples = list(
                self._fast_state_request_latencies_ms or self._fast_request_latencies_ms
            )
        if not latency_samples:
            with self._runtime_lock:
                return self._runtime.action_chunk_d
        recent = latency_samples[-self._d_window :]
        latency_estimate_ms = float(np.mean(recent))
        with self._event_lock:
            self._last_latency_estimate_ms = latency_estimate_ms
        control_period_ms = 1000.0 / self._control_hz
        estimated = int(np.rint(latency_estimate_ms / control_period_ms))
        return int(np.clip(estimated, 1, self._d_max))

    def _decode_actions(self, action: np.ndarray, model_state: np.ndarray) -> np.ndarray:
        outputs: dict[str, Any] = {
            "state": np.asarray(model_state, dtype=np.float32).copy(),
            "actions": np.asarray(action, dtype=np.float32),
        }
        for transform in (
            *self._model_output_transforms,
            _transforms.Unnormalize(
                self._data_config.norm_stats,
                use_quantiles=self._data_config.use_quantile_norm,
            ),
            *self._data_output_transforms,
        ):
            outputs = transform(outputs)

        actions = np.asarray(outputs["actions"])
        if actions.ndim != 2 or actions.shape[-1] < 14:
            raise ValueError(f"FlowPi produced an invalid RoboTwin qpos shape: {actions.shape}")
        actions = np.asarray(actions[:, :14], dtype=np.float32)
        if not np.all(np.isfinite(actions)):
            raise FloatingPointError(f"FlowPi produced non-finite RoboTwin actions: {actions}")
        return actions

    def _act_with_prompt(
        self,
        observation: dict,
        prompt: str | None,
        observation_history: list[dict] | None = None,
    ) -> np.ndarray:
        request_t0 = time.perf_counter()
        request_start_s = request_t0 - self._metrics_start
        history = list(observation_history) if observation_history else [observation]
        if not history:
            history = [observation]
        control_tick = _extract_control_tick(history[-1])
        encode_t0 = request_t0
        # The DOMINO continuous client sends every physical control frame accumulated while the
        # previous RPC was in flight. Intermediate frames update the ring/SEA-RAFT mailbox only;
        # the latest frame is the one that runs the next fast NFE.
        if not self._episode_started:
            model_observation, model_state = self._encode_observation(history[0], prompt)
        else:
            for history_observation in history[:-1]:
                intermediate_observation = self._encode_runtime_observation(history_observation)
                intermediate_tick = _extract_control_tick(history_observation)
                self._ingest_image_frame(
                    intermediate_observation,
                    update_latest=False,
                    submit_flow=True,
                    source_control_tick=intermediate_tick,
                )
            model_observation, model_state = self._encode_observation(history[-1], prompt)
        encode_ms = (time.perf_counter() - encode_t0) * 1000
        runtime_t0 = time.perf_counter()
        mode = "warm_start"
        slow_refresh_requested = False
        d_used = self._runtime.action_chunk_d
        if not self._episode_started:
            with self._runtime_lock:
                self._runtime.warm_start(model_observation)
                # warm_start owns image frame 0.
                self._image_frame_tick = 1
            for history_observation in history[1:]:
                intermediate_observation = self._encode_runtime_observation(history_observation)
                intermediate_tick = _extract_control_tick(history_observation)
                self._ingest_image_frame(
                    intermediate_observation,
                    update_latest=False,
                    submit_flow=True,
                    source_control_tick=intermediate_tick,
                )
            with self._runtime_lock:
                action_chunk = self._runtime.emit()
            self._episode_started = True
            self._episode_count += 1
            self._control_ticks = int(action_chunk.shape[0])
            while self._next_slow_refresh_tick <= self._control_ticks:
                self._next_slow_refresh_tick += self._slow_every_n
        else:
            mode = "fast_tick"
            d_used = self._estimate_action_chunk_d()
            # Ingest the latest image on the image clock, then run the NFE against that exact
            # already-ingested snapshot. This avoids making the physical control tick double as
            # a camera-frame index when the DOMINO catch-up path skips intermediate RGB captures.
            with self._runtime_lock:
                effective_control_tick = self._monotonic_control_tick(control_tick)
                self._runtime.set_action_chunk_d(d_used)
                self._runtime.ingest_observation(
                    model_observation,
                    source_control_tick=effective_control_tick,
                    source_frame_tick=self._image_frame_tick,
                )
                self._image_frame_tick += 1
                action_chunk = self._runtime.tick(
                    model_observation,
                    observation_already_ingested=True,
                    control_tick=effective_control_tick,
                )
            if effective_control_tick is None:
                self._control_ticks += int(action_chunk.shape[0])
            else:
                # DOMINO's physical clock is authoritative even when a client-side future
                # action chunk is later coalesced before execution.
                self._control_ticks = max(self._control_ticks, int(effective_control_tick))
            if self._control_ticks >= self._next_slow_refresh_tick:
                # Submit after the fast NFE has finished. The worker runs on the slow JAX replica
                # while RoboTwin executes this qpos command and while the next frame is rendered.
                self._runtime.refresh_prefix()
                slow_refresh_requested = True
                while self._next_slow_refresh_tick <= self._control_ticks:
                    self._next_slow_refresh_tick += self._slow_every_n
        runtime_ms = (time.perf_counter() - runtime_t0) * 1000
        decode_t0 = time.perf_counter()
        decoded_actions = self._decode_actions(action_chunk, model_state)
        decode_ms = (time.perf_counter() - decode_t0) * 1000
        request_ms = (time.perf_counter() - request_t0) * 1000
        if mode == "fast_tick":
            self._fast_request_latencies_ms.append(request_ms)
        next_d = self._estimate_action_chunk_d()
        self._request_times_s.append(request_start_s)
        self._request_events.append(
            {
                "request_index": len(self._request_events),
                "timestamp_s": request_start_s,
                "request_interval_ms": (
                    (request_start_s - self._request_times_s[-2]) * 1000
                    if len(self._request_times_s) > 1
                    else None
                ),
                "episode": self._episode_count,
                "mode": mode,
                "history_frames": len(history),
                "runtime_tick": len(self._runtime.telemetry),
                "action_chunk_d": d_used,
                "actions_emitted": int(decoded_actions.shape[0]),
                "next_action_chunk_d": next_d,
                "latency_estimate_ms": self._last_latency_estimate_ms,
                "slow_refresh_requested": slow_refresh_requested,
                "encode_ms": encode_ms,
                "runtime_ms": runtime_ms,
                "decode_ms": decode_ms,
                "request_ms": request_ms,
            }
        )
        if len(self._request_events) % self._metrics_flush_every == 0:
            self._write_metrics()
        return decoded_actions

    def _update_with_prompt(
        self,
        observation_history: list[dict],
        prompt: str | None,
    ) -> dict[str, Any]:
        """Ingest full RGB frames and submit SEA-RAFT work without running an NFE."""
        if not self._episode_started:
            raise RuntimeError("update_observation requires an episode started by infer")
        if not observation_history:
            raise ValueError("update_observation requires at least one observation")
        request_t0 = time.perf_counter()
        request_start_s = request_t0 - self._metrics_start
        encode_t0 = request_t0
        last_history_index = len(observation_history) - 1
        for history_index, history_observation in enumerate(observation_history):
            intermediate_observation = self._encode_runtime_observation(history_observation)
            intermediate_tick = _extract_control_tick(history_observation)
            # Ingest each physical frame separately so SEA-RAFT can consume the newest valid
            # stride while the rest of the accumulated history is being encoded. The lock is
            # deliberately per frame: a state-only NFE never waits for the whole image batch.
            self._ingest_image_frame(
                intermediate_observation,
                update_latest=history_index == last_history_index,
                submit_flow=True,
                source_control_tick=intermediate_tick,
                place_observation=False,
            )
        # The latest frame is intentionally ingested through the same lightweight path. The
        # current episode's tokenized prompt is immutable and was cached by the synchronous
        # warm-start; state-only NFEs replace this raw state with the normalized state from
        # ``_encode_state``. Thus no model-input semantics are lost, while the image worker no
        # longer runs the tokenizer/Aloha repack on every RGB update.
        encode_ms = (time.perf_counter() - encode_t0) * 1000
        runtime_ms = 0.0
        request_ms = (time.perf_counter() - request_t0) * 1000
        with self._event_lock:
            self._request_times_s.append(request_start_s)
            self._request_events.append(
                {
                    "request_index": len(self._request_events),
                    "timestamp_s": request_start_s,
                    "request_interval_ms": (
                        (request_start_s - self._request_times_s[-2]) * 1000
                        if len(self._request_times_s) > 1
                        else None
                    ),
                    "episode": self._episode_count,
                    "mode": "image_update",
                    "history_frames": len(observation_history),
                    "runtime_frame": int(getattr(self._runtime, "_frame_index", -1)),
                    "encode_ms": encode_ms,
                    "runtime_ms": runtime_ms,
                    "request_ms": request_ms,
                }
            )
            should_flush = len(self._request_events) % self._metrics_flush_every == 0
        if should_flush:
            self._write_metrics()
        return {
            "updated": True,
            "image_frames_ingested": len(observation_history),
            "runtime_frame": int(getattr(self._runtime, "_frame_index", -1)),
        }

    def _infer_fast_state(self, state: Any, control_tick: int | None = None) -> np.ndarray:
        """Run one NFE from a fresh proprioceptive state and cached visual features."""
        if not self._episode_started:
            raise RuntimeError("infer_fast_state requires an episode started by infer")
        request_t0 = time.perf_counter()
        request_start_s = request_t0 - self._metrics_start
        encode_t0 = request_t0
        model_state = self._encode_state(state)
        encode_ms = (time.perf_counter() - encode_t0) * 1000
        d_used = self._estimate_action_chunk_d()
        runtime_t0 = time.perf_counter()
        with self._runtime_lock:
            effective_control_tick = self._monotonic_control_tick(control_tick)
            self._runtime.set_action_chunk_d(d_used)
            action_chunk = self._runtime.tick_state(model_state, control_tick=effective_control_tick)
            if effective_control_tick is None:
                # The DOMINO latest-state worker may run a periodic NFE while no newer
                # simulator observation has arrived. Runtime-internal tick/RNG state still
                # advances, but slow-prefix cadence must remain tied to the explicit physical
                # simulator clock and must not turn a 25 Hz fast stream into a 25 Hz VLM stream.
                slow_refresh_requested = False
            else:
                # Keep slow-refresh cadence tied to the simulator's actual control tick rather
                # than to every speculative chunk generated by the latest-state worker.
                self._control_ticks = max(self._control_ticks, int(effective_control_tick))
                slow_refresh_requested = False
                if self._control_ticks >= self._next_slow_refresh_tick:
                    # The slow prefix intentionally refreshes from the latest full image snapshot.
                    # The new proprioception is already consumed by the fast action expert and
                    # should not force a costly VLM/image RPC.
                    self._runtime.refresh_prefix()
                    slow_refresh_requested = True
                    while self._next_slow_refresh_tick <= self._control_ticks:
                        self._next_slow_refresh_tick += self._slow_every_n
            runtime_tick = len(self._runtime.telemetry)
        runtime_ms = (time.perf_counter() - runtime_t0) * 1000
        decode_t0 = time.perf_counter()
        decoded_actions = self._decode_actions(action_chunk, model_state)
        decode_ms = (time.perf_counter() - decode_t0) * 1000
        request_ms = (time.perf_counter() - request_t0) * 1000
        with self._event_lock:
            self._fast_request_latencies_ms.append(request_ms)
            self._fast_state_request_latencies_ms.append(request_ms)
        next_d = self._estimate_action_chunk_d()
        with self._event_lock:
            latency_estimate_ms = self._last_latency_estimate_ms
            self._request_times_s.append(request_start_s)
            self._request_events.append(
                {
                    "request_index": len(self._request_events),
                    "timestamp_s": request_start_s,
                    "request_interval_ms": (
                        (request_start_s - self._request_times_s[-2]) * 1000
                        if len(self._request_times_s) > 1
                        else None
                    ),
                    "episode": self._episode_count,
                    "mode": "fast_state",
                    "history_frames": 0,
                    "runtime_tick": runtime_tick,
                    "action_chunk_d": d_used,
                    "actions_emitted": int(decoded_actions.shape[0]),
                    "next_action_chunk_d": next_d,
                    "latency_estimate_ms": latency_estimate_ms,
                    "slow_refresh_requested": slow_refresh_requested,
                    "encode_ms": encode_ms,
                    "runtime_ms": runtime_ms,
                    "decode_ms": decode_ms,
                    "request_ms": request_ms,
                }
            )
            should_flush = len(self._request_events) % self._metrics_flush_every == 0
        if should_flush:
            self._write_metrics()
        return decoded_actions

    def act(self, task_env: Any, observation: dict) -> np.ndarray:
        # The in-process evaluator API is one-action-at-a-time. The DOMINO RPC path uses
        # ``infer`` below and consumes the full chunk asynchronously.
        return self._act_with_prompt(observation, task_env.get_instruction())[0]

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Serve one FlowPi action through DOMINO's lightweight TCP protocol.

        DOMINO sends the raw RoboTwin observation and instruction to the policy process. The
        returned chunk contains the ``d`` actions that the DOMINO client can consume while the
        next query runs in the background.
        """
        if not isinstance(request, dict):
            raise TypeError("infer request must be a dictionary")
        self._raise_image_update_error()
        observation = request.get("observation")
        instruction = request.get("instruction")
        observation_history = request.get("observation_history")
        if not isinstance(observation, dict):
            raise TypeError("infer request requires an observation dictionary")
        if instruction is not None and not isinstance(instruction, str):
            raise TypeError("infer request instruction must be a string or None")
        if observation_history is None:
            observation_history = [observation]
        if not isinstance(observation_history, list) or not observation_history:
            raise TypeError("infer observation_history must be a non-empty list when provided")
        if not all(isinstance(item, dict) for item in observation_history):
            raise TypeError("infer observation_history entries must be observation dictionaries")
        actions = self._act_with_prompt(observation_history[-1], instruction, observation_history)
        return {
            "actions": actions,
            "pi0_step": int(actions.shape[0]),
            "action_chunk_d": int(self._runtime.action_chunk_d),
        }

    def update_observation(self, request: dict[str, Any]) -> dict[str, Any]:
        """Enqueue a full simulator observation without blocking the fast RPC handler."""
        if not isinstance(request, dict):
            raise TypeError("update_observation request must be a dictionary")
        self._raise_image_update_error()
        observation = request.get("observation")
        history = request.get("observation_history")
        instruction = request.get("instruction")
        if history is None:
            history = [observation]
        if not isinstance(history, list) or not history or not all(isinstance(item, dict) for item in history):
            raise TypeError("update_observation observation_history must be a non-empty list")
        if instruction is not None and not isinstance(instruction, str):
            raise TypeError("update_observation instruction must be a string or None")
        with self._image_update_queue_lock:
            if self._image_update_stop:
                raise RuntimeError("FlowPi image-update worker is stopping")
            while True:
                try:
                    self._image_update_queue.put_nowait(
                        (self._image_update_generation, history, instruction)
                    )
                    break
                except Full:
                    with suppress(Empty):
                        self._image_update_queue.get_nowait()
                    self._image_update_drops += 1
        return {
            "updated": True,
            "queued": True,
            "image_frames_ingested": len(history),
            "runtime_frame": int(getattr(self._runtime, "_frame_index", -1)),
        }

    def infer_fast_state(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run one fast closed-loop NFE from a small proprioception-only request."""
        if not isinstance(request, dict):
            raise TypeError("infer_fast_state request must be a dictionary")
        self._raise_image_update_error()
        if "state" not in request:
            raise TypeError("infer_fast_state request requires state")
        actions = self._infer_fast_state(request["state"], _extract_control_tick(request))
        return {
            "actions": actions,
            "pi0_step": int(actions.shape[0]),
            "action_chunk_d": int(self._runtime.action_chunk_d),
        }

    def reset_model(self) -> None:
        """RPC-compatible episode reset hook used by DOMINO's client evaluator."""
        self.reset()

    def reset(self) -> None:
        # Wait for an in-flight image batch to finish, then invalidate queued batches before the
        # next warm_start. Otherwise a delayed previous-episode frame could enter the new flow
        # ring after its episode reset.
        with self._image_update_execution_lock, self._image_update_queue_lock:
            self._image_update_generation += 1
            while True:
                try:
                    self._image_update_queue.get_nowait()
                except Empty:
                    break
            self._episode_started = False
            self._control_ticks = 0
            self._image_frame_tick = 0
            self._next_slow_refresh_tick = self._slow_every_n

    def close(self) -> None:
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            try:
                self._stop_image_update_worker()
                runtime.close()
            finally:
                metrics = self._metrics_payload()
                self._write_metrics(metrics)
                print(
                    _flowpi_runtime.format_metrics_table(
                        metrics,
                        title="FlowPi RoboTwin 推理指标 / Inference Metrics",
                    ),
                    flush=True,
                )
                self._runtime = None


def get_model(usr_args: dict[str, Any]) -> FlowPiRoboTwinPolicy:
    return FlowPiRoboTwinPolicy(usr_args)


def eval(task_env: Any, model: FlowPiRoboTwinPolicy, observation: dict) -> None:
    action = model.act(task_env, observation)
    # RoboTwin's qpos action layout is [left arm(6), left gripper, right arm(6), right gripper].
    # Keep this one-step: the evaluator calls eval once per environment control step.
    task_env.take_action(action, action_type="qpos")


def reset_model(model: FlowPiRoboTwinPolicy) -> None:
    model.reset()
