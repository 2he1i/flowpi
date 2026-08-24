"""FlowPi runtime: frame buffering, asynchronous optical flow, prefix refresh, and πR² streaming."""

from collections.abc import Sequence
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import contextlib
import dataclasses
import functools
import multiprocessing as mp
import os
from queue import Empty
from queue import Full
import threading
import time
from typing import Any
import unicodedata

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.shared import nnx_utils as _nnx_utils
from openpi.training import sea_raft_worker as _sea_raft_worker
from openpi.training.sea_raft import SeaRaftFlowExtractor
from openpi.transforms import compute_image_frame_offsets
from openpi.transforms import normalize_flow


def _resolve_jax_device(spec: str | None) -> jax.Device | None:
    """Parse a device spec (``None``, ``"cpu"``, ``"cuda:0"``, ``"gpu:1"``) into a jax device."""
    if spec is None:
        return None
    backend = spec
    index = 0
    if ":" in spec:
        backend, index_str = spec.split(":", 1)
        index = int(index_str)
    if backend in ("cuda", "gpu"):
        backend = "gpu"
    devices = jax.devices(backend)
    if not 0 <= index < len(devices):
        raise ValueError(f"jax_device {spec!r}: backend {backend!r} has {len(devices)} devices")
    return devices[index]


def _place_tree(tree: Any, device: jax.Device | None) -> Any:
    """Place array leaves of a model/observation tree on one device."""
    if device is None:
        return tree
    return jax.tree.map(
        lambda value: jax.device_put(value, device)
        if isinstance(value, (jax.Array, np.ndarray))
        else value,
        tree,
    )


@functools.cache
def _dataclass_fields(cls: type) -> tuple[dataclasses.Field[Any], ...]:
    """Cache field metadata for the small set of runtime dataclasses."""
    return tuple(dataclasses.fields(cls))


def _replace_dataclass_fast(instance: Any, **changes: Any) -> Any:
    """Shallow-clone an already validated dataclass without invoking its constructor.

    ``Observation`` is decorated with jaxtyping/beartype.  ``dataclasses.replace`` therefore
    re-enters the runtime type checker on every fast NFE; the checker walks Python stack frames
    and is disproportionately expensive for a 25+ Hz state-only loop.  Objects reaching this
    helper have already crossed the validated RPC/preprocessing boundary, and the replacements
    below only swap model-owned arrays.  Copy fields directly so no user data is copied and no
    constructor/type-check hook runs in the hot path.
    """
    fields = _dataclass_fields(type(instance))
    field_names = {field.name for field in fields}
    unknown = set(changes).difference(field_names)
    if unknown:
        raise TypeError(f"Unknown dataclass fields for {type(instance).__name__}: {sorted(unknown)}")
    clone = object.__new__(type(instance))
    for field in fields:
        value = changes.get(field.name, getattr(instance, field.name))
        object.__setattr__(clone, field.name, value)
    return clone


def _fold_in_rng(seed: jax.Array, tick: Any) -> jax.Array:
    """Jittable exact replacement for the per-tick eager RNG fold-in."""
    return jax.random.fold_in(seed, tick)


def _place_model(model: _pi0.Pi0, device: jax.Device | None) -> _pi0.Pi0:
    """Move an NNX model state without replicating it across the visible device set."""
    if device is None:
        return model
    graphdef, state = nnx.split(model)
    return nnx.merge(graphdef, _place_tree(state, device))


@functools.lru_cache(maxsize=1)
def _ensure_streaming_state_pytree() -> None:
    """Register the runtime state so ``jax.jit`` can carry it between ticks."""
    jax.tree_util.register_dataclass(
        _pi0.Pi0.StreamingState,
        data_fields=("action_buffer", "tau", "kv_cache", "prefix_mask", "prefix_len", "prefix_source_tick"),
        meta_fields=(),
    )


def summarize_series(values: Sequence[float]) -> dict[str, float | int]:
    """Return robust distribution statistics for a numeric metric series."""
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def summarize_rate(timestamps_s: Sequence[float]) -> dict[str, Any]:
    """Summarize an event stream as inter-event periods and effective frequency."""
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.size < 2:
        return {"event_count": int(timestamps.size), "mean_hz": None, "median_hz": None}
    intervals_ms = np.diff(timestamps) * 1000.0
    intervals_ms = intervals_ms[np.isfinite(intervals_ms) & (intervals_ms > 0)]
    if intervals_ms.size == 0:
        return {"event_count": int(timestamps.size), "mean_hz": None, "median_hz": None}
    return {
        "event_count": int(timestamps.size),
        "mean_hz": float(1000.0 / np.mean(intervals_ms)),
        "median_hz": float(1000.0 / np.median(intervals_ms)),
        "interval_ms": summarize_series(intervals_ms.tolist()),
    }


def _display_width(value: str) -> int:
    """Return a terminal display width that also keeps CJK table cells aligned."""
    return sum(2 if unicodedata.east_asian_width(char) in "WFA" else 1 for char in value)


def _pad_display(value: str, width: int) -> str:
    return value + " " * max(0, width - _display_width(value))


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a dependency-free Unicode-safe table for terminal logs."""
    string_rows = [[str(cell) for cell in row] for row in rows]
    string_headers = [str(header) for header in headers]
    widths = [
        max((_display_width(row[index]) for row in [string_headers, *string_rows]), default=0)
        for index in range(len(string_headers))
    ]

    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def line(row: Sequence[str]) -> str:
        cells = [_pad_display(str(value), widths[index]) for index, value in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    output = [border(), line(string_headers), border()]
    output.extend(line(row) for row in string_rows)
    output.append(border())
    return "\n".join(output)


def _format_metric(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value)}{suffix}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{value}{suffix}"
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}{suffix}"


def _summary_value(summary: Any, key: str) -> Any:
    if not isinstance(summary, dict):
        return None
    return summary.get(key)


def format_metrics_table(metrics: dict[str, Any], *, title: str = "FlowPi Runtime Metrics") -> str:
    """Format a runtime or full policy metrics payload as human-readable tables.

    The JSON payload remains the source of truth. This function intentionally only presents the
    compact aggregate fields, so it is safe to call at the end of a RoboTwin server process or
    when inspecting a historical ``policy_runtime.json`` file.
    """
    if not isinstance(metrics, dict):
        raise TypeError(f"metrics must be a dictionary, got {type(metrics).__name__}")

    # Accept both FlowPiRuntime.metrics_summary() and the policy adapter's wrapper payload.
    runtime = metrics.get("runtime", metrics)
    if not isinstance(runtime, dict):
        raise TypeError("metrics['runtime'] must be a dictionary")
    config = runtime.get("config", {})
    devices = metrics.get("devices", {})
    policy = metrics.get("policy", {})
    rates = runtime.get("rates_hz", {})
    latency = runtime.get("latency_ms", {})
    if not latency and isinstance(runtime.get("stats"), dict):
        latency = {
            name: summarize_series(values)
            for name, values in runtime["stats"].items()
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
        }

    sections: list[str] = [str(title)]

    config_rows: list[list[str]] = []
    duration_s = runtime.get("duration_s")
    if duration_s is not None:
        config_rows.append(["总运行时长", _format_metric(duration_s, digits=1, suffix=" s")])
    if devices:
        config_rows.extend(
            [
                ["慢通道 JAX", str(devices.get("slow_jax", "-"))],
                ["快通道 JAX", str(devices.get("fast_jax", "-"))],
                ["SEA-RAFT", str(devices.get("sea_raft", "-"))],
            ]
        )
    precision = devices.get("sea_raft_precision", config.get("sea_raft_precision"))
    if precision is not None:
        config_rows.append(["SEA-RAFT 精度", str(precision)])
    if "async_flow" in config:
        config_rows.append(["异步光流", "启用" if config["async_flow"] else "关闭"])
    if "action_chunk_d" in config:
        d_history = config.get("action_chunk_d_history", [])
        if isinstance(d_history, list) and d_history:
            d_text = (
                f"当前 {config['action_chunk_d']} / "
                f"范围 {min(d_history)}-{max(d_history)} / "
                f"样本 {len(d_history)}"
            )
        else:
            d_text = str(config["action_chunk_d"])
        config_rows.append(["πR² 动态 d", d_text])
    if policy.get("action_chunk_d_max") is not None:
        config_rows.append(
            [
                "部署 d 上限",
                (
                    f"{policy.get('action_chunk_d_max')} "
                    f"(训练上限 {policy.get('action_chunk_d_training_max', '-')})"
                ),
            ]
        )
    if "flow_stride_frames" in config or "num_flow_steps" in config:
        config_rows.append(
            [
                "光流 stride / steps",
                f"{config.get('flow_stride_frames', '-')} / {config.get('num_flow_steps', '-')} frames/steps",
            ]
        )
    if "flow_delay_max" in config or "vlm_delay_max" in config:
        config_rows.append(
            [
                "最大允许年龄",
                (
                    f"flow={config.get('flow_delay_max', '-')} ticks, "
                    f"VLM={config.get('vlm_delay_max', '-')} ticks"
                ),
            ]
        )
    if config_rows:
        sections.append("配置 / Configuration\n" + _render_table(["项目", "值"], config_rows))

    client = metrics.get("client")
    if isinstance(client, dict):
        client_rows = [
            ["DOMINO 控制模式", str(client.get("control_mode", "-"))],
            ["控制目标频率", _format_metric(client.get("control_target_hz"), suffix=" Hz")],
            [
                "动作块批处理",
                "启用" if client.get("drain_action_chunk", False) else "关闭",
            ],
            [
                "Hold-last 最大批处理",
                f"{client.get('hold_last_burst', 1)} ticks",
            ],
            [
                "RGB RPC 传输",
                (
                    f"JPEG-{client.get('jpeg_quality', 95)}"
                    if client.get("compress_rgb", False)
                    else "原始数组"
                ),
            ],
            [
                "稳态实测控制频率",
                _format_metric(client.get("steady_loop_hz", client.get("loop_hz")), suffix=" Hz"),
            ],
            [
                "稳态策略查询频率",
                _format_metric(client.get("steady_policy_query_rate", client.get("policy_query_rate")), suffix=" Hz"),
            ],
            [
                "本体状态快通道频率",
                _format_metric(client.get("fast_state_query_rate"), suffix=" Hz"),
            ],
            [
                "图像队列丢帧",
                str(client.get("pending_observation_drops", 0)),
            ],
            [
                "状态队列丢弃",
                str(client.get("pending_state_drops", 0)),
            ],
            [
                "完整图像更新频率",
                _format_metric(client.get("image_update_rate"), suffix=" Hz"),
            ],
            ["控制步数", _format_metric(client.get("control_step_count"), digits=0)],
            ["策略查询次数", _format_metric(client.get("request_count"), digits=0)],
            ["本体状态查询次数", _format_metric(client.get("fast_state_query_count"), digits=0)],
            ["完整图像更新次数", _format_metric(client.get("image_update_count"), digits=0)],
            ["观测丢弃数", _format_metric(client.get("pending_observation_drops"), digits=0)],
            ["状态丢弃数", _format_metric(client.get("pending_state_drops"), digits=0)],
        ]
        if client.get("loop_hz") is not None and client.get("steady_loop_hz") is not None:
            client_rows.append(["含启动控制频率", _format_metric(client.get("loop_hz"), suffix=" Hz")])
        rpc_summary = client.get("rpc_roundtrip_ms")
        execute_summary = client.get("action_execute_ms")
        if isinstance(rpc_summary, dict) and rpc_summary.get("count"):
            client_rows.append(["RPC p50 / p99", f"{_format_metric(rpc_summary.get('p50'))} / {_format_metric(rpc_summary.get('p99'))} ms"])
        if isinstance(execute_summary, dict) and execute_summary.get("count"):
            client_rows.append(["仿真控制步 p50 / p99", f"{_format_metric(execute_summary.get('p50'))} / {_format_metric(execute_summary.get('p99'))} ms"])
        sections.append("DOMINO 控制 / DOMINO Control\n" + _render_table(["项目", "值"], client_rows))

    rate_specs = [
        ("policy_request", "策略 RPC 请求", policy.get("request_rate")),
        # One NFE emits d actions. The physical action frequency is reported by the DOMINO
        # client; these rows deliberately separate NFE/state/image clocks.
        ("fast_nfe", "快通道 NFE / 动作块更新", rates.get("fast_nfe", rates.get("fast_control_loop"))),
        ("fast_state_loop", "快通道本体状态闭环", rates.get("fast_state_loop")),
        ("image_ingest", "完整图像帧进入环", rates.get("image_ingest")),
        ("flow_submit", "SEA-RAFT 请求提交", rates.get("flow_submit")),
        ("flow_update", "SEA-RAFT 光流更新", rates.get("flow_update")),
        ("slow_refresh_request", "慢通道刷新请求", rates.get("slow_refresh_request")),
        ("slow_prefill_start", "慢通道 prefill 开始", rates.get("slow_prefill_start")),
        ("slow_prefill_complete", "慢通道 prefill 完成", rates.get("slow_prefill_complete")),
        ("prefix_install", "VLM prefix 安装", rates.get("prefix_install")),
    ]
    rate_rows = []
    for _key, label, rate in rate_specs:
        if not isinstance(rate, dict):
            continue
        interval = rate.get("interval_ms", {})
        rate_rows.append(
            [
                label,
                _format_metric(rate.get("event_count"), digits=0),
                _format_metric(rate.get("mean_hz"), suffix=" Hz"),
                _format_metric(rate.get("median_hz"), suffix=" Hz"),
                _format_metric(_summary_value(interval, "p50"), suffix=" ms"),
            ]
        )
    if rate_rows:
        sections.append(
            "频率 / Frequency\n"
            + _render_table(["通道 / 事件", "次数", "平均", "中位", "周期 p50"], rate_rows)
        )

    latency_specs = [
        ("policy_request", "策略请求", policy.get("latency_ms")),
        ("fast_state", "本体状态快通道请求", policy.get("fast_state_latency_ms")),
        ("image_update", "完整图像更新请求", policy.get("image_update_latency_ms")),
        ("frame_ingest_ms", "帧接收与预处理", latency.get("frame_ingest_ms")),
        ("warm_start_ms", "首次 warm start", latency.get("warm_start_ms")),
        ("fast_jit_compile_ms", "快通道 JIT 编译", latency.get("fast_jit_compile_ms")),
        ("flow_queue_ms", "光流排队", latency.get("flow_queue_ms")),
        ("flow_ms", "SEA-RAFT 推理", latency.get("flow_ms") or latency.get("flow_compute_ms")),
        ("flow_age_ms", "光流年龄", latency.get("flow_age_ms")),
        ("fast_model_ms", "快通道模型 / NFE", latency.get("fast_model_ms")),
        ("tick_total_ms", "Runtime tick 总耗时", latency.get("tick_total_ms")),
        ("prefill_ms", "慢通道 prefill", latency.get("prefill_ms")),
        ("prefix_install_ms", "Prefix 安装", latency.get("prefix_install_ms")),
        ("prefix_age_ms_at_install", "VLM prefix 年龄", latency.get("prefix_age_ms_at_install")),
    ]
    latency_rows = []
    for _, label, summary in latency_specs:
        if not isinstance(summary, dict) or not summary.get("count"):
            continue
        latency_rows.append(
            [
                label,
                _format_metric(summary.get("mean"), suffix=" ms"),
                _format_metric(summary.get("p50"), suffix=" ms"),
                _format_metric(summary.get("p90"), suffix=" ms"),
                _format_metric(summary.get("p99"), suffix=" ms"),
                _format_metric(summary.get("max"), suffix=" ms"),
            ]
        )
    if latency_rows:
        sections.append(
            "延迟 / Latency\n"
            + _render_table(["指标", "平均", "p50", "p90", "p99", "最大"], latency_rows)
        )

    freshness_specs = [
        ("flow_age_ticks", "光流年龄", "ticks"),
        ("prefix_age_at_install", "VLM prefix 年龄", "ticks"),
        ("prefix_age_at_install_raw", "VLM prefix 原始年龄", "ticks"),
    ]
    freshness_rows = []
    for key, label, unit in freshness_specs:
        summary = latency.get(key)
        if not isinstance(summary, dict) or not summary.get("count"):
            continue
        freshness_rows.append(
            [
                label,
                unit,
                _format_metric(summary.get("mean")),
                _format_metric(summary.get("p50")),
                _format_metric(summary.get("p90")),
                _format_metric(summary.get("max")),
            ]
        )
    if freshness_rows:
        sections.append(
            "新鲜度 / Freshness\n"
            + _render_table(["指标", "单位", "平均", "p50", "p90", "最大"], freshness_rows)
        )

    counter_labels = {
        "fast_ticks": "快通道 NFE tick",
        "fast_state_ticks": "本体状态闭环 NFE",
        "control_ticks": "物理 control tick",
        "image_frames": "完整图像帧",
        "image_frame_gaps": "图像物理帧缺口",
        "flow_updates": "光流更新",
        "flow_submitted": "光流提交",
        "flow_completed": "光流完成",
        "flow_coalesced": "光流合并丢弃",
        "flow_generation_drops": "光流 generation 丢弃",
        "flow_nonfinite_lags": "光流非有限 lag (已屏蔽)",
        "slow_refresh_requests": "慢通道刷新请求",
        "slow_refresh_coalesced": "慢通道刷新合并",
        "slow_prefills_started": "慢通道 prefill 开始",
        "slow_prefills_completed": "慢通道 prefill 完成",
        "slow_prefix_published": "Prefix 发布",
        "slow_prefix_installs": "Prefix 安装",
        "generation_drops": "Prefix generation 丢弃",
    }
    counters = runtime.get("counters", {})
    counter_rows = [
        [label, _format_metric(counters[key], digits=0)]
        for key, label in counter_labels.items()
        if key in counters
    ]
    if counter_rows:
        sections.append("计数器 / Counters\n" + _render_table(["计数器", "数量"], counter_rows))

    if len(sections) == 1:
        sections.append("(暂无可显示的聚合指标)")
    return "\n\n".join(sections)


@dataclasses.dataclass
class PrefixGeneration:
    """A completed slow-channel prefix prefill, published for atomic installation.

    ``episode_id`` and the source clocks let the runtime drop stale generations: a prefill from a
    previous episode, or one computed from an observation older than the currently active prefix,
    is never installed. ``source_tick`` remains the image-frame clock for compatibility and flow
    telemetry; ``source_control_tick`` is the physical/control clock used by the slow-delay
    embedding.
    """

    # Episode this prefix was computed in (drops cross-episode publications).
    episode_id: int
    # Episode-relative image-frame tick of the observation this prefix was computed from.
    source_tick: int
    kv_cache: Any
    prefix_mask: jax.Array
    # Episode-relative physical control tick of the source observation. ``None`` is accepted for
    # lightweight test doubles and older callers; in that case ``source_tick`` is used.
    source_control_tick: int | None = None
    # Monotonic wall-clock time at which the source observation was ingested.
    source_wall_s: float | None = None


@dataclasses.dataclass(frozen=True)
class _ObservationSnapshot:
    """A runtime-owned observation and the exact episode tick it belongs to."""

    episode_id: int
    source_tick: int
    observation: _model.Observation
    # Physical/control clock at which this image was observed. Image updates may arrive after
    # one or more state-only NFEs, so this must not be inferred from ``source_tick``.
    source_control_tick: int | None = None
    source_wall_s: float | None = None


@dataclasses.dataclass(frozen=True)
class _FlowSnapshot:
    """Immutable flow request copied out of the mutable frame ring."""

    episode_id: int
    source_tick: int
    source_fast_tick: int
    prev: np.ndarray
    curr: np.ndarray
    valid_lags: np.ndarray
    submitted_at: float


@dataclasses.dataclass(frozen=True)
class _FlowGeneration:
    """A completed SEA-RAFT result and the tick it describes."""

    episode_id: int
    source_tick: int | None
    source_fast_tick: int | None
    flow: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    compute_ms: float | None
    queue_ms: float | None
    completed_at: float | None


class _FrameRingBuffer:
    """Sliding window of decoded camera frames in CHW uint8 layout.

    ``current`` always points to the latest frame slot. ``base_index`` is the physical
    frame-index of that latest frame. Slots also carry their physical index so a dropped RGB
    frame creates an invalid lag instead of being mistaken for a contiguous frame.
    """

    def __init__(
        self,
        cam_keys: Sequence[str],
        capacity: int,
        first_frames: dict[str, np.ndarray],
    ):
        self.cam_keys = tuple(cam_keys)
        self.capacity = capacity
        h, w = first_frames[next(iter(cam_keys))].shape[-2:]
        self.buffer: dict[str, np.ndarray] = {cam: np.zeros((capacity, 3, h, w), dtype=np.uint8) for cam in cam_keys}
        for cam in cam_keys:
            self.buffer[cam][0] = first_frames[cam]
        self.slot_indices = np.full((capacity,), -1, dtype=np.int64)
        self.slot_indices[0] = 0
        self.base_index = 0  # dataset-frame-index of the latest frame
        self.current = 0  # buffer[:, current] is the latest frame

    def push(self, cam_key: str, frame_chw: np.ndarray) -> None:
        """Write one camera frame at the current cursor position."""
        self.buffer[cam_key][self.current] = frame_chw

    def advance(self, step: int = 1) -> int:
        """Advance the cursor by physical frame ticks, invalidating skipped slots logically."""
        if step <= 0:
            raise ValueError(f"frame advance step must be positive, got {step}")
        old = self.current
        self.current = (self.current + step) % self.capacity
        self.base_index += step
        self.slot_indices[self.current] = self.base_index
        return old

    def get(self, offset: int) -> dict[str, np.ndarray]:
        """Return the frame(s) offset ticks *before* the latest (offset ≤ 0).

        Raises ``IndexError`` if the requested offset is outside the buffered
        window (i.e. before the episode started or beyond what the ring holds).
        """
        target_index = self.base_index + offset
        if target_index < 0:
            raise IndexError(f"Frame at offset {offset} is before the episode start.")
        if offset > 0 or -offset >= self.capacity:
            raise IndexError(f"Frame at offset {offset} is outside the ring buffer window.")
        idx = (self.current + offset) % self.capacity
        if self.slot_indices[idx] != target_index:
            raise IndexError(f"Frame at offset {offset} was dropped or is not available.")
        return {cam: arr[idx] for cam, arr in self.buffer.items()}

    @classmethod
    def create(
        cls,
        cam_keys: Sequence[str],
        capacity: int,
        first_frames: dict[str, np.ndarray],
    ) -> "_FrameRingBuffer":
        """Initialise with the first frame at position 0."""
        return cls(cam_keys, capacity, first_frames)


class FlowPiRuntime:
    """Offline-replay / online-deployment runtime for a flowpi model.

    The runtime expects full-resolution camera frames (the ``flow_image_size`` used at training
    time, e.g. 480x640): it computes the online SEA-RAFT flow on those frames and lets the model
    preprocess the same observation for the VLM (resizing internally). Feeding pre-resized
    (model-resolution) frames produces a wrong flow grid and raises.

    Usage (offline replay on a dataset episode, with a dynamically selected ``d``)::

        runtime = FlowPiRuntime(model, flow_config=..., sea_raft_ckpt=..., sea_raft_device="cuda")
        runtime.warm_start(first_observation)
        # warm_start already creates and installs the initial prefix
        for frame_idx in range(1, episode_length):
            obs = dataset[frame_idx]                # (Observation state, images)
            actions = runtime.tick(obs)
            if frame_idx % slow_every_n == 0:
                runtime.refresh_prefix()              # async: returns immediately
            # use actions ...
        runtime.close()                             # drain the slow worker, propagate errors

    The slow delay is `control_tick - prefix_source_control_tick`: the physical/control tick of
    the observation the active prefix was computed from, so state-only NFEs and dropped image
    frames still age the prefix correctly. The image-frame clock remains separate and is used
    only for the SEA-RAFT ring geometry.

    Slow-channel scheduling: one in-flight VLM prefill plus a single latest-pending mailbox.
    Refresh requests arriving while the worker is busy coalesce into the mailbox (the queue is
    bounded to one slot), and a completed prefill that is fresher than the active prefix is
    always published. Under a refresh storm the prefix source tick therefore advances
    monotonically instead of starving (the old "only the latest submitted generation may
    publish" scheme could drop every completed prefill and leave the prefix frozen).

    Flow scheduling uses the same bounded pattern: one in-flight SEA-RAFT request plus one latest
    pending frame snapshot. The fast channel never waits for flow; it consumes the latest completed
    flow and passes its bounded age embedding to the model.
    """

    def __init__(
        self,
        model: _pi0.Pi0,
        *,
        flow_config: Any,  # pi0_config.FlowConfig
        slow_model: _pi0.Pi0 | None = None,
        sea_raft_ckpt: str | None = None,
        sea_raft_variant: str = "M",
        sea_raft_iters: int | None = None,
        sea_raft_device: str = "cuda",
        sea_raft_precision: str = "fp16",
        jax_device: str | None = None,
        slow_jax_device: str | None = None,
        d: int = 1,
        precompile_d_max: int | None = None,
        allow_random_init: bool = False,
        async_flow: bool = True,
        flow_process: bool = False,
    ):
        self.model = model
        self.flow_config = flow_config
        self._d = self._validate_action_chunk_d(d)
        training_d_max = int(getattr(flow_config, "d_max", self._d))
        if precompile_d_max is None:
            self._precompile_d_max = training_d_max
        else:
            self._precompile_d_max = min(training_d_max, int(precompile_d_max))
            if self._precompile_d_max < 1:
                raise ValueError("precompile_d_max must be positive")
        # ``d`` is a deployment-time latency estimate, not a fixed architecture constant. Keep
        # the history so the policy/client metrics can show how the estimate evolved.
        self._action_chunk_d_history: list[int] = [self._d]
        self._async_flow = bool(async_flow)
        # Inference can isolate Torch/SEA-RAFT from the JAX policy process. Keep the direct
        # in-process path as the default for offline replay and unit-test doubles.
        self._flow_process_enabled = bool(flow_process and self._async_flow)

        # The fast replica owns the streaming action-expert calls. A separate slow replica can
        # prefill the VLM prefix concurrently on another GPU; SEA-RAFT is pinned independently
        # via ``sea_raft_device`` (torch).
        self._jax_device = _resolve_jax_device(jax_device)
        self.model = _place_model(self.model, self._jax_device)
        if slow_model is None:
            if slow_jax_device is not None:
                raise ValueError("slow_jax_device requires a separate slow_model replica")
            self._slow_model = self.model
        else:
            self._slow_jax_device = _resolve_jax_device(slow_jax_device)
            self._slow_model = _place_model(slow_model, self._slow_jax_device)
        if slow_model is None:
            self._slow_jax_device = self._jax_device

        # The generic Policy wrapper JITs model.sample_actions, but this streaming runtime calls
        # the lower-level methods directly. Freeze the inference-only NNX state into compiled
        # functions so the first call pays compilation once and later ticks execute the cached
        # executable. The model is never mutated during inference, so freezing is safe here.
        _ensure_streaming_state_pytree()
        self._fast_warm_start = _nnx_utils.module_jit(
            self.model.warm_start,
            static_argnames=("num_steps", "d"),
        )
        self._fast_denoise_step = _nnx_utils.module_jit(
            self.model.denoise_step,
            static_argnames=("d",),
        )
        self._slow_prefix_forward = _nnx_utils.module_jit(self._slow_model._prefix_forward)  # noqa: SLF001
        # ``jax.random.fold_in`` is intentionally eager in the original runtime. On the hot
        # path that creates a small host-side dispatch/compile envelope around every NFE. Keep
        # the exact same key stream as ``fold_in(key(0), tick)`` but compile the tiny primitive
        # once and feed it the fast-device seed thereafter.
        self._fast_rng_seed = _place_tree(jax.random.key(0), self._jax_device)
        self._fast_rng_fold_in = jax.jit(_fold_in_rng)
        # Keep the original bound method for a small but useful failure-injection/debugging
        # hook: when the fast and slow replicas are the same object, a caller can replace
        # ``model._prefix_forward`` and the runtime must not silently keep invoking a stale
        # compiled closure. Separate replicas retain the frozen/JIT path unconditionally.
        self._slow_prefix_method = self._slow_model._prefix_forward  # noqa: SLF001

        self._cam_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

        # Frame ring buffer geometry.
        k = flow_config.num_flow_steps
        stride = flow_config.flow_stride_frames
        # Keep enough history for the latest-result flow mailbox. Flow requests contain copied
        # frame tensors, so the mutable ring can continue ingesting frames while SEA-RAFT runs.
        self._ring_capacity = (
            max(
                k * stride + flow_config.flow_delay_max,
                flow_config.vlm_delay_max,
            )
            + 1
        )
        self._frame_offsets = compute_image_frame_offsets(
            k,
            stride,
            flow_config.vlm_delay_max,
            flow_config.flow_delay_max,
        )

        # SEA-RAFT extractor (online flow). In the process-isolated inference path the child
        # owns the Torch model and GPU2; keeping no CUDA model in this process is what prevents
        # JAX/slow-channel host scheduling from inflating the flow service time.
        self._raft = None
        self._sea_raft_ckpt = None if sea_raft_ckpt is None else str(sea_raft_ckpt)
        self._sea_raft_variant = sea_raft_variant
        self._sea_raft_iters = sea_raft_iters
        self._sea_raft_device = sea_raft_device
        self._sea_raft_allow_random_init = allow_random_init
        self._sea_raft_precision = sea_raft_precision
        if not self._flow_process_enabled:
            self._raft = SeaRaftFlowExtractor(
                ckpt_path=sea_raft_ckpt,
                variant=sea_raft_variant,
                iters=sea_raft_iters,
                device=sea_raft_device,
                allow_random_init=allow_random_init,
                precision=sea_raft_precision,
            )
            self._sea_raft_precision = getattr(self._raft, "_precision", sea_raft_precision)

        # SEA-RAFT runs on its own Torch device. One in-flight request plus one latest-pending
        # request prevents a slow optical-flow service from building an unbounded queue. The
        # fast channel consumes the freshest completed result and reports its actual age.
        self._flow_lock = threading.Lock()
        self._flow_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="flowpi-flow") if self._async_flow else None
        )
        self._flow_futures: list[Future] = []
        self._flow_busy = False
        self._flow_mailbox: _FlowSnapshot | None = None
        self._flow_process: mp.Process | None = None
        self._flow_process_request_queue: Any | None = None
        self._flow_process_result_queue: Any | None = None
        self._flow_process_shared_prev: Any | None = None
        self._flow_process_shared_curr: Any | None = None
        self._flow_process_shared_output: Any | None = None
        self._flow_process_prev_view: np.ndarray | None = None
        self._flow_process_curr_view: np.ndarray | None = None
        self._flow_process_output_view: np.ndarray | None = None
        self._flow_process_request_id = 0
        self._flow_process_timeout_s = float(os.environ.get("FLOWPI_FLOW_PROCESS_TIMEOUT", "120"))
        if self._flow_process_timeout_s <= 0:
            raise ValueError("FLOWPI_FLOW_PROCESS_TIMEOUT must be positive")
        self._active_flow: _FlowGeneration | None = None
        self._flow_submitted = 0
        self._flow_completed = 0
        self._flow_coalesced = 0
        self._flow_generation_drops = 0
        # A bad AMP/kernel result must not inject NaNs into the JAX policy.  Count invalid lag
        # results so a deployment can distinguish a conservative zero-flow fallback from a
        # genuinely fresh SEA-RAFT update.
        self._flow_nonfinite_lags = 0

        # Slow-channel publication: a completed background prefix refresh is published here
        # atomically (kv_cache + prefix_mask + source metadata as one consistent object) and
        # installed into the active StreamingState at the start of the next fast tick.
        self._pending_prefix: PrefixGeneration | None = None
        self._slow_lock = threading.Lock()
        # Single slow worker. Scheduling is "one in-flight prefill + one latest-pending
        # mailbox": when the refresh rate exceeds the VLM service rate, requests coalesce into
        # a single slot instead of queuing, and a completed prefill that is fresher than the
        # active prefix is always published (never dropped just because a newer request
        # arrived while it was computing). This prevents refresh starvation: the active prefix
        # source tick advances monotonically and the pending queue stays bounded.
        self._slow_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flowpi-slow")
        self._slow_futures: list[Future] = []
        # True while a prefill is running on the worker (only then may the mailbox accept
        # requests). Cleared by the worker itself, so a finished worker is never double-booked.
        self._slow_busy = False
        # Latest pending refresh snapshot, coalesced while the worker is busy. The snapshot keeps
        # episode id, source tick, and observation inseparable.
        self._slow_mailbox: _ObservationSnapshot | None = None

        # Per-episode streaming state. `StreamingState.prefix_source_tick` (mirrored by
        # `self._prefix_source_tick`) is the single authoritative *control* clock for the age of
        # the prefix actually used by the fast policy. The image-frame source is tracked
        # separately for stale-generation ordering and telemetry.
        self._streaming_state: _pi0.Pi0.StreamingState | None = None
        self._ring: _FrameRingBuffer | None = None
        # Monotonically increasing per-tick counter (RNG only, not a prefix clock).
        self._tick = 0
        self._episode_id = 0
        # Episode-relative index of the most recently ingested frame (0 = the warm-start frame).
        self._frame_index = 0
        # Episode-relative physical/control clock. When the caller does not provide an explicit
        # clock, one fast NFE advances this by the emitted action width ``d``. DOMINO supplies
        # its authoritative ``take_action_cnt`` so state-only requests and dropped RGB frames
        # use the simulator clock instead of the retained-image count.
        self._control_tick = 0
        # Runtime-owned latest observation. Prefix refreshes always capture this snapshot instead
        # of accepting a caller-supplied observation that could belong to another tick.
        self._latest_observation: _ObservationSnapshot | None = None
        # Episode-relative physical/control tick of the observation the active prefix was
        # computed from. ``_prefix_source_frame_tick`` is the independent image-frame clock.
        self._prefix_source_tick: int | None = None
        self._prefix_source_frame_tick: int | None = None

        # Image resolution for flow (480x640 -> 60x80 grid).
        h, w = flow_config.flow_image_size
        self._flow_grid = (h // 8, w // 8)
        self._active_flow = self._empty_flow_generation(episode_id=0)
        # The same completed SEA-RAFT generation is consumed by many state-only NFEs. Cache its
        # small flow tensors on the fast JAX device instead of copying the host numpy arrays into
        # JAX on every NFE; refresh the cache only when SEA-RAFT publishes a new generation.
        self._flow_jax_cache_generation: _FlowGeneration | None = None
        self._flow_jax_cache: tuple[dict[str, Any], dict[str, Any]] | None = None

        # Telemetry (wall-clock ms and delays; appended from the main thread and the slow worker).
        self.stats: dict[str, list[float]] = {
            "warm_start_ms": [],
            "frame_ingest_ms": [],
            "flow_ms": [],
            "flow_compute_ms": [],
            "flow_queue_ms": [],
            "flow_age_ticks": [],
            "flow_age_ms": [],
            "prefill_ms": [],
            "fast_model_ms": [],
            "tick_total_ms": [],
            "tick_wall_ms": [],
            "fast_tick_interval_ms": [],
            "flow_interval_ms": [],
            "prefix_install_ms": [],
            # Keep the original key clamped for consumers that already use it; the explicit raw
            # series is what delay fitting should use.
            "prefix_age_at_install": [],
            "prefix_age_at_install_raw": [],
            "prefix_age_at_install_clamped": [],
            "prefix_age_ms_at_install": [],
            "action_chunk_d": [float(self._d)],
        }
        # Per-tick freshness telemetry reconstructs the actual Age_VLM and Age_Flow values from
        # source ticks and wall-clock ingestion times.
        self.telemetry: list[dict[str, Any]] = []
        # Wall-clock of each frame's ingestion (indexed by episode-relative frame index), used by
        # flow telemetry; pruned so it stays bounded. Prefix generations carry their own source
        # wall-clock so state-only ticks do not need an entry in this frame-indexed map.
        self._ingest_wall: dict[int, float] = {}
        self._image_frame_gaps = 0
        # Published-but-never-installed prefix generations (stale episode or out-of-order tick).
        self.num_generation_drops: int = 0

        # Event clocks are relative to runtime construction, so telemetry can be persisted and
        # compared across processes without leaking host-monotonic timestamps.
        self._metrics_start = time.perf_counter()
        self._tick_times_s: list[float] = []
        self._flow_submit_times_s: list[float] = []
        self._flow_start_times_s: list[float] = []
        self._flow_complete_times_s: list[float] = []
        self._flow_times_s: list[float] = []
        self._image_ingest_times_s: list[float] = []
        self._fast_state_tick_times_s: list[float] = []
        self._slow_refresh_request_times_s: list[float] = []
        self._slow_prefill_start_times_s: list[float] = []
        self._slow_prefill_complete_times_s: list[float] = []
        self._prefix_install_times_s: list[float] = []
        self._slow_refresh_requests = 0
        self._slow_refresh_coalesced = 0
        self._slow_prefills_started = 0
        self._slow_prefills_completed = 0
        self._slow_prefix_published = 0
        self._slow_prefix_installs = 0
        self._last_tick_start_s: float | None = None
        self._last_flow_start_s: float | None = None
        self._last_flow_complete_s: float | None = None
        self._closed = False
        if self._flow_process_enabled:
            self._start_flow_process()

    def _validate_action_chunk_d(self, d: int) -> int:
        """Validate a runtime action-chunk latency in the training support."""
        try:
            value = int(d)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"action chunk d must be an integer, got {d!r}") from exc
        d_max = int(getattr(self.flow_config, "d_max", value))
        if not 1 <= value <= d_max:
            raise ValueError(f"action chunk d={value} is outside the training support [1, {d_max}]")
        return value

    @property
    def action_chunk_d(self) -> int:
        """Return the latency/action-chunk value used by the next fast step."""
        return self._d

    def set_action_chunk_d(self, d: int) -> int:
        """Update the deployment-time ``d`` used by the next denoising step.

        πR² trains all staircase widths in ``[1, d_max]`` and the original deployment estimates
        the width from measured query latency. JAX compiles one executable per static ``d``; the
        first use of a new width therefore pays a one-time compilation cost, after which it is
        cached normally.
        """
        value = self._validate_action_chunk_d(d)
        self._d = value
        self._action_chunk_d_history.append(value)
        if hasattr(self, "stats") and "action_chunk_d" in self.stats:
            self.stats["action_chunk_d"].append(float(value))
        return value

    def _metric_now(self) -> float | None:
        """Return runtime-relative time; tolerate lightweight test doubles."""
        start = getattr(self, "_metrics_start", None)
        return None if start is None else time.perf_counter() - start

    def _record_timestamp(self, name: str) -> float | None:
        """Append a runtime-relative event timestamp when metrics are initialized."""
        timestamp = self._metric_now()
        timestamps = getattr(self, name, None)
        if timestamp is not None and timestamps is not None:
            timestamps.append(timestamp)
        return timestamp

    @staticmethod
    def _snapshot_control_tick(snapshot: _ObservationSnapshot) -> int:
        """Return the physical clock carried by an observation snapshot.

        Older lightweight callers only populate ``source_tick``. Falling back to that frame
        clock keeps those callers functional while production DOMINO requests use the explicit
        simulator control tick.
        """
        return int(snapshot.source_control_tick if snapshot.source_control_tick is not None else snapshot.source_tick)

    @staticmethod
    def _generation_control_tick(generation: PrefixGeneration) -> int:
        """Return a prefix generation's physical source clock with a legacy fallback."""
        return int(
            generation.source_control_tick
            if generation.source_control_tick is not None
            else generation.source_tick
        )

    @classmethod
    def _generation_order(cls, generation: PrefixGeneration) -> tuple[int, int]:
        """Order prefix generations by physical time, then image-frame time."""
        return cls._generation_control_tick(generation), int(generation.source_tick)

    def _resolve_control_tick(self, control_tick: int | None) -> int:
        """Resolve the current fast-NFE control clock.

        Offline replay has no external simulator clock, so each NFE advances by its emitted
        action width. DOMINO passes ``take_action_cnt`` explicitly; it is authoritative even
        when the policy has emitted a chunk wider than one action.
        """
        if control_tick is None:
            candidate = self._control_tick + self._d
            latest = getattr(self, "_latest_observation", None)
            if latest is not None:
                candidate = max(candidate, self._snapshot_control_tick(latest))
            return candidate
        try:
            value = int(control_tick)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"control_tick must be an integer, got {control_tick!r}") from exc
        if value < 0:
            raise ValueError(f"control_tick must be non-negative, got {value}")
        if value < self._control_tick:
            raise ValueError(
                f"control_tick moved backwards from {self._control_tick} to {value}; "
                "the policy requests must be ordered by physical simulator time"
            )
        return value

    # ---- ring buffer -----------------------------------------------------------

    def _ingest_frame(self, obs: _model.Observation, *, frame_index: int | None = None) -> None:
        """Push the current frame(s) into the ring buffer at a physical frame index."""
        ingest_t0 = time.perf_counter()
        first = {cam: np.asarray(jax.device_get(obs.images[cam]))[0] for cam in self._cam_keys}
        # Convert float32 [-1,1] → uint8 [0,255] for SEA-RAFT.
        first_u8 = {cam: ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8) for cam, img in first.items()}
        # HWC → CHW.
        first_chw = {cam: np.transpose(img, (2, 0, 1)) for cam, img in first_u8.items()}
        if self._ring is None:
            if frame_index not in (None, 0):
                raise ValueError(
                    f"the first frame of an episode must have frame_index=0, got {frame_index}"
                )
            self._ring = _FrameRingBuffer.create(self._cam_keys, self._ring_capacity, first_chw)
        else:
            # ``current`` denotes the latest valid frame, so move to the next
            # slot before writing the new synchronized camera frame set. A physical frame gap
            # advances the ring by more than one and leaves the skipped lag indices invalid.
            target_index = self._ring.base_index + 1 if frame_index is None else int(frame_index)
            step = target_index - self._ring.base_index
            if step < 0:
                raise ValueError(
                    f"frame_index must not move backwards from {self._ring.base_index}, got {target_index}"
                )
            if step > 0:
                self._ring.advance(step)
            for cam in self._cam_keys:
                self._ring.push(cam, first_chw[cam])
        self._frame_index = self._ring.base_index
        if "frame_ingest_ms" in self.stats:
            self.stats["frame_ingest_ms"].append((time.perf_counter() - ingest_t0) * 1000)

    def _empty_flow_generation(self, *, episode_id: int) -> _FlowGeneration:
        """Return a correctly shaped invalid flow result for startup and episode resets."""
        k = self.flow_config.num_flow_steps
        h8, w8 = self._flow_grid
        return _FlowGeneration(
            episode_id=episode_id,
            source_tick=None,
            source_fast_tick=None,
            flow={
                cam: np.zeros((k, 2, h8, w8), dtype=np.float32)
                for cam in self._cam_keys
            },
            masks={cam: np.zeros((k,), dtype=bool) for cam in self._cam_keys},
            compute_ms=None,
            queue_ms=None,
            completed_at=None,
        )

    def _build_flow_snapshot(self) -> _FlowSnapshot:
        """Copy the current ring-buffer window into an immutable worker request.

        Copying here is intentional: the producer can then run independently while the main
        thread advances and overwrites the ring buffer on the next control ticks.
        """
        assert self._ring is not None
        k = self.flow_config.num_flow_steps
        stride = self.flow_config.flow_stride_frames
        n_cam = len(self._cam_keys)

        curr_frame = self._ring.get(0)
        sample = curr_frame[self._cam_keys[0]]
        prev_stacked = np.zeros((1, k * n_cam, *sample.shape), dtype=np.uint8)
        curr_stacked = np.empty_like(prev_stacked)
        valid_lags: list[bool] = []
        for ki in range(1, k + 1):
            offset = -ki * stride
            start = (ki - 1) * n_cam
            lag = None
            if self._ring.base_index + offset >= 0:
                try:
                    lag = self._ring.get(offset)
                except IndexError:
                    # A physical frame was dropped from the transport queue. Keep the lag
                    # masked instead of feeding SEA-RAFT a pair with the wrong temporal offset.
                    lag = None
            valid_lags.append(lag is not None)
            for ci, cam in enumerate(self._cam_keys):
                curr_stacked[0, start + ci] = curr_frame[cam]
                if lag is not None:
                    prev_stacked[0, start + ci] = lag[cam]

        return _FlowSnapshot(
            episode_id=self._episode_id,
            source_tick=self._frame_index,
            source_fast_tick=self._tick,
            prev=prev_stacked,
            curr=curr_stacked,
            valid_lags=np.asarray(valid_lags, dtype=bool),
            submitted_at=time.perf_counter(),
        )

    def _start_flow_process(self) -> None:
        """Start the Torch-only SEA-RAFT child used by deployment inference."""
        if not self._flow_process_enabled or self._flow_process is not None:
            return
        context = mp.get_context("spawn")
        request_queue = context.Queue(maxsize=1)
        result_queue = context.Queue(maxsize=1)
        h, w = self.flow_config.flow_image_size
        n_cam = len(self._cam_keys)
        k = self.flow_config.num_flow_steps
        input_shape = (1, k * n_cam, 3, h, w)
        output_shape = (1, k * n_cam, 2, h // 8, w // 8)
        input_bytes = int(np.prod(input_shape, dtype=np.int64))
        output_values = int(np.prod(output_shape, dtype=np.int64))
        # RawArray is inherited by the spawned child without pickle-copying the image tensors.
        # The flow worker is strictly single-flight, so these three buffers are safe to reuse
        # after the matching result has been received.
        shared_prev = context.RawArray("B", input_bytes)
        shared_curr = context.RawArray("B", input_bytes)
        shared_output = context.RawArray("f", output_values)
        prev_view = np.ndarray(input_shape, dtype=np.uint8, buffer=shared_prev)
        curr_view = np.ndarray(input_shape, dtype=np.uint8, buffer=shared_curr)
        output_view = np.ndarray(output_shape, dtype=np.float32, buffer=shared_output)
        process = context.Process(
            target=_sea_raft_worker.process_main,
            args=(
                self._sea_raft_ckpt,
                self._sea_raft_variant,
                self._sea_raft_iters,
                self._sea_raft_device,
                self._sea_raft_allow_random_init,
                self._sea_raft_precision,
                request_queue,
                result_queue,
                shared_prev,
                shared_curr,
                shared_output,
                input_shape,
                output_shape,
            ),
            name="flowpi-sea-raft-process",
            daemon=True,
        )
        self._flow_process_request_queue = request_queue
        self._flow_process_result_queue = result_queue
        self._flow_process_shared_prev = shared_prev
        self._flow_process_shared_curr = shared_curr
        self._flow_process_shared_output = shared_output
        self._flow_process_prev_view = prev_view
        self._flow_process_curr_view = curr_view
        self._flow_process_output_view = output_view
        self._flow_process = process
        process.start()

    def _compute_flow_in_process(self, snapshot: _FlowSnapshot) -> np.ndarray:
        """Submit one immutable flow snapshot and wait for its matching child result."""
        process = self._flow_process
        request_queue = self._flow_process_request_queue
        result_queue = self._flow_process_result_queue
        if process is None or request_queue is None or result_queue is None:
            raise RuntimeError("FlowPi SEA-RAFT process is not running")
        prev_view = self._flow_process_prev_view
        curr_view = self._flow_process_curr_view
        output_view = self._flow_process_output_view
        if prev_view is None or curr_view is None or output_view is None:
            raise RuntimeError("FlowPi SEA-RAFT shared buffers are not initialized")
        if snapshot.prev.shape != prev_view.shape or snapshot.curr.shape != curr_view.shape:
            raise ValueError(
                "FlowPi SEA-RAFT process received an unexpected frame shape: "
                f"{snapshot.prev.shape} and {snapshot.curr.shape}, expected {prev_view.shape}"
            )
        np.copyto(prev_view, snapshot.prev, casting="no")
        np.copyto(curr_view, snapshot.curr, casting="no")
        self._flow_process_request_id += 1
        request_id = self._flow_process_request_id
        request_queue.put(request_id)
        deadline = time.perf_counter() + self._flow_process_timeout_s
        while True:
            remaining_s = deadline - time.perf_counter()
            if remaining_s <= 0:
                raise TimeoutError(
                    f"FlowPi SEA-RAFT process did not return request {request_id} within "
                    f"{self._flow_process_timeout_s:.1f}s"
                )
            try:
                result = result_queue.get(timeout=min(remaining_s, 1.0))
            except Empty as exc:
                if not process.is_alive():
                    raise RuntimeError("FlowPi SEA-RAFT process exited before returning a result") from exc
                continue
            kind = result.get("kind") if isinstance(result, dict) else None
            if kind == "fatal":
                raise RuntimeError(f"FlowPi SEA-RAFT process failed: {result.get('error', result)}")
            if kind == "error":
                raise RuntimeError(f"FlowPi SEA-RAFT request failed: {result.get('error', result)}")
            if kind != "result" or int(result.get("request_id", -1)) != request_id:
                raise RuntimeError(f"FlowPi SEA-RAFT returned an invalid result: {result!r}")
            # Copy before allowing the next request to overwrite the single-flight output slot.
            return np.asarray(output_view, dtype=np.float32).copy()

    def _stop_flow_process(self) -> None:
        """Stop and reap the Torch child after all queued flow work has drained."""
        process = self._flow_process
        request_queue = self._flow_process_request_queue
        result_queue = self._flow_process_result_queue
        if process is None:
            return
        if request_queue is not None:
            try:
                request_queue.put_nowait(None)
            except Full:
                with contextlib.suppress(Empty):
                    request_queue.get_nowait()
                with contextlib.suppress(Full):
                    request_queue.put_nowait(None)
        process.join(timeout=min(self._flow_process_timeout_s, 10.0))
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        for queue in (request_queue, result_queue):
            if queue is not None:
                with contextlib.suppress(Exception):
                    queue.cancel_join_thread()
                    queue.close()
        self._flow_process = None
        self._flow_process_request_queue = None
        self._flow_process_result_queue = None
        self._flow_process_shared_prev = None
        self._flow_process_shared_curr = None
        self._flow_process_shared_output = None
        self._flow_process_prev_view = None
        self._flow_process_curr_view = None
        self._flow_process_output_view = None

    def _run_flow(self, snapshot: _FlowSnapshot) -> _FlowGeneration:
        """Run one SEA-RAFT request on the dedicated Torch device."""
        raft_t0 = time.perf_counter()
        flow_start_s = self._record_timestamp("_flow_start_times_s")
        if flow_start_s is not None:
            self._last_flow_start_s = flow_start_s
        if self._flow_process_enabled:
            flow = self._compute_flow_in_process(snapshot)
        else:
            assert self._raft is not None
            flow = self._raft.compute(snapshot.prev, snapshot.curr)  # [1, K*n_cam, 2, h8, w8]
        flow_ms = (time.perf_counter() - raft_t0) * 1000
        _, _, _, h8, w8 = flow.shape
        if (h8, w8) != self._flow_grid:
            raise ValueError(
                f"SEA-RAFT produced a {h8}x{w8} flow grid but the model expects "
                f"{self._flow_grid[0]}x{self._flow_grid[1]} (flow_image_size="
                f"{self.flow_config.flow_image_size}). Feed the runtime full-resolution camera "
                "frames; do not resize images to the model resolution before the runtime sees them."
            )
        k = self.flow_config.num_flow_steps
        n_cam = len(self._cam_keys)
        flow = flow.reshape(k, n_cam, 2, h8, w8)

        result = {}
        masks = {}
        valid_mask = snapshot.valid_lags
        for ci, cam in enumerate(self._cam_keys):
            raw = flow[:, ci].copy()  # [K, 2, h8, w8]
            finite_mask = np.isfinite(raw).all(axis=(1, 2, 3))
            effective_mask = valid_mask & finite_mask
            if hasattr(self, "_flow_nonfinite_lags"):
                self._flow_nonfinite_lags += int(np.count_nonzero(valid_mask & ~finite_mask))
            # Keep one invalid lag from poisoning the whole action NFE.  This is primarily a
            # guard for random-init smoke tests and transient CUDA/AMP numerical faults; normal
            # checkpointed inference should keep every valid SEA-RAFT lag finite.
            raw[~effective_mask] = 0.0
            result[cam] = normalize_flow(raw, self.flow_config.flow_scale, self.flow_config.flow_clamp)
            masks[cam] = effective_mask.copy()
        flow_complete_s = self._record_timestamp("_flow_complete_times_s")
        if flow_complete_s is not None:
            if self._last_flow_complete_s is not None and "flow_interval_ms" in self.stats:
                self.stats["flow_interval_ms"].append((flow_complete_s - self._last_flow_complete_s) * 1000)
            self._last_flow_complete_s = flow_complete_s
            # Keep _flow_times_s as the public flow-update event stream for compatibility; its
            # meaning is now completion time rather than synchronous request start time.
            self._flow_times_s.append(flow_complete_s)
        queue_ms = (raft_t0 - snapshot.submitted_at) * 1000
        self.stats["flow_ms"].append(flow_ms)
        self.stats["flow_compute_ms"].append(flow_ms)
        self.stats["flow_queue_ms"].append(queue_ms)
        return _FlowGeneration(
            episode_id=snapshot.episode_id,
            source_tick=snapshot.source_tick,
            source_fast_tick=snapshot.source_fast_tick,
            flow=result,
            masks=masks,
            compute_ms=flow_ms,
            queue_ms=queue_ms,
            completed_at=time.monotonic(),
        )

    def _flow_worker_run(self, snapshot: _FlowSnapshot) -> None:
        """Run flow requests back-to-back, consuming only the latest pending request."""
        try:
            while True:
                generation = self._run_flow(snapshot)
                with self._flow_lock:
                    if generation.episode_id != self._episode_id:
                        self._flow_generation_drops += 1
                    else:
                        active = self._active_flow
                        if active is None or active.source_tick is None or generation.source_tick >= active.source_tick:
                            self._active_flow = generation
                        else:
                            self._flow_generation_drops += 1
                    self._flow_completed += 1
                    next_snapshot = self._flow_mailbox
                    self._flow_mailbox = None
                    if next_snapshot is None:
                        self._flow_busy = False
                        return
                    snapshot = next_snapshot
        finally:
            with self._flow_lock:
                self._flow_busy = False

    def _check_flow_errors(self) -> None:
        """Re-raise completed flow-worker exceptions in the main control thread."""
        if not self._flow_futures:
            return
        remaining: list[Future] = []
        for future in self._flow_futures:
            if future.done():
                future.result()
            else:
                remaining.append(future)
        self._flow_futures = remaining

    def _submit_flow(self, snapshot: _FlowSnapshot) -> None:
        """Submit a flow request, coalescing while the dedicated worker is busy."""
        # Validate the camera geometry even during startup, when no valid lag exists yet and
        # the actual SEA-RAFT call is intentionally skipped. This keeps a model-resolution
        # (224x224) observation from silently entering the full-resolution flow ring.
        expected_h, expected_w = self.flow_config.flow_image_size
        actual_h, actual_w = snapshot.curr.shape[-2:]
        if (actual_h, actual_w) != (expected_h, expected_w):
            raise ValueError(
                f"SEA-RAFT input flow grid frames have shape {actual_h}x{actual_w}, but the runtime expects "
                f"{expected_h}x{expected_w} (flow_image_size={self.flow_config.flow_image_size}). "
                "Feed full-resolution camera frames; do not resize images to the model resolution "
                "before the runtime sees them."
            )
        # During the first ``K * stride`` frames every lag is invalid and the model consumes an
        # all-zero masked flow. Avoid spending a full SEA-RAFT forward pass on those zeros.
        if not bool(np.any(snapshot.valid_lags)):
            return
        submit_s = self._record_timestamp("_flow_submit_times_s")
        if submit_s is not None:
            self._flow_submitted += 1
        if not self._async_flow:
            generation = self._run_flow(snapshot)
            with self._flow_lock:
                self._active_flow = generation
                self._flow_completed += 1
            return

        assert self._flow_executor is not None
        with self._flow_lock:
            if self._flow_busy:
                self._flow_mailbox = snapshot
                self._flow_coalesced += 1
                return
            self._flow_busy = True
            future = self._flow_executor.submit(self._flow_worker_run, snapshot)
            self._flow_futures.append(future)

    def _current_flow(self) -> _FlowGeneration:
        """Read the latest completed flow atomically for the current episode."""
        with self._flow_lock:
            generation = self._active_flow
        if generation is None or generation.episode_id != self._episode_id:
            return self._empty_flow_generation(episode_id=self._episode_id)
        return generation

    def _compute_flow(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Synchronous compatibility helper used when ``async_flow=False``."""
        snapshot = self._build_flow_snapshot()
        generation = self._run_flow(snapshot)
        with self._flow_lock:
            self._active_flow = generation
            self._flow_completed += 1
        return generation.flow, generation.masks

    # ---- public API ------------------------------------------------------------

    def _check_slow_errors(self) -> None:
        """Re-raise completed slow-worker exceptions in the calling thread.

        A failed prefill must fail the replay/serving loop loudly instead of silently dropping
        the prefix generation.
        """
        if not self._slow_futures:
            return
        remaining: list[Future] = []
        for future in self._slow_futures:
            if future.done():
                future.result()  # raises the worker exception, if any
            else:
                remaining.append(future)
        self._slow_futures = remaining

    def _check_worker_errors(self) -> None:
        """Surface exceptions from either background inference worker."""
        self._check_slow_errors()
        self._check_flow_errors()

    def _prefill(self, observation: _model.Observation) -> tuple[Any, jax.Array]:
        """Run the VLM prefix encoder on a (preprocessed) observation, timing the prefill."""
        t0 = time.perf_counter()
        self._record_timestamp("_slow_prefill_start_times_s")
        if hasattr(self, "_slow_prefills_started"):
            self._slow_prefills_started += 1
        # Lightweight test doubles may construct the object without running ``__init__``. Keep
        # the helper usable for those callers while normal runtimes always provide both devices.
        slow_device = getattr(self, "_slow_jax_device", getattr(self, "_jax_device", None))
        fast_device = getattr(self, "_jax_device", slow_device)
        observation = _place_tree(observation, slow_device)
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_forward = getattr(self, "_slow_prefix_forward", None)
        slow_model = getattr(self, "_slow_model", getattr(self, "model", None))
        original_prefix_method = getattr(self, "_slow_prefix_method", None)
        current_prefix_method = getattr(slow_model, "_prefix_forward", None)
        # ``module_jit`` intentionally freezes a bound method.  For the single-replica path,
        # honor an explicit instance-level replacement (used by error propagation tests and
        # useful for serving diagnostics); otherwise keep the compiled fast path.
        if (
            slow_model is getattr(self, "model", None)
            and original_prefix_method is not None
            and current_prefix_method is not None
            and getattr(current_prefix_method, "__func__", None)
            is not getattr(original_prefix_method, "__func__", None)
        ):
            prefix_forward = current_prefix_method
        if prefix_forward is None:
            # Preserve the small direct-call surface used by unit-test doubles created with
            # ``object.__new__``; production runtimes install the jitted slow replica above.
            prefix_forward = self.model._prefix_forward  # noqa: SLF001
        kv_cache, prefix_mask = prefix_forward(observation)
        # JAX dispatches device work asynchronously. Do not publish or record the prefill as
        # complete until every cache and mask leaf is materialised; otherwise the slow worker can
        # queue multiple VLM prefills on the device and under-report the service time.
        jax.block_until_ready((kv_cache, prefix_mask))
        self.stats["prefill_ms"].append((time.perf_counter() - t0) * 1000)
        if hasattr(self, "_slow_prefills_completed"):
            self._slow_prefills_completed += 1
        self._record_timestamp("_slow_prefill_complete_times_s")
        return _place_tree(kv_cache, fast_device), _place_tree(prefix_mask, fast_device)

    def _publish(self, *, snapshot: _ObservationSnapshot, kv_cache: Any, prefix_mask: jax.Array) -> None:
        """Atomically publish a completed prefill.

        Cross-episode results are dropped. A completed generation that is *older* than the
        currently pending one never replaces it: the pending slot is monotonically fresh, so a
        slow worker finishing behind a newer synchronous refresh cannot regress the prefix.
        """
        with self._slow_lock:
            if snapshot.episode_id != self._episode_id:
                self.num_generation_drops += 1
                return
            pending = self._pending_prefix
            candidate = PrefixGeneration(
                episode_id=snapshot.episode_id,
                source_tick=snapshot.source_tick,
                kv_cache=kv_cache,
                prefix_mask=prefix_mask,
                source_control_tick=self._snapshot_control_tick(snapshot),
                source_wall_s=snapshot.source_wall_s,
            )
            if pending is not None and self._generation_order(pending) > self._generation_order(candidate):
                return
            self._pending_prefix = candidate
            if hasattr(self, "_slow_prefix_published"):
                self._slow_prefix_published += 1

    def _precompile_fast_tick(self, observation: _model.Observation) -> None:
        """Compile the flow-bearing denoise signature during episode warm-up.

        Without this probe, the first real tick pays the full XLA compilation cost while the
        evaluator is waiting for an action. The probe is pure: its returned state is discarded,
        so it does not advance the streaming trajectory or RNG sequence.
        """
        state = self._streaming_state
        if state is None:
            return
        batch_size = observation.state.shape[0]
        k = self.flow_config.num_flow_steps
        h8, w8 = self._flow_grid
        zero_flow = {
            cam: jnp.zeros((batch_size, k, 2, h8, w8), dtype=jnp.float32)
            for cam in self._cam_keys
        }
        zero_masks = {
            cam: jnp.zeros((batch_size, k), dtype=jnp.bool_) for cam in self._cam_keys
        }
        probe_observation = _replace_dataclass_fast(
            observation,
            flow=zero_flow,
            flow_masks=zero_masks,
            flow_delay=jnp.zeros((batch_size,), dtype=jnp.int32),
            vlm_delay=jnp.zeros((batch_size,), dtype=jnp.int32),
        )
        probe_observation = _place_tree(probe_observation, self._jax_device)
        # Compile the RNG helper before the first real tick. The runtime tick uses the seed key
        # with the exact physical/NFE tick as the fold-in value, preserving the former stream.
        probe_rng = self._fast_rng_fold_in(self._fast_rng_seed, 1)
        jax.block_until_ready(probe_rng)
        # ``d`` is a static argument because the emitted chunk has a static shape. Warm every
        # deployment-supported width once during episode startup so the first latency-driven d
        # change does not stall the DOMINO history mailbox with a surprise XLA compilation.
        for candidate_d in range(1, self._precompile_d_max + 1):
            compile_t0 = time.perf_counter()
            emit, probe_state = self._fast_denoise_step(
                state,
                probe_observation,
                probe_rng,
                d=candidate_d,
            )
            jax.block_until_ready((emit, probe_state))
            self.stats.setdefault("fast_jit_compile_ms", []).append((time.perf_counter() - compile_t0) * 1000)

    def ingest_observation(
        self,
        observation: _model.Observation,
        *,
        update_latest: bool = True,
        submit_flow: bool = True,
        source_control_tick: int | None = None,
        source_frame_tick: int | None = None,
        place_observation: bool = True,
    ) -> _model.Observation:
        """Ingest one RGB frame without running a fast NFE.

        The DOMINO continuous client can produce several simulator observations while one policy
        RPC is in flight. Keeping those frames in the runtime ring preserves the training-time
        camera stride and flow history; only the newest frame runs the next denoising step. The
        optional ``source_control_tick`` records the simulator clock independently of the
        retained image-frame index, which is essential when the client drops queued RGB frames.
        ``source_frame_tick`` is the physical image sequence number; gaps are represented as
        invalid ring slots so SEA-RAFT never treats a non-adjacent pair as an adjacent frame.

        ``update_latest=False`` is used for intermediate history frames. Those frames are needed
        by the optical-flow ring, but they are not valid slow-channel prefix candidates because
        they never run a fast policy tick. The following full observation (the one passed to
        ``tick``) becomes the latest slow-channel snapshot. ``submit_flow=False`` can be used for
        a batch of intermediate frames: the ring is advanced for every frame, while only the
        final frame builds/submits a SEA-RAFT snapshot. This avoids copying the same large flow
        window repeatedly when an image RPC has accumulated several physical frames.
        ``place_observation=False`` keeps the image-only path host-resident: SEA-RAFT consumes
        host-side snapshots, and the following state-only NFE supplies the device-resident state.
        """
        if self._streaming_state is None:
            raise RuntimeError("call warm_start before ingest_observation")
        self._check_worker_errors()
        if place_observation:
            observation = _place_tree(observation, self._jax_device)
        if source_control_tick is None:
            resolved_source_control_tick = self._control_tick
        else:
            try:
                resolved_source_control_tick = int(source_control_tick)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"source_control_tick must be an integer, got {source_control_tick!r}"
                ) from exc
            if resolved_source_control_tick < 0:
                raise ValueError(
                    f"source_control_tick must be non-negative, got {resolved_source_control_tick}"
                )
        if source_frame_tick is None:
            resolved_source_frame_tick = self._frame_index + 1
        else:
            try:
                resolved_source_frame_tick = int(source_frame_tick)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"source_frame_tick must be an integer, got {source_frame_tick!r}"
                ) from exc
            if resolved_source_frame_tick < 0:
                raise ValueError(
                    f"source_frame_tick must be non-negative, got {resolved_source_frame_tick}"
                )
        if resolved_source_frame_tick > self._frame_index + 1:
            self._image_frame_gaps += resolved_source_frame_tick - (self._frame_index + 1)
        self._ingest_frame(observation, frame_index=resolved_source_frame_tick)
        self._record_timestamp("_image_ingest_times_s")
        source_wall_s = time.monotonic()
        self._ingest_wall[resolved_source_frame_tick] = source_wall_s
        if update_latest:
            snapshot = _ObservationSnapshot(
                self._episode_id,
                resolved_source_frame_tick,
                observation,
                resolved_source_control_tick,
                source_wall_s,
            )
            with self._slow_lock:
                latest = self._latest_observation
                latest_order = (
                    (self._snapshot_control_tick(latest), latest.source_tick)
                    if latest is not None
                    else None
                )
                snapshot_order = (resolved_source_control_tick, resolved_source_frame_tick)
                if latest_order is None or snapshot_order >= latest_order:
                    self._latest_observation = snapshot
        if submit_flow:
            self._submit_flow(self._build_flow_snapshot())
        return observation

    def warm_start(self, observation: _model.Observation) -> None:
        """Begin a new episode and synchronously install its initial slow prefix.

        Any in-flight slow refresh of the previous episode is invalidated. The returned
        streaming state is ready for ``emit`` and the first ``tick``; an initial
        ``refresh_prefix(..., wait=True)`` is unnecessary.
        """
        warm_t0 = time.perf_counter()
        # Do not let a previous episode's background prefill use the slow replica while the new
        # episode performs its synchronous initial prefill.
        if self._slow_futures:
            futures, self._slow_futures = self._slow_futures, []
            for future in futures:
                future.result()
        self._check_worker_errors()
        observation = _place_tree(observation, self._jax_device)
        # A new episode starts from an empty ring buffer: stale frames of the previous episode
        # would otherwise leak into the first ticks' flow (cross-episode flow contamination).
        self._ring = None
        self._frame_index = 0
        self._control_tick = 0
        self._image_frame_gaps = 0
        self._prefix_source_tick = 0
        self._prefix_source_frame_tick = 0
        self._episode_id += 1
        source_wall_s = time.monotonic()
        snapshot = _ObservationSnapshot(self._episode_id, 0, observation, 0, source_wall_s)
        with self._slow_lock:
            self._latest_observation = snapshot
            self._pending_prefix = None
            self._slow_mailbox = None
        with self._flow_lock:
            # An old flow worker may still be finishing on the previous episode. Its result is
            # tagged with the old episode id and will be dropped; a new request can use the same
            # worker without blocking episode reset.
            self._active_flow = self._empty_flow_generation(episode_id=self._episode_id)
            self._flow_mailbox = None
        self._ingest_frame(observation)
        self._record_timestamp("_image_ingest_times_s")
        self._ingest_wall = {0: source_wall_s}

        initial_prefix = None
        if self._slow_model is not self.model:
            initial_prefix = self._prefill(observation)
        warm_rng = _place_tree(jax.random.key(0), self._jax_device)
        self._streaming_state = self._fast_warm_start(
            warm_rng,
            observation,
            num_steps=10,
            d=self._d,
            prefix=initial_prefix,
        )
        self._precompile_fast_tick(observation)
        self._tick = 0
        self._last_tick_start_s = None
        self._last_flow_start_s = None
        self._last_flow_complete_s = None
        if "warm_start_ms" in self.stats:
            self.stats["warm_start_ms"].append((time.perf_counter() - warm_t0) * 1000)

    def reset_metrics(self) -> None:
        """Restart runtime telemetry after an optional server-side compilation warm-up.

        A deployment prewarm intentionally runs before the first real RoboTwin episode. Its
        compilation time must not be mixed into the episode's control-rate denominator.
        Runtime state is left untouched; only clocks, samples, and counters are reset.
        """
        self._metrics_start = time.perf_counter()
        for name in (
            "_tick_times_s",
            "_image_ingest_times_s",
            "_fast_state_tick_times_s",
            "_flow_submit_times_s",
            "_flow_start_times_s",
            "_flow_complete_times_s",
            "_flow_times_s",
            "_slow_refresh_request_times_s",
            "_slow_prefill_start_times_s",
            "_slow_prefill_complete_times_s",
            "_prefix_install_times_s",
        ):
            getattr(self, name).clear()
        for values in self.stats.values():
            values.clear()
        self.stats["action_chunk_d"] = [float(self._d)]
        self._action_chunk_d_history = [self._d]
        self.telemetry.clear()
        self._ingest_wall.clear()
        self._image_frame_gaps = 0
        self.num_generation_drops = 0
        self._flow_submitted = 0
        self._flow_completed = 0
        self._flow_coalesced = 0
        self._flow_generation_drops = 0
        self._flow_nonfinite_lags = 0
        self._slow_refresh_requests = 0
        self._slow_refresh_coalesced = 0
        self._slow_prefills_started = 0
        self._slow_prefills_completed = 0
        self._slow_prefix_published = 0
        self._slow_prefix_installs = 0
        self._last_tick_start_s = None
        self._last_flow_start_s = None
        self._last_flow_complete_s = None

    def prewarm(self, observation: _model.Observation) -> None:
        """Compile all streaming signatures before the first real episode.

        The first real ``warm_start`` otherwise blocks the DOMINO evaluator while JAX compiles
        the slow prefix, fast warm-start, and every supported static action width. Calling this
        once with a shape-compatible synthetic observation moves that one-time cost to server
        startup and resets telemetry before serving real episodes.
        """
        self.warm_start(observation)
        self.reset_metrics()

    def tick(
        self,
        observation: _model.Observation,
        *,
        observation_already_ingested: bool = False,
        state_only: bool = False,
        control_tick: int | None = None,
    ) -> np.ndarray:
        """One fast NFE: fresh state + cached/latest online flow + one denoising step.

        A slow-channel refresh completed since the last tick is installed first (kv_cache +
        prefix_mask swapped together with the prefix source tick; the slow delay is then
        `control_tick - prefix_source_control_tick`, which includes state-only control steps and
        VLM compute latency. ``control_tick`` is optional for offline replay; DOMINO passes its
        authoritative physical simulator tick. Returns the ``d`` emitted actions
        (shape ``[d, action_dim]``).
        """
        if state_only and not observation_already_ingested:
            raise ValueError("state_only=True requires observation_already_ingested=True")
        current_control_tick = self._resolve_control_tick(control_tick)
        tick_t0 = time.perf_counter()
        tick_start_s = self._record_timestamp("_tick_times_s")
        state_tick_start_s = self._record_timestamp("_fast_state_tick_times_s") if state_only else None
        tick_interval_ms = None
        if tick_start_s is not None and self._last_tick_start_s is not None:
            tick_interval_ms = (tick_start_s - self._last_tick_start_s) * 1000
            if "fast_tick_interval_ms" in self.stats:
                self.stats["fast_tick_interval_ms"].append(tick_interval_ms)
        self._last_tick_start_s = tick_start_s
        self._check_worker_errors()
        # ``tick_state`` already places the cached image/state tree on the fast device. Avoid
        # traversing and re-device-putting that tree on every state-only NFE; full image ticks
        # still need the placement because their caller may provide host numpy leaves.
        if not state_only:
            observation = _place_tree(observation, self._jax_device)
        # Install the latest completed slow-prefix refresh, if any, into the active streaming
        # state. Publication is atomic: kv_cache, prefix_mask and source metadata always come
        # from the same prefix generation. A stale generation (previous episode, or computed
        # from a frame older than the active prefix) is dropped.
        install_t0 = time.perf_counter()
        with self._slow_lock:
            pending = self._pending_prefix
            self._pending_prefix = None
        state = self._streaming_state
        installed = False
        if pending is not None:
            active_order = (
                self._prefix_source_tick or 0,
                self._prefix_source_frame_tick or 0,
            )
            if pending.episode_id == self._episode_id and self._generation_order(pending) >= active_order:
                # Fresh generation (>=: an equal source tick re-computes the same prefix, which
                # is indistinguishable from the active one).
                self._prefix_source_tick = self._generation_control_tick(pending)
                self._prefix_source_frame_tick = pending.source_tick
                prefix_source_tick = jnp.full(
                    (state.action_buffer.shape[0],), self._prefix_source_tick, dtype=jnp.int32
                )
                prefix_source_tick = _place_tree(prefix_source_tick, self._jax_device)
                state = _replace_dataclass_fast(
                    state,
                    kv_cache=pending.kv_cache,
                    prefix_mask=pending.prefix_mask,
                    prefix_source_tick=prefix_source_tick,
                )
                installed = True
            else:
                # Stale generation (previous episode, or computed from a frame older than the
                # active prefix): never installed.
                self.num_generation_drops += 1

        if installed:
            if "prefix_install_ms" in self.stats:
                self.stats["prefix_install_ms"].append((time.perf_counter() - install_t0) * 1000)
            if hasattr(self, "_slow_prefix_installs"):
                self._slow_prefix_installs += 1
            self._record_timestamp("_prefix_install_times_s")

        if observation_already_ingested:
            with self._slow_lock:
                latest = self._latest_observation
            if latest is None:
                raise RuntimeError("observation_already_ingested=True requires a prior ingest_observation call")
            if state_only:
                # A state-only tick must still use the newest proprioception supplied by the
                # caller. The previous implementation silently discarded ``observation.state``
                # here and reused the old state from ``_latest_observation``, which made the
                # purported fast closed loop image/flow-only rather than state-closed-loop.
                observation = _replace_dataclass_fast(latest.observation, state=observation.state)
            else:
                # A full image was already ingested by the caller. Reuse that device-resident
                # snapshot directly; this preserves the independent image-frame clock and avoids
                # another host-state replacement before the fast NFE.
                observation = latest.observation
        else:
            # A full observation is the visual source for this NFE. Associate it with the
            # physical tick being evaluated, rather than with the number of RGB frames retained
            # by the client. This keeps the slow-delay clock correct when DOMINO drops queued
            # observations under load.
            observation = self.ingest_observation(
                observation,
                source_control_tick=current_control_tick,
                source_frame_tick=current_control_tick if control_tick is not None else None,
            )
        flow_generation = self._current_flow()
        cached_generation = getattr(self, "_flow_jax_cache_generation", None)
        cached_flow = getattr(self, "_flow_jax_cache", None)
        if cached_generation is not flow_generation or cached_flow is None:
            cached_flow = (
                {
                    cam: _place_tree(jnp.asarray(arr)[None, ...], self._jax_device)
                    for cam, arr in flow_generation.flow.items()
                },
                {
                    cam: _place_tree(jnp.asarray(mask)[None, ...], self._jax_device)
                    for cam, mask in flow_generation.masks.items()
                },
            )
            self._flow_jax_cache_generation = flow_generation
            self._flow_jax_cache = cached_flow
        flow_data_jax, flow_masks_jax = cached_flow

        # Attach the latest completed flow and report its real age. The model receives the
        # training-time bounded age embedding, while telemetry keeps the unclamped age so a slow
        # worker cannot hide that it fell behind the configured support.
        if flow_generation.source_fast_tick is None:
            flow_delay_raw = 0
            flow_age_ms = 0.0
        else:
            # State-only NFEs advance the fast closed-loop clock without necessarily producing a
            # new RGB frame. Age the flow in that same clock so the model is told when it is
            # reusing a cached visual result instead of seeing a falsely fresh delay=0.
            flow_delay_raw = max(0, self._tick - flow_generation.source_fast_tick)
            source_wall = self._ingest_wall.get(flow_generation.source_tick)
            flow_age_ms = (time.monotonic() - source_wall) * 1000 if source_wall is not None else 0.0
        flow_delay = min(flow_delay_raw, self.flow_config.flow_delay_max)
        self.stats["flow_age_ticks"].append(float(flow_delay_raw))
        self.stats["flow_age_ms"].append(float(flow_age_ms))

        # Attach the slow-channel delay using the physical/control clock. ``_frame_index`` is an
        # image-retention clock and cannot age the prefix during state-only NFEs or compensate for
        # observations dropped by the DOMINO transport queue.
        delay = max(0, current_control_tick - (self._prefix_source_tick or 0))
        delay_raw = delay
        delay = min(delay, self.flow_config.vlm_delay_max)
        if installed:
            self.stats["prefix_age_at_install"].append(delay)
            self.stats["prefix_age_at_install_raw"].append(delay_raw)
            self.stats["prefix_age_at_install_clamped"].append(delay)
            wall = pending.source_wall_s
            if wall is not None:
                self.stats["prefix_age_ms_at_install"].append((time.monotonic() - wall) * 1000)
        flow_delay_device = _place_tree(
            jnp.full((observation.state.shape[0],), flow_delay, dtype=jnp.int32),
            self._jax_device,
        )
        vlm_delay_device = _place_tree(
            jnp.full((observation.state.shape[0],), delay, dtype=jnp.int32),
            self._jax_device,
        )
        obs_with_flow = _replace_dataclass_fast(
            observation,
            flow=flow_data_jax,
            flow_masks=flow_masks_jax,
            flow_delay=flow_delay_device,
            vlm_delay=vlm_delay_device,
        )

        # Per-tick freshness telemetry. Timing fields are completed after the fast NFE below,
        # once device work is materialised.
        tick_record = {
            "tick": self._frame_index,
            "frame_index": self._frame_index,
            "control_tick": current_control_tick,
            "fast_nfe_index": self._tick,
            "mode": "state_only" if state_only else "image_tick",
            "tick_start_s": tick_start_s,
            "state_tick_start_s": state_tick_start_s,
            "tick_interval_ms": tick_interval_ms,
            "flow_source_tick": flow_generation.source_tick,
            "flow_source_fast_tick": flow_generation.source_fast_tick,
            "flow_delay_ticks": flow_delay_raw,
            "flow_delay_ticks_clamped": flow_delay,
            "flow_delay_ms": flow_age_ms,
            "flow_compute_ms": flow_generation.compute_ms,
            "flow_queue_ms": flow_generation.queue_ms,
            "flow_interval_ms": self.stats["flow_interval_ms"][-1]
            if self.stats.get("flow_interval_ms")
            else None,
            "prefix_installed": installed,
            "prefix_source_tick": self._prefix_source_tick or 0,
            "prefix_source_control_tick": self._prefix_source_tick or 0,
            "prefix_source_frame_tick": self._prefix_source_frame_tick or 0,
            "delay_ticks": delay,
            "delay_ticks_raw": delay_raw,
            "delay_ticks_clamped": delay,
        }
        # Prune the ingestion wall-clock history: only the delay window (plus a generous margin
        # for a slow VLM) is ever needed to report the prefix age in milliseconds.
        prune_before = self._frame_index - (self.flow_config.vlm_delay_max + 1024)
        if prune_before > 0 and len(self._ingest_wall) > self.flow_config.vlm_delay_max + 1024:
            for k in list(self._ingest_wall):
                if k < prune_before:
                    del self._ingest_wall[k]

        # One NFE (per-tick RNG: must not repeat after a prefix refresh resets the age).
        rng = self._fast_rng_fold_in(self._fast_rng_seed, self._tick)
        self._tick += 1
        fast_model_t0 = time.perf_counter()
        emit, new_state = self._fast_denoise_step(state, obs_with_flow, rng, d=self._d)
        self._streaming_state = new_state
        emit_np = np.asarray(emit[0])  # [d, action_dim]; materialises asynchronous JAX work.
        fast_model_ms = (time.perf_counter() - fast_model_t0) * 1000
        tick_total_ms = (time.perf_counter() - tick_t0) * 1000
        # Commit the physical clock only after the NFE has produced its action chunk. If the
        # model raises, the caller can fail the episode without silently advancing freshness
        # telemetry.
        self._control_tick = current_control_tick
        if "fast_model_ms" in self.stats:
            self.stats["fast_model_ms"].append(fast_model_ms)
        if "tick_total_ms" in self.stats:
            self.stats["tick_total_ms"].append(tick_total_ms)
        tick_record.update(
            {
                "fast_model_ms": fast_model_ms,
                "tick_total_ms": tick_total_ms,
                "tick_end_s": self._metric_now(),
                "fast_loop_hz": 1000.0 / tick_interval_ms if tick_interval_ms and tick_interval_ms > 0 else None,
            }
        )
        self.telemetry.append(tick_record)
        return emit_np

    def tick_state(self, state: Any, *, control_tick: int | None = None) -> np.ndarray:
        """Run one fast NFE with a new state and the latest image/flow cache.

        ``state`` is the already policy-preprocessed, unbatched state vector. No image is copied
        into the frame ring and no SEA-RAFT request is submitted by this method. A separate
        ``ingest_observation`` call updates the image/flow mailbox whenever a fresh simulator RGB
        frame is available. ``control_tick`` may be supplied by a simulator; otherwise the
        runtime advances its offline clock by the emitted action width.
        """
        with self._slow_lock:
            latest = self._latest_observation
        if latest is None:
            raise RuntimeError("call warm_start before tick_state")
        state_array = jnp.asarray(state)
        expected = latest.observation.state.shape
        if state_array.ndim == len(expected) - 1:
            state_array = state_array[None, ...]
        if tuple(state_array.shape) != tuple(expected):
            raise ValueError(
                f"state-only tick expected state shape {tuple(expected)}, got {tuple(state_array.shape)}"
            )
        state_array = _place_tree(state_array, self._jax_device)
        observation = _replace_dataclass_fast(latest.observation, state=state_array)
        return self.tick(
            observation,
            observation_already_ingested=True,
            state_only=True,
            control_tick=control_tick,
        )

    def refresh_prefix(self, *, wait: bool = False) -> None:
        """Slow-channel: re-run the VLM prefix encoder and produce fresh KV cache.

        By default the prefill runs on a single background worker; the result is published
        atomically and installed into the active streaming state at the start of the next fast
        tick (the prefix source control tick is carried from the source observation, so the slow
        delay keeps counting from that physical observation while the VLM is busy). With
        ``wait=True`` the
        prefill runs synchronously in the calling thread and supersedes any in-flight refresh.

        When the refresh rate exceeds the VLM service rate, requests coalesce into a single
        latest-pending mailbox slot instead of queuing: the worker finishes its current prefill,
        publishes it, and immediately picks up the latest pending snapshot (back-to-back,
        as fast as inference allows). A completed prefill that is fresher than the active prefix
        is always published, so the prefix source tick advances monotonically even under a
        refresh storm.

        The refresh always uses the latest observation ingested by warm_start or tick. Its episode
        id and both source clocks are captured with that observation as one immutable snapshot, so
        the worker cannot compute a prefix from one tick while labeling it as another. Worker
        exceptions are re-raised in the main thread at the next tick/refresh/close.
        """
        self._check_worker_errors()
        if hasattr(self, "_slow_refresh_requests"):
            self._slow_refresh_requests += 1
        self._record_timestamp("_slow_refresh_request_times_s")
        with self._slow_lock:
            snapshot = self._latest_observation
        if snapshot is None:
            raise RuntimeError("call warm_start before refresh_prefix")
        if wait:
            # warm_start and a repeated synchronous refresh for the same frame refer to the
            # already-active prefix. Keep the old call pattern a harmless no-op.
            if (
                self._streaming_state is not None
                and snapshot.source_tick == self._prefix_source_frame_tick
                and self._snapshot_control_tick(snapshot) == (self._prefix_source_tick or 0)
            ):
                return
            kv_cache, prefix_mask = self._prefill(snapshot.observation)
            self._publish(snapshot=snapshot, kv_cache=kv_cache, prefix_mask=prefix_mask)
            return
        with self._slow_lock:
            if self._slow_busy:
                # A prefill is already running: coalesce into a single latest-pending slot
                # (supersedes any older pending snapshot).
                self._slow_mailbox = snapshot
                if hasattr(self, "_slow_refresh_coalesced"):
                    self._slow_refresh_coalesced += 1
            else:
                self._slow_busy = True
                future = self._slow_executor.submit(self._slow_run, snapshot)
                self._slow_futures.append(future)

    def _slow_run(self, snapshot: _ObservationSnapshot) -> None:
        """Slow worker loop: prefill -> publish -> immediately consume the latest pending
        snapshot and repeat.

        Back-to-back scheduling: the VLM is never idle while a refresh is pending, at most one
        prefill runs at a time, and the refresh queue is bounded to a single mailbox slot
        (no queued-but-doomed jobs). A completed prefill is published whenever it belongs to the
        current episode and is no older than the pending generation; the per-tick install check
        drops anything older than the already-active prefix.
        """
        try:
            while True:
                kv_cache, prefix_mask = self._prefill(snapshot.observation)
                self._publish(snapshot=snapshot, kv_cache=kv_cache, prefix_mask=prefix_mask)
                with self._slow_lock:
                    mailbox = self._slow_mailbox
                    self._slow_mailbox = None
                    if mailbox is not None:
                        snapshot = mailbox
                        continue
                    self._slow_busy = False
                    return
        finally:
            # A worker exception must not leave the bounded mailbox permanently marked busy.
            with self._slow_lock:
                self._slow_busy = False

    def metrics_summary(self) -> dict[str, Any]:
        """Return latency, frequency, freshness, and queueing metrics for this runtime."""
        stats = {name: list(values) for name, values in self.stats.items() if values}
        telemetry = [dict(entry) for entry in self.telemetry]
        duration_s = self._metric_now()
        return {
            "schema_version": 1,
            "duration_s": duration_s,
            "config": {
                "action_chunk_d": self._d,
                "action_chunk_d_history": list(self._action_chunk_d_history),
                "control_tick": self._control_tick,
                "precompile_d_max": self._precompile_d_max,
                "async_flow": self._async_flow,
                "flow_process": self._flow_process_enabled,
                "sea_raft_precision": self._sea_raft_precision,
                "num_flow_steps": self.flow_config.num_flow_steps,
                "flow_stride_frames": self.flow_config.flow_stride_frames,
                "flow_delay_max": self.flow_config.flow_delay_max,
                "vlm_delay_max": self.flow_config.vlm_delay_max,
            },
            "counters": {
                "fast_ticks": len(self._tick_times_s),
                "fast_state_ticks": len(self._fast_state_tick_times_s),
                "control_ticks": self._control_tick,
                "image_frames": len(self._image_ingest_times_s),
                "image_frame_gaps": self._image_frame_gaps,
                "flow_updates": len(self._flow_times_s),
                "flow_submitted": self._flow_submitted,
                "flow_completed": self._flow_completed,
                "flow_coalesced": self._flow_coalesced,
                "flow_generation_drops": self._flow_generation_drops,
                "flow_nonfinite_lags": self._flow_nonfinite_lags,
                "slow_refresh_requests": self._slow_refresh_requests,
                "slow_refresh_coalesced": self._slow_refresh_coalesced,
                "slow_prefills_started": self._slow_prefills_started,
                "slow_prefills_completed": self._slow_prefills_completed,
                "slow_prefix_published": self._slow_prefix_published,
                "slow_prefix_installs": self._slow_prefix_installs,
                "generation_drops": self.num_generation_drops,
            },
            "rates_hz": {
                "fast_nfe": summarize_rate(self._tick_times_s),
                # Compatibility alias retained for older metric readers. It has always meant
                # one emitted action chunk/NFE, never the physical simulator action rate.
                "fast_control_loop": summarize_rate(self._tick_times_s),
                "fast_state_loop": summarize_rate(self._fast_state_tick_times_s),
                "image_ingest": summarize_rate(self._image_ingest_times_s),
                "flow_submit": summarize_rate(self._flow_submit_times_s),
                "flow_update": summarize_rate(self._flow_times_s),
                "slow_refresh_request": summarize_rate(self._slow_refresh_request_times_s),
                "slow_prefill_start": summarize_rate(self._slow_prefill_start_times_s),
                "slow_prefill_complete": summarize_rate(self._slow_prefill_complete_times_s),
                "prefix_install": summarize_rate(self._prefix_install_times_s),
            },
            "latency_ms": {name: summarize_series(values) for name, values in stats.items()},
            "stats": stats,
            "telemetry": telemetry,
        }

    def close(self) -> None:
        """Drain the slow worker and re-raise its first exception, if any. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._slow_executor.shutdown(wait=True)
        if self._flow_executor is not None:
            self._flow_executor.shutdown(wait=True)
        try:
            self._check_worker_errors()
            slow_futures, self._slow_futures = self._slow_futures, []
            flow_futures, self._flow_futures = self._flow_futures, []
            for future in (*slow_futures, *flow_futures):
                future.result()
        finally:
            self._stop_flow_process()

    def emit(self) -> np.ndarray:
        """Return the current action chunk (the first ``d`` actions in the buffer).

        Useful *after* ``warm_start`` to obtain the initial in-flight actions.
        """
        if self._streaming_state is None:
            raise RuntimeError("call warm_start before emit")
        return np.asarray(self._streaming_state.action_buffer[0, : self._d])  # [d, action_dim]
