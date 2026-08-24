"""Sequential three-device FlowPi inference runtime.

The runtime deliberately has one execution order and one action width.  A caller supplies a
full-resolution observation, SEA-RAFT computes the current flow on its Torch device, the slow
JAX replica computes the VLM prefix, and the fast JAX replica performs one streaming NFE.  There
are no simulator hooks, wall-clock schedulers, asynchronous mailboxes, or dynamic frequency
controllers in this module.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.shared import nnx_utils as _nnx_utils
from openpi.training.sea_raft import SeaRaftFlowExtractor
from openpi.transforms import normalize_flow

_CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _resolve_jax_device(spec: str | None) -> jax.Device | None:
    """Resolve ``gpu:0``/``cuda:0``/``cpu`` without touching other visible devices."""
    if spec is None:
        return None
    backend, separator, index_text = spec.partition(":")
    backend = "gpu" if backend == "cuda" else backend
    index = int(index_text) if separator else 0
    devices = jax.devices(backend)
    if not 0 <= index < len(devices):
        raise ValueError(f"jax device {spec!r}: backend {backend!r} has {len(devices)} devices")
    return devices[index]


def _place_tree(tree: Any, device: jax.Device | None) -> Any:
    if device is None:
        return tree
    return jax.tree.map(
        lambda value: jax.device_put(value, device) if isinstance(value, (jax.Array, np.ndarray)) else value,
        tree,
    )


@functools.cache
def _dataclass_fields(cls: type) -> tuple[dataclasses.Field[Any], ...]:
    return tuple(dataclasses.fields(cls))


def _replace_dataclass(instance: Any, **changes: Any) -> Any:
    """Replace model-owned leaves without invoking the observation type checker repeatedly."""
    fields = _dataclass_fields(type(instance))
    field_names = {field.name for field in fields}
    unknown = set(changes).difference(field_names)
    if unknown:
        raise TypeError(f"Unknown fields for {type(instance).__name__}: {sorted(unknown)}")
    clone = object.__new__(type(instance))
    for field in fields:
        object.__setattr__(clone, field.name, changes.get(field.name, getattr(instance, field.name)))
    return clone


def _place_model(model: _pi0.Pi0, device: jax.Device | None) -> _pi0.Pi0:
    if device is None:
        return model
    graphdef, state = nnx.split(model)
    return nnx.merge(graphdef, _place_tree(state, device))


@functools.lru_cache(maxsize=1)
def _register_streaming_state() -> None:
    jax.tree_util.register_dataclass(
        _pi0.Pi0.StreamingState,
        data_fields=("action_buffer", "tau", "kv_cache", "prefix_mask"),
        meta_fields=(),
    )


def _fold_in(seed: jax.Array, tick: int) -> jax.Array:
    return jax.random.fold_in(seed, tick)


def _as_uint8(value: Any) -> np.ndarray:
    """Convert one batched model image to CHW uint8 for SEA-RAFT."""
    image = np.asarray(value)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"expected one batched camera image [1,H,W,3], got {image.shape}")
    image = image[0]
    if image.shape[-1] != 3:
        raise ValueError(f"expected RGB camera image [H,W,3], got {image.shape}")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image.astype(np.float32, copy=False)
    if image.size and float(image.min()) < -0.1:
        image = (image + 1.0) * 127.5
    elif image.size and float(image.max()) <= 1.1:
        image = image * 255.0
    return np.ascontiguousarray(np.transpose(np.clip(image, 0, 255).astype(np.uint8), (2, 0, 1)))


class _FrameRing:
    """Small contiguous frame history used by the synchronous flow call."""

    def __init__(self, first: dict[str, np.ndarray], capacity: int):
        self.keys = tuple(first)
        self.capacity = capacity
        height, width = first[self.keys[0]].shape[-2:]
        self.frames = {key: np.zeros((capacity, 3, height, width), dtype=np.uint8) for key in self.keys}
        self.indices = np.full((capacity,), -1, dtype=np.int64)
        self.cursor = 0
        self.index = 0
        self.indices[0] = 0
        for key, value in first.items():
            self.frames[key][0] = value

    def append(self, frame: dict[str, np.ndarray]) -> None:
        self.cursor = (self.cursor + 1) % self.capacity
        self.index += 1
        self.indices[self.cursor] = self.index
        for key, value in frame.items():
            self.frames[key][self.cursor] = value

    def get(self, offset: int) -> dict[str, np.ndarray] | None:
        target = self.index + offset
        if target < 0 or offset > 0 or -offset >= self.capacity:
            return None
        slot = (self.cursor + offset) % self.capacity
        if self.indices[slot] != target:
            return None
        return {key: values[slot] for key, values in self.frames.items()}


class FlowPiRuntime:
    """One sequential FlowPi stream with separate slow JAX, fast JAX and SEA-RAFT devices."""

    def __init__(
        self,
        model: _pi0.Pi0,
        *,
        flow_config: Any,
        slow_model: _pi0.Pi0 | None = None,
        sea_raft_ckpt: str,
        sea_raft_variant: str = "M",
        sea_raft_iters: int | None = None,
        sea_raft_device: str = "cuda:2",
        sea_raft_precision: str = "fp16",
        jax_device: str = "gpu:1",
        slow_jax_device: str = "gpu:0",
    ):
        if not sea_raft_ckpt:
            raise ValueError("sea_raft_ckpt must be an explicit checkpoint path")
        self.flow_config = flow_config
        self.fast_device = _resolve_jax_device(jax_device)
        self.slow_device = _resolve_jax_device(slow_jax_device)
        self.model = _place_model(model, self.fast_device)
        self.slow_model = _place_model(slow_model or model, self.slow_device)
        _register_streaming_state()

        self._fast_warm_start = _nnx_utils.module_jit(
            self.model.warm_start,
            static_argnames=("num_steps", "d"),
        )
        self._fast_denoise_step = _nnx_utils.module_jit(
            self.model.denoise_step,
            static_argnames=("d",),
        )
        self._slow_prefix_forward = _nnx_utils.module_jit(self.slow_model._prefix_forward)  # noqa: SLF001
        self._fast_rng_seed = _place_tree(jax.random.key(0), self.fast_device)
        self._fast_rng_fold_in = jax.jit(_fold_in)

        self._raft = SeaRaftFlowExtractor(
            ckpt_path=sea_raft_ckpt,
            variant=sea_raft_variant,
            iters=sea_raft_iters,
            device=sea_raft_device,
            precision=sea_raft_precision,
        )
        self._sea_raft_device = sea_raft_device
        self._cam_keys = _CAMERA_KEYS
        self._ring: _FrameRing | None = None
        self._state: _pi0.Pi0.StreamingState | None = None
        self._tick = 0
        self._telemetry: list[dict[str, float | int]] = []

    def _frames_from_observation(self, observation: _model.Observation) -> dict[str, np.ndarray]:
        missing = set(self._cam_keys).difference(observation.images)
        if missing:
            raise ValueError(f"observation is missing camera keys: {sorted(missing)}")
        frames = {key: _as_uint8(observation.images[key]) for key in self._cam_keys}
        height, width = frames[self._cam_keys[0]].shape[-2:]
        expected = tuple(self.flow_config.flow_image_size)
        if (height, width) != expected:
            raise ValueError(
                f"FlowPi expects full-resolution frames {expected}, got {(height, width)}. "
                "Do not resize images before passing them to the runtime."
            )
        if self._ring is None:
            capacity = (
                max(
                    int(self.flow_config.vlm_delay_max),
                    int(self.flow_config.flow_delay_max)
                    + int(self.flow_config.num_flow_steps) * int(self.flow_config.flow_stride_frames),
                )
                + 1
            )
            self._ring = _FrameRing(frames, max(capacity, 2))
        else:
            self._ring.append(frames)
        return frames

    def _flow_observation(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compute current-target flow for every available trained lag."""
        k_num = int(self.flow_config.num_flow_steps)
        stride = int(self.flow_config.flow_stride_frames)
        height, width = tuple(self.flow_config.flow_image_size)
        low_shape = (height // 8, width // 8)
        flows = {key: np.zeros((k_num, 2, *low_shape), dtype=np.float32) for key in self._cam_keys}
        masks = {key: np.zeros((k_num,), dtype=bool) for key in self._cam_keys}
        assert self._ring is not None
        current = self._ring.get(0)
        if current is None:
            return flows, masks
        for lag in range(1, k_num + 1):
            previous = self._ring.get(-lag * stride)
            if previous is None:
                continue
            prev_batch = np.stack([previous[key] for key in self._cam_keys], axis=0)[None]
            curr_batch = np.stack([current[key] for key in self._cam_keys], axis=0)[None]
            result = self._raft.compute(prev_batch, curr_batch)[0]
            for camera_index, key in enumerate(self._cam_keys):
                flows[key][lag - 1] = normalize_flow(
                    result[camera_index],
                    self.flow_config.flow_scale,
                    self.flow_config.flow_clamp,
                )
                masks[key][lag - 1] = True
        return flows, masks

    def _prefix(self, observation: _model.Observation) -> tuple[Any, jax.Array]:
        slow_observation = _place_tree(observation, self.slow_device)
        slow_observation = _model.preprocess_observation(None, slow_observation, train=False)
        kv_cache, prefix_mask = self._slow_prefix_forward(slow_observation)
        jax.block_until_ready((kv_cache, prefix_mask))
        return _place_tree(kv_cache, self.fast_device), _place_tree(prefix_mask, self.fast_device)

    def _with_flow(
        self,
        observation: _model.Observation,
        flows: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
    ) -> _model.Observation:
        flow = {key: _place_tree(jnp.asarray(value)[None, ...], self.fast_device) for key, value in flows.items()}
        flow_masks = {key: _place_tree(jnp.asarray(value)[None, ...], self.fast_device) for key, value in masks.items()}
        zeros = _place_tree(jnp.zeros((observation.state.shape[0],), dtype=jnp.int32), self.fast_device)
        return _replace_dataclass(
            _place_tree(observation, self.fast_device),
            flow=flow,
            flow_masks=flow_masks,
            flow_delay=zeros,
            vlm_delay=zeros,
        )

    def warm_start(self, observation: _model.Observation) -> None:
        """Reset the stream and run the initial full sampler."""
        self._ring = None
        self._state = None
        self._tick = 0
        self._telemetry = []
        self._frames_from_observation(observation)
        prefix = self._prefix(observation)
        fast_observation = _place_tree(observation, self.fast_device)
        rng = _place_tree(jax.random.key(0), self.fast_device)
        self._state = self._fast_warm_start(
            rng,
            fast_observation,
            num_steps=10,
            d=1,
            prefix=prefix,
        )
        jax.block_until_ready((self._state.action_buffer, self._state.tau))

    def tick(self, observation: _model.Observation) -> np.ndarray:
        """Run one sequential flow -> slow-prefix -> fast-NFE step."""
        if self._state is None:
            raise RuntimeError("call warm_start before tick")
        start = time.perf_counter()
        self._frames_from_observation(observation)
        flows, masks = self._flow_observation()
        fast_observation = self._with_flow(observation, flows, masks)
        prefix = self._prefix(observation)
        state = _replace_dataclass(self._state, kv_cache=prefix[0], prefix_mask=prefix[1])
        rng = self._fast_rng_fold_in(self._fast_rng_seed, self._tick + 1)
        emitted, self._state = self._fast_denoise_step(
            state,
            fast_observation,
            rng,
            d=1,
        )
        result = np.asarray(emitted[0])
        self._tick += 1
        self._telemetry.append({"tick": self._tick, "step_ms": (time.perf_counter() - start) * 1000})
        return result

    def emit(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("call warm_start before emit")
        return np.asarray(self._state.action_buffer[0, :1])

    def metrics_summary(self) -> dict[str, Any]:
        return {
            "devices": {
                "slow_jax": str(self.slow_device),
                "fast_jax": str(self.fast_device),
                "sea_raft": self._sea_raft_device,
            },
            "steps": list(self._telemetry),
        }

    def close(self) -> None:
        """Release the Torch model reference; no background workers need draining."""
        self._raft = None
