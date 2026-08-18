"""FlowPi runtime: frame ring buffer, per-tick optical flow, background prefix refresh, and the πR² streaming loop."""

from collections.abc import Sequence
import dataclasses
import threading
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.training.sea_raft import SeaRaftFlowExtractor
from openpi.transforms import compute_image_frame_offsets
from openpi.transforms import normalize_flow


class _FrameRingBuffer:
    """Sliding window of decoded camera frames in CHW uint8 layout.

    ``buffer[t]`` holds the frame whose dataset-frame-index is ``base_index + t``.
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
        self.base_index = 0  # dataset-frame-index of self.buffer[:, 0]
        self.current = 0  # write cursor; buffer[:, current] is the latest frame

    def push(self, cam_key: str, frame_chw: np.ndarray) -> None:
        """Write one camera frame at the current cursor position."""
        self.buffer[cam_key][self.current] = frame_chw

    def advance(self) -> int:
        old = self.current
        self.current = (self.current + 1) % self.capacity
        self.base_index += 1
        return old

    def get(self, offset: int) -> dict[str, np.ndarray]:
        """Return the frame(s) offset ticks *before* the latest (offset ≤ 0).

        Raises ``IndexError`` if the requested offset is outside the buffered
        window (i.e. before the episode started or beyond what the ring holds).
        """
        if self.base_index + offset < 0:
            raise IndexError(f"Frame at offset {offset} is before the episode start.")
        idx = (self.current + offset) % self.capacity
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

    Usage (offline replay on a dataset episode, 50 Hz, d=1 per tick)::

        runtime = FlowPiRuntime(model, flow_config=..., sea_raft_ckpt=..., device="cuda")
        runtime.warm_start(first_observation)
        runtime.refresh_prefix(first_observation)   # initial slow-channel fill
        for frame_idx in range(1, episode_length):
            obs = dataset[frame_idx]                # (Observation state, images)
            actions = runtime.tick(obs)
            if frame_idx % slow_every_n == 0:
                runtime.refresh_prefix(obs)
            # use actions ...
    """

    def __init__(
        self,
        model: _pi0.Pi0,
        *,
        flow_config: Any,  # pi0_config.FlowConfig
        sea_raft_ckpt: str | None = None,
        sea_raft_device: str = "cuda",
        d: int = 1,
    ):
        self.model = model
        self.flow_config = flow_config
        self._d = d
        self._prefix_age = 0

        self._cam_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

        # Frame ring buffer geometry.
        k = flow_config.num_flow_steps
        stride = flow_config.flow_stride_frames
        self._ring_capacity = k * stride + 1
        self._frame_offsets = compute_image_frame_offsets(k, stride, flow_config.vlm_delay_max)

        # SEA-RAFT extractor (online flow).
        self._raft = SeaRaftFlowExtractor(
            ckpt_path=sea_raft_ckpt,
            variant="M",
            device=sea_raft_device,
        )

        # Slow-channel background state (updated by background thread).
        self._kv_cache: Any = None
        self._prefix_mask: Any = None
        self._slow_lock = threading.Lock()

        # Per-episode streaming state.
        self._streaming_state: _pi0.Pi0.StreamingState | None = None
        self._ring: _FrameRingBuffer | None = None

        # Image resolution for flow (480x640 -> 60x80 grid).
        h, w = flow_config.flow_image_size
        self._flow_grid = (h // 8, w // 8)

    # ---- ring buffer -----------------------------------------------------------

    def _ingest_frame(self, obs: _model.Observation) -> None:
        """Push the current frame(s) into the ring buffer."""
        first = {cam: np.asarray(jax.device_get(obs.images[cam]))[0] for cam in self._cam_keys}
        # Convert float32 [-1,1] → uint8 [0,255] for SEA-RAFT.
        first_u8 = {cam: ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8) for cam, img in first.items()}
        # HWC → CHW.
        first_chw = {cam: np.transpose(img, (2, 0, 1)) for cam, img in first_u8.items()}
        if self._ring is None:
            self._ring = _FrameRingBuffer.create(self._cam_keys, self._ring_capacity, first_chw)
        else:
            for cam in self._cam_keys:
                self._ring.push(cam, first_chw[cam])
            self._ring.advance()

    def _compute_flow(self) -> dict[str, np.ndarray]:
        """Compute normalised per-lag optical flow from the ring buffer.

        Lags that refer to frames before the episode start (first few ticks) are
          filled with zero flow and the caller attaches a per-lag mask (True = valid).
        """
        assert self._ring is not None
        k = self.flow_config.num_flow_steps
        stride = self.flow_config.flow_stride_frames
        n_cam = len(self._cam_keys)

        curr_frame = self._ring.get(0)

        prev_frames: list[np.ndarray] = []
        valid_lags: list[bool] = []
        for ki in range(1, k + 1):
            offset = -ki * stride
            if self._ring.base_index + offset >= 0:
                lag = self._ring.get(offset)
                valid_lags.append(True)
                prev_frames.extend(lag[cam][None] for cam in self._cam_keys)
            else:
                valid_lags.append(False)
                prev_frames.extend(np.zeros_like(curr_frame[cam][None]) for cam in self._cam_keys)

        curr_stacked = np.concatenate(
            [curr_frame[cam][None] for _ in range(k) for cam in self._cam_keys],
            axis=0,
        )[None]  # [1, K*n_cam, 3, H, W]
        prev_stacked = np.concatenate(prev_frames, axis=0)[None]

        flow = self._raft.compute(prev_stacked, curr_stacked)  # [1, K*n_cam, 2, h8, w8]
        _, _, _, h8, w8 = flow.shape
        flow = flow.reshape(k, n_cam, 2, h8, w8)

        result = {}
        for ci, cam in enumerate(self._cam_keys):
            raw = flow[:, ci]  # [K, 2, h8, w8]
            for lag_idx, valid in enumerate(valid_lags):
                if not valid:
                    raw[lag_idx] = 0.0
            result[cam] = normalize_flow(raw, self.flow_config.flow_scale, self.flow_config.flow_clamp)
        return result

    # ---- public API ------------------------------------------------------------

    def warm_start(self, observation: _model.Observation) -> None:
        """Begin a new episode: initialise ring buffer, run warm_start.

        The caller should call ``refresh_prefix`` before the first ``tick``.
        """
        self._ingest_frame(observation)

        self._streaming_state = self.model.warm_start(
            jax.random.key(0),
            observation,
            num_steps=10,
            d=self._d,
        )
        self._prefix_age = 0

    def tick(self, observation: _model.Observation) -> np.ndarray:
        """One fast control tick (50 Hz): fresh state + online flow + one NFE.

        Returns the ``d`` emitted actions (shape ``[d, action_dim]``).
        """
        self._ingest_frame(observation)
        flow_data = self._compute_flow()

        # Attach fresh flow and vlm_delay to the observation.
        obs_with_flow = dataclasses.replace(
            observation,
            flow={cam: jnp.asarray(arr)[None, ...] for cam, arr in flow_data.items()},
            flow_masks={cam: jnp.ones((1, self.flow_config.num_flow_steps), dtype=bool) for cam in self._cam_keys},
            vlm_delay=jnp.asarray([self._prefix_age], dtype=jnp.int32),
        )

        # One NFE.
        rng = jax.random.fold_in(jax.random.key(self._prefix_age), self._prefix_age)
        emit, self._streaming_state = self.model.denoise_step(
            self._streaming_state,
            obs_with_flow,
            rng,
            d=self._d,
        )

        self._prefix_age += 1
        return np.asarray(emit[0])  # [d, action_dim]

    def refresh_prefix(self, observation: _model.Observation) -> None:
        """Slow-channel: re-run the VLM prefix encoder and produce fresh KV cache.

        This should be called from a background thread (e.g. every N ticks).
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        kv_cache, prefix_mask = self.model._prefix_forward(observation)  # noqa: SLF001
        with self._slow_lock:
            self._kv_cache = kv_cache
            self._prefix_mask = prefix_mask
        self._prefix_age = 0

    def emit(self) -> np.ndarray:
        """Return the current action chunk (the first ``d`` actions in the buffer).

        Useful *after* ``warm_start`` to obtain the initial in-flight actions.
        """
        if self._streaming_state is None:
            raise RuntimeError("call warm_start before emit")
        return np.asarray(self._streaming_state.action_buffer[0, : self._d])  # [d, action_dim]
