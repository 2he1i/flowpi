"""FlowPi runtime: frame ring buffer, per-tick optical flow, async prefix refresh, and the πR² streaming loop."""

from collections.abc import Sequence
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import threading
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
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


@dataclasses.dataclass
class PrefixGeneration:
    """A completed slow-channel prefix prefill, published for atomic installation.

    ``episode_id`` and ``source_tick`` let the runtime drop stale generations: a prefill from a
    previous episode, or one computed from a frame older than the currently active prefix, is
    never installed.
    """

    # Episode this prefix was computed in (drops cross-episode publications).
    episode_id: int
    # Episode-relative tick of the observation this prefix was computed from. The slow delay is
    # `current_tick - source_tick` (includes the VLM compute latency).
    source_tick: int
    kv_cache: Any
    prefix_mask: jax.Array


class _FrameRingBuffer:
    """Sliding window of decoded camera frames in CHW uint8 layout.

    ``current`` always points to the latest valid frame. ``base_index`` is the
    dataset-frame-index of that latest frame.
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
        self.base_index = 0  # dataset-frame-index of the latest frame
        self.current = 0  # buffer[:, current] is the latest frame

    def push(self, cam_key: str, frame_chw: np.ndarray) -> None:
        """Write one camera frame at the current cursor position."""
        self.buffer[cam_key][self.current] = frame_chw

    def advance(self) -> int:
        """Advance the cursor to the slot for the next frame."""
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
        if offset > 0 or -offset >= self.capacity:
            raise IndexError(f"Frame at offset {offset} is outside the ring buffer window.")
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

    The runtime expects full-resolution camera frames (the ``flow_image_size`` used at training
    time, e.g. 480x640): it computes the online SEA-RAFT flow on those frames and lets the model
    preprocess the same observation for the VLM (resizing internally). Feeding pre-resized
    (model-resolution) frames produces a wrong flow grid and raises.

    Usage (offline replay on a dataset episode, 50 Hz, d=1 per tick)::

        runtime = FlowPiRuntime(model, flow_config=..., sea_raft_ckpt=..., device="cuda")
        runtime.warm_start(first_observation)
        runtime.refresh_prefix(first_observation, wait=True)   # initial slow-channel fill
        for frame_idx in range(1, episode_length):
            obs = dataset[frame_idx]                # (Observation state, images)
            actions = runtime.tick(obs)
            if frame_idx % slow_every_n == 0:
                runtime.refresh_prefix(obs)         # async: returns immediately
            # use actions ...
        runtime.close()                             # drain the slow worker, propagate errors

    The slow delay is `current_tick - prefix_source_tick`: the tick of the observation the
    active prefix was computed from, so it includes the VLM compute latency.

    Slow-channel scheduling: one in-flight VLM prefill plus a single latest-pending mailbox.
    Refresh requests arriving while the worker is busy coalesce into the mailbox (the queue is
    bounded to one slot), and a completed prefill that is fresher than the active prefix is
    always published. Under a refresh storm the prefix source tick therefore advances
    monotonically instead of starving (the old "only the latest submitted generation may
    publish" scheme could drop every completed prefill and leave the prefix frozen).
    """

    def __init__(
        self,
        model: _pi0.Pi0,
        *,
        flow_config: Any,  # pi0_config.FlowConfig
        sea_raft_ckpt: str | None = None,
        sea_raft_device: str = "cuda",
        jax_device: str | None = None,
        d: int = 1,
        allow_random_init: bool = False,
    ):
        self.model = model
        self.flow_config = flow_config
        self._d = d

        # Optional device placement for the JAX model (slow VLM + fast action expert run on the
        # same JAX graph and share one parameter tree, so they move together; a true VLM/AE
        # split across two devices needs process-level separation). SEA-RAFT is pinned
        # independently via ``sea_raft_device`` (torch), so the flow channel can already be
        # isolated on its own GPU.
        self._jax_device = _resolve_jax_device(jax_device)
        if self._jax_device is not None:
            graphdef, state = nnx.split(model)
            state = jax.tree.map(
                lambda x: jax.device_put(x, self._jax_device) if isinstance(x, jax.Array) else x,
                state,
            )
            self.model = nnx.merge(graphdef, state)

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
            allow_random_init=allow_random_init,
        )

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
        # Latest pending refresh observation, coalesced while the worker is busy.
        self._slow_mailbox: tuple[int, int, _model.Observation] | None = None

        # Per-episode streaming state. `StreamingState.prefix_source_tick` (mirrored by
        # `self._prefix_source_tick`) is the single authoritative clock for the age of the
        # prefix actually used by the fast policy.
        self._streaming_state: _pi0.Pi0.StreamingState | None = None
        self._ring: _FrameRingBuffer | None = None
        # Monotonically increasing per-tick counter (RNG only, not a prefix clock).
        self._tick = 0
        self._episode_id = 0
        # Episode-relative index of the most recently ingested frame (0 = the warm-start frame).
        self._frame_index = 0
        # Episode-relative tick of the observation the active prefix was computed from.
        self._prefix_source_tick: int | None = None

        # Image resolution for flow (480x640 -> 60x80 grid).
        h, w = flow_config.flow_image_size
        self._flow_grid = (h // 8, w // 8)

        # Telemetry (wall-clock ms and delays; appended from the main thread and the slow worker).
        self.stats: dict[str, list[float]] = {
            "flow_ms": [],
            "prefill_ms": [],
            "tick_total_ms": [],
            "prefix_age_at_install": [],
            "prefix_age_ms_at_install": [],
        }
        # Per-tick freshness telemetry: reconstructs Age_VLM (current - prefix_source_tick),
        # Age_Flow (always 0: flow is recomputed per tick) and the raw tick numbers from the
        # wall-clock timing recorded separately by the caller.
        self.telemetry: list[dict[str, float]] = []
        # Wall-clock of each frame's ingestion (indexed by episode-relative frame index), used to
        # report the prefix age in milliseconds; pruned so it stays bounded.
        self._ingest_wall: dict[int, float] = {}
        # Published-but-never-installed prefix generations (stale episode or out-of-order tick).
        self.num_generation_drops: int = 0

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
            # ``current`` denotes the latest valid frame, so move to the next
            # slot before writing the new synchronized camera frame set.
            self._ring.advance()
            for cam in self._cam_keys:
                self._ring.push(cam, first_chw[cam])

    def _compute_flow(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compute normalised per-lag optical flow and validity masks.

        Lags that refer to frames before the episode start (first few ticks) are
        filled with zero flow and marked invalid in the returned masks.
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

        raft_t0 = time.perf_counter()
        flow = self._raft.compute(prev_stacked, curr_stacked)  # [1, K*n_cam, 2, h8, w8]
        self.stats["flow_ms"].append((time.perf_counter() - raft_t0) * 1000)
        _, _, _, h8, w8 = flow.shape
        if (h8, w8) != self._flow_grid:
            raise ValueError(
                f"SEA-RAFT produced a {h8}x{w8} flow grid but the model expects "
                f"{self._flow_grid[0]}x{self._flow_grid[1]} (flow_image_size="
                f"{self.flow_config.flow_image_size}). Feed the runtime full-resolution camera "
                "frames; do not resize images to the model resolution before the runtime sees them."
            )
        flow = flow.reshape(k, n_cam, 2, h8, w8)

        result = {}
        masks = {}
        valid_mask = np.asarray(valid_lags, dtype=bool)
        for ci, cam in enumerate(self._cam_keys):
            raw = flow[:, ci].copy()  # [K, 2, h8, w8]
            raw[~valid_mask] = 0.0
            result[cam] = normalize_flow(raw, self.flow_config.flow_scale, self.flow_config.flow_clamp)
            masks[cam] = valid_mask.copy()
        return result, masks

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

    def _prefill(self, observation: _model.Observation) -> tuple[Any, jax.Array]:
        """Run the VLM prefix encoder on a (preprocessed) observation, timing the prefill."""
        t0 = time.perf_counter()
        observation = _model.preprocess_observation(None, observation, train=False)
        kv_cache, prefix_mask = self.model._prefix_forward(observation)  # noqa: SLF001
        self.stats["prefill_ms"].append((time.perf_counter() - t0) * 1000)
        return kv_cache, prefix_mask

    def _publish(self, *, episode_id: int, source_tick: int, kv_cache: Any, prefix_mask: jax.Array) -> None:
        """Atomically publish a completed prefill.

        Cross-episode results are dropped. A completed generation that is *older* than the
        currently pending one never replaces it: the pending slot is monotonically fresh, so a
        slow worker finishing behind a newer synchronous refresh cannot regress the prefix.
        """
        with self._slow_lock:
            if episode_id != self._episode_id:
                self.num_generation_drops += 1
                return
            pending = self._pending_prefix
            if pending is not None and pending.source_tick > source_tick:
                return
            self._pending_prefix = PrefixGeneration(episode_id, source_tick, kv_cache, prefix_mask)

    def warm_start(self, observation: _model.Observation) -> None:
        """Begin a new episode: initialise ring buffer, run warm_start.

        The caller should call ``refresh_prefix(observation, wait=True)`` before the first
        ``tick``. Any in-flight slow refresh of the previous episode is invalidated.
        """
        self._check_slow_errors()
        # A new episode starts from an empty ring buffer: stale frames of the previous episode
        # would otherwise leak into the first ticks' flow (cross-episode flow contamination).
        self._ring = None
        self._frame_index = 0
        self._prefix_source_tick = 0
        self._episode_id += 1
        self._ingest_frame(observation)
        self._ingest_wall = {0: time.monotonic()}

        self._streaming_state = self.model.warm_start(
            jax.random.key(0),
            observation,
            num_steps=10,
            d=self._d,
        )
        self._tick = 0
        # A new episode starts from a clean slow channel: drop any pending refresh of the
        # previous episode (in-flight worker results are dropped by the episode-id check, and
        # the mailbox is cleared so the worker does not waste a prefill on a stale observation).
        with self._slow_lock:
            self._pending_prefix = None
            self._slow_mailbox = None

    def tick(self, observation: _model.Observation) -> np.ndarray:
        """One fast control tick (50 Hz): fresh state + online flow + one NFE.

        A slow-channel refresh completed since the last tick is installed first (kv_cache +
        prefix_mask swapped together with the prefix source tick; the slow delay is then
        `current_tick - prefix_source_tick`, which includes the VLM compute latency). Returns
        the ``d`` emitted actions (shape ``[d, action_dim]``).
        """
        self._check_slow_errors()
        # Install the latest completed slow-prefix refresh, if any, into the active streaming
        # state. Publication is atomic: kv_cache, prefix_mask and source metadata always come
        # from the same prefix generation. A stale generation (previous episode, or computed
        # from a frame older than the active prefix) is dropped.
        with self._slow_lock:
            pending = self._pending_prefix
            self._pending_prefix = None
        state = self._streaming_state
        installed = False
        if pending is not None:
            if pending.episode_id == self._episode_id and pending.source_tick >= (self._prefix_source_tick or 0):
                # Fresh generation (>=: an equal source tick re-computes the same prefix, which
                # is indistinguishable from the active one).
                self._prefix_source_tick = pending.source_tick
                state = dataclasses.replace(
                    state,
                    kv_cache=pending.kv_cache,
                    prefix_mask=pending.prefix_mask,
                    prefix_source_tick=jnp.full((state.action_buffer.shape[0],), pending.source_tick, dtype=jnp.int32),
                )
                installed = True
            else:
                # Stale generation (previous episode, or computed from a frame older than the
                # active prefix): never installed.
                self.num_generation_drops += 1

        self._frame_index += 1
        self._ingest_frame(observation)
        self._ingest_wall[self._frame_index] = time.monotonic()
        flow_data, flow_masks = self._compute_flow()

        # Attach fresh flow, per-lag validity, and the slow-channel delay: the age of the exact
        # prefix stored in the streaming state, `current_tick - prefix_source_tick` (clamped so
        # the embedding lookup stays in range).
        delay = max(0, self._frame_index - (self._prefix_source_tick or 0))
        delay = min(delay, self.flow_config.vlm_delay_max)
        if installed:
            self.stats["prefix_age_at_install"].append(delay)
            wall = self._ingest_wall.get(pending.source_tick)
            if wall is not None:
                self.stats["prefix_age_ms_at_install"].append((time.monotonic() - wall) * 1000)
        obs_with_flow = dataclasses.replace(
            observation,
            flow={cam: jnp.asarray(arr)[None, ...] for cam, arr in flow_data.items()},
            flow_masks={cam: jnp.asarray(mask)[None, ...] for cam, mask in flow_masks.items()},
            vlm_delay=jnp.full((observation.state.shape[0],), delay, dtype=jnp.int32),
        )

        # Per-tick freshness telemetry: reconstructs Age_VLM (current - prefix_source_tick),
        # Age_Flow (always 0 here: flow is recomputed per tick) and the raw tick numbers.
        self.telemetry.append(
            {
                "tick": self._frame_index,
                "flow_source_tick": self._frame_index,
                "prefix_source_tick": self._prefix_source_tick or 0,
                "delay_ticks": delay,
            }
        )
        # Prune the ingestion wall-clock history: only the delay window (plus a generous margin
        # for a slow VLM) is ever needed to report the prefix age in milliseconds.
        prune_before = self._frame_index - (self.flow_config.vlm_delay_max + 1024)
        if prune_before > 0 and len(self._ingest_wall) > self.flow_config.vlm_delay_max + 1024:
            for k in list(self._ingest_wall):
                if k < prune_before:
                    del self._ingest_wall[k]

        # One NFE (per-tick RNG: must not repeat after a prefix refresh resets the age).
        rng = jax.random.fold_in(jax.random.key(0), self._tick)
        self._tick += 1
        emit, new_state = self.model.denoise_step(state, obs_with_flow, rng, d=self._d)
        self._streaming_state = new_state
        return np.asarray(emit[0])  # [d, action_dim]

    def refresh_prefix(self, observation: _model.Observation, *, wait: bool = False) -> None:
        """Slow-channel: re-run the VLM prefix encoder and produce fresh KV cache.

        By default the prefill runs on a single background worker; the result is published
        atomically and installed into the active streaming state at the start of the next fast
        tick (the prefix source tick is recorded at installation time, so the slow delay keeps
        counting from the observation this prefix was computed from). With ``wait=True`` the
        prefill runs synchronously in the calling thread and supersedes any in-flight refresh.

        When the refresh rate exceeds the VLM service rate, requests coalesce into a single
        latest-pending mailbox slot instead of queuing: the worker finishes its current prefill,
        publishes it, and immediately picks up the latest pending observation (back-to-back,
        as fast as inference allows). A completed prefill that is fresher than the active prefix
        is always published, so the prefix source tick advances monotonically even under a
        refresh storm.

        The observation is assumed to be the most recently ingested frame (its episode-relative
        index becomes the new prefix source tick). Worker exceptions are re-raised in the main
        thread at the next tick/refresh/close.
        """
        self._check_slow_errors()
        episode_id = self._episode_id
        source_tick = self._frame_index
        if wait:
            kv_cache, prefix_mask = self._prefill(observation)
            self._publish(episode_id=episode_id, source_tick=source_tick, kv_cache=kv_cache, prefix_mask=prefix_mask)
            return
        with self._slow_lock:
            if self._slow_busy:
                # A prefill is already running: coalesce into the single latest-pending slot
                # (supersedes any older pending observation).
                self._slow_mailbox = (episode_id, source_tick, observation)
            else:
                self._slow_busy = True
                future = self._slow_executor.submit(self._slow_run, episode_id, source_tick, observation)
                self._slow_futures.append(future)

    def _slow_run(
        self,
        episode_id: int,
        source_tick: int,
        observation: _model.Observation,
    ) -> None:
        """Slow worker loop: prefill -> publish -> immediately consume the latest pending
        observation and repeat.

        Back-to-back scheduling: the VLM is never idle while a refresh is pending, at most one
        prefill runs at a time, and the refresh queue is bounded to a single mailbox slot
        (no queued-but-doomed jobs). A completed prefill is published whenever it belongs to the
        current episode and is no older than the pending generation; the per-tick install check
        drops anything older than the already-active prefix.
        """
        while True:
            kv_cache, prefix_mask = self._prefill(observation)
            self._publish(episode_id=episode_id, source_tick=source_tick, kv_cache=kv_cache, prefix_mask=prefix_mask)
            with self._slow_lock:
                mailbox = self._slow_mailbox
                self._slow_mailbox = None
                if mailbox is not None:
                    episode_id, source_tick, observation = mailbox
                    continue
                self._slow_busy = False
                return

    def close(self) -> None:
        """Drain the slow worker and re-raise its first exception, if any. Idempotent."""
        self._slow_executor.shutdown(wait=True)
        futures, self._slow_futures = self._slow_futures, []
        for future in futures:
            future.result()

    def emit(self) -> np.ndarray:
        """Return the current action chunk (the first ``d`` actions in the buffer).

        Useful *after* ``warm_start`` to obtain the initial in-flight actions.
        """
        if self._streaming_state is None:
            raise RuntimeError("call warm_start before emit")
        return np.asarray(self._streaming_state.action_buffer[0, : self._d])  # [d, action_dim]
