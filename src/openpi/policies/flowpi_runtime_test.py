"""FlowPi runtime test with dummy model + random image sequence."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.flowpi_runtime as flowpi_runtime

pytestmark = pytest.mark.slow


def _dummy_flow_config():
    return pi0_config.FlowConfig(
        num_flow_steps=2,
        flow_stride_frames=3,
        vlm_delay_max=4,
        injection_layers=(1, 2),  # dummy has depth=4
    )


def test_frame_ring_buffer_tracks_latest_and_history():
    """The cursor must always identify the latest valid synchronized frame set."""
    cams = ("cam0", "cam1")

    def frame(value: int) -> np.ndarray:
        return np.full((3, 2, 2), value, dtype=np.uint8)

    ring = flowpi_runtime._FrameRingBuffer.create(  # noqa: SLF001
        cams,
        capacity=3,
        first_frames={cam: frame(0) for cam in cams},
    )
    assert all(np.all(x == 0) for x in ring.get(0).values())

    for value in (1, 2, 3, 4):
        ring.advance()
        for cam in cams:
            ring.push(cam, frame(value))

        assert all(np.all(x == value) for x in ring.get(0).values())
        assert all(np.all(x == value - 1) for x in ring.get(-1).values())
        if value >= 2:
            assert all(np.all(x == value - 2) for x in ring.get(-2).values())

    with pytest.raises(IndexError):
        ring.get(-3)
    with pytest.raises(IndexError):
        ring.get(1)


def _run_runtime_test(sea_raft_device="cuda"):
    """Shared test body — parameterised by device."""
    model = pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=12,
        flow=_dummy_flow_config(),
    ).create(jax.random.key(0))

    flow_cfg = _dummy_flow_config()
    h, w = 480, 640

    runtime = flowpi_runtime.FlowPiRuntime(
        model,
        flow_config=flow_cfg,
        sea_raft_ckpt=None,
        sea_raft_device=sea_raft_device,
        d=1,
        allow_random_init=True,
    )

    # Create 20 frames of dummy observations.
    n_frames = 20
    observations = []
    rng = jax.random.key(42)
    for _ in range(n_frames):
        rng, *sub = jax.random.split(rng, 4)
        img_rng, state_rng, prompt_rng = sub
        # Generate random images HWC uint8.
        images = {
            "base_0_rgb": jax.random.uniform(img_rng, (1, h, w, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32),
            "left_wrist_0_rgb": jax.random.uniform(img_rng, (1, h, w, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32),
            "right_wrist_0_rgb": jax.random.uniform(img_rng, (1, h, w, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32),
        }
        state = jax.random.uniform(state_rng, (1, 32), minval=-1.0, maxval=1.0)
        token_ids = jax.random.randint(prompt_rng, (1, 10), 0, 257152)
        token_mask = jnp.ones((1, 10), dtype=bool)
        obs = _model.Observation(
            images=images,
            image_masks={k: jnp.ones((1,), dtype=bool) for k in images},
            state=state,
            tokenized_prompt=token_ids,
            tokenized_prompt_mask=token_mask,
        )
        observations.append(obs)

    # --- warm_start ---
    runtime.warm_start(observations[0])
    runtime.refresh_prefix(observations[0], wait=True)
    initial_actions = runtime.emit()
    assert initial_actions.shape == (1, 32)
    assert np.all(np.isfinite(initial_actions))

    tick_actions = []
    kv_before_refresh = None
    for i in range(1, n_frames):
        acts = runtime.tick(observations[i])
        assert acts.shape == (1, 32), f"tick {i}: shape {acts.shape}"
        assert np.all(np.isfinite(acts)), f"tick {i}: nan"
        tick_actions.append(acts)

        # The slow delay is `current_tick - prefix_source_tick`: it counts from the tick of the
        # observation the *active* prefix was computed from (includes VLM compute latency), and
        # only advances when the runtime installs a completed refresh.
        source_tick = int(runtime._streaming_state.prefix_source_tick[0])  # noqa: SLF001
        assert runtime._prefix_source_tick == source_tick  # noqa: SLF001
        expected_age = i - ((i - 1) // 5) * 5
        assert i - source_tick == expected_age, f"tick {i}: delay {i - source_tick} != {expected_age}"

        # A refresh completed at the previous tick must have been installed by now: the active
        # KV cache changed and the delay restarted from the installation tick.
        if kv_before_refresh is not None:
            kv_now = jax.tree.leaves(runtime._streaming_state.kv_cache)  # noqa: SLF001
            changed = any(
                np.any(np.asarray(a) != np.asarray(b)) for a, b in zip(kv_now, kv_before_refresh, strict=True)
            )
            assert changed, f"tick {i}: refresh result was not installed into the streaming state"
            assert i - source_tick == 1, f"tick {i}: delay {i - source_tick} != 1 right after a prefix swap"
            kv_before_refresh = None

        # Refresh every 5 ticks: publish a fresh prefix but do NOT touch the active source tick yet.
        if i % 5 == 0:
            kv_before_refresh = jax.tree.leaves(runtime._streaming_state.kv_cache)  # noqa: SLF001
            runtime.refresh_prefix(observations[i], wait=True)
            assert runtime._prefix_source_tick == source_tick  # noqa: SLF001

    # Post-refresh age check.
    assert len(tick_actions) == n_frames - 1
    runtime.close()


def test_runtime_cpu():
    _run_runtime_test(sea_raft_device="cpu")


def test_runtime_cuda():
    _run_runtime_test(sea_raft_device="cuda")


def test_warm_start_second_episode_resets_ring():
    """warm_start must start the ring buffer from scratch: stale frames of the previous episode
    must not leak into the first ticks' flow."""
    model = pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=12,
        flow=_dummy_flow_config(),
    ).create(jax.random.key(0))
    h, w = 480, 640

    runtime = flowpi_runtime.FlowPiRuntime(
        model,
        flow_config=_dummy_flow_config(),
        sea_raft_ckpt=None,
        sea_raft_device="cuda",
        d=1,
        allow_random_init=True,
    )

    def obs_for(images_value: float, state_key: int) -> _model.Observation:
        images = {
            cam: jnp.full((1, h, w, 3), images_value, dtype=jnp.float32)
            for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        }
        return _model.Observation(
            images=images,
            image_masks={cam: jnp.ones((1,), dtype=bool) for cam in images},
            state=jax.random.uniform(jax.random.key(state_key), (1, 32), minval=-1.0, maxval=1.0),
            tokenized_prompt=jnp.zeros((1, 10), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((1, 10), dtype=bool),
        )

    # Episode A: several fast ticks on 0.25 frames (uint8 value 159).
    runtime.warm_start(obs_for(0.25, 0))
    for i in range(5):
        runtime.tick(obs_for(0.25, i + 1))

    # Episode B: warm_start again on +1.0 frames (uint8 value 255). The ring must be freshly
    # initialised: slot 0 holds the new first frame, every other slot is the zero-init (a stale
    # episode-A frame would be 159).
    runtime.warm_start(obs_for(1.0, 100))
    ring = runtime._ring  # noqa: SLF001
    assert ring.base_index == 0
    for cam in ring.cam_keys:
        buf = np.asarray(ring.buffer[cam])
        assert np.all(buf[0] == 255), f"first frame of episode B not in {cam}"
        assert np.all(buf[1:] == 0), f"stale frames from episode A leaked into {cam}"

    # The streaming state restarted as well: source tick 0 and a fresh finite action buffer.
    assert int(runtime._streaming_state.prefix_source_tick[0]) == 0  # noqa: SLF001
    assert np.all(np.isfinite(np.asarray(runtime._streaming_state.action_buffer)))  # noqa: SLF001


def _make_runtime(sea_raft_device="cuda") -> flowpi_runtime.FlowPiRuntime:
    model = pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=12,
        flow=_dummy_flow_config(),
    ).create(jax.random.key(0))
    return flowpi_runtime.FlowPiRuntime(
        model,
        flow_config=_dummy_flow_config(),
        sea_raft_ckpt=None,
        sea_raft_device=sea_raft_device,
        d=1,
        allow_random_init=True,
    )


def _obs_for(value: float, state_key: int = 0, h: int = 480, w: int = 640) -> _model.Observation:
    images = {
        cam: jnp.full((1, h, w, 3), value, dtype=jnp.float32)
        for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    return _model.Observation(
        images=images,
        image_masks={cam: jnp.ones((1,), dtype=bool) for cam in images},
        state=jax.random.uniform(jax.random.key(state_key), (1, 32), minval=-1.0, maxval=1.0),
        tokenized_prompt=jnp.zeros((1, 10), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, 10), dtype=bool),
    )


def _wait_until(condition, timeout_s=30.0) -> None:
    """Busy-wait until `condition()` is truthy (slow worker runs in a background thread)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise TimeoutError("condition not met in time")


def test_async_refresh_installed_on_next_tick():
    """An async refresh is installed at the next fast tick with the correct source tick."""
    runtime = _make_runtime()
    try:
        runtime.warm_start(_obs_for(0.25, 0))
        runtime.tick(_obs_for(0.25, 1))
        # Refresh with the most recently ingested frame (index 1) -> source tick 1.
        runtime.refresh_prefix(_obs_for(0.25, 1), wait=False)
        _wait_until(lambda: runtime._pending_prefix is not None)  # noqa: SLF001
        assert runtime._pending_prefix.source_tick == 1  # noqa: SLF001
        assert runtime._prefix_source_tick == 0  # noqa: SLF001

        runtime.tick(_obs_for(0.25, 2))
        assert runtime._prefix_source_tick == 1  # noqa: SLF001
        # delay = 2 (current frame) - 1 (prefix source) = 1.
        assert int(runtime._streaming_state.prefix_source_tick[0]) == 1  # noqa: SLF001
    finally:
        runtime.close()


def test_stale_prefix_generation_is_dropped():
    """A generation computed from a frame older than the active prefix is never installed."""
    runtime = _make_runtime()
    try:
        runtime.warm_start(_obs_for(0.25, 0))
        for i in range(1, 6):
            runtime.tick(_obs_for(0.25, i))
        # Active prefix comes from frame 5.
        runtime.refresh_prefix(_obs_for(0.25, 5), wait=True)
        runtime.tick(_obs_for(0.25, 6))
        assert runtime._prefix_source_tick == 5  # noqa: SLF001

        # Publish a generation computed from frame 1 (older than the active source tick).
        kv_cache, prefix_mask = runtime.model._prefix_forward(  # noqa: SLF001
            _model.preprocess_observation(None, _obs_for(0.25, 1), train=False)
        )
        runtime._publish(episode_id=runtime._episode_id, source_tick=1, kv_cache=kv_cache, prefix_mask=prefix_mask)  # noqa: SLF001
        runtime.tick(_obs_for(0.25, 7))
        assert runtime._prefix_source_tick == 5  # noqa: SLF001
        assert runtime.num_generation_drops >= 1
    finally:
        runtime.close()


def test_cross_episode_prefix_is_dropped():
    """A refresh of the previous episode must never be installed into the new episode."""
    runtime = _make_runtime()
    try:
        runtime.warm_start(_obs_for(0.25, 0))
        # A generation from a previous episode (forged episode id) must be dropped at install.
        kv_cache, prefix_mask = runtime.model._prefix_forward(  # noqa: SLF001
            _model.preprocess_observation(None, _obs_for(0.25, 0), train=False)
        )
        runtime._publish(episode_id=runtime._episode_id - 1, source_tick=0, kv_cache=kv_cache, prefix_mask=prefix_mask)  # noqa: SLF001
        kv_old = jax.tree.leaves(runtime._streaming_state.kv_cache)  # noqa: SLF001
        runtime.tick(_obs_for(0.25, 1))
        assert runtime._prefix_source_tick == 0  # noqa: SLF001
        kv_now = jax.tree.leaves(runtime._streaming_state.kv_cache)  # noqa: SLF001
        assert all(
            np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(kv_now, kv_old, strict=True)
        ), "stale episode prefix was installed"
        assert runtime.num_generation_drops >= 1
    finally:
        runtime.close()


def test_slow_worker_exception_propagates():
    """A failed background prefill must raise in the main thread at the next tick/refresh."""
    runtime = _make_runtime()

    def boom(*args, **kwargs):
        raise RuntimeError("prefill exploded")

    try:
        runtime.warm_start(_obs_for(0.25, 0))
        runtime.model._prefix_forward = boom  # noqa: SLF001
        runtime.refresh_prefix(_obs_for(0.25, 1), wait=False)
        _wait_until(lambda: runtime._slow_futures and runtime._slow_futures[0].done())  # noqa: SLF001
        with pytest.raises(RuntimeError, match="prefill exploded"):
            runtime.tick(_obs_for(0.25, 2))
    finally:
        # close() drains the worker and re-raises the same failure.
        with pytest.raises(RuntimeError, match="prefill exploded"):
            runtime.close()


def test_resized_frames_raise_flow_grid_error():
    """Feeding model-resolution (224x224) frames must raise a clear flow-grid error instead of
    silently producing a 28x28 flow grid."""
    runtime = _make_runtime()
    try:
        runtime.warm_start(_obs_for(0.25, 0, h=224, w=224))
        with pytest.raises(ValueError, match="flow grid"):
            runtime.tick(_obs_for(0.25, 1, h=224, w=224))
    finally:
        runtime.close()
