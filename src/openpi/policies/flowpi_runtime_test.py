"""FlowPi runtime test with dummy model + random image sequence."""

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


def _run_runtime_test(sea_raft_device="cpu"):
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
    runtime.refresh_prefix(observations[0])
    initial_actions = runtime.emit()
    assert initial_actions.shape == (1, 32)
    assert np.all(np.isfinite(initial_actions))

    tick_actions = []
    for i in range(1, n_frames):
        acts = runtime.tick(observations[i])
        assert acts.shape == (1, 32), f"tick {i}: shape {acts.shape}"
        assert np.all(np.isfinite(acts)), f"tick {i}: nan"
        tick_actions.append(acts)
        # _prefix_age counts ticks since last refresh; resets on refresh.
        expected_age = i - ((i - 1) // 5) * 5
        assert runtime._prefix_age == expected_age  # noqa: SLF001

        # Refresh every 5 ticks.
        if i % 5 == 0:
            runtime.refresh_prefix(observations[i])
            assert runtime._prefix_age == 0  # noqa: SLF001

    # Post-refresh age check.
    assert len(tick_actions) == n_frames - 1


def test_runtime_cpu():
    _run_runtime_test(sea_raft_device="cpu")


def test_runtime_cuda():
    _run_runtime_test(sea_raft_device="cuda")
