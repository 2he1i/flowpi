"""Regression coverage for FlowPi warm-start AdaRMS conditioning."""

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _make_model():
    flow = _pi0_config.FlowConfig(
        num_flow_steps=2,
        flow_stride_frames=3,
        d_max=2,
        injection_layers=(1, 2),
        vlm_delay_max=3,
        use_delay=False,
    )
    return _pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=12,
        flow=flow,
    ).create(jax.random.key(0))


def _time_condition(model, timestep):
    time_emb = _pi0.posemb_sincos(
        jnp.full((1,), timestep, dtype=jnp.float32),
        model.action_in_proj.out_features,
        min_period=4e-3,
        max_period=4.0,
    )
    time_emb = model.time_mlp_in(time_emb)
    time_emb = _pi0.nnx.swish(time_emb)
    time_emb = model.time_mlp_out(time_emb)
    return _pi0.nnx.swish(time_emb)


def test_scalar_warm_start_conditions_state_at_zero_time():
    """The scalar suffix path used by warm_start gives state t=0 and actions their current t."""
    model = _make_model()
    observation = model.config.fake_obs(batch_size=1)
    actions = jnp.zeros((1, model.action_horizon, model.action_dim), dtype=jnp.float32)

    _, _, _, adarms_cond = model.embed_suffix(observation, actions, jnp.full((1,), 0.5))

    assert adarms_cond is not None
    assert adarms_cond.shape[1] == model.action_horizon + 1
    np.testing.assert_allclose(np.asarray(adarms_cond[:, 0]), np.asarray(_time_condition(model, 0.0)))
    action_cond = np.asarray(adarms_cond[:, 1:])
    np.testing.assert_allclose(action_cond, np.broadcast_to(np.asarray(adarms_cond[:, 1:2]), action_cond.shape))
    assert not np.array_equal(np.asarray(adarms_cond[:, 0]), np.asarray(adarms_cond[:, 1]))
