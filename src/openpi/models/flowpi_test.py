"""flowpi model tests (dummy variants, fast on GPU/CPU).

Covers: zero-gate equivalence, init gradient invariants, parameter budget, staircase
construction/self-similarity, per-position RMSNorm, and the streaming runtime.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _flow_config(**overrides) -> _pi0_config.FlowConfig:
    defaults = dict(
        num_flow_steps=2,
        flow_stride_frames=3,
        d_max=2,
        injection_layers=(1, 2),
        vlm_delay_max=3,
    )
    defaults.update(overrides)
    return _pi0_config.FlowConfig(**defaults)


def _make_model(flow: _pi0_config.FlowConfig | None, key=0):
    config = _pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=12,
        flow=flow,
    )
    return config.create(jax.random.key(key))


def _copy_shared_params(src_model, dst_model):
    """Copies every parameter of src into dst where names and shapes match."""
    gdef, sdst = nnx.split(dst_model)
    _, ssrc = nnx.split(src_model)
    src_pure = ssrc.to_pure_dict()
    dst_pure = sdst.to_pure_dict()

    def merge(dst, src):
        out = {}
        for k, v in dst.items():
            if isinstance(v, dict):
                out[k] = merge(v, src.get(k, {}))
            elif k in src and hasattr(src[k], "shape") and hasattr(v, "shape") and src[k].shape == v.shape:
                out[k] = src[k]
            else:
                out[k] = v
        return out

    sdst.replace_by_pure_dict(merge(dst_pure, src_pure))
    return nnx.merge(gdef, sdst)


def _fake_obs(config, batch_size=2, seed=1):
    obs = config.fake_obs(batch_size=batch_size)
    return jax.tree.map(lambda x: jax.random.normal(jax.random.key(seed), x.shape, x.dtype) if x.dtype == jnp.float32 else x, obs)


def test_zero_gate_equivalence():
    """With flow gates at zero, the flow model's suffix forward must equal the baseline's for the
    same (x_t, tau): the flow branch and delay embedding are exact no-ops at initialization."""
    base = _make_model(None)
    flow_model = _make_model(_flow_config())
    flow_model = _copy_shared_params(base, flow_model)

    config = flow_model.config
    obs = _fake_obs(config)
    actions = jax.random.normal(jax.random.key(2), (2, config.action_horizon, config.action_dim))

    # Same scalar time for both models.
    time = jnp.full((2,), 0.5)

    # Baseline v_t via one forward pass.
    prefix_tokens, prefix_mask, prefix_ar_mask = base.embed_prefix(obs)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = base.embed_suffix(obs, actions, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (p_out, s_out), _ = base.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_base = base.action_out_proj(s_out[:, -base.action_horizon :])

    # flowpi: per-position tau broadcast from the same scalar time, flow input present.
    tau = jnp.broadcast_to(time[:, None], (2, config.action_horizon))
    prefix_tokens_f, prefix_mask_f, prefix_ar_f = flow_model.embed_prefix(obs)
    suffix_tokens_f, suffix_mask_f, suffix_ar_f, adarms_cond_f = flow_model.embed_suffix(obs, actions, tau)
    input_mask_f = jnp.concatenate([prefix_mask_f, suffix_mask_f], axis=1)
    ar_mask_f = jnp.concatenate([prefix_ar_f, suffix_ar_f], axis=0)
    attn_mask_f = _pi0.make_attn_mask(input_mask_f, ar_mask_f)
    positions_f = jnp.cumsum(input_mask_f, axis=1) - 1
    flow_tokens, flow_token_mask = flow_model.embed_flow(obs)
    (p_out_f, s_out_f), _ = flow_model.PaliGemma.llm(
        [prefix_tokens_f, suffix_tokens_f],
        mask=attn_mask_f,
        positions=positions_f,
        adarms_cond=[None, adarms_cond_f],
        flow=flow_tokens,
        flow_mask=flow_token_mask,
    )
    v_flow = flow_model.action_out_proj(s_out_f[:, -flow_model.action_horizon :])

    np.testing.assert_allclose(np.asarray(v_flow), np.asarray(v_base), rtol=1e-4, atol=1e-4)


def test_zero_gate_equivalence_no_flow_input():
    """Same as above but with flow=None in the observation (placeholder token path)."""
    base = _make_model(None)
    flow_model = _make_model(_flow_config())
    flow_model = _copy_shared_params(base, flow_model)
    config = flow_model.config

    obs = _fake_obs(config)
    obs_noflow = type(obs)(
        images=obs.images, image_masks=obs.image_masks, state=obs.state,
        tokenized_prompt=obs.tokenized_prompt, tokenized_prompt_mask=obs.tokenized_prompt_mask,
        flow=None, flow_masks=None, vlm_delay=None,
    )
    actions = jax.random.normal(jax.random.key(2), (2, config.action_horizon, config.action_dim))
    time = jnp.full((2,), 0.3)

    prefix_tokens, prefix_mask, prefix_ar_mask = base.embed_prefix(obs_noflow)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = base.embed_suffix(obs_noflow, actions, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, s_out), _ = base.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_base = base.action_out_proj(s_out[:, -base.action_horizon :])

    tau = jnp.broadcast_to(time[:, None], (2, config.action_horizon))
    flow_embedded = flow_model.embed_flow(obs_noflow)
    assert flow_embedded is None
    prefix_tokens_f, prefix_mask_f, prefix_ar_f = flow_model.embed_prefix(obs_noflow)
    suffix_tokens_f, suffix_mask_f, suffix_ar_f, adarms_cond_f = flow_model.embed_suffix(obs_noflow, actions, tau)
    input_mask_f = jnp.concatenate([prefix_mask_f, suffix_mask_f], axis=1)
    ar_mask_f = jnp.concatenate([prefix_ar_f, suffix_ar_f], axis=0)
    attn_mask_f = _pi0.make_attn_mask(input_mask_f, ar_mask_f)
    positions_f = jnp.cumsum(input_mask_f, axis=1) - 1
    (_, s_out_f), _ = flow_model.PaliGemma.llm(
        [prefix_tokens_f, suffix_tokens_f], mask=attn_mask_f, positions=positions_f, adarms_cond=[None, adarms_cond_f]
    )
    v_flow = flow_model.action_out_proj(s_out_f[:, -flow_model.action_horizon :])
    np.testing.assert_allclose(np.asarray(v_flow), np.asarray(v_base), rtol=1e-4, atol=1e-4)


def _flow_grads(grads) -> dict[str, jax.Array]:
    """Flattens an nnx grad tree to {path: array} for flow-related params."""
    flat = jax.tree_util.tree_flatten_with_path(grads)[0]
    return {jax.tree_util.keystr(path): g for path, g in flat}


def test_init_gradient_invariants():
    """Step 0: flow_gate grads nonzero (CA output nonzero); tokenizer/QKV grads zero. After one
    update (gate becomes nonzero), tokenizer/QKV grads become nonzero."""
    model = _make_model(_flow_config())
    config = model.config
    obs = _fake_obs(config)
    actions = jax.random.normal(jax.random.key(2), (2, config.action_horizon, config.action_dim))

    def loss_fn(m):
        return jnp.mean(m.compute_loss(jax.random.key(3), obs, actions))

    grads = _flow_grads(nnx.grad(loss_fn)(model))

    gate = next(v for k, v in grads.items() if k.endswith("flow_gate'].value"))
    assert bool(jnp.any(gate != 0)), "flow_gate grad must be nonzero at step 0"

    for name in ("flow_q", "flow_kv", "flow_out", "flow_pre_norm_scale"):
        v = next(v for k, v in grads.items() if f"'{name}'].value" in k)
        assert bool(jnp.all(v == 0)), f"{name} grads must be zero at step 0 (gate=0)"
    conv0 = next(v for k, v in grads.items() if "convs'][0]['kernel" in k)
    assert bool(jnp.all(conv0 == 0)), "tokenizer grads must be zero at step 0 (gate=0)"

    # One optimizer step on the full grad, then flow grads must reach the tokenizer and QKV.
    optimizer = nnx.Optimizer(model, optax.sgd(1e-2))
    optimizer.update(nnx.grad(loss_fn)(model))
    grads2 = _flow_grads(nnx.grad(loss_fn)(model))
    conv0_2 = next(v for k, v in grads2.items() if "convs'][0]['kernel" in k)
    q_2 = next(v for k, v in grads2.items() if k.endswith("flow_q'].value"))
    assert bool(jnp.any(conv0_2 != 0)), "tokenizer grads must be nonzero after one update"
    assert bool(jnp.any(q_2 != 0)), "flow_q grads must be nonzero after one update"


def test_flow_param_budget():
    """Flow CA params exist only as slot stacks of shape [n_slots, ...]; nothing flow-related in the
    scanned depth stack."""
    model = _make_model(_flow_config())
    _, state = nnx.split(model)
    flat = state.to_pure_dict()

    def walk(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                yield from walk(v, f"{path}/{k}")
        else:
            yield path, d

    flow_params = {p: v for p, v in walk(flat) if "flow" in p.lower()}
    assert len(flow_params) > 0

    total = 0
    for path, value in flow_params.items():
        if "llm" in path:
            # Slot-stacked CA params: first dim must equal the number of injection layers.
            assert value.shape[0] == 2, f"{path}: {value.shape}"
        total += int(np.prod(value.shape))
    # Roughly: q 3*8*1024*128(dummy 64) + kv + out + gate + norm ~ O(10^6) for real config.
    assert total > 0

    # Scanned Block params must not contain flow params (they live outside the scan).
    for path in flow_params:
        assert "/layers/" not in path, f"flow param leaked into scanned layers: {path}"


def test_staircase_construction():
    h, d = 12, 2
    tau = _pi0.make_staircase_tau(h, d)
    expected = jnp.array([0, 0, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0])
    np.testing.assert_allclose(np.asarray(tau), np.asarray(expected), atol=1e-6)


def test_staircase_self_similarity():
    """For the V1 per-tick shift (d=1): t -= 1/(H-2), left-shift by 1, append fresh noise — the
    staircase profile is restored exactly. (Larger single-shift d skips intermediate staircase
    states and is only approximately self-similar; V1 uses d=1 per tick.)"""
    h = 12
    tau = _pi0.make_staircase_tau(h, 1)
    dt = 1 / (h - 2)
    tau_new = jnp.maximum(tau - dt, 0.0)
    shifted = jnp.concatenate([tau_new[1:], jnp.ones((1,))])
    np.testing.assert_allclose(np.asarray(shifted), np.asarray(tau), atol=1e-6)


def test_per_position_rmsnorm_shapes():
    """adaRMS RMSNorm accepts cond [b, d] (legacy) and [b, s, d] (per-position); the [b, d] path
    broadcasts over positions exactly as before (checkpoint compatibility)."""
    import openpi.models.gemma as _gemma  # noqa: PLC0415

    b, s, d = 2, 5, 64
    x = jax.random.normal(jax.random.key(0), (b, s, d))
    cond = jax.random.normal(jax.random.key(1), (b, d))
    norm = _gemma.RMSNorm()
    params = norm.init(jax.random.key(2), x, cond)["params"]

    # Legacy [b, d] cond.
    out_legacy, gate_legacy = norm.apply({"params": params}, x, cond)
    assert out_legacy.shape == (b, s, d)
    assert gate_legacy.shape == (b, 1, d)  # broadcastable over positions

    # Per-position [b, s, d] cond.
    cond_pp = jax.random.normal(jax.random.key(2), (b, s, d))
    out_pp, gate_pp = norm.apply({"params": params}, x, cond_pp)
    assert out_pp.shape == (b, s, d)
    assert gate_pp.shape == (b, s, d)

    # A constant per-position cond reproduces the legacy broadcast result.
    cond_const = jnp.broadcast_to(cond[:, None, :], (b, s, d))
    out_const, _ = norm.apply({"params": params}, x, cond_const)
    np.testing.assert_allclose(np.asarray(out_const), np.asarray(out_legacy), rtol=1e-5, atol=1e-5)


def test_streaming_runtime():
    """warm_start + N ticks of denoise_step(d=1): tau profile cycles, one action per tick, tail is
    fresh noise, prefix_age increments and resets on refresh."""
    model = _make_model(_flow_config())
    config = model.config
    h = config.action_horizon
    obs = _fake_obs(config)

    state = model.warm_start(jax.random.key(0), obs, num_steps=3, d=1)
    tau0 = state.tau[0]
    # Staircase in-flight prefix has at least d zeros (p=d lands exactly on t=0 too).
    assert int((tau0 == 0).sum()) >= 1
    assert float(tau0[-1]) == 1.0

    emitted = []
    for tick in range(h - 2):
        acts, state = model.denoise_step(state, obs, jax.random.key(tick + 10), d=1)
        assert acts.shape == (2, 1, config.action_dim)
        assert bool(jnp.all(jnp.isfinite(acts)))
        emitted.append(acts)
        # tau profile after shift matches the original staircase (self-similarity).
        np.testing.assert_allclose(np.asarray(state.tau[0]), np.asarray(tau0), atol=1e-5)
        assert int(state.prefix_age[0]) == tick + 1

    # Tail of the buffer holds fresh noise ~ N(0, 1).
    tail = state.action_buffer[0, -2]
    assert float(jnp.std(state.action_buffer[:, -2:])) > 0.1

    refreshed = model.refresh_prefix(state, obs)
    assert int(refreshed.prefix_age[0]) == 0
