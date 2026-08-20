"""flowpi model tests (dummy variants, fast on GPU/CPU).

Covers: zero-gate equivalence, init gradient invariants, parameter budget, staircase
construction/self-similarity, per-position RMSNorm, the streaming runtime, action-emission
progression, all-invalid flow cross-attention, loss-scale algebra, prefix-refresh KV swapping,
and fast-path delay conditioning.
"""

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _flow_config(**overrides) -> _pi0_config.FlowConfig:
    defaults = {
        "num_flow_steps": 2,
        "flow_stride_frames": 3,
        "d_max": 2,
        "injection_layers": (1, 2),
        "vlm_delay_max": 3,
    }
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
    return jax.tree.map(
        lambda x: jax.random.normal(jax.random.key(seed), x.shape, x.dtype) if x.dtype == jnp.float32 else x, obs
    )


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
        images=obs.images,
        image_masks=obs.image_masks,
        state=obs.state,
        tokenized_prompt=obs.tokenized_prompt,
        tokenized_prompt_mask=obs.tokenized_prompt_mask,
        flow=None,
        flow_masks=None,
        vlm_delay=None,
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
    import openpi.models.gemma as _gemma

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


def test_streaming_action_progression_no_duplication():
    """Each action-buffer position is emitted exactly once: warm_start emits buffer positions
    [0:d), and the k-th streaming tick emits the positions [k*d:(k+1)*d) of the original
    warm-start buffer. Regression test for the emit off-by-`d` (the already-executed prefix must
    never be re-emitted)."""
    model = _make_model(_flow_config())
    config = model.config
    h = config.action_horizon
    obs = _fake_obs(config)
    d = 1

    state = model.warm_start(jax.random.key(0), obs, num_steps=3, d=d)
    buffer0 = np.asarray(state.action_buffer)  # [B, H, D] of the warm-start generation
    warm_emit = np.asarray(state.action_buffer[0, :d])

    # Deterministic zero velocity: the denoise step then only shifts the buffer, so the emitted
    # values are exactly the original buffer positions that reached the execution boundary.
    model._suffix_forward = lambda *a, **k: jnp.zeros((2, h, config.action_dim))  # noqa: SLF001
    for tick in range(3):
        acts, state = model.denoise_step(state, obs, jax.random.key(100 + tick), d=d)
        expected = buffer0[0, (tick + 1) * d : (tick + 2) * d]
        np.testing.assert_array_equal(np.asarray(acts[0]), expected)
        assert not np.array_equal(np.asarray(acts[0]), warm_emit), f"tick {tick}: re-emitted warm-start actions"


def test_flow_cross_attn_all_invalid_is_exactly_zero():
    """All-invalid flow masks must produce exactly zero flow cross-attention output (regression
    for the all-invalid flow handling)."""
    import openpi.models.gemma as _gemma

    b, s, n_heads, head_dim, width, n_flow = 2, 5, 8, 16, 64, 4
    q_hidden = jax.random.normal(jax.random.key(0), (b, s, width))
    flow = jax.random.normal(jax.random.key(1), (b, n_flow, width))
    # Slot-stacked parameters: [slot, n_heads, width, head_dim] (one slot used below).
    params = {
        "flow_q": jax.random.normal(jax.random.key(2), (1, n_heads, width, head_dim)),
        "flow_kv": jax.random.normal(jax.random.key(3), (1, 2, n_heads, width, head_dim)),
        "flow_out": jax.random.normal(jax.random.key(4), (1, n_heads, head_dim, width)),
    }

    out = _gemma._flow_cross_attn(  # noqa: SLF001
        q_hidden, flow, jnp.zeros((b, n_flow), dtype=bool), params, 0, head_dim=head_dim
    )
    np.testing.assert_array_equal(np.asarray(out), 0.0)

    # With at least one valid flow token the output is nonzero (sanity: the test is not vacuous).
    mask = jnp.zeros((b, n_flow), dtype=bool).at[:, 0].set(True)
    out_valid = _gemma._flow_cross_attn(q_hidden, flow, mask, params, 0, head_dim=head_dim)  # noqa: SLF001
    assert np.any(np.asarray(out_valid) != 0.0)


def test_loss_scale_invariants():
    """The FlowPi masked-loss reduction keeps the baseline π0.5 scale. For standard-FM rows with
    all positions valid the renormalization factor is exactly 1 (loss = mean(sq), i.e. the
    baseline reduction); for πR² rows the loss reduces to the mean over the non-inpainted
    positions only (factor horizon / valid_count)."""
    for p_standard, tau_jitter in ((1.0, 0.01), (0.0, 0.0)):
        model = _make_model(_flow_config(p_standard=p_standard, tau_jitter=tau_jitter))
        config = model.config
        h = config.action_horizon
        batch_shape = (2,)
        obs = _fake_obs(config)
        actions = jax.random.normal(jax.random.key(2), (2, h, config.action_dim))
        rng = jax.random.key(7)

        computed = np.asarray(model.compute_loss(rng, obs, actions))

        # Replicate the exact rng splits of compute_loss.
        preprocess_rng, noise_rng, time_rng, mix_rng = jax.random.split(rng, 4)
        obs_p = _model.preprocess_observation(preprocess_rng, obs, train=False)
        noise = jax.random.normal(noise_rng, actions.shape)
        d_rng, jitter_rng, std_rng = jax.random.split(time_rng, 3)

        if p_standard == 1.0:
            t_std = jax.random.beta(std_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            tau = einops.repeat(t_std, "b -> b h", h=h)
            loss_mask = jnp.ones((2, h))
        else:
            d = jax.random.randint(d_rng, batch_shape, minval=1, maxval=config.flow.d_max + 1)
            pos = jnp.arange(h)
            tau = jnp.where(
                pos[None, :] < d[:, None],
                0.0,
                jnp.where(pos[None, :] >= h - d[:, None], 1.0, (pos[None, :] - d[:, None]) / (h - 2 * d[:, None])),
            )
            jitter = jax.random.uniform(jitter_rng, actions.shape[:-1], minval=-tau_jitter, maxval=tau_jitter)
            mid = (tau > 0) & (tau < 1)
            tau = jnp.where(mid, jnp.clip(tau + jitter, 0.0, 1.0), tau)
            loss_mask = ~(tau == 0.0)

        inpaint = tau == 0.0
        x_t = tau[..., None] * noise + (1 - tau[..., None]) * actions
        x_t = jnp.where(inpaint[..., None], actions, x_t)
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs_p)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(obs_p, x_t, tau)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        flow_tokens, flow_token_mask = model.embed_flow(obs_p)
        (_, s_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            flow=flow_tokens,
            flow_mask=flow_token_mask,
        )
        v_t = model.action_out_proj(s_out[:, -h:])

        sq = jnp.square(v_t - u_t)
        valid_count = jnp.maximum(jnp.sum(loss_mask, axis=-1, keepdims=True), 1.0)
        manual = jnp.mean(sq, axis=-1) * loss_mask * (h / valid_count)
        np.testing.assert_allclose(computed, np.asarray(manual), rtol=1e-5, atol=1e-5)
        if p_standard == 1.0:
            # All positions valid => the renormalization factor is exactly 1 (baseline π0.5 scale).
            np.testing.assert_allclose(computed, np.asarray(jnp.mean(sq, axis=-1)), rtol=1e-5, atol=1e-5)


def test_prefix_refresh_swaps_kv_and_carries_source_tick():
    """refresh_prefix must swap (kv_cache, prefix_mask) together and never touch the action
    buffer; the prefix source tick is carried through fast ticks and only the runtime installs a
    new value (the model cannot know the current tick)."""
    model = _make_model(_flow_config())
    config = model.config
    obs = _fake_obs(config, seed=1)
    # Different prompt => different prefix tokens => different KV (the dummy SigLIP tower is a
    # zero-output no-op, so the prompt is the controllable prefix differentiator here).
    obs2 = dataclasses.replace(obs, tokenized_prompt=jnp.zeros_like(obs.tokenized_prompt))

    state = model.warm_start(jax.random.key(0), obs, num_steps=3, d=1)
    kv_old = jax.tree.leaves(state.kv_cache)
    mask_old = np.asarray(state.prefix_mask)
    # The warm-start prefix comes from the episode-start observation (tick 0).
    assert int(state.prefix_source_tick[0]) == 0

    # A fast tick must not touch the prefix: same KV, same mask, source tick carried through.
    _, state = model.denoise_step(state, obs, jax.random.key(5), d=1)
    assert int(state.prefix_source_tick[0]) == 0
    np.testing.assert_array_equal(np.asarray(state.prefix_mask), mask_old)
    assert all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(state.kv_cache), kv_old, strict=True)
    )

    # Refresh: new KV + matching mask; the action buffer and the source tick are untouched (the
    # runtime records the observation's tick and installs it with the new prefix).
    refreshed = model.refresh_prefix(state, obs2)
    kv_new = jax.tree.leaves(refreshed.kv_cache)
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(kv_new, kv_old, strict=True))
    expected_mask = np.asarray(model._prefix_forward(obs2)[1])  # noqa: SLF001
    np.testing.assert_array_equal(np.asarray(refreshed.prefix_mask), expected_mask)
    assert int(refreshed.prefix_source_tick[0]) == 0
    np.testing.assert_array_equal(np.asarray(refreshed.action_buffer), np.asarray(state.action_buffer))


def test_delay_conditions_fast_suffix():
    """The fast Action Expert must be conditioned on the slow-channel delay: identical inputs
    with different vlm_delay must change the fast suffix path (via the adaRMS conditioning), and
    out-of-range delays clamp to the max-delay embedding."""
    model = _make_model(_flow_config())
    config = model.config
    obs = _fake_obs(config)
    actions = jax.random.normal(jax.random.key(2), (2, config.action_horizon, config.action_dim))
    tau = jnp.broadcast_to(jnp.full((2,), 0.4)[:, None], (2, config.action_horizon))

    delay_0 = jnp.zeros((2,), dtype=jnp.int32)
    delay_max = jnp.full((2,), config.flow.vlm_delay_max, dtype=jnp.int32)

    # At initialization the delay embedding is zero: the fast AE is an exact no-op w.r.t. delay
    # (pretrained-path preservation).
    _, _, _, cond_a = model.embed_suffix(dataclasses.replace(obs, vlm_delay=delay_0), actions, tau)
    _, _, _, cond_b = model.embed_suffix(dataclasses.replace(obs, vlm_delay=delay_max), actions, tau)
    np.testing.assert_array_equal(np.asarray(cond_a), np.asarray(cond_b))

    # With a nonzero (learned) embedding, different delays must change the fast conditioning.
    model.flow_vlm_delay_fast.embedding.value = jax.random.normal(
        jax.random.key(3), model.flow_vlm_delay_fast.embedding.value.shape
    )
    _, _, _, cond_a = model.embed_suffix(dataclasses.replace(obs, vlm_delay=delay_0), actions, tau)
    _, _, _, cond_b = model.embed_suffix(dataclasses.replace(obs, vlm_delay=delay_max), actions, tau)
    assert not np.allclose(np.asarray(cond_a), np.asarray(cond_b))

    # Delays beyond vlm_delay_max clamp to the max-delay embedding (never an OOB lookup).
    delay_big = jnp.full((2,), 100, dtype=jnp.int32)
    _, _, _, cond_big = model.embed_suffix(dataclasses.replace(obs, vlm_delay=delay_big), actions, tau)
    np.testing.assert_array_equal(np.asarray(cond_big), np.asarray(cond_b))


def test_streaming_runtime():
    """warm_start + N ticks of denoise_step(d=1): tau profile cycles, one action per tick, tail is
    fresh noise, the prefix source tick is carried through and never touched by fast ticks."""
    model = _make_model(_flow_config())
    config = model.config
    h = config.action_horizon
    obs = _fake_obs(config)

    state = model.warm_start(jax.random.key(0), obs, num_steps=3, d=1)
    tau0 = state.tau[0]
    # Staircase in-flight prefix has at least d zeros (p=d lands exactly on t=0 too).
    assert int((tau0 == 0).sum()) >= 1
    assert float(tau0[-1]) == 1.0
    assert int(state.prefix_source_tick[0]) == 0

    emitted = []
    for tick in range(h - 2):
        acts, state = model.denoise_step(state, obs, jax.random.key(tick + 10), d=1)
        assert acts.shape == (2, 1, config.action_dim)
        assert bool(jnp.all(jnp.isfinite(acts)))
        emitted.append(acts)
        # tau profile after shift matches the original staircase (self-similarity).
        np.testing.assert_allclose(np.asarray(state.tau[0]), np.asarray(tau0), atol=1e-5)
        # Fast ticks never touch the prefix source tick (the runtime owns the delay clock).
        assert int(state.prefix_source_tick[0]) == 0

    # Tail of the buffer holds fresh noise ~ N(0, 1).
    assert float(jnp.std(state.action_buffer[:, -2:])) > 0.1

    # refresh_prefix keeps the source tick as well (the runtime installs it on publication).
    refreshed = model.refresh_prefix(state, obs)
    assert int(refreshed.prefix_source_tick[0]) == 0


def test_ablation_toggles_gate_usage_not_layout():
    """The `use_*` toggles must disable a channel in the forward pass without changing the
    parameter layout (every flowpi configuration shares one architecture and one checkpoint)."""
    obs = _fake_obs(_make_model(_flow_config()).config)

    # 1) use_fresh_state=False: no state token in the suffix (H tokens instead of H+1).
    on = _make_model(_flow_config())
    no_fresh = _make_model(_flow_config(use_fresh_state=False))
    no_delay = _make_model(_flow_config(use_delay=False))
    no_flow = _make_model(_flow_config(use_flow=False))
    actions = jax.random.normal(jax.random.key(7), (2, on.config.action_horizon, on.config.action_dim))
    tau = jnp.broadcast_to(jnp.full((2,), 0.4)[:, None], (2, on.config.action_horizon))
    tokens_on, _, _, _ = on.embed_suffix(obs, actions, tau)
    tokens_off, _, _, _ = no_fresh.embed_suffix(obs, actions, tau)
    assert tokens_on.shape[1] == on.config.action_horizon + 1
    assert tokens_off.shape[1] == on.config.action_horizon

    # 2) use_delay=False: the delay embedding is never consulted, even with vlm_delay present.
    delayed_obs = dataclasses.replace(obs, vlm_delay=jnp.zeros((2,), dtype=jnp.int32))
    assert on._vlm_delay(delayed_obs) is not None  # noqa: SLF001
    assert no_delay._vlm_delay(delayed_obs) is None  # noqa: SLF001

    # 3) use_flow=False: no flow tokens even with flow in the observation.
    flow_obs = dataclasses.replace(
        obs,
        flow={
            cam: jnp.zeros((2, on.config.flow.num_flow_steps, 2, 60, 80))
            for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        flow_masks={
            cam: jnp.ones((2, on.config.flow.num_flow_steps), dtype=jnp.bool_)
            for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
    )
    assert on.embed_flow(flow_obs) is not None
    assert no_flow.embed_flow(flow_obs) is None

    # 4) use_pir2=False: the loss still runs (baseline β(t) path) at the baseline scale.
    no_pir2 = _make_model(_flow_config(use_pir2=False))
    loss = no_pir2.compute_loss(
        jax.random.key(3),
        obs,
        jax.random.normal(jax.random.key(4), (2, no_pir2.config.action_horizon, no_pir2.config.action_dim)),
    )
    assert loss.shape == (2, no_pir2.config.action_horizon)
    assert bool(jnp.all(jnp.isfinite(loss)))

    # 5) Parameter layout is identical across toggle settings (same checkpoint format).
    def param_count(model) -> int:
        _, state = nnx.split(model)
        return sum(int(x.size) for x in jax.tree.leaves(state))

    counts = {
        param_count(_make_model(_flow_config())),
        param_count(_make_model(_flow_config(use_fresh_state=False))),
        param_count(_make_model(_flow_config(use_delay=False))),
        param_count(_make_model(_flow_config(use_flow=False))),
        param_count(_make_model(_flow_config(use_pir2=False))),
    }
    assert len(counts) == 1, f"toggle variants have different parameter layouts: {counts}"


def test_flow_config_validation():
    """Invalid FlowConfig parameters must be rejected at construction time."""
    invalid = (
        {"num_flow_steps": 0},
        {"flow_stride_frames": 0},
        {"flow_scale": 0.0},
        {"flow_clamp": 0.0},
        {"tokenizer_channels": ()},
        {"tokenizer_channels": (32, 0)},
        {"tokenizer_channels": (32, 64, 30)},
        {"tokenizer_mlp_hidden": 0},
        {"p_standard": 1.5},
        {"p_standard": -0.1},
        {"tau_jitter": 1.0},
        {"vlm_delay_max": -1},
        {"d_max": 0},
        {"injection_layers": (1, 1)},
        {"injection_layers": (-1, 2)},
        {"flow_image_size": (480, 641)},
        {"num_cross_heads": 0},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError, match="must be|must have|flow_image_size"):
            _pi0_config.FlowConfig(**kwargs)
    # A valid config (and defaults) construct fine.
    _pi0_config.FlowConfig()
    _pi0_config.FlowConfig(injection_layers=(1, 3))

    # Injection layers must fit inside the action expert depth (fail at config time).
    with pytest.raises(AssertionError, match="out of range"):
        _pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            flow=_pi0_config.FlowConfig(injection_layers=(3, 4)),  # dummy depth is 4
        )
