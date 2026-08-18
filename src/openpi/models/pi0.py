import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import flow_tokenizer as _flow_tokenizer
from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_staircase_tau(horizon: int, d: int) -> jax.Array:
    """πR² staircase noise schedule (this repo's convention: t=1 noise, t=0 clean actions).

    Positions [0, d):      t=0   (already executed / in-flight; clean inpainting, no loss)
    Positions [d, H-d):    t=(p-d)/(H-2d)   (progressively noised future)
    Positions [H-d, H):    t=1   (fresh Gaussian noise)
    """
    pos = jnp.arange(horizon)
    mid = (pos - d) / (horizon - 2 * d)
    tau = jnp.where(pos < d, 0.0, jnp.where(pos >= horizon - d, 1.0, mid))
    return tau


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.config = config
        self.flow_config = flow_cfg = config.flow if (config.flow is not None and config.flow.enabled) else None

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        flow_geom = None
        if flow_cfg is not None:
            injection_layers = flow_cfg.injection_layers
            if injection_layers is None:
                injection_layers = tuple(round(f * action_expert_config.depth) for f in (0.4, 0.65, 0.9))
            flow_geom = _gemma.FlowGeom(
                num_heads=flow_cfg.num_cross_heads,
                head_dim=flow_cfg.cross_head_dim,
                injection_layers=tuple(injection_layers),
            )

        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                flow_geom=flow_geom,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        if flow_cfg is not None:
            assert config.pi05, "flowpi requires the pi05 (adaRMS) architecture"
            h, w = flow_cfg.flow_image_size[0] // 8, flow_cfg.flow_image_size[1] // 8
            self.flow_tokenizer = _flow_tokenizer.FlowTokenizer(
                num_flow_steps=flow_cfg.num_flow_steps,
                flow_grid_size=(h, w),
                width=action_expert_config.width,
                channels=flow_cfg.tokenizer_channels,
                mlp_hidden=flow_cfg.tokenizer_mlp_hidden,
                rngs=rngs,
            )
            # Fresh-state fast channel: state token re-encoded into the suffix at every NFE.
            self.flow_state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            # Slow-channel delay embedding (zeros init: purely additive, exact no-op at start).
            self.flow_vlm_delay = nnx.Embed(
                num_embeddings=flow_cfg.vlm_delay_max + 1,
                features=paligemma_config.width,
                embedding_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            # flowpi: slow-channel delay embedding (zeros init => exact no-op when unused).
            if self.flow_config is not None and obs.vlm_delay is not None:
                image_tokens = image_tokens + self.flow_vlm_delay(obs.vlm_delay)[:, None, :]

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"] | at.Float[at.Array, "b h"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | at.Float[at.Array, "b _s emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        adarms_cond = None

        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1].
        # `timestep` may be [B] (scalar time shared by the chunk, standard path) or [B, H]
        # (per-position noise level, flowpi πR² path).
        per_position = timestep.ndim == 2
        time_emb = posemb_sincos(
            timestep.reshape(-1), self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        ).reshape(*timestep.shape, -1)
        if self.pi05:
            # time MLP (for adaRMS), applied per position
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            if per_position:
                adarms_cond = time_emb  # [B, H, emb]
            else:
                adarms_cond = time_emb  # [B, emb]
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            assert not per_position, "per-position timesteps require the pi05 architecture"
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_expert_tokens = action_time_tokens

        if self.pi05 and self.flow_config is not None:
            # flowpi fresh-state fast channel: prepend a state token to the suffix (re-encoded at
            # every NFE) and give the adaRMS cond for it the t=0 embedding (it is a conditioning
            # token, not a denoising target).
            state_token = self.flow_state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

            if adarms_cond is not None:
                if per_position:
                    zero_time = posemb_sincos(
                        jnp.zeros((noisy_actions.shape[0],), dtype=jnp.float32),
                        self.action_in_proj.out_features,
                        min_period=4e-3,
                        max_period=4.0,
                    )
                    zero_time = self.time_mlp_in(zero_time)
                    zero_time = nnx.swish(zero_time)
                    zero_time = self.time_mlp_out(zero_time)
                    zero_time = nnx.swish(zero_time)
                    adarms_cond = jnp.concatenate([zero_time[:, None, :], adarms_cond], axis=1)  # [B, H+1, emb]
                else:
                    adarms_cond = jnp.concatenate([adarms_cond[:, None, :]] * 1, axis=1)  # [B, 1, emb] broadcastable

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def embed_flow(self, obs: _model.Observation) -> tuple[at.Float[at.Array, "b f emb"], at.Bool[at.Array, "b f"]] | None:
        """Tokenizes the (normalized) optical flow into cross-attention tokens. Returns None when
        the flow fast-path is disabled or the observation carries no flow."""
        if self.flow_config is None or obs.flow is None:
            return None
        return self.flow_tokenizer(obs.flow, obs.flow_masks or dict.fromkeys(obs.flow))

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, mix_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        horizon = self.action_horizon

        if self.flow_config is not None and self.pi05:
            # πR² training: per-batch-element mix of staircase samples (p_pir2) and standard
            # scalar-time FM samples (p_standard), vectorized with per-row masks.
            cfg = self.flow_config
            is_pir2 = jax.random.bernoulli(mix_rng, p=1.0 - cfg.p_standard, shape=batch_shape)  # [B]

            # Staircase branch.
            d = jax.random.randint(time_rng, batch_shape, minval=1, maxval=cfg.d_max + 1)  # [B]
            pos = jnp.arange(horizon)
            tau_stair = jnp.where(
                pos[None, :] < d[:, None],
                0.0,
                jnp.where(pos[None, :] >= horizon - d[:, None], 1.0, (pos[None, :] - d[:, None]) / (horizon - 2 * d[:, None])),
            )  # [B, H]
            # τ jitter on the middle segment only (endpoints stay exact).
            jitter = jax.random.uniform(time_rng, actions.shape[:-1], minval=-cfg.tau_jitter, maxval=cfg.tau_jitter)
            mid = (tau_stair > 0) & (tau_stair < 1)
            tau_stair = jnp.where(mid, jnp.clip(tau_stair + jitter, 0.0, 1.0), tau_stair)

            # Standard branch: t ~ Beta(1.5, 1) shared across positions.
            t_std = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # [B]
            tau_std = einops.repeat(t_std, "b -> b h", h=horizon)

            tau = jnp.where(is_pir2[:, None], tau_stair, tau_std)  # [B, H]

            # Noisy actions: staircase rows inpaint [0, d) with the clean GT actions.
            inpaint = is_pir2[:, None] & (tau_stair == 0.0)
            t_exp = tau[..., None]
            x_t = t_exp * noise + (1 - t_exp) * actions
            x_t = jnp.where(inpaint[..., None], actions, x_t)
            u_t = noise - actions

            # Loss mask: staircase rows exclude the inpainted [0, d) prefix; standard rows keep all.
            loss_mask = jnp.where(is_pir2[:, None], ~inpaint, 1.0)
        else:
            # Baseline π0.5 path (unchanged).
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
            tau = time
            loss_mask = jnp.ones(actions.shape[:-1])

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, tau)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        flow_embedded = self.embed_flow(observation)
        if flow_embedded is not None:
            flow_tokens, flow_token_mask = flow_embedded
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
                flow=flow_tokens,
                flow_mask=flow_token_mask,
            )
        else:
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
            )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        sq = jnp.square(v_t - u_t)
        if self.flow_config is not None and self.pi05:
            # Masked mean over positions (per-position weights); unbatched dims keep [B, H] shape.
            return jnp.sum(sq * loss_mask[..., None], axis=-1) / jnp.maximum(
                jnp.sum(loss_mask, axis=-1, keepdims=True) * sq.shape[-1], 1.0
            )
        return jnp.mean(sq, axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # flow tokens are computed once per full denoising (standard path: warm-start / baselines).
        flow_embedded = self.embed_flow(observation)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            if flow_embedded is not None:
                flow_tokens, flow_token_mask = flow_embedded
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                    flow=flow_tokens,
                    flow_mask=flow_token_mask,
                )
            else:
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    # ----------------------------------------------------------------------------------------------
    # flowpi πR² streaming runtime.
    # ----------------------------------------------------------------------------------------------

    @dataclasses.dataclass
    class StreamingState:
        """Per-episode streaming state for the πR² single-NFE runtime."""

        # Action buffer [B, H, D] holding the staircase state (in-flight prefix at t=0).
        action_buffer: jax.Array
        # Per-position noise levels (this repo's convention: t=1 noise, t=0 clean), [B, H].
        tau: jax.Array
        # Cached slow-channel KV cache from the last prefix prefill.
        kv_cache: tuple | None
        # Prefix input mask (needed to build suffix attention masks against the cached prefix).
        prefix_mask: jax.Array
        # Prefix token count per batch element.
        prefix_len: jax.Array | None = None
        # Ticks elapsed since the last prefix refresh (for the d_vlm delay embedding).
        prefix_age: jax.Array | None = None

    def _prefix_forward(self, observation: _model.Observation):
        """Runs the prefix (slow channel) once and returns (kv_cache, prefix_mask)."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return kv_cache, prefix_mask

    def _suffix_forward(self, observation, x_t, tau, kv_cache, prefix_mask, flow_embedded):
        """One forward pass over the suffix (state token + action tokens) against the cached prefix."""
        batch_size = observation.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, tau)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        kwargs = {}
        if flow_embedded is not None:
            kwargs["flow"], kwargs["flow_mask"] = flow_embedded
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            **kwargs,
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def warm_start(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        d: int = 1,
        noise: jax.Array | None = None,
    ) -> StreamingState:
        """Episode start: full standard denoising, then re-noise into the πR² staircase.

        Returns a StreamingState whose buffer holds the re-noised chunk; the caller should emit
        `buffer[0:d]` (the just-cleaned in-flight actions) before entering the streaming loop.
        """
        assert self.flow_config is not None, "warm_start requires the flowpi configuration"
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        horizon = self.action_horizon

        clean = self.sample_actions(rng, observation, num_steps=num_steps, noise=noise)

        # Prefill the prefix KV cache (age 0).
        kv_cache, prefix_mask = self._prefix_forward(observation)

        renoise_rng, _ = jax.random.split(rng)
        eps = jax.random.normal(renoise_rng, clean.shape)
        tau = make_staircase_tau(horizon, d)[None, :].repeat(batch_size, axis=0)  # [B, H]
        x = tau[..., None] * eps + (1 - tau[..., None]) * clean
        # In-flight prefix stays exactly clean.
        x = jnp.where(tau[..., None] == 0.0, clean, x)

        return self.StreamingState(
            action_buffer=x,
            tau=tau,
            kv_cache=kv_cache,
            prefix_mask=prefix_mask,
            prefix_age=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    def denoise_step(
        self,
        state: StreamingState,
        observation: _model.Observation,
        rng: at.KeyArrayLike,
        *,
        d: int = 1,
    ) -> tuple[jax.Array, StreamingState]:
        """One control tick = exactly one NFE.

        1. Re-encode the fresh state token and the fresh flow tokens from the latest observation.
        2. Euler step on all positions with t > 0 using dt = d / (H - 2d), then clamp t to >= 0.
        3. Shift the buffer left by d and append d fresh Gaussians (t=1) at the tail.
        4. Return the d actions at the front of the *shift* buffer (just cleaned).

        The staircase is self-similar under this operation: all t decrease by dt and the left
        shift by d restores the original profile.
        """
        assert self.flow_config is not None, "denoise_step requires the flowpi configuration"
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        horizon = self.action_horizon
        cfg = self.flow_config
        assert 1 <= d <= cfg.d_max, f"d={d} out of range [1, {cfg.d_max}]"

        tau = state.tau
        x = state.action_buffer
        # Step size from the self-similarity derivation: all t decrease by d/(H-2d), and the left
        # shift by d restores the staircase.
        dt = d / (horizon - 2 * d)

        # Fresh flow tokens — recomputed at EVERY NFE. This is the point of FlowPi.
        flow_embedded = self.embed_flow(observation)

        v = self._suffix_forward(observation, x, tau, state.kv_cache, state.prefix_mask, flow_embedded)
        # Euler step toward t=0 (velocity convention: u = eps - a, x <- x - dt * v).
        x_new = x - dt * v
        tau_new = jnp.maximum(tau - dt, 0.0)
        # Positions that were already clean (t=0) stay clean — no updates below t=0.
        x_new = jnp.where(tau[..., None] > 0, x_new, x)

        # Shift left by d; emit the d front actions; append d fresh noise at the tail.
        emit = jax.lax.stop_gradient(x_new[:, :d])
        shifted = jax.lax.dynamic_slice(x_new, (0, d, 0), (batch_size, horizon - d, self.action_dim))
        tail_rng, _ = jax.random.split(rng)
        tail = jax.random.normal(tail_rng, (batch_size, d, self.action_dim))
        x_next = jnp.concatenate([shifted, tail], axis=1)
        tau_shifted = jax.lax.dynamic_slice(tau_new, (0, d), (batch_size, horizon - d))
        tau_next = jnp.concatenate([tau_shifted, jnp.ones((batch_size, d))], axis=1)

        new_state = self.StreamingState(
            action_buffer=x_next,
            tau=tau_next,
            kv_cache=state.kv_cache,
            prefix_mask=state.prefix_mask,
            prefix_age=(state.prefix_age + 1) if state.prefix_age is not None else None,
        )
        return emit, new_state

    def refresh_prefix(self, state: StreamingState, observation: _model.Observation) -> StreamingState:
        """Slow-channel refresh: re-run the prefix with the latest observation; reset the age."""
        assert self.flow_config is not None, "refresh_prefix requires the flowpi configuration"
        observation = _model.preprocess_observation(None, observation, train=False)
        kv_cache, prefix_mask = self._prefix_forward(observation)
        batch_size = observation.state.shape[0]
        return self.StreamingState(
            action_buffer=state.action_buffer,
            tau=state.tau,
            kv_cache=kv_cache,
            prefix_mask=prefix_mask,
            prefix_age=jnp.zeros((batch_size,), dtype=jnp.int32),
        )
