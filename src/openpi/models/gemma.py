# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm. `cond` may be [b, d] (one modulation shared by all positions, legacy
        # behavior) or [b, s, d] (per-position modulation, flowpi πR² path).
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        if modulation.ndim == 2:
            scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        else:
            scale, shift, gate = jnp.split(modulation, 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


def _flow_cross_attn(q_hidden, flow, flow_mask, flow_params, slot, *, head_dim):
    """Gated cross-attention from expert-1 hidden states to flow tokens.

    q_hidden: [b, s, D] expert-1 hidden states (already pre-normalized).
    flow: [b, n_flow, D] flow tokens. flow_mask: [b, n_flow] bool validity.
    flow_params: dict of slot-stacked parameters; slot: int (which slot to use).
    """
    dtype = q_hidden.dtype
    q = jnp.einsum("bsd,ndh->bsnh", q_hidden, flow_params["flow_q"][slot])  # [b, s, n, h]
    k = jnp.einsum("bfd,ndh->bfnh", flow, flow_params["flow_kv"][slot][0])  # [b, n_flow, n, h]
    v = jnp.einsum("bfd,ndh->bfnh", flow, flow_params["flow_kv"][slot][1])  # [b, n_flow, n, h]
    q = q * (head_dim**-0.5)
    logits = jnp.einsum(
        "bsnh,bfnh->bnsf", q.astype(jnp.float32), k.astype(jnp.float32), preferred_element_type=jnp.float32
    )
    big_neg = -2.3819763e38  # See gemma/modules.py
    logits = jnp.where(flow_mask[:, None, None, :], logits, big_neg)  # mask over k tokens
    probs = jax.nn.softmax(logits, axis=-1).astype(dtype)
    has_valid_flow = jnp.any(flow_mask, axis=-1)[:, None, None, None]
    probs = jnp.where(has_valid_flow, probs, jnp.zeros_like(probs))
    out = jnp.einsum("bnsf,bfnh->bsnh", probs, v)  # [b, s, n, h]
    return jnp.einsum("bsnh,nhd->bsd", out, flow_params["flow_out"][slot])  # [b, s, D]


def _flow_rmsnorm(x, scale):
    dtype = x.dtype
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = x * jnp.reciprocal(jnp.sqrt(var + 1e-06)) * (1 + scale)
    return normed.astype(dtype)


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    # flowpi: if true, apply flow cross-attention injection at the end of this block for expert 1.
    flow_enabled: bool = False

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        flow,
        flow_mask,
        flow_params,
        flow_slot,
        positions,
        attn_mask,
        adarms_cond,
        deterministic=True,  # noqa: FBT002
    ):
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]

        if self.flow_enabled:
            head_dim = flow_params["flow_q"].shape[-1]

            def inject(h):
                hn = _flow_rmsnorm(h, flow_params["flow_pre_norm_scale"][flow_slot])
                ca = _flow_cross_attn(hn, flow, flow_mask, flow_params, flow_slot, head_dim=head_dim)
                g = jnp.tanh(flow_params["flow_gate"][flow_slot]).astype(h.dtype)
                h_out = h + g * ca.astype(h.dtype)
                # Telemetry: magnitude of the gated CA residual relative to the hidden state at
                # this layer (r_l = |g_l C_l| / |h_l|), a float32 scalar per batch element.
                ratio = jnp.mean(jnp.abs(g * ca.astype(h.dtype)), dtype=jnp.float32) / (
                    jnp.mean(jnp.abs(h), dtype=jnp.float32) + 1e-6
                )
                return h_out, ratio

            def no_inject(h):
                return h, jnp.zeros((), dtype=jnp.float32)

            # Non-injection layers skip the cross-attention matmul entirely. Prefix-only passes
            # (expert 1 is None) skip injection as well.
            if xs[1] is not None:
                new_h, flow_stat = jax.lax.cond(flow_slot >= 0, inject, no_inject, xs[1])
                xs = [xs[0], new_h]
            else:
                flow_stat = jnp.zeros((), dtype=jnp.float32)
        else:
            flow_stat = jnp.zeros((), dtype=jnp.float32)

        xs = sharding.activation_sharding_constraint(xs)

        return xs, (kv_cache, flow_stat)


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@dataclasses.dataclass(frozen=True)
class FlowGeom:
    """Geometry of the flowpi flow cross-attention (parameters live in the outer Module, one set
    per injection layer slot — NOT stacked along depth)."""

    num_heads: int = 8
    head_dim: int = 128
    injection_layers: tuple[int, ...] = (7, 12, 16)


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False
    # flowpi: flow cross-attention geometry. None keeps the module identical to the baseline.
    flow_geom: FlowGeom | None = None

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(10,),  # 0=self, 10=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,  # kv_cache
                nn.broadcast,  # flow
                nn.broadcast,  # flow_mask
                nn.broadcast,  # flow_params
                0,  # flow_slot
                nn.broadcast,  # positions
                nn.broadcast,  # mask
                nn.broadcast,  # adarms_cond
                nn.broadcast,  # deterministic
            ),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            flow_enabled=self.flow_geom is not None,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

        if self.flow_geom is not None:
            n_slots = len(self.flow_geom.injection_layers)
            n_heads, head_dim = self.flow_geom.num_heads, self.flow_geom.head_dim
            width_e1 = self.configs[1].width
            lecun_2d = nn.initializers.lecun_normal(in_axis=-2, out_axis=-1)
            lecun_3d = nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1))
            # Slot-stacked flow cross-attention parameters. Created in the outer module (outside
            # the depth scan) so they exist exactly once per slot, not depth times.
            self.flow_q = self.param("flow_q", lecun_2d, (n_slots, n_heads, width_e1, head_dim))
            self.flow_kv = self.param("flow_kv", lecun_3d, (n_slots, 2, n_heads, width_e1, head_dim))
            self.flow_out = self.param(
                "flow_out",
                nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                (n_slots, n_heads, head_dim, width_e1),
            )
            # The ONLY zero-initialized flow parameters (initial tanh(gate)=0 => exact π0.5 equivalence).
            self.flow_gate = self.param("flow_gate", nn.initializers.zeros_init(), (n_slots, width_e1))
            self.flow_pre_norm_scale = self.param(
                "flow_pre_norm_scale", nn.initializers.zeros_init(), (n_slots, width_e1)
            )

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    def _make_flow_slot(self) -> jax.Array:
        depth = self.configs[0].depth
        slots = jnp.full((depth,), -1, dtype=jnp.int32)
        for slot, layer in enumerate(self.flow_geom.injection_layers):
            if layer >= depth:
                raise ValueError(f"flow injection layer {layer} out of range for depth {depth}")
            slots = slots.at[layer].set(slot)
        return slots

    def _make_flow_params(self) -> dict[str, jax.Array]:
        return {
            "flow_q": self.flow_q,
            "flow_kv": self.flow_kv,
            "flow_out": self.flow_out,
            "flow_gate": self.flow_gate,
            "flow_pre_norm_scale": self.flow_pre_norm_scale,
        }

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | at.Float[at.Array, "b _s _d"] | None] | None = None,
        *,
        flow: at.Float[at.Array, "b _f _d"] | None = None,
        flow_mask: at.Bool[at.Array, "b _f"] | None = None,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
        return_flow_stats: bool = False,
    ) -> (
        tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]
        | tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache, at.Float[at.Array, " _slots"]]
    ):
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        if self.flow_geom is not None:
            flow_slot = self._make_flow_slot()
            flow_params = self._make_flow_params()
            if flow is None:
                # No flow tokens (e.g. `use_flow=False` ablation, or an observation without
                # flow). Substitute a dummy token with an *all-invalid* mask so the gated
                # cross-attention output is exactly zero (identity) and the flow branch receives
                # no gradient — never learned attention over a dummy token.
                batch_size = next(e.shape[0] for e in embedded if e is not None)
                flow = jnp.zeros(
                    (batch_size, 1, self.configs[1].width),
                    dtype=jnp.dtype(self.embed_dtype) if isinstance(self.embed_dtype, str) else self.embed_dtype,
                )
                flow_mask = jnp.zeros((batch_size, 1), dtype=jnp.bool_)
        else:
            flow_slot = jnp.full((self.configs[0].depth,), -1, dtype=jnp.int32)
            flow_params = None
            flow = None
            flow_mask = None

        embedded, (kv_cache, flow_stats) = self.layers(
            embedded, kv_cache, flow, flow_mask, flow_params, flow_slot, positions, mask, adarms_cond, deterministic
        )

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        outputs = [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ]
        if not return_flow_stats:
            return outputs, kv_cache
        # Per-slot CA residual ratios (0.0 for non-injection layers); when the flow branch is
        # disabled the per-slot values are all zero.
        if self.flow_geom is None:
            flow_stats = jnp.zeros((len(self.flow_geom or ()),), dtype=jnp.float32)
        else:
            flow_stats = flow_stats[jnp.asarray(self.flow_geom.injection_layers)]
        return outputs, kv_cache, flow_stats

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
