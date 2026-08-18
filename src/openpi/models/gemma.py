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
        return Config(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    if variant == "gemma_300m":
        return Config(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b":
        return Config(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256)
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
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype), None
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        if modulation.ndim == 2:
            scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        else:
            scale, shift, gate = jnp.split(modulation, 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding", nn.initializers.normal(), (self.vocab_size, self.embed_dim)
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
        dtype = next(x.dtype for x in xs if x is not None)
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
        big_neg = -2.3819763e38
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
    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
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
            "linear", nn.initializers.lecun_normal(in_axis=-2, out_axis=-1), (self.hidden_dim, self.features)
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


def _flow_cross_attn(q_hidden, flow, flow_mask, flow_params, slot, *, head_dim):
    """Gated cross-attention from expert-1 hidden states to flow tokens."""
    dtype = q_hidden.dtype
    q = jnp.einsum("bsd,ndh->bsnh", q_hidden, flow_params["flow_q"][slot])
    k = jnp.einsum("bfd,ndh->bfnh", flow, flow_params["flow_kv"][slot][0])
    v = jnp.einsum("bfd,ndh->bfnh", flow, flow_params["flow_kv"][slot][1])
    q = q * (head_dim**-0.5)
    logits = jnp.einsum(
        "bsnh,bfnh->bnsf", q.astype(jnp.float32), k.astype(jnp.float32), preferred_element_type=jnp.float32
    )
    big_neg = -2.3819763e38
    logits = jnp.where(flow_mask[:, None, None, :], logits, big_neg)
    probs = jax.nn.softmax(logits, axis=-1).astype(dtype)
    # A fully-masked row otherwise becomes an artificial uniform distribution because all
    # logits are the same large negative value. Force such rows to the exact no-flow output.
    has_valid_flow = jnp.any(flow_mask, axis=-1)[:, None, None, None]
    probs = jnp.where(has_valid_flow, probs, jnp.zeros_like(probs))
    out = jnp.einsum("bnsf,bfnh->bsnh", probs, v)
    return jnp.einsum("bsnh,nhd->bsd", out, flow_params["flow_out"][slot])


def _flow_rmsnorm(x, scale):
    dtype = x.dtype
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = x * jnp.reciprocal(jnp.sqrt(var + 1e-06)) * (1 + scale)
    return normed.astype(dtype)


@at.typecheck
class Block(nn.Module):
    configs: tuple[Config, ...]
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
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
        deterministic=True,
    ):
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x
        attn = Attention(configs=self.configs, name="attn")
        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])
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
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])
                x = lora.FeedForward(
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
                return h + jnp.tanh(flow_params["flow_gate"][flow_slot]).astype(h.dtype) * ca.astype(h.dtype)

            if xs[1] is not None:
                new_h = jax.lax.cond(flow_slot >= 0, inject, lambda h: h, xs[1])
                xs = [xs[0], new_h]
        xs = sharding.activation_sharding_constraint(xs)
        return xs, kv_cache


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@dataclasses.dataclass(frozen=True)
class FlowGeom:
    num_heads: int = 8
    head_dim: int = 128
    injection_layers: tuple[int, ...] = (7, 12, 16)


@at.typecheck
class Module(nn.Module):
    configs: Sequence[Config]
    embed_dtype: str
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False
    flow_geom: FlowGeom | None = None

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        self.embedder = Embedder(vocab_size=PALIGEMMA_VOCAB_SIZE, embed_dim=self.configs[0].width, name="embedder")
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(10,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, 0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
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
            self.flow_q = self.param("flow_q", lecun_2d, (n_slots, n_heads, width_e1, head_dim))
            self.flow_kv = self.param("flow_kv", lecun_3d, (n_slots, 2, n_heads, width_e1, head_dim))
            self.flow_out = self.param(
                "flow_out",
                nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                (n_slots, n_heads, head_dim, width_e1),
            )
            self.flow_gate = self.param("flow_gate", nn.initializers.zeros_init(), (n_slots, width_e1))
            self.flow_pre_norm_scale = self.param("flow_pre_norm_scale", nn.initializers.zeros_init(), (n_slots, width_e1))

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
        embedded,
        positions,
        mask,
        adarms_cond=None,
        *,
        flow=None,
        flow_mask=None,
        kv_cache=None,
        deterministic: bool = True,
    ):
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        if self.flow_geom is not None:
            flow_slot = self._make_flow_slot()
            flow_params = self._make_flow_params()
            if flow is None:
                batch_size = next(e.shape[0] for e in embedded if e is not None)
                flow = jnp.zeros(
                    (batch_size, 1, self.configs[1].width),
                    dtype=jnp.dtype(self.embed_dtype) if isinstance(self.embed_dtype, str) else self.embed_dtype,
                )
                flow_mask = jnp.ones((batch_size, 1), dtype=jnp.bool_)
        else:
            flow_slot = jnp.full((self.configs[0].depth,), -1, dtype=jnp.int32)
            flow_params = None
            flow = None
            flow_mask = None
        embedded, kv_cache = self.layers(
            embedded, kv_cache, flow, flow_mask, flow_params, flow_slot, positions, mask, adarms_cond, deterministic
        )
        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)
        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    return res.astype(x.dtype)


def _name(name, i):
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
