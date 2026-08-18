"""flowpi Flow Tokenizer: CNN + positional/lag/camera embeddings -> per-camera flow tokens.

Consumes `obs.flow` (normalized, per camera `[B, K, 2, H//8, W//8]`) and `obs.flow_masks`
(per-lag validity `[B, K]`), and produces a single token sequence `[B, n_cam*K*grid, D]` for the
gated flow cross-attention inside the action expert.

All parameters are normally initialized (no zero-init here — the only zero-initialized flow
parameters are the cross-attention gates in gemma.Module, which keep the model exactly equivalent
to the baseline π0.5 at initialization while still admitting gradient flow from step 0).
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.pi0 as _pi0

# Dataset camera order (must match model.IMAGE_KEYS).
_CAMERA_ORDER = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

# Module-level initializers (shared, stable objects). Constructing these fresh inside `__init__`
# would create new closure objects on every model instantiation, which leaks into the nnx graphdef
# static fields and breaks pjit out_shardings / output structural comparison.
_LECUN = nnx.initializers.lecun_normal()


def _normal_small(rng, shape, dtype):
    return jnp.asarray(0.02 * jax.random.normal(rng, shape), dtype)


class FlowTokenizer(nnx.Module):
    def __init__(
        self,
        *,
        num_flow_steps: int,
        flow_grid_size: tuple[int, int],  # (H//8, W//8)
        width: int,  # action expert width (output dim)
        channels: tuple[int, ...] = (32, 64, 128),
        mlp_hidden: int = 512,
        rngs: nnx.Rngs,
    ):
        self.num_flow_steps = num_flow_steps
        self.flow_grid_size = tuple(flow_grid_size)
        self.width = width

        convs = []
        in_ch = 2
        for i, out_ch in enumerate(channels):
            convs.append(
                nnx.Conv(
                    in_features=in_ch,
                    out_features=out_ch,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="SAME",
                    kernel_init=_LECUN,
                    rngs=rngs,
                )
            )
            in_ch = out_ch
        self.convs = convs

        # Spatial size after len(channels) stride-2 SAME convs. Flax k=3 s=2 SAME: 60->30->15->8,
        # i.e. even inputs halve exactly and odd inputs round up ((x+1)//2).
        def _sz(x: int) -> int:
            for _ in channels:
                x = (x + 1) // 2
            return x

        self.grid_h = _sz(flow_grid_size[0])
        self.grid_w = _sz(flow_grid_size[1])

        hidden = channels[-1]
        self.norm = nnx.LayerNorm(num_features=hidden, rngs=rngs)
        self.mlp_in = nnx.Linear(hidden, mlp_hidden, kernel_init=_LECUN, rngs=rngs)
        self.mlp_out = nnx.Linear(mlp_hidden, width, kernel_init=_LECUN, rngs=rngs)

        # Fixed 2D sincos positional embedding over the reduced grid (row + col, 64 dims each).
        # Recomputed on the fly (cheap sincos tables) instead of being stored as a module
        # attribute, so nnx.split works (plain array leaves are not supported).
        pos_dim = hidden // 2
        rows = _pi0.posemb_sincos(jnp.arange(self.grid_h, dtype=jnp.float32), pos_dim, 1e-3, 1e3)
        cols = _pi0.posemb_sincos(jnp.arange(self.grid_w, dtype=jnp.float32), pos_dim, 1e-3, 1e3)
        self._pos_spec = (pos_dim, self.grid_h, self.grid_w)

        # Learned lag / camera embeddings (normal init 0.02).
        self.lag_emb = nnx.Embed(
            num_embeddings=num_flow_steps,
            features=hidden,
            embedding_init=_normal_small,
            rngs=rngs,
        )
        self.cam_emb = nnx.Embed(
            num_embeddings=len(_CAMERA_ORDER),
            features=hidden,
            embedding_init=_normal_small,
            rngs=rngs,
        )

    def _pos_emb(self) -> jax.Array:
        """Fixed 2D sincos positional embedding [gh, gw, hidden]."""
        pos_dim, gh, gw = self._pos_spec
        rows = _pi0.posemb_sincos(jnp.arange(gh, dtype=jnp.float32), pos_dim, 1e-3, 1e3)
        cols = _pi0.posemb_sincos(jnp.arange(gw, dtype=jnp.float32), pos_dim, 1e-3, 1e3)
        return jnp.concatenate(
            [jnp.broadcast_to(rows[:, None, :], (gh, gw, pos_dim)),
             jnp.broadcast_to(cols[None, :, :], (gh, gw, pos_dim))],
            axis=-1,
        )

    def embed_camera(self, flow: jax.Array) -> jax.Array:
        """flow: [B, K, 2, H, W] -> tokens [B, K*gh*gw, hidden]."""
        b, k = flow.shape[0], flow.shape[1]
        x = flow.reshape((b * k, *flow.shape[2:]))
        x = x.transpose(0, 2, 3, 1)  # [B*K, H, W, 2] (channel-last for nnx.Conv)
        for conv in self.convs:
            x = conv(x)
            x = nnx.silu(x)
        # x: [B*K, gh, gw, C] -> [B*K, gh*gw, C] (grid size read from the actual conv output)
        gh, gw = x.shape[1], x.shape[2]
        if (gh, gw) != (self.grid_h, self.grid_w):
            raise ValueError(f"Flow grid mismatch: convs produced {gh}x{gw}, expected {self.grid_h}x{self.grid_w}")
        x = x.reshape(b * k, gh * gw, -1)
        x = x + self._pos_emb().reshape(gh * gw, -1)
        tokens = x.reshape(b, k * gh * gw, -1)
        return tokens

    def __call__(self, flow: dict, flow_masks: dict) -> tuple[jax.Array, jax.Array]:
        """Returns (tokens [B, n_cam*K*gh*gw, width], token_mask [B, n_cam*K*gh*gw])."""
        token_lists = []
        mask_lists = []
        for cam_i, cam in enumerate(_CAMERA_ORDER):
            if cam not in flow:
                continue
            tokens = self.embed_camera(flow[cam])  # [B, K*G, hidden]
            b = tokens.shape[0]
            k = self.num_flow_steps
            g = self.grid_h * self.grid_w
            hidden = tokens.shape[-1]

            lag = self.lag_emb(jnp.arange(k))  # [K, hidden]
            cam_e = self.cam_emb(jnp.array(cam_i))  # [hidden]
            tokens = tokens.reshape(b, k, g, hidden) + lag[None, :, None, :] + cam_e[None, None, None, :]
            tokens = tokens.reshape(b, k * g, hidden)

            # Per-lag validity broadcast to the G spatial tokens of each lag.
            lag_mask = jnp.asarray(flow_masks[cam], dtype=jnp.bool_)  # [B, K]
            token_mask = jnp.broadcast_to(lag_mask[:, :, None], (b, k, g)).reshape(b, k * g)

            token_lists.append(tokens)
            mask_lists.append(token_mask)

        tokens = jnp.concatenate(token_lists, axis=1)  # [B, F, hidden]
        token_mask = jnp.concatenate(mask_lists, axis=1)  # [B, F]

        tokens = self.norm(tokens)
        tokens = nnx.silu(self.mlp_in(tokens))
        tokens = self.mlp_out(tokens)  # [B, F, width]
        return jnp.asarray(tokens, dtype=jnp.float32), token_mask
