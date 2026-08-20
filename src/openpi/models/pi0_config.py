import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class FlowConfig:
    """Configuration for the flowpi flow fast-path, πR² streaming, and slow-channel delay.

    When `Pi0Config.flow is None`, the model graph is identical to the baseline π0.5.
    """

    enabled: bool = True
    # Optical flow.
    num_flow_steps: int = 2  # K
    flow_stride_frames: int = 3  # Δ (dataset frames; Δt = Δ/fps)
    flow_scale: float = 4.0
    flow_clamp: float = 8.0
    flow_image_size: tuple[int, int] = (480, 640)
    tokenizer_channels: tuple[int, ...] = (32, 64, 128)
    tokenizer_mlp_hidden: int = 512
    # Cross-attention.
    num_cross_heads: int = 8
    cross_head_dim: int = 128
    injection_layers: tuple[int, ...] | None = None  # None => (7, 12, 16)
    # πR².
    d_max: int = 5  # must be < action_horizon / 2
    p_standard: float = 0.2
    tau_jitter: float = 0.01
    # Slow channel delay.
    vlm_delay_max: int = 10
    # Ablation toggles. These gate the *use* of a channel in the forward pass but never the
    # parameter layout: the flow modules are always created when `enabled`, so every flowpi
    # configuration (including ablations) shares one architecture and can load the same
    # checkpoint. A channel that is switched off receives no gradient and stays exactly at its
    # initialization.
    use_fresh_state: bool = True  # flow_state_proj state token in the suffix at every NFE.
    use_delay: bool = True  # flow_vlm_delay_fast adaRMS conditioning.
    use_flow: bool = True  # flow tokenizer + FlowGeom cross-attention.
    use_pir2: bool = True  # πR² staircase noise; False falls back to baseline β(t) noise.
    # Image augmentation strategy during training. The flow cache is computed offline on the raw
    # frames, so any geometric augmentation of the VLM image (RandomCrop / Rotate) would put the
    # image and the flow in different coordinate systems. Default (False) keeps only photometric
    # augmentation (ColorJitter) for flowpi models; set to True to opt back into the upstream
    # geometric augmentation at the cost of that spatial mismatch.
    image_geometric_aug: bool = False

    def __post_init__(self):
        if self.injection_layers is None:
            object.__setattr__(self, "injection_layers", (7, 12, 16))
        if self.num_flow_steps <= 0:
            raise ValueError(f"num_flow_steps must be positive, got {self.num_flow_steps}")
        if self.flow_stride_frames <= 0:
            raise ValueError(f"flow_stride_frames must be positive, got {self.flow_stride_frames}")
        if self.flow_scale <= 0:
            raise ValueError(f"flow_scale must be positive, got {self.flow_scale}")
        if self.flow_clamp <= 0:
            raise ValueError(f"flow_clamp must be positive, got {self.flow_clamp}")
        if not self.tokenizer_channels:
            raise ValueError("tokenizer_channels must be non-empty")
        if any(c <= 0 for c in self.tokenizer_channels):
            raise ValueError(f"tokenizer_channels must be positive, got {self.tokenizer_channels}")
        if self.tokenizer_mlp_hidden <= 0:
            raise ValueError(f"tokenizer_mlp_hidden must be positive, got {self.tokenizer_mlp_hidden}")
        if not 0.0 <= self.p_standard <= 1.0:
            raise ValueError(f"p_standard must be in [0, 1], got {self.p_standard}")
        if not 0.0 <= self.tau_jitter < 1.0:
            raise ValueError(f"tau_jitter must be in [0, 1), got {self.tau_jitter}")
        if self.vlm_delay_max < 0:
            raise ValueError(f"vlm_delay_max must be non-negative, got {self.vlm_delay_max}")
        if self.num_cross_heads <= 0:
            raise ValueError(f"num_cross_heads must be positive, got {self.num_cross_heads}")
        if self.cross_head_dim <= 0:
            raise ValueError(f"cross_head_dim must be positive, got {self.cross_head_dim}")
        if self.injection_layers is not None:
            if any(layer < 0 for layer in self.injection_layers):
                raise ValueError(f"injection_layers must be non-negative, got {self.injection_layers}")
            if len(set(self.injection_layers)) != len(self.injection_layers):
                raise ValueError(f"injection_layers must be unique, got {self.injection_layers}")
        if len(self.flow_image_size) != 2 or any(s <= 0 or s % 8 != 0 for s in self.flow_image_size):
            raise ValueError(f"flow_image_size must be (h, w) with h, w multiples of 8, got {self.flow_image_size}")
        if self.d_max <= 0:
            raise ValueError(f"d_max must be positive, got {self.d_max}")


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # flowpi: flow fast-path / πR² streaming / slow-channel delay configuration.
    # None (default) keeps the model graph identical to the baseline π0.5.
    flow: FlowConfig | None = None

    # Freeze the SigLIP vision tower. Upstream default (False) trains everything; the flowpi
    # configs opt in explicitly so the frozen-vision policy stays an experiment decision and
    # baseline π0.5 configs keep upstream semantics.
    freeze_vision_encoder: bool = False

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.flow is not None and self.flow.enabled:
            assert self.flow.d_max < self.action_horizon / 2, (
                f"flow.d_max ({self.flow.d_max}) must be < action_horizon/2 ({self.action_horizon / 2})"
            )
            depth = _gemma.get_config(self.action_expert_variant).depth
            for layer in self.flow.injection_layers:
                assert 0 <= layer < depth, (
                    f"flow injection layer {layer} is out of range for action expert depth {depth}"
                )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        flow = self.flow
        flow_spec = None
        flow_masks_spec = None
        if flow is not None and flow.enabled:
            h, w = flow.flow_image_size[0] // 8, flow.flow_image_size[1] // 8
            flow_spec = {
                "base_0_rgb": jax.ShapeDtypeStruct([batch_size, flow.num_flow_steps, 2, h, w], jnp.float32),
                "left_wrist_0_rgb": jax.ShapeDtypeStruct([batch_size, flow.num_flow_steps, 2, h, w], jnp.float32),
                "right_wrist_0_rgb": jax.ShapeDtypeStruct([batch_size, flow.num_flow_steps, 2, h, w], jnp.float32),
            }
            flow_masks_spec = {
                "base_0_rgb": jax.ShapeDtypeStruct([batch_size, flow.num_flow_steps], jnp.bool_),
                "left_wrist_0_rgb": jax.ShapeDtypeStruct([batch_size, flow.num_flow_steps], jnp.bool_),
                "right_wrist_0_rgb": jax.ShapeDtypeStruct([batch_size, flow.num_flow_steps], jnp.bool_),
            }

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                flow=flow_spec,
                flow_masks=flow_masks_spec,
                vlm_delay=(
                    jax.ShapeDtypeStruct([batch_size], jnp.int32)
                    if flow is not None and flow.enabled and flow.vlm_delay_max > 0
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            # With no LoRA the upstream default is to train everything. The flowpi frozen-vision
            # policy opts in via `freeze_vision_encoder` (flowpi additionally keeps SEA-RAFT
            # frozen outside the JAX parameter tree).
            if self.freeze_vision_encoder:
                return nnx_utils.PathRegex("PaliGemma/img.*")
            return nnx.Nothing
        return nnx.All(*filters)
