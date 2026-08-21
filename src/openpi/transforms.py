from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import pathlib
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


# --------------------------------------------------------------------------------------------------
# flowpi: flow / slow-channel-delay transforms.
# --------------------------------------------------------------------------------------------------


def compute_image_frame_offsets(
    num_flow_steps: int,
    flow_stride_frames: int,
    vlm_delay_max: int,
    flow_delay_max: int = 0,
) -> tuple[int, ...]:
    """Computes the negative frame offsets needed by both asynchronous channels.

    For a current action tick ``t``, the VLM may read ``t - d_vlm`` and a flow observation with
    age ``d_flow`` needs the target frame ``t - d_flow`` plus its internal history
    ``t - d_flow - k * flow_stride_frames``.  ``flow_delay_max=0`` preserves the original
    FlowPI offsets and therefore the d_flow=0 data path.

    Returned offsets are ascending (oldest-first), matching the ``delta_timestamps`` convention.
    """
    offsets = {0}
    offsets.update(range(1, vlm_delay_max + 1))
    offsets.update(range(flow_delay_max + 1))
    offsets.update(
        flow_delay + k * flow_stride_frames
        for flow_delay in range(flow_delay_max + 1)
        for k in range(1, num_flow_steps + 1)
    )
    return tuple(-o for o in sorted(offsets, reverse=True))


def frame_offset_index(frame_offsets: tuple[int, ...], offset: int) -> int:
    """Returns the stacking index of `offset` within a stack ordered like `frame_offsets`."""
    return frame_offsets.index(offset)


def normalize_flow(flow: np.ndarray, flow_scale: float, flow_clamp: float) -> np.ndarray:
    """Scales and clamps raw pixel flow to the normalized range used by the model."""
    return np.clip(flow.astype(np.float32) / flow_scale, -flow_clamp, flow_clamp)


def _delay_rng(seed: int, worker_id: int, call_index: int, stream_id: int) -> np.random.Generator:
    """Returns a worker-safe RNG whose stream is distinct for each asynchronous channel."""
    # Keep the existing VLM stream byte-for-byte compatible (stream 0 used
    # ``[seed, worker_id, call_index]`` before flow age was added). Flow uses a domain-separated
    # stream so the two channels never draw the same categorical sequence by construction.
    entropy = [seed, worker_id, call_index] if stream_id == 0 else [seed, stream_id, worker_id, call_index]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _worker_id() -> int:
    try:
        import torch

        worker_info = torch.utils.data.get_worker_info()
        return 0 if worker_info is None else worker_info.id
    except ImportError:
        return 0


def _sample_reachable_delay(
    rng: np.random.Generator,
    frame_index: int,
    delay_max: int,
    distribution: np.ndarray | None,
) -> int:
    """Samples a delay after restricting its support to frames in the current episode."""
    max_delay = min(delay_max, max(frame_index, 0))
    if distribution is None:
        return int(rng.integers(0, max_delay + 1))

    reachable = distribution[: max_delay + 1]
    total = reachable.sum()
    if total <= 0:
        return int(rng.integers(0, max_delay + 1))
    return int(rng.choice(max_delay + 1, p=reachable / total))


def _resolve_delay(
    data: DataDict,
    key: str,
    *,
    frame_index: int,
    delay_max: int,
    distribution: np.ndarray | None,
    rng: np.random.Generator,
) -> int:
    """Reads or samples a channel age and clamps it to the episode-reachable support."""
    if key in data:
        requested = int(np.asarray(data[key]).item())
        delay = min(max(requested, 0), delay_max, max(frame_index, 0))
    else:
        delay = _sample_reachable_delay(rng, frame_index, delay_max, distribution)
    data[key] = delay
    return delay


class LoadFlowCache(DataTransformFn):
    """Loads precomputed raw SEA-RAFT flow from the offline cache (training path).

    Expected cache layout (produced by `scripts/precompute_flow_cache.py`):
      {flow_cache_dir}/episode-{ep:06d}/{cam_key}.npy   # [T, K, 2, H//8, W//8] float16 (raw)
      {flow_cache_dir}/episode-{ep:06d}/valid.npy       # [T, K] bool
      {flow_cache_dir}/meta.json                        # K / stride / resolution checks

    Produces `data["flow"] = {cam_key: [K, 2, h, w]}` (normalized) and
    `data["flow_masks"] = {cam_key: [K]}` (per-lag validity).

    For action tick ``t``, ``flow_delay=d`` selects cache row ``s=t-d``. That row remains the
    complete observation targeted at ``s``: its K entries are ``F_(s-k*stride -> s)``.

    Flow arrays are memory-mapped (never loaded whole into RAM); the open per-episode mappings
    are bounded by an LRU so that long/many-episode datasets cannot exhaust file descriptors.
    """

    def __init__(
        self,
        flow_cache_dir: str,
        cam_keys: Sequence[str],
        *,
        num_flow_steps: int,
        flow_stride_frames: int,
        flow_image_size: tuple[int, int],
        flow_scale: float,
        flow_clamp: float,
        flow_delay_max: int = 0,
        flow_delay_distribution: Sequence[float] | None = None,
        seed: int = 1,
        sea_raft_ckpt: str | pathlib.Path | None = None,
        sea_raft_variant: str | None = None,
        sea_raft_iters: int | None = None,
        max_cached_episodes: int = 8,
    ):
        self.flow_cache_dir = pathlib.Path(flow_cache_dir)
        self.cam_keys = tuple(cam_keys)
        self.num_flow_steps = num_flow_steps
        self.flow_stride_frames = flow_stride_frames
        self.flow_image_size = tuple(flow_image_size)
        self.flow_scale = flow_scale
        self.flow_clamp = flow_clamp
        self.flow_delay_max = flow_delay_max
        self.flow_delay_distribution = (
            None if flow_delay_distribution is None else np.asarray(list(flow_delay_distribution), dtype=np.float64)
        )
        if self.flow_delay_max < 0:
            raise ValueError(f"flow_delay_max must be non-negative, got {self.flow_delay_max}")
        if self.flow_delay_distribution is not None:
            if len(self.flow_delay_distribution) != self.flow_delay_max + 1:
                raise ValueError(
                    f"flow delay distribution must have flow_delay_max+1={self.flow_delay_max + 1} weights "
                    f"(one per delay in [0, flow_delay_max]), got {len(self.flow_delay_distribution)}"
                )
            if np.any(self.flow_delay_distribution < 0) or self.flow_delay_distribution.sum() <= 0:
                raise ValueError(
                    "flow delay distribution weights must be non-negative with a positive sum, "
                    f"got {self.flow_delay_distribution}"
                )
        self.seed = seed
        self._calls = 0
        self.sea_raft_ckpt = str(sea_raft_ckpt) if sea_raft_ckpt else None
        self.sea_raft_variant = sea_raft_variant
        self.sea_raft_iters = sea_raft_iters
        self._max_cached = max(1, max_cached_episodes)
        self._validate_meta()
        # OrderedDict doubles as the LRU: hits move the key to the end, eviction drops the front.
        self._mmaps: OrderedDict[int, tuple[dict[str, np.ndarray], np.ndarray]] = OrderedDict()

    def _validate_meta(self) -> None:
        import json

        meta_path = self.flow_cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Flow cache meta not found: {meta_path}. Run scripts/precompute_flow_cache.py first."
            )
        with open(meta_path) as f:
            meta = json.load(f)
        expected = {
            "num_flow_steps": self.num_flow_steps,
            "flow_stride_frames": self.flow_stride_frames,
            "image_size": list(self.flow_image_size),
        }
        for key, value in expected.items():
            if key in meta and meta[key] != value:
                raise ValueError(
                    f"Flow cache mismatch for '{key}': cache has {meta[key]}, config expects {value}. "
                    "Recompute the flow cache or update the config."
                )

        provenance_keys = (
            "sea_raft_variant",
            "sea_raft_iters",
            "sea_raft_checkpoint_sha256",
            "camera_keys",
        )
        strict_provenance = self.sea_raft_ckpt is not None
        if strict_provenance or any(key in meta for key in provenance_keys):
            from openpi.training.sea_raft import checkpoint_sha256

            expected_provenance = {
                "sea_raft_variant": self.sea_raft_variant,
                "sea_raft_iters": self.sea_raft_iters,
                "sea_raft_checkpoint_sha256": checkpoint_sha256(self.sea_raft_ckpt),
                "camera_keys": sorted(self.cam_keys),
            }
            for key, value in expected_provenance.items():
                if value is None:
                    continue
                if key not in meta:
                    if strict_provenance:
                        raise ValueError(f"Flow cache is missing provenance field '{key}'. Recompute the flow cache.")
                    continue
                actual = sorted(meta[key]) if key == "camera_keys" else meta[key]
                if actual != value:
                    raise ValueError(
                        f"Flow cache mismatch for '{key}': cache has {actual}, config expects {value}. "
                        "Recompute the flow cache or update the config."
                    )

    def _episode(self, episode_index: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        cached = self._mmaps.get(episode_index)
        if cached is not None:
            # LRU hit: refresh the recency order.
            self._mmaps.move_to_end(episode_index)
            return cached
        ep_dir = self.flow_cache_dir / f"episode-{episode_index:06d}"
        flows = {cam: np.load(ep_dir / f"{cam}.npy", mmap_mode="r") for cam in self.cam_keys}
        valid = np.load(ep_dir / "valid.npy")
        cached = (flows, valid)
        self._mmaps[episode_index] = cached
        while len(self._mmaps) > self._max_cached:
            # Dropping the entry releases the memmap file handles (the arrays are never fully
            # loaded into memory, so eviction is cheap).
            self._mmaps.popitem(last=False)
        return cached

    def __call__(self, data: DataDict) -> DataDict:
        episode_index = int(np.asarray(data["episode_index"]).item())
        frame_index = int(np.asarray(data["frame_index"]).item())
        flow_delay = _resolve_delay(
            data,
            "flow_delay",
            frame_index=frame_index,
            delay_max=self.flow_delay_max,
            distribution=self.flow_delay_distribution,
            rng=_delay_rng(self.seed, _worker_id(), self._calls, stream_id=1),
        )
        self._calls += 1
        source_frame_index = frame_index - flow_delay
        flows, valid = self._episode(episode_index)

        data["flow"] = {}
        data["flow_masks"] = {}
        for cam in self.cam_keys:
            if not 0 <= source_frame_index < flows[cam].shape[0]:
                raise IndexError(
                    f"Flow source tick {source_frame_index} is outside episode {episode_index} "
                    f"cache range [0, {flows[cam].shape[0]})."
                )
            # Cache row s is the flow whose target/current frame is episode tick s.  Selecting
            # row t-d_flow makes the channel age real; the K lag entries inside that row are
            # still F_(s-k*stride -> s), never reinterpreted around t.
            raw = np.asarray(flows[cam][source_frame_index], dtype=np.float32)  # [K, 2, h, w]
            if raw.shape[0] != self.num_flow_steps:
                raise ValueError(f"Flow cache for {cam} has {raw.shape[0]} lags, expected {self.num_flow_steps}")
            lag_valid = valid[source_frame_index].astype(bool)  # [K]
            flow = normalize_flow(raw, self.flow_scale, self.flow_clamp)
            flow = flow * lag_valid[:, None, None, None]
            data["flow"][cam] = flow
            data["flow_masks"][cam] = lag_valid
        return data


class ComputeFlow(DataTransformFn):
    """Computes SEA-RAFT flow online from stacked camera history (inference / cache precomputation path).

    Expects `data["images"][cam_key]` to be a stacked `[T, 3, H, W]` uint8 array ordered like
    `frame_offsets` (oldest first). For action tick ``t``, ``flow_delay=d`` targets the image at
    offset ``-d`` and computes each internal lag from ``-(d + k * flow_stride_frames)`` to that
    target. Produces the same `data["flow"]` / `data["flow_masks"]` structure as `LoadFlowCache`.
    """

    def __init__(
        self,
        extractor,
        cam_keys: Sequence[str],
        *,
        num_flow_steps: int,
        flow_stride_frames: int,
        flow_scale: float,
        flow_clamp: float,
        frame_offsets: tuple[int, ...],
        flow_delay_max: int = 0,
        flow_delay_distribution: Sequence[float] | None = None,
        seed: int = 1,
    ):
        self.extractor = extractor
        self.cam_keys = tuple(cam_keys)
        self.num_flow_steps = num_flow_steps
        self.flow_stride_frames = flow_stride_frames
        self.flow_scale = flow_scale
        self.flow_clamp = flow_clamp
        self.frame_offsets = tuple(frame_offsets)
        self.flow_delay_max = flow_delay_max
        self.flow_delay_distribution = (
            None if flow_delay_distribution is None else np.asarray(list(flow_delay_distribution), dtype=np.float64)
        )
        if self.flow_delay_max < 0:
            raise ValueError(f"flow_delay_max must be non-negative, got {self.flow_delay_max}")
        if self.flow_delay_distribution is not None:
            if len(self.flow_delay_distribution) != self.flow_delay_max + 1:
                raise ValueError(
                    f"flow delay distribution must have flow_delay_max+1={self.flow_delay_max + 1} weights "
                    f"(one per delay in [0, flow_delay_max]), got {len(self.flow_delay_distribution)}"
                )
            if np.any(self.flow_delay_distribution < 0) or self.flow_delay_distribution.sum() <= 0:
                raise ValueError(
                    "flow delay distribution weights must be non-negative with a positive sum, "
                    f"got {self.flow_delay_distribution}"
                )
        self.seed = seed
        self._calls = 0

    def __call__(self, data: DataDict) -> DataDict:
        frame_index = int(np.asarray(data["frame_index"]).item())
        flow_delay = _resolve_delay(
            data,
            "flow_delay",
            frame_index=frame_index,
            delay_max=self.flow_delay_max,
            distribution=self.flow_delay_distribution,
            rng=_delay_rng(self.seed, _worker_id(), self._calls, stream_id=1),
        )
        self._calls += 1
        curr_idx = frame_offset_index(self.frame_offsets, -flow_delay)

        prev_frames = []
        curr_frames = []
        lag_valid = np.ones(self.num_flow_steps, dtype=bool)
        for k in range(1, self.num_flow_steps + 1):
            lag_offset = flow_delay + k * self.flow_stride_frames
            lag_idx = frame_offset_index(self.frame_offsets, -lag_offset)
            for cam in self.cam_keys:
                stack = np.asarray(data["images"][cam])
                prev_frames.append(stack[lag_idx])
                curr_frames.append(stack[curr_idx])
            lag_valid[k - 1] = frame_index - flow_delay >= k * self.flow_stride_frames

        prev = np.stack(prev_frames, axis=0)[None]  # [1, K*n_cam, 3, H, W]
        curr = np.stack(curr_frames, axis=0)[None]
        flow = self.extractor.compute(prev, curr)  # [1, K*n_cam, 2, H//8, W//8]
        _, _, _, h8, w8 = flow.shape
        flow = flow.reshape(self.num_flow_steps, len(self.cam_keys), 2, h8, w8)

        data["flow"] = {}
        data["flow_masks"] = {}
        for cam_i, cam in enumerate(self.cam_keys):
            raw = flow[:, cam_i]  # [K, 2, h, w]
            normalized = normalize_flow(raw, self.flow_scale, self.flow_clamp)
            normalized = normalized * lag_valid[:, None, None, None]
            data["flow"][cam] = normalized
            data["flow_masks"][cam] = lag_valid
        return data


class DelaySlowImage(DataTransformFn):
    """Samples a slow-channel VLM delay `d_vlm ~ U{0..min(vlm_delay_max, frame_index)}` and selects
    the corresponding delayed frame from the stacked camera history as the (single) prefix image.

    Must run *before* the robot-specific inputs transform (e.g. `AlohaInputs`), while the images are
    still stacked `[T, 3, H, W]` in `frame_offsets` order. When images are already single frames
    (no history loaded), sets `data["vlm_delay"] = 0` and does nothing else.

    The delay is drawn from a per-sample RNG stream derived from ``(seed, data-loader worker id,
    call counter)``, not a fixed per-instance stream: torch DataLoader workers fork the transform,
    so a single shared stream would sample the *same* delays in every worker (cross-worker
    correlation). Deriving the stream from the worker id makes the workers independent, and the
    per-call counter keeps consecutive samples within one worker varied — like the online
    runtime. The sampled delay never exceeds the frame index (a runtime refresh can never reach
    further back than the episode start).
    """

    def __init__(
        self,
        vlm_delay_max: int,
        frame_offsets: tuple[int, ...],
        *,
        seed: int = 0,
        distribution: Sequence[float] | None = None,
    ):
        self.vlm_delay_max = vlm_delay_max
        self.frame_offsets = tuple(frame_offsets)
        self.seed = seed
        self._calls = 0
        self.distribution = None if distribution is None else np.asarray(list(distribution), dtype=np.float64)
        if self.distribution is not None:
            if len(self.distribution) != vlm_delay_max + 1:
                raise ValueError(
                    f"delay distribution must have vlm_delay_max+1={vlm_delay_max + 1} weights "
                    f"(one per delay in [0, vlm_delay_max]), got {len(self.distribution)}"
                )
            if np.any(self.distribution < 0) or self.distribution.sum() <= 0:
                raise ValueError(
                    f"delay distribution weights must be non-negative with a positive sum, got {self.distribution}"
                )

    def __call__(self, data: DataDict) -> DataDict:
        images = data.get("images", {})
        stacked = next(iter(images.values()), None)
        if stacked is None or np.asarray(stacked).ndim != 4:
            data["vlm_delay"] = 0
            return data

        frame_index = int(np.asarray(data["frame_index"]).item())
        rng = _delay_rng(self.seed, _worker_id(), self._calls, stream_id=0)
        self._calls += 1
        d_vlm = _sample_reachable_delay(rng, frame_index, self.vlm_delay_max, self.distribution)

        idx = frame_offset_index(self.frame_offsets, -d_vlm) if d_vlm > 0 else frame_offset_index(self.frame_offsets, 0)
        data["images"] = {cam: np.asarray(stack)[idx] for cam, stack in images.items()}
        data["vlm_delay"] = d_vlm
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
