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


def compute_image_frame_offsets(num_flow_steps: int, flow_stride_frames: int, vlm_delay_max: int) -> tuple[int, ...]:
    """Computes the set of *negative* frame offsets (into the past) that must be loaded for the image
    keys: one frame per flow lag (`k * flow_stride_frames` for `k = 1..num_flow_steps`) plus the slow
    channel delay window (`1..vlm_delay_max`), always including the current frame (offset 0).

    Returned in ascending (oldest-first) order, matching the `delta_timestamps` convention
    `[(-i) / fps for i in vlm_delay_max..0]`.
    """
    offsets = {0}
    offsets.update(k * flow_stride_frames for k in range(1, num_flow_steps + 1))
    offsets.update(range(1, vlm_delay_max + 1))
    return tuple(-o for o in sorted(offsets, reverse=True))


def frame_offset_index(frame_offsets: tuple[int, ...], offset: int) -> int:
    """Returns the stacking index of `offset` within a stack ordered like `frame_offsets`."""
    return frame_offsets.index(offset)


def normalize_flow(flow: np.ndarray, flow_scale: float, flow_clamp: float) -> np.ndarray:
    """Scales and clamps raw pixel flow to the normalized range used by the model."""
    return np.clip(flow.astype(np.float32) / flow_scale, -flow_clamp, flow_clamp)


class LoadFlowCache(DataTransformFn):
    """Loads precomputed raw SEA-RAFT flow from the offline cache (training path).

    Expected cache layout (produced by `scripts/precompute_flow_cache.py`):
      {flow_cache_dir}/episode-{ep:06d}/{cam_key}.npy   # [T, K, 2, H//8, W//8] float16 (raw)
      {flow_cache_dir}/episode-{ep:06d}/valid.npy       # [T, K] bool
      {flow_cache_dir}/meta.json                        # K / stride / resolution checks

    Produces `data["flow"] = {cam_key: [K, 2, h, w]}` (normalized) and
    `data["flow_masks"] = {cam_key: [K]}` (per-lag validity).
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
    ):
        self.flow_cache_dir = pathlib.Path(flow_cache_dir)
        self.cam_keys = tuple(cam_keys)
        self.num_flow_steps = num_flow_steps
        self.flow_stride_frames = flow_stride_frames
        self.flow_image_size = tuple(flow_image_size)
        self.flow_scale = flow_scale
        self.flow_clamp = flow_clamp
        self._validate_meta()
        self._mmaps: dict[int, tuple[dict[str, np.ndarray], np.ndarray]] = {}

    def _validate_meta(self) -> None:
        import json

        meta_path = self.flow_cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Flow cache meta not found: {meta_path}. Run scripts/precompute_flow_cache.py first.")
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

    def _episode(self, episode_index: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        cached = self._mmaps.get(episode_index)
        if cached is None:
            ep_dir = self.flow_cache_dir / f"episode-{episode_index:06d}"
            flows = {cam: np.load(ep_dir / f"{cam}.npy", mmap_mode="r") for cam in self.cam_keys}
            valid = np.load(ep_dir / "valid.npy")
            cached = (flows, valid)
            self._mmaps[episode_index] = cached
        return cached

    def __call__(self, data: DataDict) -> DataDict:
        episode_index = int(np.asarray(data["episode_index"]).item())
        frame_index = int(np.asarray(data["frame_index"]).item())
        flows, valid = self._episode(episode_index)

        data["flow"] = {}
        data["flow_masks"] = {}
        for cam in self.cam_keys:
            raw = np.asarray(flows[cam][frame_index], dtype=np.float32)  # [K, 2, h, w]
            if raw.shape[0] != self.num_flow_steps:
                raise ValueError(f"Flow cache for {cam} has {raw.shape[0]} lags, expected {self.num_flow_steps}")
            lag_valid = valid[frame_index].astype(bool)  # [K]
            flow = normalize_flow(raw, self.flow_scale, self.flow_clamp)
            flow = flow * lag_valid[:, None, None, None]
            data["flow"][cam] = flow
            data["flow_masks"][cam] = lag_valid
        return data


class ComputeFlow(DataTransformFn):
    """Computes SEA-RAFT flow online from stacked camera history (inference / cache precomputation path).

    Expects `data["images"][cam_key]` to be a stacked `[T, 3, H, W]` uint8 array ordered like
    `frame_offsets` (oldest first, current frame at `frame_offset_index(frame_offsets, 0)`).
    Produces the same `data["flow"]` / `data["flow_masks"]` structure as `LoadFlowCache`.
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
    ):
        self.extractor = extractor
        self.cam_keys = tuple(cam_keys)
        self.num_flow_steps = num_flow_steps
        self.flow_stride_frames = flow_stride_frames
        self.flow_scale = flow_scale
        self.flow_clamp = flow_clamp
        self.frame_offsets = tuple(frame_offsets)

    def __call__(self, data: DataDict) -> DataDict:
        frame_index = int(np.asarray(data["frame_index"]).item())
        curr_idx = frame_offset_index(self.frame_offsets, 0)

        prev_frames = []
        curr_frames = []
        lag_valid = np.ones(self.num_flow_steps, dtype=bool)
        for k in range(1, self.num_flow_steps + 1):
            lag_idx = frame_offset_index(self.frame_offsets, -k * self.flow_stride_frames)
            for cam in self.cam_keys:
                stack = np.asarray(data["images"][cam])
                prev_frames.append(stack[lag_idx])
                curr_frames.append(stack[curr_idx])
            lag_valid[k - 1] = frame_index >= k * self.flow_stride_frames

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
    """Samples a slow-channel VLM delay `d_vlm ~ U{0..vlm_delay_max}` and selects the corresponding
    delayed frame from the stacked camera history as the (single) prefix image.

    Must run *before* the robot-specific inputs transform (e.g. `AlohaInputs`), while the images are
    still stacked `[T, 3, H, W]` in `frame_offsets` order. When images are already single frames
    (no history loaded), sets `data["vlm_delay"] = 0` and does nothing else.
    """

    def __init__(self, vlm_delay_max: int, frame_offsets: tuple[int, ...], *, seed: int = 0):
        self.vlm_delay_max = vlm_delay_max
        self.frame_offsets = tuple(frame_offsets)
        self.seed = seed

    def __call__(self, data: DataDict) -> DataDict:
        images = data.get("images", {})
        stacked = next(iter(images.values()), None)
        if stacked is None or np.asarray(stacked).ndim != 4:
            data["vlm_delay"] = 0
            return data

        episode_index = int(np.asarray(data["episode_index"]).item())
        frame_index = int(np.asarray(data["frame_index"]).item())
        rng = np.random.default_rng(self.seed * 1_000_003 + episode_index * 100_003 + frame_index)
        d_vlm = int(rng.integers(0, self.vlm_delay_max + 1))

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
