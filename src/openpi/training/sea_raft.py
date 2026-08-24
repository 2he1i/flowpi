"""Frozen SEA-RAFT optical flow extractor (torch), used outside of the JAX parameter tree.

Computes 1/8-resolution optical flow `flow_8x` between consecutive (or lagged) image pairs.
The model is completely frozen: it is never differentiated, never checkpointed, and never
part of the JAX model. During training, flow is consumed from a precomputed offline cache;
this extractor is only used for cache precomputation, inference, and tests.
"""

import contextlib
import functools
import hashlib
import inspect
import os
import pathlib
import sys

import numpy as np
import torch

_SEA_RAFT_CORE_DIR = pathlib.Path(__file__).resolve().parents[3] / "SEA-RAFT" / "core"

_VARIANT_CONFIGS = {
    # variant: (dim, iters, radius, block_dims)
    "S": (96, 4, 4, [64, 96, 128]),
    "M": (128, 4, 4, [64, 128, 256]),
    "L": (192, 6, 4, [64, 160, 224]),
}


def resolve_sea_raft_iters(variant: str, iters: int | None = None) -> int:
    """Return the effective refinement iteration count for a SEA-RAFT variant."""
    if variant not in _VARIANT_CONFIGS:
        raise ValueError(f"Unknown SEA-RAFT variant: {variant}. Choose from {list(_VARIANT_CONFIGS)}")
    if iters is None:
        return _VARIANT_CONFIGS[variant][1]
    if iters <= 0:
        raise ValueError(f"SEA-RAFT iters must be positive, got {iters}")
    return iters


def checkpoint_sha256(path: str | pathlib.Path | None) -> str | None:
    """Return a checkpoint's SHA256 digest, or None when random initialization is used."""
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_model_state_dict(checkpoint: object) -> dict:
    """Extract model weights from raw SEA-RAFT or training-wrapper checkpoints."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}")

    for key in ("state_dict", "model_state_dict"):
        weights = checkpoint.get(key)
        if isinstance(weights, dict):
            return weights

    # Official SEA-RAFT checkpoints are saved as the raw model state dict.
    return checkpoint


def _make_raft_args(dim: int, iters: int, radius: int, block_dims: list[int]):
    # SEA-RAFT's RAFT constructor mutates the args namespace (e.g. `args.corr_levels = 4`), so we
    # use a plain namespace instead of a frozen dataclass.
    from types import SimpleNamespace  # noqa: PLC0415

    return SimpleNamespace(
        dim=dim,
        iters=iters,
        radius=radius,
        block_dims=list(block_dims),
        use_var=True,
        var_min=0,
        var_max=10,
        initial_dim=64,
        num_blocks=2,
        num_head=1,
        pretrain="resnet34",
        corr_levels=4,
    )


@functools.lru_cache(maxsize=1)
def _import_raft():
    """Import and validate the patched SEA-RAFT RAFT module."""
    if not _SEA_RAFT_CORE_DIR.is_dir():
        raise RuntimeError(
            "SEA-RAFT submodule is missing. Run `git submodule update --init SEA-RAFT` and "
            "`uv run python scripts/setup_sea_raft.py`."
        )
    core_dir = str(_SEA_RAFT_CORE_DIR)
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    from raft import RAFT  # noqa: PLC0415

    if "return_low_res" not in inspect.signature(RAFT.forward).parameters:
        raise RuntimeError(
            "SEA-RAFT is missing FlowPi's return_low_res API. Run `uv run python scripts/setup_sea_raft.py`."
        )
    return RAFT


class SeaRaftFlowExtractor:
    """Wraps a frozen SEA-RAFT (Tartan-M by default) model with a numpy interface.

    Args:
        ckpt_path: Path to raw model weights or a training-wrapper checkpoint (.pt/.pth). If None,
            random weights are used.
        allow_random_init: If False (default), raise when `ckpt_path` is None. Set to True only
            for tests and smoke runs where random weights are acceptable.
        variant: Model variant ("S" | "M" | "L"). The user's fine-tuned weights use "M".
        device: Torch device for inference.
        iters: Number of refinement iterations (default from the variant config).
        precision: Inference autocast mode (``fp32``, ``fp16``, ``bf16``, or ``auto``). The
            default remains ``fp32`` so offline training-cache generation is numerically stable;
            deployment can opt into Tensor Core inference without changing the training path.
    """

    def __init__(
        self,
        ckpt_path: str | pathlib.Path | None = None,
        variant: str = "M",
        device: str = "cpu",
        iters: int | None = None,
        *,
        allow_random_init: bool = False,
        precision: str = "fp32",
    ):
        effective_iters = resolve_sea_raft_iters(variant, iters)
        dim, _, radius, block_dims = _VARIANT_CONFIGS[variant]
        args = _make_raft_args(
            dim=dim,
            iters=effective_iters,
            radius=radius,
            block_dims=block_dims,
        )
        self._iters = args.iters
        self._device = torch.device(device)
        precision = str(precision).lower()
        if precision not in {"fp32", "fp16", "bf16", "auto"}:
            raise ValueError(f"Unsupported SEA-RAFT inference precision {precision!r}; use fp32, fp16, bf16, or auto")
        if precision == "auto":
            precision = "fp16" if self._device.type == "cuda" else "fp32"
        if self._device.type != "cuda" and precision != "fp32":
            # CPU autocast is not a useful or consistent deployment path for SEA-RAFT. Keep the
            # extractor usable in CPU smoke tests while making the fallback explicit in metrics.
            precision = "fp32"
        self._precision = precision
        self._amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
        if self._device.type == "cuda":
            # The policy server already owns JAX's CPU dispatch pools. Limiting Torch's host
            # helper pools for inference prevents 64 inter-op/intra-op workers from competing
            # with the fast RPC and slow-channel scheduler. This changes no CUDA kernels or
            # model inputs; it is opt-in so offline training/cache generation keeps its default.
            thread_setting = os.environ.get("FLOWPI_SEA_RAFT_TORCH_THREADS")
            if thread_setting:
                thread_count = int(thread_setting)
                if thread_count <= 0:
                    raise ValueError("FLOWPI_SEA_RAFT_TORCH_THREADS must be positive")
                torch.set_num_threads(thread_count)
                with contextlib.suppress(RuntimeError):
                    torch.set_num_interop_threads(thread_count)
            # Deployment uses one fixed geometry (six 480x640 images per request). Let cuDNN
            # select the fastest convolution algorithms for that stable shape.
            torch.backends.cudnn.benchmark = True

        raft_cls = _import_raft()
        # Random init when no checkpoint is given. Random weights produce garbage flow, so this
        # must be an explicit opt-in (tests/smoke runs only), and is seeded for reproducibility.
        if ckpt_path is None:
            if not allow_random_init:
                raise ValueError(
                    "SEA-RAFT would run with random weights: `ckpt_path` is None. Pass a checkpoint, "
                    "or set allow_random_init=True for tests/smoke runs."
                )
            torch.manual_seed(0)
        self._model = raft_cls(args).to(self._device).eval()
        for p in self._model.parameters():
            p.requires_grad = False

        if ckpt_path is not None:
            # FlowPi only needs tensor weights from the raw SEA-RAFT or training-wrapper
            # dictionary. Restrict deserialization to PyTorch's safe weights-only mode so an
            # uploaded inference checkpoint cannot execute arbitrary pickle payloads.
            checkpoint = torch.load(str(ckpt_path), map_location=self._device, weights_only=True)
            state_dict = _extract_model_state_dict(checkpoint)
            self._model.load_state_dict(state_dict, strict=True)

    @contextlib.contextmanager
    def _autocast(self):
        if self._amp_dtype is None:
            yield
            return
        with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
            yield

    @torch.inference_mode()
    def compute(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Compute 1/8-resolution optical flow between two image stacks.

        Args:
            prev: uint8 array [B, n_cam, 3, H, W] (earlier frames).
            curr: uint8 array [B, n_cam, 3, H, W] (later frames; flow is defined as
                displacement from `prev` to `curr`).

        Returns:
            float32 array [B, n_cam, 2, H//8, W//8] of optical flow (in pixels at 1/8 scale).
        """
        prev = np.asarray(prev)
        curr = np.asarray(curr)
        if prev.shape != curr.shape:
            raise ValueError(f"prev/curr shape mismatch: {prev.shape} vs {curr.shape}")
        if prev.ndim != 5 or prev.shape[2] != 3:
            raise ValueError(f"Expected [B, n_cam, 3, H, W] uint8 inputs, got {prev.shape}")
        if prev.dtype != np.uint8:
            raise ValueError(f"Expected uint8 inputs, got {prev.dtype}")

        b, n_cam = prev.shape[:2]
        t1 = torch.from_numpy(prev.reshape(b * n_cam, *prev.shape[2:])).to(self._device)
        t2 = torch.from_numpy(curr.reshape(b * n_cam, *curr.shape[2:])).to(self._device)

        with self._autocast():
            out = self._model(t1, t2, iters=self._iters, test_mode=True, return_low_res=True)
        if "flow_8x" not in out:
            raise RuntimeError(
                "SEA-RAFT return_low_res=True did not return flow_8x. Run `uv run python scripts/setup_sea_raft.py`."
            )
        # Keep the public contract stable even when AMP is enabled: downstream normalization and
        # JAX transfer always receive float32 flow.
        flow_8x = out["flow_8x"].float().cpu().numpy()
        return flow_8x.reshape(b, n_cam, *flow_8x.shape[1:])
