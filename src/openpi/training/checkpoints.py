from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import datetime
import json
import logging
import pathlib
from typing import TYPE_CHECKING, Any, Protocol

from etils import epath
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils

if TYPE_CHECKING:
    import openpi.training.config as _config


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "config": ocp.JsonCheckpointHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    config: _config.TrainConfig,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "config": _training_config_metadata(config, step),
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def _training_config_metadata(config: _config.TrainConfig, step: int) -> dict[str, Any]:
    """Return a JSON-compatible snapshot of the resolved training recipe."""
    config_dict = json.loads(json.dumps(dataclasses.asdict(config), default=str))
    return {
        "schema_version": 1,
        "saved_step": step,
        "saved_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "training": {
            "batch_size": config.batch_size,
            "num_train_steps": config.num_train_steps,
            "num_workers": config.num_workers,
            "seed": config.seed,
            "ema_decay": config.ema_decay,
            "fsdp_devices": config.fsdp_devices,
            "log_interval": config.log_interval,
            "save_interval": config.save_interval,
            "keep_period": config.keep_period,
            "resume_step": config.resume_step,
        },
        "lr_schedule_type": type(config.lr_schedule).__name__,
        "lr_schedule": config_dict["lr_schedule"],
        "optimizer_type": type(config.optimizer).__name__,
        "optimizer": config_dict["optimizer"],
        "config": config_dict,
    }


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    """Normalize a checkpoint path to a restoreable params directory.

    Accepts (in order of preference):
    - a released checkpoint (e.g. ``gs://.../pi05_base/params`` or ``.../params``),
    - an orbax training step directory (``<dir>/step_000123`` or a numeric ``<dir>/123``),
    - a training checkpoint root with a ``latest`` symlink (``<dir>/latest -> step_...``),
    - a step directory's ``params`` item (``<dir>/step_000123/params``),
    - an orbax CheckpointManager root with numeric step dirs (``<dir>/<step>/params``).
    """
    if str(checkpoint_path).startswith("gs://"):
        return str(checkpoint_path)
    path = pathlib.Path(checkpoint_path)
    candidates = [path]
    latest = path / "latest"
    if latest.is_symlink():
        candidates.append(latest.resolve())
    candidates.append(path / "params")
    for candidate in candidates:
        if (candidate / "metadata.json").exists():
            return str(candidate)
    if path.is_dir():
        # The path itself is an orbax step dir (<step>/params + train_state + assets): resolve
        # its params item.
        if (path / "_CHECKPOINT_METADATA").exists() and (path / "params").is_dir():
            return str(path / "params")
        # Orbax CheckpointManager root: <root>/<step>/params (+ train_state, assets,
        # _CHECKPOINT_METADATA). Pick the largest numeric step with a restorable params item so
        # a checkpoint root resolves without hand-writing the step number.
        steps = sorted(
            (d for d in path.iterdir() if d.name.isdigit() and (d / "params").is_dir()),
            key=lambda d: int(d.name),
            reverse=True,
        )
        if steps:
            return str(steps[0] / "params")
    # No metadata found; let `restore_params` raise the precise error.
    return str(path)


def resolve_checkpoint_assets_dir(checkpoint_path: str | pathlib.Path) -> pathlib.Path | None:
    """Find the local assets item belonging to a checkpoint path.

    Training checkpoints are addressed in several ways by the inference tools: as a
    checkpoint root, a numeric step directory, a latest symlink, or directly as
    the params item. The assets live next to params in all of those layouts.
    Return None for remote checkpoints or checkpoints without a local assets item
    so callers can fall back to their configured assets source.
    """
    if str(checkpoint_path).startswith("gs://"):
        return None

    path = pathlib.Path(checkpoint_path).expanduser()
    candidates: list[pathlib.Path] = []
    if path.name == "params":
        candidates.append(path.parent)
    if (path / "params").is_dir():
        candidates.append(path)
    if (path / "assets").is_dir():
        candidates.append(path)

    latest = path / "latest"
    if latest.is_symlink():
        candidates.append(latest.resolve())

    if path.is_dir():
        candidates.extend(
            sorted(
                (directory for directory in path.iterdir() if directory.name.isdigit()),
                key=lambda directory: int(directory.name),
                reverse=True,
            )
        )

    for candidate in candidates:
        assets_dir = candidate / "assets"
        if assets_dir.is_dir():
            return assets_dir
    return None


def load_model_from_checkpoint(
    model_config: _model.BaseModelConfig,
    checkpoint_path: str,
    *,
    dtype: jnp.dtype | None = None,
) -> _model.BaseModel:
    """Create a model with parameters restored from an openpi checkpoint.

    This is the single entry point for replay / serving / eval. It restores the params and
    verifies (via the model-config equality check) that the checkpoint contains exactly the
    model's parameters — a checkpoint without the flowpi weights (e.g. a plain π0.5 checkpoint)
    fails loudly instead of silently running with random weights.

    Args:
        model_config: The model config (e.g. ``train_config.model``).
        checkpoint_path: Path to a released checkpoint, an orbax step directory, or a training
            checkpoint root with a ``latest`` symlink.
        dtype: Optional dtype override for the restored params.

    Returns:
        The model with restored parameters.
    """
    params = _model.restore_params(_resolve_checkpoint_path(checkpoint_path), dtype=dtype)
    return model_config.load(params)


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
