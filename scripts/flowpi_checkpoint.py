"""Checkpoint-specific configuration compatibility helpers for FlowPi inference."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)


def _replace_known_fields(instance: Any, saved: dict[str, Any]) -> Any:
    """Replace only fields that still exist in the current dataclass.

    Training metadata is a JSON snapshot, so tuple-valued dataclass fields arrive as
    lists.  Convert those back to tuples before constructing the current config.
    """
    field_names = {field.name for field in dataclasses.fields(instance)}
    overrides = {}
    for name in field_names.intersection(saved):
        value = saved[name]
        current = getattr(instance, name)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        overrides[name] = value
    return dataclasses.replace(instance, **overrides) if overrides else instance


def _checkpoint_step_dirs(checkpoint: str | pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Return local checkpoint step directories in restore preference order.

    The restore helper accepts an exact step directory, a ``params`` item, a root with a
    ``latest`` symlink, or a numeric-step CheckpointManager root. Configuration metadata lives
    next to ``params`` at the selected step, so config alignment must resolve the same step
    before reading ``config/metadata``.
    """
    if str(checkpoint).startswith(("gs://", "s3://", "obs://")):
        return ()

    path = pathlib.Path(checkpoint).expanduser()
    candidates: list[pathlib.Path] = []

    def add(candidate: pathlib.Path) -> None:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate not in candidates:
            candidates.append(candidate)

    if path.name == "params":
        add(path.parent)
    add(path)

    latest = path / "latest"
    if latest.is_symlink():
        add(latest.resolve())

    if path.is_dir():
        numeric_steps = sorted(
            (child for child in path.iterdir() if child.name.isdigit() and child.is_dir()),
            key=lambda child: int(child.name),
            reverse=True,
        )
        for step in numeric_steps:
            add(step)

    return tuple(candidates)


def _checkpoint_metadata_path(checkpoint: str | pathlib.Path) -> pathlib.Path | None:
    """Locate the selected checkpoint step's resolved training metadata."""
    for step_dir in _checkpoint_step_dirs(checkpoint):
        metadata_path = step_dir / "config" / "metadata"
        if metadata_path.is_file():
            return metadata_path
    return None


def align_train_config_with_checkpoint(train_config: Any, checkpoint: str | pathlib.Path) -> Any:
    """Apply the checkpoint's saved FlowPi architecture/data settings to inference config.

    FlowPi's Orbax checkpoint contains the resolved training recipe.  This matters when a
    default changes after training (for example ``flow_delay_max`` changing from 3 to 2):
    constructing the current default model would then fail shape validation even though the
    checkpoint is complete.  Missing metadata is normal for released checkpoints, so those
    checkpoints continue to use the explicitly selected config unchanged.
    """
    metadata_path = _checkpoint_metadata_path(checkpoint)
    if metadata_path is None:
        return train_config

    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read checkpoint config metadata from %s: %s", metadata_path, exc)
        return train_config

    saved_config = metadata.get("config")
    if not isinstance(saved_config, dict):
        return train_config

    model_config = train_config.model
    saved_model = saved_config.get("model")
    saved_flow = saved_model.get("flow") if isinstance(saved_model, dict) else None
    current_flow = getattr(model_config, "flow", None)
    if isinstance(saved_flow, dict) and current_flow is not None:
        model_config = dataclasses.replace(
            model_config,
            flow=_replace_known_fields(current_flow, saved_flow),
        )

    data_config = train_config.data
    saved_data = saved_config.get("data")
    saved_data_flow = saved_data.get("flow") if isinstance(saved_data, dict) else None
    current_data_flow = getattr(data_config, "flow", None)
    if isinstance(saved_data_flow, dict) and current_data_flow is not None:
        data_config = dataclasses.replace(
            data_config,
            flow=_replace_known_fields(current_data_flow, saved_data_flow),
        )

    return dataclasses.replace(train_config, model=model_config, data=data_config)
