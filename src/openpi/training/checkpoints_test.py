"""Checkpoint path resolution tests (orbax CheckpointManager numeric step dirs)."""

import pathlib

from openpi.training import checkpoints as _checkpoints

_resolve = _checkpoints._resolve_checkpoint_path  # noqa: SLF001


def _write_step(root: pathlib.Path, step: int) -> None:
    """Create an orbax-style step dir: <root>/<step>/{_CHECKPOINT_METADATA, params, train_state}."""
    step_dir = root / str(step)
    step_dir.mkdir(parents=True)
    (step_dir / "_CHECKPOINT_METADATA").write_text("{}")
    (step_dir / "params").mkdir()
    (step_dir / "train_state").mkdir()


def test_resolve_checkpoint_root_picks_largest_numeric_step(tmp_path):
    """A CheckpointManager root must resolve to the largest numeric step's params item
    (regression: previously the user had to hand-write the full `<root>/<step>/params` path)."""
    _write_step(tmp_path, 3)
    _write_step(tmp_path, 9)
    _write_step(tmp_path, 7)
    assert _resolve(str(tmp_path)) == str(tmp_path / "9" / "params")


def test_resolve_checkpoint_root_single_numeric_step(tmp_path):
    _write_step(tmp_path, 5)
    assert _resolve(str(tmp_path)) == str(tmp_path / "5" / "params")


def test_resolve_numeric_step_dir_directly(tmp_path):
    """Passing a numeric step dir itself must resolve to its params item."""
    _write_step(tmp_path, 9)
    assert _resolve(str(tmp_path / "9")) == str(tmp_path / "9" / "params")


def test_resolve_params_item_passthrough(tmp_path):
    """The full `<root>/<step>/params` path (release-checkpoint style) is passed through."""
    _write_step(tmp_path, 9)
    params = tmp_path / "9" / "params"
    assert _resolve(str(params)) == str(params)


def test_resolve_metadata_json_released_checkpoint(tmp_path):
    """Released checkpoints (metadata.json inside) keep resolving as before."""
    params = tmp_path / "params"
    params.mkdir(parents=True)
    (params / "metadata.json").write_text("{}")
    assert _resolve(str(tmp_path)) == str(params)


def test_resolve_latest_symlink(tmp_path):
    """A training root with a `latest` symlink still resolves (existing behavior)."""
    _write_step(tmp_path, 4)
    target = tmp_path / "4"
    latest = tmp_path / "latest"
    latest.symlink_to(target.name, target_is_directory=True)
    (target / "metadata.json").write_text("{}")
    assert _resolve(str(tmp_path)) == str(target)
