"""Regression tests for the FlowPi data and cache-precompute paths."""

import io
import json
import pathlib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

import openpi.models.pi0_config as _pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.transforms import compute_image_frame_offsets
from scripts.precompute_flow_cache import _create_precompute_data_config

_H, _W = 8, 8
_CAM = "observation.images.cam_a"


def _frame_bytes(value: int) -> bytes:
    image = np.full((_H, _W, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _build_v3_dataset(root: pathlib.Path, length: int = 12) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir()
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "fps": 50,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [1]},
            "action": {"dtype": "float32", "shape": [1]},
            _CAM: {"dtype": "image", "shape": [3, _H, _W]},
            "episode_index": {"dtype": "int64"},
            "index": {"dtype": "int64"},
            "task_index": {"dtype": "int64"},
        },
    }
    with open(root / "meta" / "info.json", "w") as f:
        json.dump(info, f)

    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    schema = pa.schema(
        [
            ("observation.state", pa.list_(pa.float32())),
            ("action", pa.list_(pa.float32())),
            (_CAM, image_type),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
        ]
    )
    table = pa.table(
        {
            "observation.state": [[float(i)] for i in range(length)],
            "action": [[float(i)] for i in range(length)],
            _CAM: pa.array(
                [{"bytes": _frame_bytes(i + 1), "path": f"frame-{i:06d}.png"} for i in range(length)],
                type=image_type,
            ),
            "episode_index": [0] * length,
            "index": list(range(length)),
            "task_index": [0] * length,
        },
        schema=schema,
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "length": [length],
                "data/chunk_index": [0],
                "data/file_index": [0],
                "dataset_from_index": [0],
                "dataset_to_index": [length],
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )


def test_create_torch_dataset_loads_past_camera_frames(tmp_path):
    """The production factory and v3 loader must map a negative offset to t + offset."""
    _build_v3_dataset(tmp_path)
    flow_model = _pi0_config.FlowConfig(
        num_flow_steps=2,
        flow_stride_frames=3,
        d_max=2,
        injection_layers=(1, 2),
        vlm_delay_max=4,
    )
    data_config = _config.DataConfig(
        repo_id=str(tmp_path),
        action_sequence_keys=(),
        flow=_config.FlowDataConfig(
            mode="cache",
            flow_cache_dir=str(tmp_path / "missing-cache"),
            load_flow_cache=True,
            sample_vlm_delay=True,
        ),
    )

    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=4,
        model_config=SimpleNamespace(flow=flow_model),
    )
    frame_offsets = compute_image_frame_offsets(2, 3, 4)
    np.testing.assert_allclose(dataset.delta_timestamps[_CAM], np.asarray(frame_offsets) / 50.0)

    # At t=8 every requested frame is in bounds. The image pixel value is global frame + 1,
    # so this directly checks the actual q = idx + delta loader behavior.
    sample = dataset[8]
    observed = [int(np.unique(frame)[0]) - 1 for frame in sample[_CAM]]
    assert observed == [8 + offset for offset in frame_offsets]
    assert observed[frame_offsets.index(-3)] == 5
    assert observed[frame_offsets.index(0)] == 8


def test_precompute_config_does_not_require_existing_cache(tmp_path):
    """Cache generation must disable cache-consuming transforms only while bootstrapping."""
    flow_model = _pi0_config.FlowConfig(
        num_flow_steps=2,
        flow_stride_frames=3,
        d_max=2,
        injection_layers=(1, 2),
        vlm_delay_max=3,
    )
    flow_data = _config.FlowDataConfig(
        mode="cache",
        flow_cache_dir=str(tmp_path / "cache-does-not-exist"),
        load_flow_cache=True,
        sample_vlm_delay=True,
    )
    train_config = _config.TrainConfig(
        name="flowpi-review",
        model=_pi0_config.Pi0Config(
            pi05=True,
            discrete_state_input=False,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_dim=32,
            action_horizon=12,
            flow=flow_model,
        ),
        data=_config.LeRobotAlohaDataConfig(repo_id="bootstrap-test", flow=flow_data),
        assets_base_dir=str(tmp_path),
        wandb_enabled=False,
    )

    created = _create_precompute_data_config(train_config)

    assert created.flow is not None
    assert created.flow.load_flow_cache is False
    assert created.flow.sample_vlm_delay is False
    assert flow_data.load_flow_cache is True
    assert flow_data.sample_vlm_delay is True
