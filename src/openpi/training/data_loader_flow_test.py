"""flowpi data pipeline tests: online flow, cache roundtrip, slow-channel delay."""

import dataclasses
import pathlib

import numpy as np
import pytest
import torch

import openpi.models.pi0_config as _pi0_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

pytestmark = pytest.mark.slow

_TEST_DATA = pathlib.Path(__file__).resolve().parents[3] / "test_data" / "adjust_bottle_ep0"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_ALOHA_REPACK = _transforms.Group(
    inputs=[
        _transforms.RepackTransform(
            {
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
            }
        )
    ]
)


def _make_train_config(mode: str, vlm_delay_max: int, flow_cache_dir: str | None = None):
    model = _pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        flow=_pi0_config.FlowConfig(
            num_flow_steps=2,
            flow_stride_frames=3,
            vlm_delay_max=vlm_delay_max,
        ),
    )
    data = _config.LeRobotAlohaDataConfig(
        repo_id=str(_TEST_DATA),
        default_prompt="Adjust the bottle on the table",
        repack_transforms=_ALOHA_REPACK,
        flow=_config.FlowDataConfig(
            mode=mode,
            flow_cache_dir=flow_cache_dir,
            sea_raft_ckpt=None,
            sea_raft_device=_DEVICE,
        ),
    )
    return _config.TrainConfig(
        name="test_flowpi_data",
        exp_name="test",
        model=model,
        data=data,
        batch_size=1,
    )


def _get_sample(train_config: _config.TrainConfig, index: int) -> dict:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    transformed = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=True)
    return transformed[index]


def test_online_flow_sample():
    train_config = _make_train_config("online", vlm_delay_max=2)

    # Episode start: both lags invalid (t=1 < 3 and < 6).
    sample = _get_sample(train_config, 1)
    assert set(sample["flow"].keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    for cam, flow in sample["flow"].items():
        assert flow.shape == (2, 2, 60, 80), cam
        assert flow.dtype == np.float32
        assert np.all(flow == 0.0)  # invalid lags are zeroed
        assert list(sample["flow_masks"][cam]) == [False, False]
    assert sample["image"]["base_0_rgb"].shape == (224, 224, 3)  # single frame, resized by model transforms
    assert sample["actions"].shape == (50, 32)  # padded to action_dim by model transforms
    assert "vlm_delay" in sample and 0 <= sample["vlm_delay"] <= 2

    # Interior frame: both lags valid.
    sample = _get_sample(train_config, 10)
    for cam, flow in sample["flow"].items():
        assert list(sample["flow_masks"][cam]) == [True, True]
        assert np.all(np.isfinite(flow))
    assert sample["flow"]["base_0_rgb"].shape == (2, 2, 60, 80)


def test_cache_roundtrip_matches_online(tmp_path):
    # Precompute the cache for the single episode using the script's worker logic.
    from scripts.precompute_flow_cache import _episode_entries, _process_episode  # noqa: PLC0415

    entries = _episode_entries(_TEST_DATA)
    cache_dir = str(tmp_path / "flow_cache")
    task = {
        "root": str(_TEST_DATA),
        "entry": entries[0],
        "cam_keys": ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"],
        "num_flow_steps": 2,
        "flow_stride_frames": 3,
        "flow_cache_dir": cache_dir,
        "sea_raft_ckpt": None,
        "sea_raft_device": _DEVICE,
        "batch_size": 16,
        "verbose": False,
    }
    _process_episode(task)

    # Write the meta file that LoadFlowCache validates.
    import json  # noqa: PLC0415

    with open(pathlib.Path(cache_dir) / "meta.json", "w") as f:
        json.dump({"num_flow_steps": 2, "flow_stride_frames": 3, "image_size": [480, 640]}, f)

    online_config = _make_train_config("online", vlm_delay_max=0)
    cache_config = _make_train_config("cache", vlm_delay_max=0, flow_cache_dir=cache_dir)

    for index in (10, 50):
        online_sample = _get_sample(online_config, index)
        cache_sample = _get_sample(cache_config, index)
        for cam in online_sample["flow"]:
            np.testing.assert_allclose(
                cache_sample["flow"][cam], online_sample["flow"][cam], atol=0.1, rtol=0.1
            )
            np.testing.assert_array_equal(cache_sample["flow_masks"][cam], online_sample["flow_masks"][cam])


def test_delay_slow_image_selection():
    frame_offsets = _transforms.compute_image_frame_offsets(num_flow_steps=2, flow_stride_frames=3, vlm_delay_max=4)
    # offsets: {-6, -4, -3, -2, -1, 0} -> ascending
    assert frame_offsets == (-6, -4, -3, -2, -1, 0)

    transform = _transforms.DelaySlowImage(4, frame_offsets, seed=7)
    t_stack, h, w = len(frame_offsets), 4, 5
    images = {f"cam_{i}": np.arange(t_stack * h * w * 3, dtype=np.uint8).reshape(t_stack, 3, h, w) for i in range(2)}
    data = {"images": {k: v.copy() for k, v in images.items()}, "episode_index": 0, "frame_index": 13}

    out = transform(dict(data))
    d = out["vlm_delay"]
    assert 0 <= d <= 4
    expected_index = frame_offsets.index(-d)
    for cam in out["images"]:
        np.testing.assert_array_equal(out["images"][cam], images[cam][expected_index])
        assert out["images"][cam].shape == (3, h, w)

    # Deterministic: same sample -> same selection.
    out2 = transform(dict(data))
    assert out2["vlm_delay"] == d
    for cam in out2["images"]:
        np.testing.assert_array_equal(out2["images"][cam], out["images"][cam])

    # vlm_delay_max=0 with unstacked (single-frame) images is a no-op.
    no_delay = _transforms.DelaySlowImage(0, (0,))
    single = {"images": {"cam": np.zeros((3, h, w), dtype=np.uint8)}, "episode_index": 0, "frame_index": 5}
    out3 = no_delay(dict(single))
    assert out3["vlm_delay"] == 0
    assert out3["images"]["cam"].shape == (3, h, w)


def test_flow_disabled_matches_baseline():
    model = _pi0_config.Pi0Config(pi05=True, discrete_state_input=False)
    baseline = _config.LeRobotAlohaDataConfig(
        repo_id=str(_TEST_DATA),
        default_prompt="Adjust the bottle",
        repack_transforms=_ALOHA_REPACK,
    )
    train_config = _config.TrainConfig(name="test_flowpi_data", exp_name="test", model=model, data=baseline)
    sample = _get_sample(train_config, 10)
    assert "flow" not in sample and "flow_masks" not in sample and "vlm_delay" not in sample
    assert "episode_index" not in sample and "frame_index" not in sample
    assert sample["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert sample["actions"].shape == (50, 32)
