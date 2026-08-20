"""flowpi data pipeline tests: online flow, cache roundtrip, slow-channel delay."""

import pathlib

import numpy as np
import pytest
import torch

import openpi.models.pi0_config as _pi0_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

pytestmark = pytest.mark.slow

_TEST_DATA = pathlib.Path(__file__).resolve().parents[3] / "data" / "adjust_bottle_ep0"

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


def _make_train_config(
    mode: str,
    vlm_delay_max: int,
    flow_cache_dir: str | None = None,
    *,
    load_flow_cache: bool = True,
    sample_vlm_delay: bool = True,
):
    model = _pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        flow=_pi0_config.FlowConfig(
            num_flow_steps=2,
            flow_stride_frames=3,
            vlm_delay_max=vlm_delay_max,
            injection_layers=(1, 2),  # dummy action expert has depth 4
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
            sea_raft_allow_random_init=True,
            load_flow_cache=load_flow_cache,
            sample_vlm_delay=sample_vlm_delay,
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
    assert "vlm_delay" in sample
    assert 0 <= sample["vlm_delay"] <= 2

    # Interior frame: both lags valid.
    sample = _get_sample(train_config, 10)
    for cam, flow in sample["flow"].items():
        assert list(sample["flow_masks"][cam]) == [True, True]
        assert np.all(np.isfinite(flow))
    assert sample["flow"]["base_0_rgb"].shape == (2, 2, 60, 80)


def test_cache_roundtrip_matches_online(tmp_path):
    # Precompute the cache for the single episode using the script's worker logic.
    from scripts.precompute_flow_cache import _episode_entries
    from scripts.precompute_flow_cache import _process_episode

    entries = _episode_entries(_TEST_DATA)
    cache_dir = str(tmp_path / "flow_cache")
    task = {
        "root": str(_TEST_DATA),
        "entry": entries[0],
        "cam_keys": [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ],
        "num_flow_steps": 2,
        "flow_stride_frames": 3,
        "flow_image_size": [480, 640],
        "max_frames": None,
        "flow_cache_dir": cache_dir,
        "sea_raft_ckpt": None,
        "sea_raft_device": _DEVICE,
        "sea_raft_allow_random_init": True,
        "batch_size": 16,
        "verbose": False,
    }
    _process_episode(task)

    # Write the meta file that LoadFlowCache validates.
    import json

    with open(pathlib.Path(cache_dir) / "meta.json", "w") as f:
        json.dump({"num_flow_steps": 2, "flow_stride_frames": 3, "image_size": [480, 640]}, f)

    online_config = _make_train_config("online", vlm_delay_max=0)
    cache_config = _make_train_config("cache", vlm_delay_max=0, flow_cache_dir=cache_dir)

    for index in (10, 50):
        online_sample = _get_sample(online_config, index)
        cache_sample = _get_sample(cache_config, index)
        for cam in online_sample["flow"]:
            np.testing.assert_allclose(cache_sample["flow"][cam], online_sample["flow"][cam], atol=0.1, rtol=0.1)
            np.testing.assert_array_equal(cache_sample["flow_masks"][cam], online_sample["flow_masks"][cam])


def test_precompute_rejects_resolution_mismatch(tmp_path):
    """The precompute worker must fail fast when the dataset resolution != flow_image_size
    instead of silently writing a cache whose flow grid disagrees with the model."""
    from scripts.precompute_flow_cache import _episode_entries
    from scripts.precompute_flow_cache import _process_episode

    entries = _episode_entries(_TEST_DATA)
    cache_dir = str(tmp_path / "flow_cache")
    task = {
        "root": str(_TEST_DATA),
        "entry": entries[0],
        "cam_keys": [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ],
        "num_flow_steps": 2,
        "flow_stride_frames": 3,
        "flow_image_size": [320, 240],  # wrong: dataset is 480x640
        "max_frames": None,
        "flow_cache_dir": cache_dir,
        "sea_raft_ckpt": None,
        "sea_raft_device": _DEVICE,
        "sea_raft_allow_random_init": True,
        "batch_size": 16,
        "verbose": False,
    }
    with pytest.raises(ValueError, match="flow_image_size"):
        _process_episode(task)
    assert not pathlib.Path(cache_dir).exists() or not any(pathlib.Path(cache_dir).iterdir())


def test_delay_slow_image_selection():
    frame_offsets = _transforms.compute_image_frame_offsets(num_flow_steps=2, flow_stride_frames=3, vlm_delay_max=4)
    # offsets: {-6, -4, -3, -2, -1, 0} -> ascending
    assert frame_offsets == (-6, -4, -3, -2, -1, 0)

    transform = _transforms.DelaySlowImage(4, frame_offsets, seed=7)
    t_stack, h, w = len(frame_offsets), 4, 5
    images = {f"cam_{i}": np.arange(t_stack * h * w * 3, dtype=np.uint8).reshape(t_stack, 3, h, w) for i in range(2)}
    data = {"images": {k: v.copy() for k, v in images.items()}, "episode_index": 0, "frame_index": 13}

    # Stochastic sampling: one transform instance draws a fresh delay per call, and every drawn
    # delay selects the matching stacked frame.
    delays = set()
    for _ in range(64):
        out = transform(dict(data))
        d = out["vlm_delay"]
        assert 0 <= d <= 4
        expected_index = frame_offsets.index(-d)
        for cam in out["images"]:
            np.testing.assert_array_equal(out["images"][cam], images[cam][expected_index])
            assert out["images"][cam].shape == (3, h, w)
        delays.add(int(d))
    assert len(delays) >= 3, f"expected varied delays over 64 draws, got {sorted(delays)}"

    # Early in the episode the delay can never reach further back than the frame index.
    early_transform = _transforms.DelaySlowImage(4, frame_offsets, seed=3)
    early_data = {"images": {k: v.copy() for k, v in images.items()}, "episode_index": 0, "frame_index": 2}
    for _ in range(64):
        out = early_transform(dict(early_data))
        assert 0 <= out["vlm_delay"] <= 2

    # vlm_delay_max=0 with unstacked (single-frame) images is a no-op.
    no_delay = _transforms.DelaySlowImage(0, (0,))
    single = {"images": {"cam": np.zeros((3, h, w), dtype=np.uint8)}, "episode_index": 0, "frame_index": 5}
    out3 = no_delay(dict(single))
    assert out3["vlm_delay"] == 0
    assert out3["images"]["cam"].shape == (3, h, w)


def test_delay_slow_image_worker_aware_rng(monkeypatch):
    """DataLoader workers must sample *uncorrelated* delays: torch forks the transform into each
    worker process, so a fixed per-instance RNG stream would replay the identical delay sequence
    in every worker. The stream must be derived from the worker id instead."""
    import torch

    frame_offsets = _transforms.compute_image_frame_offsets(num_flow_steps=2, flow_stride_frames=3, vlm_delay_max=4)
    t_stack, h, w = len(frame_offsets), 4, 5
    images = {f"cam_{i}": np.arange(t_stack * h * w * 3, dtype=np.uint8).reshape(t_stack, 3, h, w) for i in range(2)}
    data = {"images": {k: v.copy() for k, v in images.items()}, "episode_index": 0, "frame_index": 13}

    class FakeWorkerInfo:
        def __init__(self, wid: int):
            self.id = wid

    def draw_sequence(worker_id: int, n: int = 32) -> list[int]:
        transform = _transforms.DelaySlowImage(4, frame_offsets, seed=7)
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: FakeWorkerInfo(worker_id))
        return [transform(dict(data))["vlm_delay"] for _ in range(n)]

    worker0 = draw_sequence(0)
    worker1 = draw_sequence(1)
    worker2 = draw_sequence(2)

    # Different workers must not replay the same delay sequence (cross-worker correlation).
    assert worker0 != worker1, "worker 0 and worker 1 drew identical delays"
    assert worker0 != worker2, "worker 0 and worker 2 drew identical delays"
    # Each worker's sequence is still stochastic and stays within the allowed range.
    for seq in (worker0, worker1, worker2):
        assert all(0 <= d <= 4 for d in seq)
        assert len(set(seq)) >= 3, f"worker sequence has no diversity: {seq}"
    # Seeded determinism is preserved: same seed + same worker id reproduces the sequence.
    assert draw_sequence(0) == worker0


def test_flow_disabled_matches_baseline():
    model = _pi0_config.Pi0Config(pi05=True, discrete_state_input=False)
    baseline = _config.LeRobotAlohaDataConfig(
        repo_id=str(_TEST_DATA),
        default_prompt="Adjust the bottle",
        repack_transforms=_ALOHA_REPACK,
    )
    train_config = _config.TrainConfig(name="test_flowpi_data", exp_name="test", model=model, data=baseline)
    sample = _get_sample(train_config, 10)
    assert "flow" not in sample
    assert "flow_masks" not in sample
    assert "vlm_delay" not in sample
    assert "episode_index" not in sample
    assert "frame_index" not in sample
    assert sample["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert sample["actions"].shape == (50, 32)


def test_ablation_data_stream_semantics(tmp_path):
    """The ablation configs must decouple the *data stream* from the *model channels*: B/C
    (load_flow_cache=False + sample_vlm_delay=False) keep the exact π0.5 single-frame stream
    (no camera history, no vlm_delay, no flow); D (delay only) samples vlm_delay without the
    flow cache; E (full) keeps both. Regression: B/C previously still applied DelaySlowImage
    and loaded the flow cache, contaminating the ablation with a delayed-image shift."""
    # Build a real flow cache so the E config's LoadFlowCache has something to load.
    from scripts.precompute_flow_cache import _episode_entries
    from scripts.precompute_flow_cache import _process_episode

    entries = _episode_entries(_TEST_DATA)
    cache_dir = str(tmp_path / "flow_cache")
    task = {
        "root": str(_TEST_DATA),
        "entry": entries[0],
        "cam_keys": [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ],
        "num_flow_steps": 2,
        "flow_stride_frames": 3,
        "flow_image_size": [480, 640],
        "max_frames": None,
        "flow_cache_dir": cache_dir,
        "sea_raft_ckpt": None,
        "sea_raft_device": _DEVICE,
        "sea_raft_allow_random_init": True,
        "batch_size": 16,
        "verbose": False,
    }
    _process_episode(task)

    # Write the meta file that LoadFlowCache validates.
    import json

    with open(pathlib.Path(cache_dir) / "meta.json", "w") as f:
        json.dump({"num_flow_steps": 2, "flow_stride_frames": 3, "image_size": [480, 640]}, f)

    # B/C data stream: π0.5-identical (fresh image, no delay, no camera history).
    bc = _make_train_config("cache", vlm_delay_max=4, load_flow_cache=False, sample_vlm_delay=False)
    bc_created = bc.data.create(bc.assets_dirs, bc.model)
    assert not any(
        isinstance(t, _transforms.DelaySlowImage | _transforms.LoadFlowCache) for t in bc_created.data_transforms.inputs
    )
    sample = _get_sample(bc, 10)
    assert "vlm_delay" not in sample
    assert "flow" not in sample
    assert "flow_masks" not in sample
    assert "episode_index" not in sample
    assert "frame_index" not in sample
    assert sample["image"]["base_0_rgb"].shape == (224, 224, 3)

    # D data stream: delay only (vlm_delay sampled from the stacked history, no flow cache).
    d = _make_train_config("cache", vlm_delay_max=4, load_flow_cache=False, sample_vlm_delay=True)
    d_created = d.data.create(d.assets_dirs, d.model)
    assert any(isinstance(t, _transforms.DelaySlowImage) for t in d_created.data_transforms.inputs)
    assert not any(isinstance(t, _transforms.LoadFlowCache) for t in d_created.data_transforms.inputs)
    sample = _get_sample(d, 10)
    assert "vlm_delay" in sample
    assert 0 <= sample["vlm_delay"] <= 4
    assert "flow" not in sample
    assert sample["image"]["base_0_rgb"].shape == (224, 224, 3)

    # E data stream: full pipeline (flow cache + delay).
    e = _make_train_config("cache", vlm_delay_max=4, flow_cache_dir=cache_dir, sample_vlm_delay=True)
    e_created = e.data.create(e.assets_dirs, e.model)
    assert any(isinstance(t, _transforms.DelaySlowImage) for t in e_created.data_transforms.inputs)
    assert any(isinstance(t, _transforms.LoadFlowCache) for t in e_created.data_transforms.inputs)
    sample = _get_sample(e, 10)
    assert "vlm_delay" in sample
    assert 0 <= sample["vlm_delay"] <= 4
    assert "flow" in sample

    # The ablation config table wires the toggles correctly (B/C off/off, D off/on, E on/on).
    from openpi.training.config import _flowpi_ablation_configs

    by_name = {c.name: c for c in _flowpi_ablation_configs()}
    for name in ("flowpi_abl_b_fresh_state", "flowpi_abl_c_pir2"):
        flow = by_name[name].data.flow
        assert flow is not None
        assert not flow.load_flow_cache
        assert not flow.sample_vlm_delay
    d_flow = by_name["flowpi_abl_d_delay"].data.flow
    assert not d_flow.load_flow_cache
    assert d_flow.sample_vlm_delay
    e_flow = by_name["flowpi_abl_e_flow"].data.flow
    assert e_flow.load_flow_cache
    assert e_flow.sample_vlm_delay


def test_load_flow_cache_lru_eviction(tmp_path):
    """The open per-episode memmaps must be bounded by the LRU cache."""
    import json

    cache_dir = tmp_path / "flow_cache"
    cache_dir.mkdir()
    with open(cache_dir / "meta.json", "w") as f:
        json.dump({"num_flow_steps": 2, "flow_stride_frames": 3, "image_size": [480, 640]}, f)
    cams = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    for ep in range(3):
        ep_dir = cache_dir / f"episode-{ep:06d}"
        ep_dir.mkdir()
        for cam in cams:
            np.save(ep_dir / f"{cam}.npy", np.zeros((5, 2, 2, 60, 80), dtype=np.float16))
        np.save(ep_dir / "valid.npy", np.ones((5, 2), dtype=bool))

    transform = _transforms.LoadFlowCache(
        str(cache_dir),
        cams,
        num_flow_steps=2,
        flow_stride_frames=3,
        flow_image_size=(480, 640),
        flow_scale=20.0,
        flow_clamp=8.0,
        max_cached_episodes=2,
    )

    for ep in range(3):
        out = transform({"episode_index": ep, "frame_index": 0, "images": {}})
        assert "flow" in out
        assert len(transform._mmaps) <= 2  # noqa: SLF001
    # Episode 0 was evicted (LRU order: 1, 2).
    assert list(transform._mmaps.keys()) == [1, 2]  # noqa: SLF001

    # Re-accessing an evicted episode works and refreshes the recency order.
    out = transform({"episode_index": 0, "frame_index": 0, "images": {}})
    assert out["flow"]["cam_high"].shape == (2, 2, 60, 80)
    assert list(transform._mmaps.keys()) == [2, 0]  # noqa: SLF001
