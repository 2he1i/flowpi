"""Unit tests for independent asynchronous-channel delay semantics."""

import json
import pathlib

import numpy as np

import openpi.transforms as _transforms

_CAM = "cam_high"
_K = 2
_STRIDE = 3
_IMAGE_SIZE = (8, 8)


def _make_cache(root: pathlib.Path, length: int = 12) -> str:
    cache_dir = root / "flow_cache"
    cache_dir.mkdir()
    with open(cache_dir / "meta.json", "w") as f:
        json.dump({"num_flow_steps": _K, "flow_stride_frames": _STRIDE, "image_size": list(_IMAGE_SIZE)}, f)

    episode_dir = cache_dir / "episode-000000"
    episode_dir.mkdir()
    flow = np.zeros((length, _K, 2, 1, 1), dtype=np.float16)
    valid = np.zeros((length, _K), dtype=bool)
    for target in range(length):
        for lag in range(_K):
            flow[target, lag, :, 0, 0] = 100 * target + lag
            valid[target, lag] = target >= (lag + 1) * _STRIDE
    np.save(episode_dir / f"{_CAM}.npy", flow)
    np.save(episode_dir / "valid.npy", valid)
    return str(cache_dir)


def _cache_transform(
    cache_dir: str,
    *,
    flow_delay_max: int = 0,
    flow_delay_distribution: tuple[float, ...] | None = None,
) -> _transforms.LoadFlowCache:
    return _transforms.LoadFlowCache(
        cache_dir,
        (_CAM,),
        num_flow_steps=_K,
        flow_stride_frames=_STRIDE,
        flow_image_size=_IMAGE_SIZE,
        flow_scale=1.0,
        flow_clamp=10000.0,
        flow_delay_max=flow_delay_max,
        flow_delay_distribution=flow_delay_distribution,
    )


def test_stale_flow_cache_selects_source_tick_and_preserves_internal_lags(tmp_path):
    transform = _cache_transform(str(_make_cache(tmp_path)), flow_delay_max=3)

    stale = transform({"episode_index": 0, "frame_index": 8, "flow_delay": 2})
    np.testing.assert_array_equal(stale["flow"][_CAM][:, 0, 0, 0], [600, 601])
    np.testing.assert_array_equal(stale["flow_masks"][_CAM], [True, True])
    assert stale["flow_delay"] == 2

    synchronous = transform({"episode_index": 0, "frame_index": 8, "flow_delay": 0})
    np.testing.assert_array_equal(synchronous["flow"][_CAM][:, 0, 0, 0], [800, 801])
    np.testing.assert_array_equal(synchronous["flow_masks"][_CAM], [True, True])
    assert synchronous["flow_delay"] == 0


def test_flow_delay_sampling_is_clamped_at_episode_start(tmp_path):
    transform = _cache_transform(
        str(_make_cache(tmp_path)),
        flow_delay_max=3,
        flow_delay_distribution=(0.0, 0.0, 0.0, 1.0),
    )
    for _ in range(32):
        out = transform({"episode_index": 0, "frame_index": 1})
        assert 0 <= out["flow_delay"] <= 1
        # Both reachable source ticks precede the first valid internal lag (stride=3), so the
        # cache must return zeroed invalid flow rather than borrowing the previous episode.
        np.testing.assert_array_equal(out["flow_masks"][_CAM], [False, False])
        assert np.all(out["flow"][_CAM] == 0.0)


def test_flow_and_vlm_delays_are_independent(tmp_path):
    cache_dir = _make_cache(tmp_path)
    frame_offsets = _transforms.compute_image_frame_offsets(_K, _STRIDE, vlm_delay_max=4, flow_delay_max=3)
    current = 10
    images = {
        _CAM: np.stack([np.full((3, 2, 2), current + offset, dtype=np.uint8) for offset in frame_offsets], axis=0)
    }
    data = {"images": images, "episode_index": 0, "frame_index": current}

    flow_transform = _cache_transform(
        cache_dir,
        flow_delay_max=3,
        flow_delay_distribution=(0.0, 1.0, 0.0, 0.0),
    )
    vlm_transform = _transforms.DelaySlowImage(
        4,
        frame_offsets,
        distribution=(0.0, 0.0, 0.0, 0.0, 1.0),
    )
    out = vlm_transform(flow_transform(data))

    assert out["flow_delay"] == 1
    assert out["vlm_delay"] == 4
    np.testing.assert_array_equal(out["images"][_CAM], images[_CAM][frame_offsets.index(-4)])
    np.testing.assert_array_equal(out["flow"][_CAM][:, 0, 0, 0], [900, 901])


def test_online_flow_uses_stale_target_but_keeps_internal_stride():
    frame_offsets = _transforms.compute_image_frame_offsets(_K, _STRIDE, vlm_delay_max=0, flow_delay_max=3)
    current = 8
    stack = np.stack([np.full((3, 2, 2), current + offset, dtype=np.uint8) for offset in frame_offsets], axis=0)
    captured = {}

    class _Extractor:
        def compute(self, prev, curr):
            captured["prev"] = prev.copy()
            captured["curr"] = curr.copy()
            return np.zeros((1, _K, 2, 1, 1), dtype=np.float32)

    transform = _transforms.ComputeFlow(
        _Extractor(),
        (_CAM,),
        num_flow_steps=_K,
        flow_stride_frames=_STRIDE,
        flow_scale=1.0,
        flow_clamp=8.0,
        frame_offsets=frame_offsets,
        flow_delay_max=3,
    )
    out = transform({"images": {_CAM: stack}, "frame_index": current, "flow_delay": 2})

    # Source target is t-d=6; internal pairs are (3 -> 6) and (0 -> 6), not pairs ending at t=8.
    np.testing.assert_array_equal(captured["curr"][0, :, 0, 0, 0], [6, 6])
    np.testing.assert_array_equal(captured["prev"][0, :, 0, 0, 0], [3, 0])
    np.testing.assert_array_equal(out["flow_masks"][_CAM], [True, True])
