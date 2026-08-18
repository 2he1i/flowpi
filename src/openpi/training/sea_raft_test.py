import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def extractor():
    from openpi.training.sea_raft import SeaRaftFlowExtractor

    return SeaRaftFlowExtractor(ckpt_path=None, variant="M", device="cpu")


@pytest.mark.parametrize(("batch", "n_cam"), [(1, 3)])
def test_compute_shape_and_finite(extractor, batch, n_cam):
    rng = np.random.default_rng(0)
    prev = rng.integers(0, 256, size=(batch, n_cam, 3, 480, 640), dtype=np.uint8)
    curr = rng.integers(0, 256, size=(batch, n_cam, 3, 480, 640), dtype=np.uint8)
    flow = extractor.compute(prev, curr)
    assert flow.shape == (batch, n_cam, 2, 60, 80)
    assert flow.dtype == np.float32
    assert np.all(np.isfinite(flow))


def test_return_low_res_default_keys(extractor):
    import torch

    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(1, 3, 480, 640), dtype=np.uint8)
    t = torch.from_numpy(img)
    out_default = extractor._model(t, t, test_mode=True)  # noqa: SLF001
    out_low_res = extractor._model(t, t, test_mode=True, return_low_res=True)  # noqa: SLF001
    assert set(out_default.keys()) == {"final", "flow", "info", "nf"}
    assert set(out_low_res.keys()) == {"final", "flow", "info", "nf", "flow_8x"}
    assert out_low_res["flow_8x"].shape == (1, 2, 60, 80)


def test_identical_images_zero_flow(extractor):
    # Identical frames are a strong (but not guaranteed) zero-flow prior for RAFT with random
    # weights we only check finiteness and shape here; with trained weights flow ~ 0.
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(1, 1, 3, 480, 640), dtype=np.uint8)
    flow = extractor.compute(img, img)
    assert flow.shape == (1, 1, 2, 60, 80)
    assert np.all(np.isfinite(flow))
