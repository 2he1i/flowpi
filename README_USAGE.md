# FlowPi Usage Guide

> FlowPi = π0.5 + πR² streaming + Fresh State + Optical Flow fast channel

## 1. Environment Setup

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all dependencies
uv sync --group dev

# Verify JAX GPU access
uv run python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0), CudaDevice(id=1)]  (2×4090)
```

All dependencies are installed in `.venv/` (JAX GPU, PyTorch cu126, Flax, lerobot).

## 2. Dataset

FlowPi uses LeRobot **v3.0** format (parquet with pixel-embedded images).
A sample episode is at `test_data/adjust_bottle_ep0`.

For full training, replace `<dataset_path>` with your actual dataset path.

## 3. Precompute Optical Flow Cache (Required Before Training)

SEA-RAFT is frozen during training; optical flow is precomputed as offline cache.

```bash
# Smoke test (random SEA-RAFT weights, 20 frames)
uv run python scripts/precompute_flow_cache.py \
  --config-name flowpi_aloha \
  --data.flow.sea-raft-ckpt "" \
  --max-frames 20

# Full precomputation (REPLACE paths below)
uv run python scripts/precompute_flow_cache.py \
  --config-name flowpi_aloha \
  --data.flow.sea-raft-ckpt /path/to/sea_raft_weights.pth \
  --data.flow.flow_cache_dir /path/to/flow_cache \
  --data.repo_id /path/to/dataset \
  --num-workers 8
```

Cache structure per episode:
```
{flow_cache_dir}/
  episode-{ep:06d}/
    base_0_rgb.npy        # [T, K, 2, 60, 80] float16 raw flow
    left_wrist_0_rgb.npy
    right_wrist_0_rgb.npy
    valid.npy             # [T, K] bool (per-lag validity mask)
  meta.json               # K, delta, resolution, ckpt hash for validation
```

**IMPORTANT**: Changing `K` (num_flow_steps) or `Δ` (flow_stride_frames) requires recomputing the cache.

## 4. Compute Normalization Stats

```bash
uv run python scripts/compute_norm_stats.py \
  --config-name flowpi_aloha \
  --data.flow.enabled false
```

Set `--data.flow.enabled false` to skip flow/delay transforms during norm stats computation.

## 5. Training

### Smoke Test (debug config, 10 steps)
```bash
uv run python scripts/train.py debug_flowpi --exp_name smoke
```

### Full Training
```bash
uv run python scripts/train.py flowpi_aloha \
  --exp_name my_run \
  --data.repo_id /path/to/dataset \
  --data.flow.flow_cache_dir /path/to/flow_cache \
  --data.flow.sea-raft-ckpt /path/to/sea_raft_weights.pth
```

Key hyperparameters (configure via CLI overrides):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.flow.d_max` | 5 | Max in-flight actions per tick |
| `model.flow.p_standard` | 0.2 | Fraction of standard FM samples |
| `model.flow.vlm_delay_max` | 10 | Max slow-channel delay (ticks) |
| `model.flow.num_flow_steps` | 2 | K: number of lagged flow steps |
| `model.flow.flow_stride_frames` | 3 | Δ: frames between flow pairs |
| `optimizer.peak_lr` | 2.5e-5 | Learning rate |

### Optional: Flow branch independent LR
```bash
uv run python scripts/train.py flowpi_aloha --optimizer-flow-lr 1e-4 ...
```

## 6. Inference / Offline Replay

```bash
uv run python scripts/flowpi_infer.py \
  --config-name flowpi_aloha \
  --checkpoint /path/to/checkpoint \
  --dataset test_data/adjust_bottle_ep0 \
  --slow-every-n 10 \
  --max-frames 100
```

Output: `{checkpoint}/replay_actions.npz` with timing stats (RAFT/prefill/NFE).

## 7. Running Tests

```bash
# Fast tests (non-slow)
uv run python -m pytest src scripts -q -m "not slow"

# Model core tests (GPU, ~6 min)
uv run python -m pytest src/openpi/models/flowpi_test.py -q

# Runtime test (GPU, ~7 min)
uv run python -m pytest src/openpi/policies/flowpi_runtime_test.py -q

# Data pipeline tests (GPU, ~1.5 min)
uv run python -m pytest src/openpi/training/data_loader_flow_test.py -q

# SEA-RAFT tests (GPU, ~25s)
uv run python -m pytest src/openpi/training/sea_raft_test.py -q
```

## 8. Key Design Notes

### discrete_state_input=False
FlowPi sets `discrete_state_input=False` (state goes through the fast channel as a suffix token), whereas `pi05_base` was trained with `discrete_state_input=True` (state discretized into prompt tokens). This is a fine-tuning adaptation difference — the `pi05_libero` config established this precedent.

### V1 Fixed at 50Hz
Control frequency is fixed at 50Hz (matching the aloha dataset FPS). Changing K/Δ without adjusting
fps would break the flow=motion×Δt assumption. Future versions may support variable frequencies.

### PyTorch Path
The `models_pytorch` training path does **not** support FlowPi (flow cross-attention, flow tokenizer, πR² streaming). Use the JAX path.

## 9. Troubleshooting

| Symptom | Solution |
|---------|----------|
| `flow cache meta.json mismatch` | Re-run `precompute_flow_cache.py` |
| `CUDA out of memory` | Reduce `batch_size` or use `--fsdp-devices N` |
| NaN loss on step 0 | Expected with debug_flowpi (dummy + random weights); fine with real data + pretrained weights |
| `SEA-RAFT ckpt not found` | Pass `--data.flow.sea-raft-ckpt ""` for random weights (testing only) |