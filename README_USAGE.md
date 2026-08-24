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
# Training/replay may use the GPUs selected by the command. The DOMINO policy server uses
# CUDA_VISIBLE_DEVICES=0,1,2 and expects three CudaDevice entries.
```

All dependencies are installed in `.venv/` (JAX GPU, PyTorch cu126, Flax, lerobot).

FlowPi pins the SEA-RAFT source at commit
`9137517ba24e628442aec097d3afe71d03503b75` and applies the checked-in
`return_low_res`/`flow_8x` API patch. Initialize and bootstrap it after cloning:

```bash
git submodule update --init SEA-RAFT
uv run python scripts/setup_sea_raft.py
```

The bootstrap script verifies the exact submodule commit, applies the patch once, and fails loudly
if the dependency is missing, changed, or lacks the required low-resolution output. Training uses
the resulting frozen implementation through the offline flow cache; inference uses the same code
online.

## 2. Dataset

FlowPi uses LeRobot **v3.0** format (parquet with pixel-embedded images).
The repository-local `flowpi_data` entry points to `../../../data/flowpi_data` on the data disk,
with separate locations for the training dataset and optical-flow cache:

```
flowpi_data/train_dataset/
flowpi_data/flow_cache/
```

The existing local sample episode remains at `data/adjust_bottle_ep0`.

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
  --data.flow.flow_cache_dir flowpi_data/flow_cache \
  --data.repo_id flowpi_data/train_dataset \
  --num-workers 8
```

`--max-frames` is a global smoke-run limit across the dataset, not a per-episode limit.

Cache structure per episode:
```
{flow_cache_dir}/
  episode-{ep:06d}/
    base_0_rgb.npy        # [T, K, 2, 60, 80] float16 raw flow
    left_wrist_0_rgb.npy
    right_wrist_0_rgb.npy
    valid.npy             # [T, K] bool (per-lag validity mask)
  meta.json               # K, delta, resolution, SEA-RAFT provenance for validation
```

**IMPORTANT**: Changing `K` (num_flow_steps) or `Δ` (flow_stride_frames) requires recomputing the cache.
Changing the SEA-RAFT checkpoint, variant, refinement iterations, or camera set also requires recomputing it.

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
  --data.repo_id flowpi_data/train_dataset \
  --data.flow.flow_cache_dir flowpi_data/flow_cache \
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

All trainable parameters currently use the same optimizer and learning rate. The old
`--optimizer-flow-lr` example is not supported by the current training code.

## 6. Inference / Offline Replay

```bash
uv run python scripts/flowpi_infer.py \
  --config-name flowpi_aloha \
  --checkpoint /path/to/checkpoint \
  --dataset flowpi_data/train_dataset \
  --slow-every-n 10 \
  --max-frames 100
```

Inference uses the raw SEA-RAFT model checkpoint at
`SEA-RAFT/ckpt/shadow-24k.pth` by default. This is intentionally different from the
training/cache checkpoint passed with `--data.flow.sea-raft-ckpt`, which may contain optimizer
and scheduler state for resuming SEA-RAFT. Override the inference file with
`--sea-raft-ckpt /path/to/raw_sea_raft_weights.pth` when needed.

Replay accepts one or more contiguous LeRobot v3 episodes and automatically resets the
streaming ring/prefix/action state at each `episode_index` boundary. Output is
`{checkpoint}/replay_actions.npz` with timing stats (RAFT/prefill/NFE).

## 7. RoboTwin / DOMINO Simulation Inference (Four GPUs)

The adapter is for RoboTwin-based simulation evaluation, not the real ALOHA control loop. For the
DOMINO benchmark, use the double-process launcher below: the DOMINO simulation/client process is
kept separate from the JAX policy server so DOMINO's Python package set never imports OpenPI/JAX.
The policy server sends one 14-D qpos command per environment control step.

The default device layout uses three visible GPUs:

| Role | Device |
|---|---|
| slow VLM prefix replica | JAX `gpu:0` |
| fast action/NFE replica | JAX `gpu:1` |
| online SEA-RAFT | Torch `cuda:2` |

The policy server uses the first three physical GPUs and the DOMINO client uses the fourth. The
model checkpoint is the completed FlowPi checkpoint; SEA-RAFT is loaded independently from the raw
inference checkpoint at `SEA-RAFT/ckpt/shadow-24k.pth`.

```bash
scripts/run_flowpi_domino.sh \
  <task_name> <task_config> /path/to/flowpi_checkpoint [seed]
```

`DOMINO` is expected at `../DOMINO` beside this repository. For example:

```bash
scripts/run_flowpi_domino.sh adjust_bottle demo_clean_dynamic \
  /path/to/flowpi_checkpoint 0
```

The task/config names are the ones defined by the local DOMINO checkout. Override runtime settings
after the four positional arguments when needed, for example:

```bash
scripts/run_flowpi_domino.sh <task_name> <task_config> /path/to/flowpi_checkpoint 0 \
  --test_num 1 \
  --flowpi_slow_every_n 10 \
  --flowpi_sea_raft_ckpt /path/to/raw_sea_raft_weights.pth
```

`run_flowpi_domino.sh` starts `DOMINO/script/policy_model_server.py` with
`CUDA_VISIBLE_DEVICES=0,1,2` and starts `DOMINO/script/eval_policy_client.py` with
`CUDA_VISIBLE_DEVICES=3`. Set `FLOWPI_POLICY_GPUS`, `FLOWPI_DOMINO_GPU`, `POLICY_PYTHON`,
`DOMINO_PYTHON`, or `DOMINO_ROOT` to override these defaults. Both processes may use the same
Python installation; the process boundary is what keeps the DOMINO client dependency-light.

Each DOMINO run records metrics by default under
`data/flowpi_metrics/<timestamp>/`: `policy_runtime.json` contains policy request latency,
fast-tick latency/frequency, SEA-RAFT flow latency/update frequency, slow-prefix prefill and
installation frequency, prefix age, queue coalescing, and generation drops. `domino_client.json`
contains RPC round-trip latency, action execution time, and the client-side loop frequency.
Override the location with `FLOWPI_METRICS_DIR`, or provide explicit
`FLOWPI_POLICY_METRICS_PATH` / `FLOWPI_DOMINO_METRICS_PATH` paths. The first response may include
JAX compilation, so the launcher uses a 180-second DOMINO RPC timeout by default; override it with
`FLOWPI_DOMINO_TIMEOUT`.

For a direct RoboTwin evaluator checkout (without DOMINO's client/server layer), the legacy
`scripts/run_flowpi_robotwin.sh` entry point and `scripts/flowpi_robotwin_deploy.yml` remain
available.

## 8. Running Tests

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
