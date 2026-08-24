# FlowPi

This repository contains the FlowPi model, its JAX training pipeline, and the SEA-RAFT optical-flow
integration. Runtime evaluation is intentionally small: `scripts/flowpi_infer.py` is a sequential
three-GPU offline runner with one slow VLM replica, one fast streaming-NFE replica, and one
SEA-RAFT Torch device. Dataset, normalization, cache, model-checkpoint, and output paths are always
provided by the command line.

## Setup

```bash
uv sync --group dev
git submodule update --init SEA-RAFT
uv run python scripts/setup_sea_raft.py
```

The pinned SEA-RAFT submodule is patched by
`third_party/sea_raft/flowpi_return_low_res.patch`. The patch adds the low-resolution flow API,
avoids downloading ImageNet initialization for a restored model, and makes large correlation
sampling safe. SEA-RAFT weights are external files and are not stored in this repository.

## Training

FlowPi uses a LeRobot v3 dataset and an offline SEA-RAFT cache. Replace every path below with a
path on the machine running the command.

```bash
uv run python scripts/compute_norm_stats.py flowpi_aloha \
  --data.repo-id /path/to/lerobot-dataset \
  --data.assets.assets-dir /path/to/assets \
  --data.assets.asset-id flowpi \
  --data.flow.enabled false

uv run python scripts/precompute_flow_cache.py flowpi_aloha \
  --data.repo-id /path/to/lerobot-dataset \
  --data.assets.assets-dir /path/to/assets \
  --data.assets.asset-id flowpi \
  --data.flow.flow-cache-dir /path/to/flow-cache \
  --data.flow.sea-raft-ckpt /path/to/sea-raft.pth \
  --data.flow.sea-raft-device cuda:0

uv run python scripts/train.py flowpi_aloha \
  --exp-name flowpi-run \
  --checkpoint-base-dir /path/to/checkpoints \
  --data.repo-id /path/to/lerobot-dataset \
  --data.assets.assets-dir /path/to/assets \
  --data.assets.asset-id flowpi \
  --data.flow.flow-cache-dir /path/to/flow-cache \
  --data.flow.sea-raft-ckpt /path/to/sea-raft.pth \
  --weight-loader.params-path gs://openpi-assets/checkpoints/pi05_base/params
```

`--checkpoint-base-dir`, `--data.repo-id`, `--data.flow.flow-cache-dir`,
`--data.flow.sea-raft-ckpt`, and `--weight-loader.params-path` are independent paths. The training
recipe does not assume a repository-local dataset, cache, or checkpoint.

## Minimal inference

```bash
uv run python scripts/flowpi_infer.py \
  --config-name flowpi_aloha \
  --checkpoint /path/to/checkpoint/step \
  --dataset /path/to/lerobot-dataset \
  --output /path/to/results/actions.npz \
  --sea-raft-ckpt /path/to/sea-raft.pth \
  --slow-jax-device gpu:0 \
  --fast-jax-device gpu:1 \
  --sea-raft-device cuda:2
```

The runner processes frames in order, performs one NFE per frame, refreshes the slow prefix
synchronously, and writes only the requested `.npz` output. It has no simulator adapter,
frequency controller, action-width scheduler, or evaluation-result directory.

## Verification

The repository keeps CPU/static checks separate from hardware validation. For a local code check:

```bash
uv run ruff check \
  src/openpi/models/pi0.py \
  src/openpi/policies/flowpi_runtime.py \
  src/openpi/training/sea_raft.py \
  scripts/flowpi_infer.py \
  scripts/flowpi_checkpoint.py \
  scripts/compute_norm_stats.py
uv run python -m compileall -q src scripts
```

Training and GPU inference are deliberately not part of the basic verification command.
