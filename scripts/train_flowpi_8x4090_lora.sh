#!/usr/bin/env bash
# Memory-aware FlowPI training launcher for 8x RTX 4090.
#
# The durable logging, heartbeat, GPU telemetry, resume, and checkpoint behavior lives in
# train_flowpi_8xh200.sh. This wrapper only supplies an isolated recipe for the 4090 host:
# VLM LoRA, full Action Expert + FlowPI modules, frozen SigLIP, and global batch 32.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FLOWPI_CONFIG_NAME="${FLOWPI_CONFIG_NAME:-flowpi_aloha_8x4090_lora}"
export FLOWPI_EXP_NAME="${FLOWPI_EXP_NAME:-flowpi_8x4090_lora}"
export FLOWPI_LAUNCHER_LABEL="${FLOWPI_LAUNCHER_LABEL:-8xRTX4090-LoRA}"
export FLOWPI_LOG_ROOT="${FLOWPI_LOG_ROOT:-$SCRIPT_DIR/../logs/flowpi_8x4090_lora}"
export FLOWPI_CHECKPOINT_BASE_DIR="${FLOWPI_CHECKPOINT_BASE_DIR:-$SCRIPT_DIR/../checkpoints_8x4090_lora}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FLOWPI_EXPECTED_GPUS="${FLOWPI_EXPECTED_GPUS:-8}"
export FLOWPI_GLOBAL_BATCH="${FLOWPI_GLOBAL_BATCH:-32}"
export FLOWPI_NUM_STEPS="${FLOWPI_NUM_STEPS:-30000}"
export FLOWPI_WARMUP_STEPS="${FLOWPI_WARMUP_STEPS:-1000}"
export FLOWPI_PEAK_LR="${FLOWPI_PEAK_LR:-5e-5}"
export FLOWPI_DECAY_LR="${FLOWPI_DECAY_LR:-5e-6}"
export FLOWPI_GRAD_CLIP="${FLOWPI_GRAD_CLIP:-1.0}"

# LoRA recipes do not keep a second full-model EMA copy. This leaves more 24 GB memory for
# the full Action Expert and FlowPI branches while still saving complete train_state checkpoints.
export FLOWPI_EMA_DECAY="${FLOWPI_EMA_DECAY:-None}"
export FLOWPI_SAVE_INTERVAL="${FLOWPI_SAVE_INTERVAL:-2000}"
export FLOWPI_KEEP_PERIOD="${FLOWPI_KEEP_PERIOD:-2000}"
export FLOWPI_XLA_MEM_FRACTION="${FLOWPI_XLA_MEM_FRACTION:-0.92}"
export FLOWPI_NUM_WORKERS="${FLOWPI_NUM_WORKERS:-8}"

# Latest FlowPI training recipe: independently sampled ages plus Flow-required stale-prefix
# coverage and a small non-zero Flow cross-attention gate initialization.
export FLOWPI_FLOW_DELAY_MAX="${FLOWPI_FLOW_DELAY_MAX:-3}"
export FLOWPI_VLM_DELAY_MAX="${FLOWPI_VLM_DELAY_MAX:-10}"
export FLOWPI_FLOW_REQUIRED_PROB="${FLOWPI_FLOW_REQUIRED_PROB:-0.5}"
export FLOWPI_FLOW_REQUIRED_VLM_DELAY_MIN="${FLOWPI_FLOW_REQUIRED_VLM_DELAY_MIN:-5}"
export FLOWPI_FLOW_GATE_INIT="${FLOWPI_FLOW_GATE_INIT:-0.01}"
export FLOWPI_RESUME="${FLOWPI_RESUME:-1}"
export FLOWPI_OVERWRITE="${FLOWPI_OVERWRITE:-0}"
export FLOWPI_WANDB_ENABLED="${FLOWPI_WANDB_ENABLED:-0}"

exec bash "$SCRIPT_DIR/train_flowpi_8xh200.sh" "$@"
