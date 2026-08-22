#!/usr/bin/env bash
# Full FlowPI training launcher for one host with 8 H200 GPUs.
#
# This file is prepared on one machine and executed on another H-platform machine with the same
# shared filesystem. It forces the precomputed Flow cache and never starts SEA-RAFT. Training
# output is appended to RUN_DIR/run.log; heartbeat and per-GPU telemetry are separate files so a
# remote observer can inspect the run without attaching to the training terminal.
#
# Typical launch:
#   bash scripts/train_flowpi_8xh200.sh
#
# Example overrides:
#   FLOWPI_EXP_NAME=flowpi_8xh200_seed43 FLOWPI_SEED=43 \
#   FLOWPI_DATASET_ROOT=/shared/flowpi_data/train_dataset \
#   FLOWPI_FLOW_CACHE_DIR=/shared/flowpi_data/flow_cache \
#   FLOWPI_FLOW_DELAY_DISTRIBUTION="1 1 1 1" \
#   FLOWPI_LOG_ROOT=/shared/flowpi_logs \
#   bash scripts/train_flowpi_8xh200.sh
#
# Safe default: resume=true and overwrite=false. Use a new FLOWPI_EXP_NAME for a new run.

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

SCRIPT_DIR="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

env_or() {
    local name=$1
    local default=$2
    local value
    value="$(printenv "$name" 2>/dev/null || true)"
    if [[ -n "$value" ]]; then
        printf '%s' "$value"
    else
        printf '%s' "$default"
    fi
}

CONFIG_NAME="$(env_or FLOWPI_CONFIG_NAME flowpi_aloha)"
EXP_NAME="$(env_or FLOWPI_EXP_NAME flowpi_8xh200)"
RUN_ID="$(env_or FLOWPI_RUN_ID "$(date -u +%Y%m%dT%H%M%SZ)")"
LOG_ROOT="$(env_or FLOWPI_LOG_ROOT "$REPO_ROOT/logs/flowpi_8xh200")"
RUN_DIR="$(env_or FLOWPI_RUN_DIR "$LOG_ROOT/$EXP_NAME/$RUN_ID")"
mkdir -p "$RUN_DIR"

RUN_LOG="$RUN_DIR/run.log"
HEARTBEAT_LOG="$RUN_DIR/heartbeat.log"
GPU_LOG="$RUN_DIR/gpu_utilization.csv"
ENV_LOG="$RUN_DIR/environment.txt"
COMMAND_LOG="$RUN_DIR/command.txt"
STATUS_LOG="$RUN_DIR/exit_status.txt"

log() {
    local level=$1
    shift
    printf '%s [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$level" "$*" | tee -a "$RUN_LOG"
}

die() {
    log ERROR "$*"
    exit 1
}

log INFO "FlowPI 8xH200 launcher started. repo=$REPO_ROOT"
log INFO "Run directory: $RUN_DIR"
log INFO "Main log: $RUN_LOG"

# ------------------------------- training recipe -------------------------------

EXPECTED_GPUS="$(env_or FLOWPI_EXPECTED_GPUS 8)"
GLOBAL_BATCH="$(env_or FLOWPI_GLOBAL_BATCH 256)"
NUM_STEPS="$(env_or FLOWPI_NUM_STEPS 30000)"
WARMUP_STEPS="$(env_or FLOWPI_WARMUP_STEPS 1000)"
PEAK_LR="$(env_or FLOWPI_PEAK_LR 5e-5)"
DECAY_LR="$(env_or FLOWPI_DECAY_LR 5e-6)"
GRAD_CLIP="$(env_or FLOWPI_GRAD_CLIP 1.0)"
EMA_DECAY="$(env_or FLOWPI_EMA_DECAY 0.999)"
FLOW_DELAY_MAX="$(env_or FLOWPI_FLOW_DELAY_MAX 3)"
VLM_DELAY_MAX="$(env_or FLOWPI_VLM_DELAY_MAX 10)"
NUM_WORKERS="$(env_or FLOWPI_NUM_WORKERS 8)"
LOG_INTERVAL="$(env_or FLOWPI_LOG_INTERVAL 10)"
SAVE_INTERVAL="$(env_or FLOWPI_SAVE_INTERVAL 5000)"
KEEP_PERIOD="$(env_or FLOWPI_KEEP_PERIOD 5000)"
HEARTBEAT_INTERVAL="$(env_or FLOWPI_HEARTBEAT_INTERVAL 60)"
SEED="$(env_or FLOWPI_SEED 42)"

CHECKPOINT_BASE_DIR="$(env_or FLOWPI_CHECKPOINT_BASE_DIR "$REPO_ROOT/checkpoints")"
CHECKPOINT_DIR="$CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME"
DATASET_ROOT="$(env_or FLOWPI_DATASET_ROOT "$REPO_ROOT/flowpi_data/train_dataset")"
FLOW_CACHE_DIR="$(env_or FLOWPI_FLOW_CACHE_DIR "$REPO_ROOT/flowpi_data/flow_cache")"
ASSETS_DIR="$(env_or FLOWPI_ASSETS_DIR "$REPO_ROOT/assets/flowpi_aloha")"
ASSET_ID="$(env_or FLOWPI_ASSET_ID flowpi_data/train_dataset)"
WEIGHT_LOADER_PATH="$(env_or FLOWPI_WEIGHT_LOADER_PATH gs://openpi-assets/checkpoints/pi05_base/params)"
RESUME="$(env_or FLOWPI_RESUME 1)"
OVERWRITE="$(env_or FLOWPI_OVERWRITE 0)"
WANDB_ENABLED="$(env_or FLOWPI_WANDB_ENABLED 0)"

[[ "$EXPECTED_GPUS" =~ ^[1-9][0-9]*$ ]] || die "FLOWPI_EXPECTED_GPUS must be a positive integer"
[[ "$GLOBAL_BATCH" =~ ^[1-9][0-9]*$ ]] || die "FLOWPI_GLOBAL_BATCH must be a positive integer"
(( GLOBAL_BATCH % EXPECTED_GPUS == 0 )) || die "Global batch is not divisible by GPU count"
[[ "$FLOW_DELAY_MAX" =~ ^[0-9]+$ ]] || die "FLOWPI_FLOW_DELAY_MAX must be a non-negative integer"
[[ "$VLM_DELAY_MAX" =~ ^[0-9]+$ ]] || die "FLOWPI_VLM_DELAY_MAX must be a non-negative integer"
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || die "FLOWPI_RESUME must be 0 or 1"
[[ "$OVERWRITE" == 0 || "$OVERWRITE" == 1 ]] || die "FLOWPI_OVERWRITE must be 0 or 1"
[[ "$WANDB_ENABLED" == 0 || "$WANDB_ENABLED" == 1 ]] || die "FLOWPI_WANDB_ENABLED must be 0 or 1"
[[ "$RESUME:$OVERWRITE" != 1:1 ]] || die "Resume and overwrite cannot both be enabled"

FLOW_DELAY_DISTRIBUTION_RAW="$(env_or FLOWPI_FLOW_DELAY_DISTRIBUTION "")"
if [[ -n "$FLOW_DELAY_DISTRIBUTION_RAW" ]]; then
    IFS=' ' read -r -a FLOW_DELAY_DISTRIBUTION <<< "$FLOW_DELAY_DISTRIBUTION_RAW"
else
    FLOW_DELAY_DISTRIBUTION=()
    for ((delay = 0; delay <= FLOW_DELAY_MAX; delay++)); do
        FLOW_DELAY_DISTRIBUTION+=(1)
    done
fi
(( ${#FLOW_DELAY_DISTRIBUTION[@]} == FLOW_DELAY_MAX + 1 )) || die \
    "Flow delay distribution must contain FLOWPI_FLOW_DELAY_MAX+1=$((FLOW_DELAY_MAX + 1)) weights"

PER_GPU_BATCH=$((GLOBAL_BATCH / EXPECTED_GPUS))
log INFO "Target: $EXPECTED_GPUS GPUs, global batch=$GLOBAL_BATCH, per-GPU batch=$PER_GPU_BATCH"
log INFO "Schedule: steps=$NUM_STEPS, warmup=$WARMUP_STEPS, peak_lr=$PEAK_LR, end_lr=$DECAY_LR"
log INFO "Optimizer: AdamW, grad_clip=$GRAD_CLIP, ema=$EMA_DECAY, fsdp_devices=1"
log INFO "Delays: flow=0..$FLOW_DELAY_MAX, VLM=0..$VLM_DELAY_MAX, sampled independently"
log INFO "Flow: cache=$FLOW_CACHE_DIR, SEA-RAFT is offline"
log INFO "Trainable: VLM language backbone + Action Expert + Flow modules; frozen: SigLIP vision tower"
log INFO "Geometry: image geometric augmentation disabled to preserve raw-cache coordinates"
log INFO "Checkpoint directory: $CHECKPOINT_DIR"

# ----------------------------- accelerator/runtime -----------------------------

export CUDA_VISIBLE_DEVICES="$(env_or CUDA_VISIBLE_DEVICES 0,1,2,3,4,5,6,7)"
export JAX_PLATFORMS="$(env_or JAX_PLATFORMS gpu)"
export JAX_TRACEBACK_FILTERING="$(env_or JAX_TRACEBACK_FILTERING off)"
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING="$(env_or NCCL_ASYNC_ERROR_HANDLING 1)"
export NCCL_DEBUG="$(env_or NCCL_DEBUG INFO)"
export WANDB_DIR="$(env_or WANDB_DIR "$RUN_DIR/wandb")"
mkdir -p "$WANDB_DIR"

PYTHON_BIN="$(env_or FLOWPI_PYTHON_BIN "")"
if [[ -n "$PYTHON_BIN" ]]; then
    PYTHON_MODE=direct
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    PYTHON_MODE=direct
elif command -v uv >/dev/null 2>&1; then
    export UV_CACHE_DIR="$(env_or FLOWPI_UV_CACHE_DIR "$REPO_ROOT/.cache/uv")"
    mkdir -p "$UV_CACHE_DIR"
    PYTHON_MODE=uv
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
    PYTHON_MODE=direct
else
    die "No Python interpreter found. Set FLOWPI_PYTHON_BIN or install uv/python3."
fi

if [[ "$PYTHON_MODE" == uv ]]; then
    PYTHON_DISPLAY="uv run python"
else
    PYTHON_DISPLAY="$PYTHON_BIN"
fi

run_python() {
    if [[ "$PYTHON_MODE" == uv ]]; then
        uv run python "$@"
    else
        "$PYTHON_BIN" "$@"
    fi
}

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>>"$RUN_LOG" | awk 'NF {n++} END {print n+0}')"
[[ "$GPU_COUNT" == "$EXPECTED_GPUS" ]] || die \
    "Expected $EXPECTED_GPUS visible GPUs, detected $GPU_COUNT; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi -L > "$RUN_DIR/gpu_list.txt" 2>&1 || true
nvidia-smi -q > "$RUN_DIR/nvidia-smi-q.txt" 2>&1 || true
nvidia-smi topo -m > "$RUN_DIR/nvidia-smi-topology.txt" 2>&1 || true

if [[ "$DATASET_ROOT" != *://* && ! -e "$DATASET_ROOT" ]]; then
    die "Dataset path does not exist: $DATASET_ROOT"
fi
if [[ "$FLOW_CACHE_DIR" != *://* && ! -e "$FLOW_CACHE_DIR/meta.json" ]]; then
    die "Flow cache metadata does not exist: $FLOW_CACHE_DIR/meta.json"
fi
if [[ "$ASSETS_DIR" != *://* && ! -e "$ASSETS_DIR" ]]; then
    die "Assets path does not exist: $ASSETS_DIR"
fi

# -------------------------------- preflight ------------------------------------

{
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo_root=%s\n' "$REPO_ROOT"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    printf 'git_branch=%s\n' "$(git branch --show-current 2>/dev/null || echo unavailable)"
    printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'expected_gpus=%s\n' "$EXPECTED_GPUS"
    printf 'global_batch=%s\n' "$GLOBAL_BATCH"
    printf 'per_gpu_batch=%s\n' "$PER_GPU_BATCH"
    printf 'num_steps=%s\n' "$NUM_STEPS"
    printf 'warmup_steps=%s\n' "$WARMUP_STEPS"
    printf 'peak_lr=%s\n' "$PEAK_LR"
    printf 'decay_lr=%s\n' "$DECAY_LR"
    printf 'grad_clip=%s\n' "$GRAD_CLIP"
    printf 'ema_decay=%s\n' "$EMA_DECAY"
    printf 'flow_delay_max=%s\n' "$FLOW_DELAY_MAX"
    printf 'flow_delay_distribution=%s\n' "${FLOW_DELAY_DISTRIBUTION[*]}"
    printf 'vlm_delay_max=%s\n' "$VLM_DELAY_MAX"
    printf 'num_workers=%s\n' "$NUM_WORKERS"
    printf 'resume=%s\n' "$RESUME"
    printf 'overwrite=%s\n' "$OVERWRITE"
    printf 'wandb_enabled=%s\n' "$WANDB_ENABLED"
    printf 'dataset_root=%s\n' "$DATASET_ROOT"
    printf 'flow_cache_dir=%s\n' "$FLOW_CACHE_DIR"
    printf 'assets_dir=%s\n' "$ASSETS_DIR"
    printf 'checkpoint_dir=%s\n' "$CHECKPOINT_DIR"
    printf 'python_command=%s\n' "$PYTHON_DISPLAY"
    printf '\n[git status]\n'
    git status --short --untracked-files=all 2>&1 || true
    printf '\n[GPU list]\n'
    cat "$RUN_DIR/gpu_list.txt"
    printf '\n[GPU snapshot]\n'
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,temperature.gpu --format=csv 2>&1 || true
    printf '\n[flow cache meta]\n'
    if [[ -f "$FLOW_CACHE_DIR/meta.json" ]]; then
        sed -n '1,240p' "$FLOW_CACHE_DIR/meta.json"
    else
        printf 'metadata unavailable\n'
    fi
    printf '\n[Python/JAX preflight]\n'
    run_python -c '
import jax
import jaxlib
import sys
print("python=" + sys.executable)
print("jax=" + jax.__version__)
print("jaxlib=" + jaxlib.__version__)
print("backend=" + jax.default_backend())
print("devices=" + repr(jax.devices()))
print("JAX_DEVICE_COUNT=" + str(jax.device_count()))
' 2>&1
} > "$ENV_LOG"

JAX_DEVICE_COUNT="$(awk -F= '/^JAX_DEVICE_COUNT=/{value=$2} END {print value}' "$ENV_LOG")"
[[ "$JAX_DEVICE_COUNT" == "$EXPECTED_GPUS" ]] || die \
    "JAX sees $JAX_DEVICE_COUNT devices, expected $EXPECTED_GPUS; see $ENV_LOG"

# ------------------------------- train command ---------------------------------

if [[ "$WANDB_ENABLED" == 1 ]]; then
    WANDB_FLAG=--wandb-enabled
else
    WANDB_FLAG=--no-wandb-enabled
fi

if [[ "$RESUME" == 1 ]]; then
    RESUME_FLAG=--resume
    OVERWRITE_FLAG=--no-overwrite
elif [[ "$OVERWRITE" == 1 ]]; then
    RESUME_FLAG=--no-resume
    OVERWRITE_FLAG=--overwrite
else
    RESUME_FLAG=--no-resume
    OVERWRITE_FLAG=--no-overwrite
fi

write_command_log() {
    if [[ "$PYTHON_MODE" == uv ]]; then
        printf 'uv run python '
    else
        printf '%q ' "$PYTHON_BIN"
    fi
    printf '%q ' scripts/train.py "$CONFIG_NAME" \
        --project-name "$(env_or FLOWPI_PROJECT_NAME flowpi)" \
        --exp-name "$EXP_NAME" \
        --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
        --batch-size "$GLOBAL_BATCH" \
        --num-train-steps "$NUM_STEPS" \
        --num-workers "$NUM_WORKERS" \
        --log-interval "$LOG_INTERVAL" \
        --save-interval "$SAVE_INTERVAL" \
        --keep-period "$KEEP_PERIOD" \
        --seed "$SEED" \
        --fsdp-devices 1 \
        --model.pi05 --model.dtype bfloat16 --model.freeze-vision-encoder \
        --model.flow.vlm-delay-max "$VLM_DELAY_MAX" \
        --model.flow.flow-delay-max "$FLOW_DELAY_MAX" \
        --model.flow.flow-delay-distribution "${FLOW_DELAY_DISTRIBUTION[@]}" \
        --model.flow.no-image-geometric-aug \
        --lr-schedule.warmup-steps "$WARMUP_STEPS" \
        --lr-schedule.peak-lr "$PEAK_LR" \
        --lr-schedule.decay-steps "$NUM_STEPS" \
        --lr-schedule.decay-lr "$DECAY_LR" \
        --optimizer.clip-gradient-norm "$GRAD_CLIP" \
        --ema-decay "$EMA_DECAY" \
        --data.repo-id "$DATASET_ROOT" \
        --data.assets.assets-dir "$ASSETS_DIR" \
        --data.assets.asset-id "$ASSET_ID" \
        --data.flow.mode cache \
        --data.flow.flow-cache-dir "$FLOW_CACHE_DIR" \
        --data.flow.sea-raft-ckpt None \
        --data.flow.sea-raft-device cpu \
        --data.flow.load-flow-cache --data.flow.sample-vlm-delay \
        --weight-loader.params-path "$WEIGHT_LOADER_PATH" \
        "$WANDB_FLAG" "$RESUME_FLAG" "$OVERWRITE_FLAG"
    printf '%s\n' ""
}

write_command_log > "$COMMAND_LOG"
log INFO "Resolved command written to $COMMAND_LOG"
log INFO "Environment snapshot written to $ENV_LOG"
log INFO "Starting training. Follow with: tail -F $RUN_LOG"

nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv \
    --loop=30 > "$GPU_LOG" 2>&1 &
GPU_MONITOR_PID=$!

TRAIN_PID=""
HEARTBEAT_PID=""

heartbeat_loop() {
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        {
            printf '\n=== heartbeat %s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'train_pid=%s\n' "$TRAIN_PID"
            ps -p "$TRAIN_PID" -o pid,ppid,etime,stat,pcpu,pmem,rss,args --no-headers 2>&1 || true
            printf '\nlatest_training_log_tail:\n'
            tail -n 8 "$RUN_LOG" 2>&1 || true
            printf '\ncheckpoint_steps:\n'
            if [[ -d "$CHECKPOINT_DIR" ]]; then
                find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -n 2>/dev/null || true
            else
                printf 'checkpoint directory not created yet\n'
            fi
            printf '\nGPU snapshot:\n'
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv 2>&1 || true
            printf '\nDisk snapshot:\n'
            df -h "$RUN_DIR" 2>&1 || true
        } >> "$HEARTBEAT_LOG" 2>&1
        sleep "$HEARTBEAT_INTERVAL"
    done
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$HEARTBEAT_PID" ]] && kill -0 "$HEARTBEAT_PID" 2>/dev/null; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    {
        printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'exit_status=%s\n' "$status"
        printf 'train_pid=%s\n' "$TRAIN_PID"
        printf 'checkpoint_dir=%s\n' "$CHECKPOINT_DIR"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv 2>&1 || true
    } > "$STATUS_LOG"
    if (( status == 0 )); then
        log INFO "Training exited successfully. Status: $STATUS_LOG"
    else
        log ERROR "Training exited with status $status. Inspect $RUN_LOG, $HEARTBEAT_LOG, and $STATUS_LOG"
    fi
    exit "$status"
}

on_signal() {
    if [[ -n "$TRAIN_PID" ]]; then
        log WARNING "Received termination signal; forwarding SIGTERM to training PID $TRAIN_PID"
        if kill -0 "$TRAIN_PID" 2>/dev/null; then
            kill -TERM "$TRAIN_PID" 2>/dev/null || true
        fi
    else
        log WARNING "Received termination signal before training started"
    fi
    exit 143
}

trap cleanup EXIT
trap on_signal INT TERM

run_training() {
    run_python scripts/train.py "$CONFIG_NAME" \
        --project-name "$(env_or FLOWPI_PROJECT_NAME flowpi)" \
        --exp-name "$EXP_NAME" \
        --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
        --batch-size "$GLOBAL_BATCH" \
        --num-train-steps "$NUM_STEPS" \
        --num-workers "$NUM_WORKERS" \
        --log-interval "$LOG_INTERVAL" \
        --save-interval "$SAVE_INTERVAL" \
        --keep-period "$KEEP_PERIOD" \
        --seed "$SEED" \
        --fsdp-devices 1 \
        --model.pi05 \
        --model.dtype bfloat16 \
        --model.freeze-vision-encoder \
        --model.flow.vlm-delay-max "$VLM_DELAY_MAX" \
        --model.flow.flow-delay-max "$FLOW_DELAY_MAX" \
        --model.flow.flow-delay-distribution "${FLOW_DELAY_DISTRIBUTION[@]}" \
        --model.flow.no-image-geometric-aug \
        --lr-schedule.warmup-steps "$WARMUP_STEPS" \
        --lr-schedule.peak-lr "$PEAK_LR" \
        --lr-schedule.decay-steps "$NUM_STEPS" \
        --lr-schedule.decay-lr "$DECAY_LR" \
        --optimizer.clip-gradient-norm "$GRAD_CLIP" \
        --ema-decay "$EMA_DECAY" \
        --data.repo-id "$DATASET_ROOT" \
        --data.assets.assets-dir "$ASSETS_DIR" \
        --data.assets.asset-id "$ASSET_ID" \
        --data.flow.mode cache \
        --data.flow.flow-cache-dir "$FLOW_CACHE_DIR" \
        --data.flow.sea-raft-ckpt None \
        --data.flow.sea-raft-device cpu \
        --data.flow.load-flow-cache \
        --data.flow.sample-vlm-delay \
        --weight-loader.params-path "$WEIGHT_LOADER_PATH" \
        "$WANDB_FLAG" "$RESUME_FLAG" "$OVERWRITE_FLAG"
}

# The training process writes only to the durable shared log. Heartbeat and GPU monitoring remain
# independent, so a Python/JAX crash still leaves diagnostics on disk.
run_training >> "$RUN_LOG" 2>&1 &
TRAIN_PID=$!
log INFO "Training PID=$TRAIN_PID"

heartbeat_loop &
HEARTBEAT_PID=$!
log INFO "Heartbeat PID=$HEARTBEAT_PID; interval=$HEARTBEAT_INTERVAL seconds"

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
exit "$TRAIN_STATUS"
