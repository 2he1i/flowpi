#!/usr/bin/env bash
# One-shot FlowPI cache + training launcher.
#
# Modes:
#   cache  - precompute the SEA-RAFT cache on four logical CUDA devices.
#   train  - require a matching cache and launch the policy training script.
#   all    - run cache first, then training (default; useful on an 8-GPU host).
#
# The SEA-RAFT checkpoint is deliberately checked at runtime because it is expected to be moved
# into the shared filesystem after this script is committed.

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

SCRIPT_DIR="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_flowpi_cache_and_train.sh [cache|train|all]

Modes:
  cache  precompute the SEA-RAFT cache on four logical CUDA devices.
  train  require a matching cache and launch policy training.
  all    run cache first, then training (default; needs the training GPUs too).

Examples:
  bash scripts/run_flowpi_cache_and_train.sh cache
  bash scripts/run_flowpi_cache_and_train.sh train
  bash scripts/run_flowpi_cache_and_train.sh all

Environment overrides:
  FLOWPI_SEA_RAFT_CKPT       SEA-RAFT M checkpoint path.
  FLOWPI_CACHE_CUDA_DEVICES  Physical CUDA_VISIBLE_DEVICES for caching (default: 0,1,2,3).
  FLOWPI_CACHE_DEVICES       Logical devices passed to precompute (default: 0,1,2,3).
  FLOWPI_CACHE_OVERWRITE      Set to 1 to recompute an existing cache (default: 0).
  FLOWPI_TRAIN_CUDA_DEVICES  Physical CUDA_VISIBLE_DEVICES for training (default: 0,...,7).
  FLOWPI_TRAIN_EXPECTED_GPUS Number of GPUs expected by the training launcher (default: 8).
  FLOWPI_GLOBAL_BATCH         Training global batch size (default: 128).
  FLOWPI_RESUME_STEP          Optional exact checkpoint step to resume.
EOF
}

MODE="${1:-all}"
case "$MODE" in
    cache|train|all) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

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
LOG_ROOT="$(env_or FLOWPI_CACHE_TRAIN_LOG_ROOT "$REPO_ROOT/logs/flowpi_cache_train")"
RUN_DIR="$(env_or FLOWPI_CACHE_TRAIN_RUN_DIR "$LOG_ROOT/$EXP_NAME/$RUN_ID")"
mkdir -p "$RUN_DIR"

MAIN_LOG="$RUN_DIR/run.log"
CACHE_LOG="$RUN_DIR/cache.log"
CACHE_COMMAND_LOG="$RUN_DIR/cache_command.txt"
CACHE_GPU_LOG="$RUN_DIR/cache_gpu_utilization.csv"
CACHE_STATUS_LOG="$RUN_DIR/cache_status.txt"
TRAIN_LOG="$RUN_DIR/train_launcher.log"
TRAIN_RUN_DIR="$RUN_DIR/train"
ENV_LOG="$RUN_DIR/environment.txt"
STATUS_LOG="$RUN_DIR/exit_status.txt"

log() {
    local level=$1
    shift
    printf '%s [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$level" "$*" | tee -a "$MAIN_LOG"
}

die() {
    log ERROR "$*"
    exit 1
}

EXPECTED_CACHE_CKPT="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/SEA-RAFT-FT/SEA-RAFT/checkpoints/24000_robot-ft-M-4gpu-shadow-15k-to-25k_robot-ft-M-4gpu-shadow-15k-to-25k-20260822-053345.pth"
SEA_RAFT_CKPT="$(env_or FLOWPI_SEA_RAFT_CKPT "$EXPECTED_CACHE_CKPT")"
SEA_RAFT_VARIANT="$(env_or FLOWPI_SEA_RAFT_VARIANT M)"
SEA_RAFT_ITERS="$(env_or FLOWPI_SEA_RAFT_ITERS 4)"
DATASET_ROOT="$(env_or FLOWPI_DATASET_ROOT "$REPO_ROOT/flowpi_data/train_dataset")"
FLOW_CACHE_DIR="$(env_or FLOWPI_FLOW_CACHE_DIR "$REPO_ROOT/flowpi_data/flow_cache")"
ASSETS_DIR="$(env_or FLOWPI_ASSETS_DIR "$REPO_ROOT/assets/flowpi_aloha")"
ASSET_ID="$(env_or FLOWPI_ASSET_ID flowpi_data/train_dataset)"
CHECKPOINT_BASE_DIR="$(env_or FLOWPI_CHECKPOINT_BASE_DIR "$REPO_ROOT/checkpoints")"
WEIGHT_LOADER_PATH="$(env_or FLOWPI_WEIGHT_LOADER_PATH gs://openpi-assets/checkpoints/pi05_base/params)"
CACHE_CUDA_DEVICES="$(env_or FLOWPI_CACHE_CUDA_DEVICES 0,1,2,3)"
CACHE_DEVICES="$(env_or FLOWPI_CACHE_DEVICES 0,1,2,3)"
CACHE_WORKERS="$(env_or FLOWPI_CACHE_WORKERS 4)"
CACHE_OVERWRITE="$(env_or FLOWPI_CACHE_OVERWRITE 0)"
TRAIN_CUDA_DEVICES="$(env_or FLOWPI_TRAIN_CUDA_DEVICES 0,1,2,3,4,5,6,7)"
TRAIN_EXPECTED_GPUS="$(env_or FLOWPI_TRAIN_EXPECTED_GPUS 8)"
GLOBAL_BATCH="$(env_or FLOWPI_GLOBAL_BATCH 128)"

[[ "$SEA_RAFT_VARIANT" == S || "$SEA_RAFT_VARIANT" == M || "$SEA_RAFT_VARIANT" == L ]] || die \
    "FLOWPI_SEA_RAFT_VARIANT must be S, M, or L"
[[ "$SEA_RAFT_ITERS" =~ ^[1-9][0-9]*$ ]] || die "FLOWPI_SEA_RAFT_ITERS must be a positive integer"
[[ "$CACHE_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "FLOWPI_CACHE_WORKERS must be a positive integer"
[[ "$CACHE_OVERWRITE" == 0 || "$CACHE_OVERWRITE" == 1 ]] || die "FLOWPI_CACHE_OVERWRITE must be 0 or 1"
[[ "$TRAIN_EXPECTED_GPUS" =~ ^[1-9][0-9]*$ ]] || die "FLOWPI_TRAIN_EXPECTED_GPUS must be a positive integer"
[[ "$GLOBAL_BATCH" =~ ^[1-9][0-9]*$ ]] || die "FLOWPI_GLOBAL_BATCH must be a positive integer"

IFS=',' read -r -a CACHE_DEVICE_ARRAY <<< "$CACHE_DEVICES"
(( ${#CACHE_DEVICE_ARRAY[@]} == CACHE_WORKERS )) || die \
    "CACHE_WORKERS=$CACHE_WORKERS must equal the number of CACHE_DEVICES=${#CACHE_DEVICE_ARRAY[@]}"
IFS=',' read -r -a CACHE_CUDA_DEVICE_ARRAY <<< "$CACHE_CUDA_DEVICES"
(( ${#CACHE_CUDA_DEVICE_ARRAY[@]} == CACHE_WORKERS )) || die \
    "CACHE_WORKERS=$CACHE_WORKERS must equal the number of FLOWPI_CACHE_CUDA_DEVICES=${#CACHE_CUDA_DEVICE_ARRAY[@]}"
IFS=',' read -r -a TRAIN_CUDA_DEVICE_ARRAY <<< "$TRAIN_CUDA_DEVICES"
(( ${#TRAIN_CUDA_DEVICE_ARRAY[@]} == TRAIN_EXPECTED_GPUS )) || die \
    "FLOWPI_TRAIN_EXPECTED_GPUS=$TRAIN_EXPECTED_GPUS must equal the number of FLOWPI_TRAIN_CUDA_DEVICES=${#TRAIN_CUDA_DEVICE_ARRAY[@]}"
(( GLOBAL_BATCH % TRAIN_EXPECTED_GPUS == 0 )) || die \
    "FLOWPI_GLOBAL_BATCH=$GLOBAL_BATCH must be divisible by FLOWPI_TRAIN_EXPECTED_GPUS=$TRAIN_EXPECTED_GPUS"

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
else
    die "No Python interpreter found. Set FLOWPI_PYTHON_BIN or install uv."
fi

run_python() {
    if [[ "$PYTHON_MODE" == uv ]]; then
        uv run python "$@"
    else
        "$PYTHON_BIN" "$@"
    fi
}

print_python_command() {
    if [[ "$PYTHON_MODE" == uv ]]; then
        printf 'uv run python '
    else
        printf '%q ' "$PYTHON_BIN"
    fi
}

[[ -f "$SEA_RAFT_CKPT" ]] || die "SEA-RAFT checkpoint is not available yet: $SEA_RAFT_CKPT"
[[ -f "$DATASET_ROOT/meta/info.json" ]] || die "Dataset metadata not found: $DATASET_ROOT/meta/info.json"
[[ -d "$ASSETS_DIR" ]] || die "Assets directory not found: $ASSETS_DIR"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is unavailable"

CKPT_SHA256="$(sha256sum "$SEA_RAFT_CKPT" | awk '{print $1}')"
[[ -n "$CKPT_SHA256" ]] || die "Could not calculate SEA-RAFT checkpoint SHA256"

{
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo_root=%s\n' "$REPO_ROOT"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    printf 'git_branch=%s\n' "$(git branch --show-current 2>/dev/null || echo unavailable)"
    printf 'mode=%s\n' "$MODE"
    printf 'sea_raft_ckpt=%s\n' "$SEA_RAFT_CKPT"
    printf 'sea_raft_variant=%s\n' "$SEA_RAFT_VARIANT"
    printf 'sea_raft_iters=%s\n' "$SEA_RAFT_ITERS"
    printf 'sea_raft_checkpoint_sha256=%s\n' "$CKPT_SHA256"
    printf 'dataset_root=%s\n' "$DATASET_ROOT"
    printf 'flow_cache_dir=%s\n' "$FLOW_CACHE_DIR"
    printf 'assets_dir=%s\n' "$ASSETS_DIR"
    printf 'checkpoint_base_dir=%s\n' "$CHECKPOINT_BASE_DIR"
    printf 'cache_cuda_visible_devices=%s\n' "$CACHE_CUDA_DEVICES"
    printf 'cache_devices=%s\n' "$CACHE_DEVICES"
    printf 'cache_workers=%s\n' "$CACHE_WORKERS"
    printf 'cache_overwrite=%s\n' "$CACHE_OVERWRITE"
    printf 'train_cuda_visible_devices=%s\n' "$TRAIN_CUDA_DEVICES"
    printf 'train_expected_gpus=%s\n' "$TRAIN_EXPECTED_GPUS"
    printf 'global_batch=%s\n' "$GLOBAL_BATCH"
    printf 'python_mode=%s\n' "$PYTHON_MODE"
    printf '\n[GPU inventory]\n'
    nvidia-smi -L 2>&1 || true
    printf '\n[git status]\n'
    git status --short --untracked-files=all 2>&1 || true
} > "$ENV_LOG"

GPU_MONITOR_PID=""

stop_gpu_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    GPU_MONITOR_PID=""
}

cache_metadata_sha() {
    local metadata="$FLOW_CACHE_DIR/meta.json"
    [[ -f "$metadata" ]] || return 1
    awk -F'"' '/"sea_raft_checkpoint_sha256"/ {print $4; exit}' "$metadata"
}

cache_matches_checkpoint() {
    local cached_sha
    cached_sha="$(cache_metadata_sha || true)"
    [[ -n "$cached_sha" && "$cached_sha" == "$CKPT_SHA256" ]]
}

write_cache_command() {
    local -a args=(
        scripts/precompute_flow_cache.py
        "$CONFIG_NAME"
        --data.repo-id "$DATASET_ROOT"
        --data.flow.flow-cache-dir "$FLOW_CACHE_DIR"
        --data.flow.sea-raft-ckpt "$SEA_RAFT_CKPT"
        --data.flow.sea-raft-variant "$SEA_RAFT_VARIANT"
        --data.flow.sea-raft-iters "$SEA_RAFT_ITERS"
        --data.flow.sea-raft-device cuda
        --devices "$CACHE_DEVICES"
        --num-workers "$CACHE_WORKERS"
    )
    if [[ "$CACHE_OVERWRITE" == 1 ]]; then
        args+=(--overwrite)
    fi
    {
        print_python_command
        printf '%q ' "${args[@]}"
        printf '\n'
    } > "$CACHE_COMMAND_LOG"
}

run_cache() {
    write_cache_command
    if [[ -f "$FLOW_CACHE_DIR/meta.json" && "$CACHE_OVERWRITE" == 0 ]]; then
        if cache_matches_checkpoint; then
            log INFO "Flow cache already matches SEA-RAFT checkpoint; skipping recomputation."
            printf 'skipped=1\nreason=matching_checkpoint_sha256\n' > "$CACHE_STATUS_LOG"
            return 0
        fi
        die "Flow cache exists but was generated by a different SEA-RAFT checkpoint. Set FLOWPI_CACHE_OVERWRITE=1 to recompute."
    fi

    local -a args=(
        scripts/precompute_flow_cache.py
        "$CONFIG_NAME"
        --data.repo-id "$DATASET_ROOT"
        --data.flow.flow-cache-dir "$FLOW_CACHE_DIR"
        --data.flow.sea-raft-ckpt "$SEA_RAFT_CKPT"
        --data.flow.sea-raft-variant "$SEA_RAFT_VARIANT"
        --data.flow.sea-raft-iters "$SEA_RAFT_ITERS"
        --data.flow.sea-raft-device cuda
        --devices "$CACHE_DEVICES"
        --num-workers "$CACHE_WORKERS"
    )
    if [[ "$CACHE_OVERWRITE" == 1 ]]; then
        args+=(--overwrite)
    fi
    write_cache_command
    log INFO "Starting ${CACHE_WORKERS}-GPU SEA-RAFT cache generation."
    log INFO "Cache command: $CACHE_COMMAND_LOG"

    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv \
        --loop=30 > "$CACHE_GPU_LOG" 2>&1 &
    GPU_MONITOR_PID=$!

    set +e
    CUDA_VISIBLE_DEVICES="$CACHE_CUDA_DEVICES" run_python "${args[@]}" 2>&1 | tee -a "$CACHE_LOG"
    local cache_status=${PIPESTATUS[0]}
    set -e
    stop_gpu_monitor

    printf 'finished_utc=%s\nexit_status=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$cache_status" > "$CACHE_STATUS_LOG"
    (( cache_status == 0 )) || die "Flow cache generation failed; inspect $CACHE_LOG"
    [[ -f "$FLOW_CACHE_DIR/meta.json" ]] || die "Cache command completed without $FLOW_CACHE_DIR/meta.json"
    cache_matches_checkpoint || die "Generated cache SHA256 does not match SEA-RAFT checkpoint"
    log INFO "Flow cache generation completed and checkpoint SHA256 matches."
}

run_train() {
    [[ -f "$FLOW_CACHE_DIR/meta.json" ]] || die "Flow cache metadata is missing: $FLOW_CACHE_DIR/meta.json"
    cache_matches_checkpoint || die \
        "Flow cache SHA256 does not match SEA-RAFT checkpoint; run cache mode with FLOWPI_CACHE_OVERWRITE=1."
    log INFO "Starting FlowPI training with global batch $GLOBAL_BATCH."
    log INFO "Training logs will be under: $TRAIN_RUN_DIR"

    set +e
    CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_DEVICES" \
    FLOWPI_EXPECTED_GPUS="$TRAIN_EXPECTED_GPUS" \
    FLOWPI_GLOBAL_BATCH="$GLOBAL_BATCH" \
    FLOWPI_EXP_NAME="$EXP_NAME" \
    FLOWPI_DATASET_ROOT="$DATASET_ROOT" \
    FLOWPI_FLOW_CACHE_DIR="$FLOW_CACHE_DIR" \
    FLOWPI_ASSETS_DIR="$ASSETS_DIR" \
    FLOWPI_ASSET_ID="$ASSET_ID" \
    FLOWPI_CHECKPOINT_BASE_DIR="$CHECKPOINT_BASE_DIR" \
    FLOWPI_SEA_RAFT_CKPT="$SEA_RAFT_CKPT" \
    FLOWPI_SEA_RAFT_VARIANT="$SEA_RAFT_VARIANT" \
    FLOWPI_SEA_RAFT_ITERS="$SEA_RAFT_ITERS" \
    FLOWPI_WEIGHT_LOADER_PATH="$WEIGHT_LOADER_PATH" \
    FLOWPI_RUN_DIR="$TRAIN_RUN_DIR" \
    bash "$SCRIPT_DIR/train_flowpi_8xh200.sh" 2>&1 | tee -a "$TRAIN_LOG"
    local train_status=${PIPESTATUS[0]}
    set -e
    (( train_status == 0 )) || die "FlowPI training failed; inspect $TRAIN_LOG and $TRAIN_RUN_DIR/run.log"
    log INFO "FlowPI training completed successfully."
}

cleanup() {
    local status=$?
    stop_gpu_monitor
    {
        printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'exit_status=%s\n' "$status"
        printf 'mode=%s\n' "$MODE"
        printf 'flow_cache_dir=%s\n' "$FLOW_CACHE_DIR"
        printf 'train_run_dir=%s\n' "$TRAIN_RUN_DIR"
    } > "$STATUS_LOG"
    if (( status == 0 )); then
        log INFO "Run completed successfully. Status: $STATUS_LOG"
    else
        log ERROR "Run failed with status $status. Inspect $MAIN_LOG and phase logs."
    fi
}

trap cleanup EXIT
trap 'exit 143' INT TERM

log INFO "FlowPI cache/train launcher started: mode=$MODE"
log INFO "SEA-RAFT checkpoint: $SEA_RAFT_CKPT"
log INFO "Run directory: $RUN_DIR"

case "$MODE" in
    cache)
        run_cache
        ;;
    train)
        run_train
        ;;
    all)
        run_cache
        run_train
        ;;
esac
