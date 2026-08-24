#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <task_name> <task_config> <flowpi_checkpoint> [seed] [extra RoboTwin overrides...]" >&2
  exit 2
fi

POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${POLICY_ROOT}/../RoboTwin}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/home/xiangyushun/miniforge3/envs/RoboTwin/bin/python}"

TASK_NAME="$1"
TASK_CONFIG="$2"
CHECKPOINT="$3"
SEED="${4:-0}"
if [[ $# -ge 4 ]]; then
  shift 4
else
  shift 3
fi

# Three policy GPUs: JAX slow prefix, JAX fast NFE, and Torch SEA-RAFT. If the caller already
# selected CUDA_VISIBLE_DEVICES, the device indices below are relative to that visible list.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${POLICY_ROOT}/scripts:${POLICY_ROOT}/src:${POLICY_ROOT}:${PYTHONPATH:-}"

cd "${ROBOTWIN_ROOT}"
exec "${ROBOTWIN_PYTHON}" script/eval_policy.py \
  --config "${POLICY_ROOT}/scripts/flowpi_robotwin_deploy.yml" \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --ckpt_setting "flowpi" \
  --seed "${SEED}" \
  --policy_name flowpi_robotwin_policy \
  --flowpi_checkpoint "${CHECKPOINT}" \
  "$@"
