#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <task_name> <task_config> <flowpi_checkpoint> [seed] [extra DOMINO/FlowPi overrides...]" >&2
  exit 2
fi

POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOMINO_ROOT="${DOMINO_ROOT:-${POLICY_ROOT}/../DOMINO}"

# The policy environment must contain JAX/OpenPI/SEA-RAFT. The DOMINO environment only needs
# the simulator and the dependency-light flowpi_domino_client module. They may be the same Python
# environment, but they remain separate OS processes and separate CUDA visibility domains.
POLICY_PYTHON="${POLICY_PYTHON:-/home/xiangyushun/miniforge3/envs/RoboTwin/bin/python}"
DOMINO_PYTHON="${DOMINO_PYTHON:-/home/xiangyushun/miniforge3/envs/RoboTwin/bin/python}"

TASK_NAME="$1"
TASK_CONFIG="$2"
CHECKPOINT="$3"
SEED="${4:-0}"
if [[ $# -ge 4 ]]; then
  shift 4
else
  shift 3
fi

# The policy server sees three logical devices: slow JAX, fast JAX, and Torch SEA-RAFT. The
# DOMINO client sees only the remaining simulation GPU. All IDs are physical IDs unless the
# caller deliberately provides a different CUDA_VISIBLE_DEVICES mapping to the child process.
POLICY_GPU_IDS="${FLOWPI_POLICY_GPUS:-0,1,2}"
DOMINO_GPU_ID="${FLOWPI_DOMINO_GPU:-3}"
PORT="${FLOWPI_PORT:-29876}"
# The first action may include JAX compilation and online SEA-RAFT initialization. Keep the
# dependency-light DOMINO client alive long enough for that first response; callers can lower or
# raise this without changing the benchmark checkout.
FLOWPI_DOMINO_TIMEOUT="${FLOWPI_DOMINO_TIMEOUT:-180}"
FLOWPI_DOMINO_CONTROL_HZ="${FLOWPI_DOMINO_CONTROL_HZ:-50}"
# The FlowPi launcher intentionally uses fixed-step high-rate control. The standalone client
# defaults to official DOMINO take_action semantics unless a mode is selected explicitly. Keep
# the old boolean override working when callers have not migrated to CONTROL_MODE yet.
FLOWPI_DOMINO_CONTROL_MODE="${FLOWPI_DOMINO_CONTROL_MODE:-}"
if [[ -z "${FLOWPI_DOMINO_CONTROL_MODE}" ]]; then
  if [[ "${FLOWPI_DOMINO_DIRECT_CONTROL:-1}" == "0" || "${FLOWPI_DOMINO_DIRECT_CONTROL:-1}" == "false" || "${FLOWPI_DOMINO_DIRECT_CONTROL:-1}" == "no" || "${FLOWPI_DOMINO_DIRECT_CONTROL:-1}" == "off" ]]; then
    FLOWPI_DOMINO_CONTROL_MODE="official"
  else
    FLOWPI_DOMINO_CONTROL_MODE="direct"
  fi
fi
FLOWPI_DOMINO_DIRECT_CONTROL="${FLOWPI_DOMINO_DIRECT_CONTROL:-}"
FLOWPI_DOMINO_MAX_READY_CHUNKS="${FLOWPI_DOMINO_MAX_READY_CHUNKS:-1}"
FLOWPI_DOMINO_MAX_PENDING_OBS="${FLOWPI_DOMINO_MAX_PENDING_OBS:-16}"
FLOWPI_DOMINO_WORKER_JOIN_TIMEOUT="${FLOWPI_DOMINO_WORKER_JOIN_TIMEOUT:-10}"
FLOWPI_DOMINO_COPY_OBS="${FLOWPI_DOMINO_COPY_OBS:-0}"
# Consume the d actions emitted by each policy NFE in one callback. Every action still advances
# the configured fixed simulator interval (normally 5 SAPIEN steps = 20 ms), while avoiding a
# redundant three-camera get_obs() between consecutive actions in the same trained action chunk.
# Set to 0 only for a one-action-per-render diagnostic run.
FLOWPI_DOMINO_DRAIN_ACTION_CHUNK="${FLOWPI_DOMINO_DRAIN_ACTION_CHUNK:-1}"
# Bound repeated hold-last fallback controls to the same three-tick deployment cadence used by
# the action chunks. Set to 1 to disable this wall-clock batching diagnostic.
FLOWPI_DOMINO_HOLD_LAST_BURST="${FLOWPI_DOMINO_HOLD_LAST_BURST:-3}"
# Catch up the physical 50 Hz clock after a slow multi-camera get_obs() by running the exact
# number of overdue simulator intervals in the next callback. The cap bounds fallback bursts.
FLOWPI_DOMINO_WALL_CLOCK_CATCHUP="${FLOWPI_DOMINO_WALL_CLOCK_CATCHUP:-1}"
FLOWPI_DOMINO_MAX_CATCHUP_ACTIONS="${FLOWPI_DOMINO_MAX_CATCHUP_ACTIONS:-8}"
FLOWPI_DOMINO_STATE_PROCESS="${FLOWPI_DOMINO_STATE_PROCESS:-1}"
FLOWPI_DOMINO_FAST_STATE_HZ="${FLOWPI_DOMINO_FAST_STATE_HZ:-25}"
FLOWPI_DOMINO_STATE_IDLE_TIMEOUT="${FLOWPI_DOMINO_STATE_IDLE_TIMEOUT:-5}"
FLOWPI_DOMINO_THROTTLE_OBS="${FLOWPI_DOMINO_THROTTLE_OBS:-1}"
FLOWPI_DOMINO_IMAGE_HZ="${FLOWPI_DOMINO_IMAGE_HZ:-10}"
FLOWPI_DOMINO_COMPRESS_RGB="${FLOWPI_DOMINO_COMPRESS_RGB:-1}"
FLOWPI_DOMINO_JPEG_QUALITY="${FLOWPI_DOMINO_JPEG_QUALITY:-95}"
# Keep the policy-side latency-to-d conversion in the same control-clock units as the simulator.
FLOWPI_CONTROL_HZ="${FLOWPI_CONTROL_HZ:-${FLOWPI_DOMINO_CONTROL_HZ}}"
FLOWPI_METRICS_DIR="${FLOWPI_METRICS_DIR:-${POLICY_ROOT}/../data/flowpi_metrics/$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "${FLOWPI_METRICS_DIR}"
FLOWPI_METRICS_DIR="$(cd "${FLOWPI_METRICS_DIR}" && pwd)"
FLOWPI_METRICS_PATH="${FLOWPI_POLICY_METRICS_PATH:-${FLOWPI_METRICS_PATH:-${FLOWPI_METRICS_DIR}/policy_runtime.json}}"
FLOWPI_DOMINO_METRICS_PATH="${FLOWPI_DOMINO_METRICS_PATH:-${FLOWPI_METRICS_DIR}/domino_client.json}"
FLOWPI_METRICS_FLUSH_EVERY="${FLOWPI_METRICS_FLUSH_EVERY:-10}"

if [[ ! -d "${DOMINO_ROOT}" ]]; then
  echo "DOMINO checkout not found: ${DOMINO_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" && ! -d "${CHECKPOINT}" ]]; then
  echo "FlowPi checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
CHECKPOINT="$(realpath "${CHECKPOINT}")"

export PYTHONPATH="${POLICY_ROOT}/scripts:${POLICY_ROOT}/src:${POLICY_ROOT}:${PYTHONPATH:-}"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${CLIENT_PID:-}" ]] && kill -0 "${CLIENT_PID}" 2>/dev/null; then
    kill "${CLIENT_PID}" 2>/dev/null || true
    wait "${CLIENT_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -f "${FLOWPI_METRICS_PATH}" ]]; then
    echo "[FlowPi] human-readable inference metrics:"
    "${POLICY_PYTHON}" "${POLICY_ROOT}/scripts/print_flowpi_metrics.py" \
      "${FLOWPI_METRICS_PATH}" || true
  fi
  if [[ -f "${FLOWPI_DOMINO_METRICS_PATH}" ]]; then
    echo "[FlowPi] human-readable DOMINO control metrics:"
    "${POLICY_PYTHON}" "${POLICY_ROOT}/scripts/print_flowpi_metrics.py" \
      "${FLOWPI_DOMINO_METRICS_PATH}" --title "DOMINO 控制指标 / DOMINO Control Metrics" || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

echo "[FlowPi] starting policy server on CUDA_VISIBLE_DEVICES=${POLICY_GPU_IDS}"
echo "[FlowPi] metrics: ${FLOWPI_METRICS_PATH} and ${FLOWPI_DOMINO_METRICS_PATH}"
export FLOWPI_METRICS_PATH FLOWPI_DOMINO_METRICS_PATH FLOWPI_METRICS_FLUSH_EVERY FLOWPI_CONTROL_HZ
CUDA_VISIBLE_DEVICES="${POLICY_GPU_IDS}" \
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
"${POLICY_PYTHON}" "${POLICY_ROOT}/scripts/flowpi_policy_model_server.py" \
  --port "${PORT}" \
  --config "${POLICY_ROOT}/scripts/flowpi_domino_server.yml" \
  --overrides \
  --policy_name flowpi_robotwin_policy \
  --flowpi_checkpoint "${CHECKPOINT}" \
  "$@" &
SERVER_PID=$!

# Give the server a chance to bind before the client starts its retry loop. Model loading may take
# considerably longer; the DOMINO client continues retrying while the server loads the checkpoint.
sleep 1
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  wait "${SERVER_PID}"
  exit 1
fi

echo "[FlowPi] starting DOMINO client on CUDA_VISIBLE_DEVICES=${DOMINO_GPU_ID}"
(
  export CUDA_VISIBLE_DEVICES="${DOMINO_GPU_ID}"
  export FLOWPI_DOMINO_TIMEOUT FLOWPI_DOMINO_CONTROL_HZ FLOWPI_DOMINO_CONTROL_MODE FLOWPI_DOMINO_DIRECT_CONTROL FLOWPI_DOMINO_MAX_READY_CHUNKS FLOWPI_DOMINO_MAX_PENDING_OBS FLOWPI_DOMINO_WORKER_JOIN_TIMEOUT FLOWPI_DOMINO_COPY_OBS FLOWPI_DOMINO_DRAIN_ACTION_CHUNK FLOWPI_DOMINO_HOLD_LAST_BURST FLOWPI_DOMINO_WALL_CLOCK_CATCHUP FLOWPI_DOMINO_MAX_CATCHUP_ACTIONS FLOWPI_DOMINO_STATE_PROCESS FLOWPI_DOMINO_FAST_STATE_HZ FLOWPI_DOMINO_STATE_IDLE_TIMEOUT FLOWPI_DOMINO_THROTTLE_OBS FLOWPI_DOMINO_IMAGE_HZ FLOWPI_DOMINO_COMPRESS_RGB FLOWPI_DOMINO_JPEG_QUALITY
  cd "${DOMINO_ROOT}"
  exec "${DOMINO_PYTHON}" script/eval_policy_client.py \
    --port "${PORT}" \
    --config "${POLICY_ROOT}/scripts/flowpi_domino_client.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting flowpi \
    --seed "${SEED}" \
    --policy_name flowpi_domino_client \
    "$@"
) &
CLIENT_PID=$!

wait "${CLIENT_PID}"
