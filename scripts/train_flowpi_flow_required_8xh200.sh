#!/usr/bin/env bash
# Comparison recipe for FlowPI modality-laziness mitigation.
#
# It uses a separate experiment/checkpoint namespace, so the original
# flowpi_8xh200 logs and step-2000 checkpoint remain untouched. The base launcher still owns
# all validation, durable logging, GPU telemetry, resume, and checkpoint behavior.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# On selected samples, force the slow VLM prefix into the stale tail [5, 10]. Flow cache age
# remains independently sampled from the existing 0..3 distribution.
export FLOWPI_EXP_NAME="${FLOWPI_EXP_NAME:-flowpi_8xh200_flow_required}"
export FLOWPI_LOG_ROOT="${FLOWPI_LOG_ROOT:-$SCRIPT_DIR/../logs/flowpi_cache_train}"
export FLOWPI_FLOW_REQUIRED_PROB="${FLOWPI_FLOW_REQUIRED_PROB:-0.5}"
export FLOWPI_FLOW_REQUIRED_VLM_DELAY_MIN="${FLOWPI_FLOW_REQUIRED_VLM_DELAY_MIN:-5}"

# Start a fresh comparison run. Set these explicitly when resuming this comparison experiment.
export FLOWPI_RESUME="${FLOWPI_RESUME:-0}"
export FLOWPI_OVERWRITE="${FLOWPI_OVERWRITE:-0}"
export FLOWPI_SAVE_INTERVAL="${FLOWPI_SAVE_INTERVAL:-2000}"

# Target |tanh(flow_gate)| ~= 1e-2 at initialization, without changing the parameter layout.
export FLOWPI_FLOW_GATE_INIT="${FLOWPI_FLOW_GATE_INIT:-0.01}"

exec bash "$SCRIPT_DIR/train_flowpi_8xh200.sh" "$@"
