#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$(cd "$BASELINE_DIR/.." && pwd)"
CONFIG_PATH="${BASELINE_MILR_CONFIG:-$BASELINE_DIR/configs/geneval.json}"
OUTPUT_ROOT="$SRC_DIR/../outputs/baseline_milr"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${BASELINE_MILR_RUN_NAME:-geneval_milr5_seed42}"
RUN_DIR="$OUTPUT_ROOT/${TIMESTAMP}_${RUN_NAME}"
LOG_FILE="$RUN_DIR/run.log"
PID_FILE="$RUN_DIR/run.pid"
CONDA_ROOT="/ivi/zfs/s0/original_homes/ydu/miniconda3"
CONDA_ENV="$CONDA_ROOT/envs/milr_latentseek"

mkdir -p "$RUN_DIR"
cd "$SRC_DIR"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python main_janus.py \
  --config "$CONFIG_PATH" \
  --output_dir "$RUN_DIR" "$@" \
  >"$LOG_FILE" 2>&1 </dev/null &

PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
printf 'Started baseline MILR (PID %s)\nOutput: %s\nLog: %s\n' \
  "$PID" "$RUN_DIR" "$LOG_FILE"
