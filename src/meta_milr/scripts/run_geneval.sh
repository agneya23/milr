#!/usr/bin/env bash
set -euo pipefail

META_MILR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$(cd "$META_MILR_DIR/.." && pwd)"
CONFIG_PATH="${META_MILR_CONFIG:-$META_MILR_DIR/configs/geneval.json}"
LOG_DIR="$SRC_DIR/../outputs/meta_milr/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TIMESTAMP}_geneval.log"
PID_FILE="$LOG_DIR/${TIMESTAMP}_geneval.pid"
CONDA_ROOT="/ivi/zfs/s0/original_homes/ydu/miniconda3"
CONDA_ENV="$CONDA_ROOT/envs/milr_latentseek"

mkdir -p "$LOG_DIR"
cd "$SRC_DIR"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python -m meta_milr.main_meta_milr \
  --config "$CONFIG_PATH" "$@" \
  >"$LOG_FILE" 2>&1 </dev/null &

PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
printf 'Started Meta-MILR (PID %s)\nLog: %s\n' "$PID" "$LOG_FILE"
