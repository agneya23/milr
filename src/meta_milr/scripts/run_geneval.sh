#!/usr/bin/env bash
set -euo pipefail

META_MILR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$(cd "$META_MILR_DIR/.." && pwd)"
CONFIG_PATH="${META_MILR_CONFIG:-$META_MILR_DIR/configs/geneval.json}"
LOG_DIR="$SRC_DIR/../outputs/meta_milr/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"
cd "$SRC_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python -m meta_milr.main_meta_milr \
  --config "$CONFIG_PATH" "$@" 2>&1 | tee "$LOG_DIR/${TIMESTAMP}_geneval.log"
