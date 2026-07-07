#!/usr/bin/env bash
set -euo pipefail

# Sanity check for the SMART model: preprocess scenarios, run training,
# and validate that loss decreased.
#
# Run from the repository root.
#
# Configuration via environment variables (defaults shown):
#   CONFIG             training config relative to packages/smart  [configs/train/sanity_check.yaml]
#   CPU                force CPU training                          [true]
#   PREPROCESS_INPUT   preprocess input dir (rel. to packages/smart) [data/raw/train]
#   PREPROCESS_OUTPUT  preprocess output dir (rel. to packages/smart) [data/processed/train]
#   METRICS_PATH       metrics JSON output  (rel. to packages/smart) [metrics.json]
#   MIN_ACCURACY       minimum required accuracy (empty = skip check) []

CONFIG="${CONFIG:-configs/train/sanity_check.yaml}"
CPU="${CPU:-true}"
PREPROCESS_INPUT="${PREPROCESS_INPUT:-data/raw/train}"
PREPROCESS_OUTPUT="${PREPROCESS_OUTPUT:-data/processed/train}"
METRICS_PATH="${METRICS_PATH:-metrics.json}"
MIN_ACCURACY="${MIN_ACCURACY:-}"

# Ensure the pinned Rust toolchain is installed before uv sync invokes maturin.
if command -v rustup &>/dev/null; then
    rustup toolchain install
fi

uv sync --all-packages

(
  cd packages/smart

  uv run python -c "
import smart_preprocess, sys
n_ok, n_total = smart_preprocess.process_directory('$PREPROCESS_INPUT', '$PREPROCESS_OUTPUT')
print(f'Preprocessed {n_ok}/{n_total} scenarios.')
if n_ok == 0: sys.exit(1)
"

  if [ "$CPU" = "true" ]; then
    uv run python train.py --cpu --return_metrics "$METRICS_PATH" --config "$CONFIG"
  else
    uv run python train.py --return_metrics "$METRICS_PATH" --config "$CONFIG"
  fi

  cat "$METRICS_PATH"
  loss_dec=$(jq '.loss_decrease' "$METRICS_PATH")
  awk -v d="$loss_dec" \
    'BEGIN { if (d < 0) { print "\nFAIL: loss increased by " -d; exit 1 } }'

  if [ -n "$MIN_ACCURACY" ]; then
    accuracy=$(jq '.accuracy' "$METRICS_PATH")
    awk -v a="$accuracy" -v m="$MIN_ACCURACY" \
      'BEGIN { if (a < m) { print "\nFAIL: accuracy " a " < " m; exit 1 } }'
  fi
)
