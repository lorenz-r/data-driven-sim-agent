#!/usr/bin/env bash

set -euxo pipefail

# Determine package root (packages/smart) relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Accept optional positional arguments
INPUT_DIR=${1:-"$SCRIPT_DIR/data/raw/train"}
OUTPUT_DIR=${2:-"$SCRIPT_DIR/data/processed/train"}

PYTHON_SCRIPT=${PYTHON_SCRIPT:-"$SCRIPT_DIR/data_preprocess_av2.py"}

# place the venv in the package root (packages/smart/.venv_av2) by default
VENV_DIR=${VENV_DIR:-"$SCRIPT_DIR/.venv_av2"}

echo "INPUT_DIR=$INPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"

# create a dedicated venv for av2 if it doesn't exist yet
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv for AV2..."
    uv venv "$VENV_DIR" --python 3.11
fi

# install dependencies
echo "Installing/Syncing dependencies into $VENV_DIR..."
uv pip install \
    --python "$VENV_DIR/bin/python" \
    av2 \
    numpy==1.23.5 \
    torch \
    tqdm \
    pandas \
    opencv-python-headless

echo "Starting conversion: AV2 Parquet -> SMART Pickle..."
"$VENV_DIR/bin/python" "$PYTHON_SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 1

echo "------------------------------------------------------------------------"
echo "Done! Processed file is in $OUTPUT_DIR"
echo "------------------------------------------------------------------------"