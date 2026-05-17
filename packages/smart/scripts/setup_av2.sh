#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODE="full"
SCENARIOS=5
SKIP_PREPROCESS=0

CREATE_SPLITS=0

TRAIN_SIZE=16000
VAL_SIZE=4000
SPLIT_SEED=42

TRAIN_SPLIT_OUT="train_split"
VAL_SPLIT_OUT="validation_split"

HASH_FILE_CONTENTS=0

echo "Checking for required Python packages..."
python -c "import av2, torch, pandas, numpy" \
    || { echo "Required Python packages missing."; exit 1; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            MODE="dev"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --scenarios)
            SCENARIOS="$2"
            shift 2
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=1
            shift
            ;;
                --create-splits)
            CREATE_SPLITS=1
            shift
            ;;

        --train-size)
            TRAIN_SIZE="$2"
            shift 2
            ;;

        --val-size)
            VAL_SIZE="$2"
            shift 2
            ;;

        --split-seed)
            SPLIT_SEED="$2"
            shift 2
            ;;

        --train-split-out)
            TRAIN_SPLIT_OUT="$2"
            shift 2
            ;;

        --val-split-out)
            VAL_SPLIT_OUT="$2"
            shift 2
            ;;

        --hash-file-contents)
            HASH_FILE_CONTENTS=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

cd "$SCRIPT_DIR"

echo "==============================================="
echo "SMART AV2 SETUP"
echo "MODE=$MODE"
echo "SCENARIOS=$SCENARIOS"
echo "==============================================="

echo "Starting AV2 download step..."

if [[ "$MODE" == "dev" ]]; then

    bash "$SCRIPT_DIR/scripts/download_av2.sh" \
        --dev \
        --scenarios "$SCENARIOS"

else

    bash "$SCRIPT_DIR/scripts/download_av2.sh" --full

fi

if [[ "$SKIP_PREPROCESS" == "1" ]]; then
    echo "Skipping preprocessing."
    exit 0
fi

echo "==============================================="
echo "Starting preprocessing"
echo "==============================================="

PYTHON_SCRIPT="$SCRIPT_DIR/data_preprocess_av2.py"

mkdir -p "$SCRIPT_DIR/data/processed/train"
mkdir -p "$SCRIPT_DIR/data/processed/val"

echo "Processing TRAIN..."
python "$PYTHON_SCRIPT" \
    --input_dir "$SCRIPT_DIR/data/raw/train" \
    --output_dir "$SCRIPT_DIR/data/processed/train" \
    --num_workers 1

echo "Processing VAL..."
python "$PYTHON_SCRIPT" \
    --input_dir "$SCRIPT_DIR/data/raw/val" \
    --output_dir "$SCRIPT_DIR/data/processed/val" \
    --num_workers 1

echo "CREATE_SPLITS=$CREATE_SPLITS"

if [[ "$CREATE_SPLITS" == "1" ]]; then
    echo "TRAIN_SIZE=$TRAIN_SIZE"
    echo "VAL_SIZE=$VAL_SIZE"
    echo "SPLIT_SEED=$SPLIT_SEED"
fi

if [[ "$CREATE_SPLITS" == "1" ]]; then

    echo "==============================================="
    echo "Creating train/validation dataset splits"
    echo "==============================================="

    export SOURCE_DIR="$SCRIPT_DIR/data/processed/train"
    export SOURCE_VAL_DIR="$SCRIPT_DIR/data/processed/val"

    export TRAIN_OUT="$SCRIPT_DIR/$TRAIN_SPLIT_OUT"
    export VAL_OUT="$SCRIPT_DIR/$VAL_SPLIT_OUT"

    export TRAIN_SIZE
    export VAL_SIZE
    export SEED="$SPLIT_SEED"

    export HASH_FILE_CONTENTS

    python "$SCRIPT_DIR/create_datasets.py" \
        --train_size "$TRAIN_SIZE" \
        --val_size "$VAL_SIZE" \
        --seed "$SPLIT_SEED" \

fi

echo "==============================================="
echo "SMART AV2 setup complete"
echo "==============================================="