#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DOWNLOAD_DIR="$SCRIPT_DIR/data/downloads"
RAW_DIR="$SCRIPT_DIR/data/raw"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$RAW_DIR"

TRAIN_URL="https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/train.tar"
VAL_URL="https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/val.tar"

MODE="full"
SCENARIOS=5

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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done


echo "==============================================="
echo "AV2 Download Configuration"
echo "MODE=$MODE"
echo "SCENARIOS=$SCENARIOS"
echo "DOWNLOAD_DIR=$DOWNLOAD_DIR"
echo "RAW_DIR=$RAW_DIR"
echo "==============================================="


if [[ "$MODE" == "dev" ]]; then

    echo "Running DEV/PARTIAL dataset extraction..."

    python "$SCRIPT_DIR/partial_tar_extract.py" \
        --url "$TRAIN_URL" \
        --scenarios "$SCENARIOS" \
        --out "$RAW_DIR"

    python "$SCRIPT_DIR/partial_tar_extract.py" \
        --url "$VAL_URL" \
        --scenarios "$SCENARIOS" \
        --out "$RAW_DIR"

    echo "DEV dataset extraction complete."
    exit 0
fi


TRAIN_TAR="$DOWNLOAD_DIR/train.tar"
VAL_TAR="$DOWNLOAD_DIR/val.tar"


if [[ ! -f "$TRAIN_TAR" ]]; then
    echo "Downloading AV2 train.tar..."
    wget -c "$TRAIN_URL" -O "$TRAIN_TAR"
else
    echo "train.tar already exists. Skipping download."
fi


if [[ ! -f "$VAL_TAR" ]]; then
    echo "Downloading AV2 val.tar..."
    wget -c "$VAL_URL" -O "$VAL_TAR"
else
    echo "val.tar already exists. Skipping download."
fi


if [[ ! -d "$RAW_DIR/train" ]]; then
    echo "Extracting train.tar..."
    tar -xf "$TRAIN_TAR" -C "$RAW_DIR"
else
    echo "Raw train directory already exists. Skipping extraction."
fi


if [[ ! -d "$RAW_DIR/val" ]]; then
    echo "Extracting val.tar..."
    tar -xf "$VAL_TAR" -C "$RAW_DIR"
else
    echo "Raw val directory already exists. Skipping extraction."
fi


echo "==============================================="
echo "AV2 download/extraction complete"
echo "==============================================="