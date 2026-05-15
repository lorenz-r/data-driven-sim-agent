#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODE="full"
SCENARIOS=5
SKIP_PREPROCESS=0

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


if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    python -m pip install uv
fi


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


bash "$SCRIPT_DIR/scripts/run_preprocess.sh" \
    "$SCRIPT_DIR/data/raw/train" \
    "$SCRIPT_DIR/data/processed/train"


bash "$SCRIPT_DIR/scripts/run_preprocess.sh" \
    "$SCRIPT_DIR/data/raw/val" \
    "$SCRIPT_DIR/data/processed/val"


echo "==============================================="
echo "SMART AV2 setup complete"
echo "==============================================="