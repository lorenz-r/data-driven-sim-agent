#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run SMART inference, convert predictions back to AV2, and visualize the result.

Usage:
  ./run_av2_inference_pipeline.sh --scene-id SCENE_ID --checkpoint CHECKPOINT [options]

Required:
  -s, --scene-id ID         AV2 scenario id, e.g. 1b0c5826-1614-4996-8a12-f6df12beaa0e
  -c, --checkpoint PATH     SMART checkpoint to load

Options:
      --input-pkl PATH      SMART-format input pickle
                            default: packages/smart/data/av2_processed/training/<scene-id>.pkl
      --map-json PATH       Original AV2 static map json
                            default: packages/smart/data/av2_partial_py/train/<scene-id>/log_map_archive_<scene-id>.json
      --output-root DIR     Root directory for generated outputs
                            default: packages/smart/data/preds
      --config PATH         Inference config
                            default: packages/smart/configs/validation/validation_scalable.yaml
      --device DEVICE       Inference device: cpu or cuda
                            default: cpu
      --timestep N          Static visualization timestep
                            default: 109
      --animate             Also save an animated GIF
  -h, --help                Show this help

Examples:
  ./run_av2_inference_pipeline.sh \
    --scene-id 1b0c5826-1614-4996-8a12-f6df12beaa0e \
    --checkpoint packages/smart/smart_checkpoints/epoch=199.ckpt

  ./run_av2_inference_pipeline.sh \
    --scene-id 1b0c5826-1614-4996-8a12-f6df12beaa0e \
    --checkpoint packages/smart/smart_checkpoints/epoch=199.ckpt \
    --device cuda \
    --animate
EOF
}

SCENE_ID=""
CHECKPOINT=""
INPUT_PKL=""
MAP_JSON=""
OUTPUT_ROOT="packages/smart/data/preds"
CONFIG="packages/smart/configs/validation/validation_scalable.yaml"
DEVICE="cpu"
TIMESTEP="109"
ANIMATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--scene-id)
      SCENE_ID="${2:-}"
      shift 2
      ;;
    -c|--checkpoint)
      CHECKPOINT="${2:-}"
      shift 2
      ;;
    --input-pkl)
      INPUT_PKL="${2:-}"
      shift 2
      ;;
    --map-json)
      MAP_JSON="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --timestep)
      TIMESTEP="${2:-}"
      shift 2
      ;;
    --animate)
      ANIMATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SCENE_ID" || -z "$CHECKPOINT" ]]; then
  echo "Both --scene-id and --checkpoint are required." >&2
  usage >&2
  exit 1
fi

if [[ "$DEVICE" != "cpu" && "$DEVICE" != "cuda" ]]; then
  echo "--device must be either 'cpu' or 'cuda'." >&2
  exit 1
fi

if [[ -z "$INPUT_PKL" ]]; then
  INPUT_PKL="packages/smart/data/av2_processed/training/${SCENE_ID}.pkl"
fi

if [[ -z "$MAP_JSON" ]]; then
  MAP_JSON="packages/smart/data/av2_partial_py/train/${SCENE_ID}/log_map_archive_${SCENE_ID}.json"
fi

SMART_DIR="${OUTPUT_ROOT}/${SCENE_ID}/smart"
AV2_DIR="${OUTPUT_ROOT}/${SCENE_ID}/av2"
VIS_DIR="${OUTPUT_ROOT}/${SCENE_ID}/vis"
SMART_PRED_PKL="${SMART_DIR}/${SCENE_ID}.pkl"
AV2_PARQUET="${AV2_DIR}/scenario_${SCENE_ID}.parquet"

for required_path in "$INPUT_PKL" "$CHECKPOINT" "$CONFIG" "$MAP_JSON"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required file not found: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$SMART_DIR" "$AV2_DIR" "$VIS_DIR"

echo "1/4 Running SMART inference..."
uv run python packages/smart/infer.py \
  --input "$INPUT_PKL" \
  --output "$OUTPUT_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --device "$DEVICE"

echo "2/4 Converting SMART predictions to AV2 parquet..."
uv run python packages/smart/convert_smart_av2.py \
  --input "$SMART_PRED_PKL" \
  --output "$AV2_PARQUET"

echo "3/4 Copying static map next to the converted parquet..."
cp "$MAP_JSON" "$AV2_DIR/"

echo "4/4 Saving static visualization..."
uv run python packages/smart/visualisation/av2_format_vis.py \
  --input "$AV2_PARQUET" \
  --output "$VIS_DIR" \
  --timestep "$TIMESTEP"

if [[ "$ANIMATE" -eq 1 ]]; then
  echo "Saving animated visualization..."
  uv run python packages/smart/visualisation/av2_format_vis.py \
    --input "$AV2_PARQUET" \
    --output "$VIS_DIR" \
    --animate
fi

echo
echo "Done."
echo "SMART prediction: $SMART_PRED_PKL"
echo "AV2 parquet:      $AV2_PARQUET"
echo "Visualizations:   $VIS_DIR"
