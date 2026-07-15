#!/usr/bin/env python3
"""KiGRAS-AM inference CLI.

Runs observer-centric KiGRAS-AM rollout over one or more preprocessed AM
scenarios and writes a SMART-compatible collected ``.npz`` archive. The archive
keeps full-scene ground truth/context, but only observer-agent rows contain valid
model predictions. See ``infer_utils.py`` for the schema contract and rationale.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

KIGRAS_AM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = KIGRAS_AM_ROOT.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(KIGRAS_AM_ROOT) not in sys.path:
    sys.path.insert(0, str(KIGRAS_AM_ROOT))

try:
    from infer_utils import run_inference_pipeline
except ImportError as exc:
    if exc.name != "infer_utils":
        raise
    from .infer_utils import run_inference_pipeline


DEFAULT_INPUT_PATH = REPO_ROOT / "av2_data" / "kigras_processed" / "val"
DEFAULT_OUTPUT_FOLDER = KIGRAS_AM_ROOT / "data" / "preds"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "last.ckpt"
DEFAULT_VISUALIZATION_LIMIT = 10


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Run KiGRAS-AM inference over a preprocessed AM .npy split or split "
            "directory and save SMART-compatible full-scene prediction artifacts."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help=(
            "AM preprocessed .npy split or split directory "
            f"(default: {DEFAULT_INPUT_PATH})"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_FOLDER),
        help=f"Output directory for predictions (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--raw-data-folder",
        type=str,
        default=None,
        help=(
            "Optional raw AV2 scenario folder stored in metadata for downstream "
            "map-aware evaluation. It is not used to load AM inference input."
        ),
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Scenarios rolled out per batched forward pass (default: 1).",
    )
    parser.add_argument(
        "--idx",
        type=int,
        nargs="+",
        default=None,
        help="Run only these scenario indices (default: all, subject to --limit).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of scenarios to run (default: all).",
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        default=True,
        help="Save full-scene GT plus observer prediction PNG figures (default).",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Run inference without saving PNG visualizations.",
    )
    parser.add_argument(
        "--visualization-limit",
        type=int,
        default=DEFAULT_VISUALIZATION_LIMIT,
        help=(
            "Maximum number of scenarios to visualize. Use a negative value for "
            f"all (default: {DEFAULT_VISUALIZATION_LIMIT})."
        ),
    )
    parser.add_argument(
        "--grid-power",
        type=float,
        default=1.0,
        help="CTRA action-grid power used by preprocessing/training (default: 1.0).",
    )
    parser.add_argument(
        "--max-accel",
        type=float,
        default=5.0,
        help="CTRA maximum acceleration in m/s^2 (default: 5.0).",
    )
    parser.add_argument(
        "--max-yaw-rate",
        type=float,
        default=1.5,
        help="CTRA maximum yaw rate in rad/s (default: 1.5).",
    )
    parser.add_argument(
        "--delta-time",
        type=float,
        default=0.1,
        help="CTRA rollout timestep in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="CTRA inverse-solve max_iter stored for config parity (default: 20).",
    )
    args = parser.parse_args()

    visualization_limit = args.visualization_limit
    if visualization_limit is not None and visualization_limit < 0:
        visualization_limit = None

    run_inference_pipeline(
        input_path=args.input,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
        device=args.device,
        raw_data_folder=args.raw_data_folder,
        idx=args.idx,
        limit=args.limit,
        visualize=args.visualize,
        visualization_limit=visualization_limit,
        batch_size=args.batch_size,
        grid_power=args.grid_power,
        max_accel=args.max_accel,
        max_yaw_rate=args.max_yaw_rate,
        delta_time=args.delta_time,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
