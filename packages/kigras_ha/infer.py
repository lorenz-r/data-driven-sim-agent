#!/usr/bin/env python3
"""KiGRAS_HA inference CLI.

Thin command-line wrapper around ``infer_utils.run_inference_pipeline``. Runs
autoregressive CTRA rollouts for one or many Argoverse 2 scenarios and saves
predictions as ``.npz`` and/or native AV2 parquet, plus a GT-vs-prediction PNG.

Structurally this mirrors ``packages/smart/infer2.py`` (a thin CLI) and
``packages/smart/infer2_utils.py`` (the pipeline), adapted to KiGRAS_HA's data
model: the ``ArgoDataset`` loader, the CTRA action-token rollout used in
``eval2.py``, and ``scenario_reconstruction`` for AV2 export.

Examples
--------
    # All raw scenarios in a folder, greedy rollout, save everything
    python infer.py -i data/val -o data/preds

    # A single preprocessed split, 6 stochastic samples for minADE/minFDE
    python infer.py -i data/val_split.npy --raw-data-folder data/val \\
        -o data/preds --num-samples 6 --temperature 0.8 --limit 20
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
import torch

KIGRAS_HA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = KIGRAS_HA_ROOT.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(KIGRAS_HA_ROOT) not in sys.path:
    sys.path.insert(0, str(KIGRAS_HA_ROOT))


try:
    from infer_utils import run_inference_pipeline
except ImportError as exc:
    if exc.name != "infer_utils":
        raise
    from .infer_utils import run_inference_pipeline


DEFAULT_INPUT_PATH = KIGRAS_HA_ROOT / "data" / "val"
DEFAULT_OUTPUT_FOLDER = KIGRAS_HA_ROOT / "data" / "preds"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "last.ckpt"
DEFAULT_NUM_SAMPLES = 1
DEFAULT_TEMPERATURE = 1.0
DEFAULT_VISUALIZATION_LIMIT = 10


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Run KiGRAS_HA inference over a raw AV2 scenario folder or a "
            "preprocessed .npy split."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help=(
            "Raw AV2 scenario folder or preprocessed .npy split "
            f"(default: {DEFAULT_INPUT_PATH})"
        ),
    )
    parser.add_argument(
        "--raw-data-folder",
        type=str,
        default=None,
        help=(
            "Folder with raw AV2 scenario dirs, used to recover AV2 metadata "
            "when --input is a .npy split (defaults to --input when it is a folder)."
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
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--num-samples",
        "-k",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=(
            "Rollout samples per scenario for minADE/minFDE. Use 1 for greedy "
            f"argmax (default: {DEFAULT_NUM_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Softmax temperature for sampling (ignored when --num-samples=1).",
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
        "--format",
        "-f",
        choices=["npz", "av2", "both"],
        default="both",
        help="Prediction output format (default: both).",
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        default=True,
        help="Save GT-vs-prediction PNG figures (default).",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Run inference without saving visualizations.",
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
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    visualization_limit = args.visualization_limit
    if visualization_limit is not None and visualization_limit < 0:
        visualization_limit = None

    run_inference_pipeline(
        input_path=args.input,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_samples=args.num_samples,
        temperature=args.temperature,
        raw_data_folder=args.raw_data_folder,
        idx=args.idx,
        limit=args.limit,
        visualize=args.visualize,
        visualization_limit=visualization_limit,
        format_av2=args.format in ("av2", "both"),
        format_npz=args.format in ("npz", "both"),
    )


if __name__ == "__main__":
    main()
