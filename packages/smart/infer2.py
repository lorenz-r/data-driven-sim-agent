import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

SMART_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SMART_ROOT.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

try:
    from infer2_utils import run_inference_pipeline
except ImportError as exc:
    if exc.name != "infer2_utils":
        raise
    from .infer2_utils import run_inference_pipeline


DEFAULT_INPUT_PATH = SMART_ROOT / "data" / "processed" / "current" / "val"
DEFAULT_RAW_DATA_FOLDER = SMART_ROOT / "data" / "raw" / "val"
DEFAULT_OUTPUT_FOLDER = SMART_ROOT / "data" / "preds"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "model.ckpt"
DEFAULT_BATCH_VISUALIZATION_LIMIT = 10
DEFAULT_ROLLOUTS_PER_SCENARIO = 16


def _prompt_path(label: str, default: Path | str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or str(default)


def _prompt_text(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _prompt_optional_int(label: str, default: Optional[int]) -> Optional[int]:
    default_label = "all" if default is None else str(default)
    value = input(f"{label} [{default_label}]: ").strip().lower()
    if not value:
        return default
    if value in ("all", "no-limit", "unlimited"):
        return None
    if value in ("none", "no"):
        return 0
    try:
        parsed = int(value)
    except ValueError:
        print(f"Invalid integer '{value}', using {default_label}.")
        return default
    if parsed < 0:
        print(f"Negative values are not supported, using {default_label}.")
        return default
    return parsed


def _prompt_bool(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    value = input(f"{label} [{suffix}]: ").strip().lower()
    if not value:
        return default
    return value in ("y", "yes", "true", "1", "on")


def _run_setup_dialog(args) -> None:
    print("\nSMART inference setup")
    print("Press Enter at any prompt to keep the shown default.\n")
    args.input = _prompt_path("Processed scenario file or folder", args.input)
    args.raw_data_folder = _prompt_path(
        "Raw AV2 data folder (ground truth parquet + map json)",
        args.raw_data_folder,
    )
    args.output = _prompt_path("Output folder", args.output)
    args.visualize = _prompt_bool(
        "Save ground-truth and prediction GIFs",
        args.visualize,
    )
    if args.visualize:
        args.batch_visualization_limit = _prompt_optional_int(
            "How many scenarios should be visualized",
            args.batch_visualization_limit,
        )
    args.checkpoint = _prompt_path("Checkpoint path", args.checkpoint)
    args.device = _prompt_text("Device (cpu/cuda)", args.device)
    args.sample_tokens = _prompt_bool("Sample rollout tokens", args.sample_tokens)
    args.rollouts_per_scenario = _prompt_optional_int(
        "Rollouts per scenario",
        args.rollouts_per_scenario,
    )
    if args.rollouts_per_scenario is None:
        args.rollouts_per_scenario = DEFAULT_ROLLOUTS_PER_SCENARIO


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Run SMART inference for one processed scenario file or all .pkl "
            "scenarios in a processed scenario folder."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help=(
            "Processed SMART scenario .pkl file or folder of .pkl files "
            f"(default: {DEFAULT_INPUT_PATH})"
        ),
    )
    parser.add_argument(
        "--processed-folder",
        type=str,
        default=None,
        help="Deprecated alias for --input when the input is a folder.",
    )
    parser.add_argument(
        "--raw-data-folder",
        type=str,
        default=str(DEFAULT_RAW_DATA_FOLDER),
        help=(
            "Folder containing raw AV2 scenario folders with scenario parquet and "
            f"map json files (default: {DEFAULT_RAW_DATA_FOLDER})"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_FOLDER),
        help=f"Output directory for predictions and visualizations (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["smart", "av2", "both"],
        default="both",
        help="Output format: smart, av2, or both (default: both)",
    )
    parser.add_argument(
        "--batch-visualization-limit",
        type=int,
        default=DEFAULT_BATCH_VISUALIZATION_LIMIT,
        help=(
            "Maximum number of scenarios to save GIFs for. Use 0 for none or a "
            f"negative value for all (default: {DEFAULT_BATCH_VISUALIZATION_LIMIT})."
        ),
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        default=True,
        help="Save gt_<scenario>.GIF and pred_<scenario>.GIF files (default)",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Run inference without saving visualization GIFs",
    )
    parser.add_argument(
        "--setup-dialog",
        dest="setup_dialog",
        action="store_true",
        default=None,
        help="Run the interactive setup dialogue",
    )
    parser.add_argument(
        "--no-setup-dialog",
        dest="setup_dialog",
        action="store_false",
        help="Skip the interactive setup dialogue and use configured defaults/flags",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (shorthand for --device cpu)",
    )
    parser.add_argument(
        "--batch-directory",
        type=str,
        default=None,
        help="Deprecated alias for --input when the input is a folder.",
    )
    parser.add_argument(
        "--sample-tokens",
        dest="sample_tokens",
        action="store_true",
        default=True,
        help=("Sample among top-k rollout tokens for diverse rollouts (default)."),
    )
    parser.add_argument(
        "--no-sample-tokens",
        dest="sample_tokens",
        action="store_false",
        help="Use deterministic top-1 tokens for every rollout.",
    )
    parser.add_argument(
        "--rollouts-per-scenario",
        type=int,
        default=DEFAULT_ROLLOUTS_PER_SCENARIO,
        help=(
            "Number of rollouts to generate for each scenario "
            f"(default: {DEFAULT_ROLLOUTS_PER_SCENARIO})."
        ),
    )

    args = parser.parse_args()

    if args.cpu:
        args.device = "cpu"

    if args.processed_folder is not None:
        args.input = args.processed_folder
    if args.batch_directory is not None:
        args.input = args.batch_directory

    if args.setup_dialog is None and sys.stdin.isatty():
        args.setup_dialog = _prompt_bool("Run setup dialogue", False)
    elif args.setup_dialog is None:
        args.setup_dialog = False

    if args.setup_dialog:
        _run_setup_dialog(args)

    if (
        args.batch_visualization_limit is not None
        and args.batch_visualization_limit < 0
    ):
        args.batch_visualization_limit = None
    if args.rollouts_per_scenario < 1:
        raise ValueError("--rollouts-per-scenario must be at least 1")

    format_av2 = args.format in ("av2", "both")

    run_inference_pipeline(
        input_path=args.input,
        raw_data_folder=args.raw_data_folder,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
        visualize=args.visualize,
        device=args.device,
        sample_tokens=args.sample_tokens,
        rollouts_per_scenario=args.rollouts_per_scenario,
        format_av2=format_av2,
        batch_visualization_limit=args.batch_visualization_limit,
    )


if __name__ == "__main__":
    main()
