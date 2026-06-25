import argparse
import yaml
import wandb
import sys
from trainers.profiling import add_profiler_args
from trainers.trainer_registry import TRAINER_REGISTRY
from core.logging.logger_factory import (
    create_logger,
)
from pathlib import Path
from trajectory_data.av2.download import ensure_downloaded
from trajectory_data.av2.setup import (
    run_preprocess,
    read_cache_from_blob,
    write_cache_to_blob,
)

from packages.smart.scripts.setup_av2 import (
    _preprocess_smart_split,
    _hash_preprocessor,
)


def _warn_cache_recommended(context: str) -> None:
    print(
        "\n"
        "WARNING: Processed AV2/SMART data is not being cached.\n"
        f"Context: {context}\n"
        "This may waste time and compute because preprocessing can be expensive.\n"
        "Recommended usage:\n"
        "  - Use --read-cache with --blob-cache-name when a processed cache already exists.\n"
        "  - Use --write-cache with --blob-cache-name when creating a processed cache.\n",
        file=sys.stderr,
    )


# ============================================================
# Config Utilities
# ============================================================


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def update_nested_dict(config: dict, key_path: str, value):
    """
    Updates nested config values using dot notation.

    Example:
        key_path = "training.max_epochs"
    """
    keys = key_path.split(".")
    d = config

    for key in keys[:-1]:
        d = d[key]

    d[keys[-1]] = value


# ============================================================
# av2 Download Utilities
# ============================================================


REPO_ROOT = Path(__file__).resolve().parent


SIZE_DICT = {
    "single": {"scenarios": 2, "train_size": 1, "val_size": 1},
    "small": {"scenarios": 100, "train_size": 80, "val_size": 20},
    "medium": {"scenarios": 1000, "train_size": 800, "val_size": 200},
    "large": {"scenarios": 10000, "train_size": 8000, "val_size": 2000},
    "massive": {"scenarios": 20000, "train_size": 16000, "val_size": 4000},
}


def download_av2_data(
    # Config commented out for now; will be useful later for specifying file paths for different models, etc.
    # config,
    size: str = "single",
    raw_download_folder: str = "raw_av2",
):
    """
    Download AV2 data into av2_data/<raw_download_folder>.
    """

    data_root = REPO_ROOT / "av2_data"
    raw_dir = data_root / raw_download_folder

    data_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    mode = "dev" if size != "full" else "full"

    ensure_downloaded(
        out_dir=raw_dir,
        mode=mode,
        train_scenarios=SIZE_DICT[size]["train_size"],
        val_scenarios=SIZE_DICT[size]["val_size"],
    )

    return raw_dir


def process_2_smart(
    # config,
    # size: str = "single",
    raw_folder: str | Path | None = None,
    manifest_folder: str | Path = "manifests",
):
    """
    This function will be responsible for processing av2 data into the format expected by Smart.
    It will automatically create train and validation splits, according to the paths specified in the config.
    Later, this function should be modular and allow for processing from already existing folders (or maybe we choose on-the-fly processing?).

    However, right now it simply serves the purpose of allowing a quick start on a clean machine. It essentially just calls the existing SMART setup methods.
    """

    # train_path = Path(config["data"]["train_path"]).resolve()
    # val_path = Path(config["data"]["val_path"]).resolve()

    if raw_folder is None:
        raw_folder = REPO_ROOT / "av2_data" / "raw_av2"
    else:
        raw_folder = Path(raw_folder).resolve()

    manifest_root = Path(manifest_folder)
    if not manifest_root.is_absolute():
        manifest_root = REPO_ROOT / manifest_root

    # train_path.mkdir(parents=True, exist_ok=True)
    # val_path.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    processed_root = REPO_ROOT / "av2_data" / "smart_processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    cache_dir = run_preprocess(
        raw_dir=raw_folder,
        processed_dir=processed_root,
        dataset_etag="local",
        preprocessor_fingerprint=_hash_preprocessor(),
        preprocess_split=_preprocess_smart_split,
        splits=("train", "val"),
        use_cache=False,
    )

    return cache_dir


# ============================================================
# CLI Argument Parsing
# ============================================================


def parse_args():
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        additional_overrides = sys.argv[idx + 1 :]
        sys.argv = sys.argv[:idx]
    else:
        additional_overrides = []

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Download First
    # --------------------------------------------------------
    parser.add_argument(
        "--download-first",
        action="store_true",
        help="Whether to download AV2 data before training",
    )
    parser.add_argument(
        "--download-size",
        type=str,
        default="single",
        choices=SIZE_DICT.keys(),
        help="Size of AV2 data to download (if --download-first is set)",
    )
    parser.add_argument(
        "--process-smart",
        action="store_true",
        help="Whether to process AV2 data into Smart format before training (if --download-first is set or if data is already downloaded but not processed)",
    )

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------
    parser.add_argument(
        "--read-cache",
        action="store_true",
        help="Use an existing processed dataset cache instead of downloading/preprocessing.",
    )

    parser.add_argument(
        "--write-cache",
        action="store_true",
        help="Write the processed dataset cache after preprocessing.",
    )

    parser.add_argument(
        "--blob-cache-name",
        type=str,
        default=None,
        help="Name of the processed dataset cache, e.g. smart-av2-massive-v1.",
    )

    # --------------------------------------------------------
    # General Config
    # --------------------------------------------------------
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_params.yaml",
        help="Path to YAML config file",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model/trainer to use",
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    parser.add_argument("--train-path", type=str)
    parser.add_argument("--val-path", type=str)

    parser.add_argument("--train-manifest", type=str)
    parser.add_argument("--val-manifest", type=str)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--data-workers", type=int)
    parser.add_argument("--token-processor-backend", choices=["rust", "python", "auto"])

    parser.add_argument("--val-batch-size", type=int)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    parser.add_argument("--devices", type=int)

    parser.add_argument("--precision", type=str)

    parser.add_argument("--dropout", type=float)

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)

    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--scheduler", type=str)

    # --------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------
    parser.add_argument(
        "--save-ckpt-path",
        "--save_ckpt_path",
        dest="save_ckpt_path",
        type=str,
        help="Directory where Lightning checkpoints are written",
    )
    parser.add_argument(
        "--return-metrics",
        "--return_metrics",
        dest="return_metrics",
        type=str,
        help="Path to write JSON training metrics",
    )
    parser.add_argument("--top-k-checkpoints", type=int)
    parser.add_argument("--checkpoint-metric", type=str)

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    parser.add_argument("--wandb-logging", action="store_true")
    parser.add_argument("--no-wandb-logging", action="store_true")
    parser.add_argument("--project-name", type=str)
    parser.add_argument("--run-name", type=str)

    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------
    parser.add_argument("--early-stopping", action="store_true")

    # --------------------------------------------------------
    # Profiling
    # --------------------------------------------------------
    add_profiler_args(parser)

    args = parser.parse_args()
    args.additional_overrides = additional_overrides
    return args


# ============================================================
# Config Merge Logic
# ============================================================


def build_config(args=None) -> dict:
    # args = parse_args()

    # Load YAML defaults
    config = load_yaml_config(args.config)
    config.setdefault("profiling", {})
    config.setdefault("output", {})

    # Mapping:
    # CLI arg -> YAML key path
    arg_to_config = {
        "train_path": "data.train_path",
        "val_path": "data.val_path",
        "train_manifest": "data.train_manifest",
        "val_manifest": "data.val_manifest",
        "batch_size": "data.batch_size",
        "val_batch_size": "data.val_batch_size",
        "data_workers": "data.data_workers",
        "token_processor_backend": "data.token_processor_backend",
        "model": "model.model_name",
        "dropout": "model.dropout",
        "max_epochs": "training.max_epochs",
        "devices": "training.devices",
        "precision": "training.precision",
        "learning_rate": "optimization.learning_rate",
        "weight_decay": "optimization.weight_decay",
        "optimizer": "optimization.optimizer",
        "scheduler": "optimization.scheduler",
        "save_ckpt_path": "checkpointing.save_dir",
        "return_metrics": "output.metrics_path",
        "top_k_checkpoints": "checkpointing.top_k_checkpoints",
        "checkpoint_metric": "checkpointing.checkpoint_metric",
        "profile": "profiling.profile",
        "profile_dir": "profiling.profile_dir",
        "profile_name": "profiling.profile_name",
        "project_name": "logging.project_name",
        "run_name": "logging.run_name",
    }

    # Apply CLI overrides
    for arg_name, config_key in arg_to_config.items():
        value = getattr(args, arg_name)

        if value is not None:
            update_nested_dict(config, config_key, value)

    # Special handling for booleans
    if args.wandb_logging:
        config["logging"]["wandb_logging"] = True

    if args.no_wandb_logging:
        config["logging"]["wandb_logging"] = False

    if args.early_stopping:
        config["early_stopping"]["enabled"] = True

    if args.cpu:
        config["training"]["accelerator"] = "cpu"

    if args.no_profile_wandb:
        config["profiling"]["upload_to_wandb"] = False

    if args.keep_profile_action_files:
        config["profiling"]["keep_action_files"] = True

    return config


def get_trainer(config, wandb_logger):

    model_name = config["model"]["model_name"].lower()

    if model_name not in TRAINER_REGISTRY:
        available = list(TRAINER_REGISTRY.keys())

        raise ValueError(
            f"Unknown trainer/model '{model_name}'. Available: {available}"
        )

    trainer_cls = TRAINER_REGISTRY[model_name]

    return trainer_cls(config, wandb_logger)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    if args.read_cache and (
        args.download_first or args.process_smart or args.write_cache
    ):
        raise ValueError(
            "--read-cache should not be combined with --download-first, "
            "--process-smart, or --write-cache."
        )

    if args.write_cache and not args.process_smart:
        raise ValueError("--write-cache requires --process-smart.")

    if (args.read_cache or args.write_cache) and not args.blob_cache_name:
        raise ValueError(
            "--blob-cache-name is required with --read-cache or --write-cache."
        )

    if args.process_smart and not args.write_cache:
        _warn_cache_recommended(
            "train.py will preprocess SMART data with --process-smart, "
            "but --write-cache is not set."
        )

    elif not args.read_cache and not args.write_cache:
        _warn_cache_recommended(
            "train.py is running without --read-cache or --write-cache."
        )

    cache = None

    if args.read_cache:
        cache = read_cache_from_blob(
            args.blob_cache_name,
            local_cache_dir=REPO_ROOT / "av2_data" / "smart_processed",
        )
    else:
        if args.download_first:
            download_av2_data(
                size=args.download_size,
                raw_download_folder="raw_av2",
            )

        if args.process_smart:
            cache = process_2_smart(
                raw_folder=REPO_ROOT / "av2_data" / "raw_av2",
                # The manifest might not yet work correctly
                manifest_folder=REPO_ROOT / "av2_data" / "manifests",
            )

            if args.write_cache:
                write_cache_to_blob(
                    local_cache_dir=cache,
                    blob_cache_name=args.blob_cache_name,
                    args=args,
                )

        if args.model == "kigras":
            # TODO: implement kigras preprocessing
            # Reads raw AV2 from av2_data/raw_av2/, writes to data/kigras_processed/
            # currently, this is done in trainers/kigras_trainer.py
            pass

    if cache is not None:
        # Dynamically adjust file paths based on the SMART preprocessing/cache.
        args.train_path = str(cache / "train")
        args.val_path = str(cache / "val")

    config = build_config(args=args)

    # [AM] NOTE: Add configs after '--' as overrides (currently only used in kigras)
    # Current usage of CLI includes additional parameters which, if passed, override kigras hydra.
    # Example:
    #   uv run train.py --model kigras -- training.trainer.max_epochs=200 paths.processed_train=/custom/path
    config["model"]["overrides"] = [a for a in args.additional_overrides if a != "--"]

    print("\n========== FINAL CONFIG ==========")
    print(yaml.dump(config, sort_keys=False))

    # Example access
    batch_size = config["data"]["batch_size"]

    print(f"Using batch size: {batch_size}")

    wandb_logger = create_logger(config)

    wandb_status = "success"
    try:
        trainer = get_trainer(config, wandb_logger)
        results = trainer.train()
        print("Training results:", results)
    except BaseException:
        wandb_status = "failed"
        raise
    finally:
        if wandb_logger is not None:
            wandb_logger.finalize(wandb_status)
        if wandb.run is not None:
            wandb.finish()
