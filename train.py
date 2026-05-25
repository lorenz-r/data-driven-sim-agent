import argparse
import yaml
from trainers.profiling import add_profiler_args
from trainers.trainer_registry import TRAINER_REGISTRY

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
# CLI Argument Parsing
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--top-k-checkpoints", type=int)
    parser.add_argument("--checkpoint-metric", type=str)

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    parser.add_argument("--wandb-logging", action="store_true")
    parser.add_argument("--no-wandb-logging", action="store_true")

    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------
    parser.add_argument("--early-stopping", action="store_true")

    # --------------------------------------------------------
    # Profiling
    # --------------------------------------------------------
    add_profiler_args(parser)

    return parser.parse_args()


# ============================================================
# Config Merge Logic
# ============================================================


def build_config():
    args = parse_args()

    # Load YAML defaults
    config = load_yaml_config(args.config)
    config.setdefault("profiling", {})

    # Mapping:
    # CLI arg -> YAML key path
    arg_to_config = {
        "train_path": "data.train_path",
        "val_path": "data.val_path",
        "train_manifest": "data.train_manifest",
        "val_manifest": "data.val_manifest",
        "batch_size": "data.batch_size",
        "data_workers": "data.data_workers",
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
        "top_k_checkpoints": "checkpointing.top_k_checkpoints",
        "checkpoint_metric": "checkpointing.checkpoint_metric",
        "profile": "profiling.profile",
        "profile_dir": "profiling.profile_dir",
        "profile_name": "profiling.profile_name",
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


def get_trainer(config):

    model_name = config["model"]["model_name"].lower()

    if model_name not in TRAINER_REGISTRY:
        available = list(TRAINER_REGISTRY.keys())

        raise ValueError(
            f"Unknown trainer/model '{model_name}'. Available: {available}"
        )

    trainer_cls = TRAINER_REGISTRY[model_name]

    return trainer_cls(config)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    config = build_config()

    print("\n========== FINAL CONFIG ==========")
    print(yaml.dump(config, sort_keys=False))

    # Example access
    batch_size = config["data"]["batch_size"]

    print(f"Using batch size: {batch_size}")

    trainer = get_trainer(config)
    results = trainer.train()
    print("Training results:", results)
