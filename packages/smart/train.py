from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import json
import os
import pytorch_lightning as pl
from dotenv import load_dotenv
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from smart.utils.config import load_config_act
from smart.datamodules import MultiDataModule
from smart.model import SMART
from smart.utils.log import Logging
from smart.metrics.utils import get_training_metrics
from smart.utils.profiling import TrainingProfiler
from smart.utils.profiling import add_profiler_args
from smart.utils.agent_token_builder import ensure_agent_tokens
from smart.utils.token_paths import (
    default_agent_token_path,
    default_map_token_path,
    resolve_token_path,
)
import pickle
import wandb
from tqdm import tqdm


class EpochLossTracker(pl.Callback):
    """Records the training loss at the end of every epoch.

    Stores the raw sequence so callers can derive start loss, end loss,
    and total decrease without any additional bookkeeping.
    """

    def __init__(self, loss_key: str = "train_loss"):
        self.loss_key = loss_key
        self.epoch_losses: list = []

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        val = trainer.callback_metrics.get(self.loss_key)
        if val is not None:
            try:
                loss = float(val.item()) if hasattr(val, "item") else float(val)
                self.epoch_losses.append(loss)
            except Exception:
                pass

    # Convenience properties consumed by get_training_metrics
    @property
    def start_loss(self) -> Optional[float]:
        return self.epoch_losses[0] if self.epoch_losses else None

    @property
    def end_loss(self) -> Optional[float]:
        return self.epoch_losses[-1] if self.epoch_losses else None

    @property
    def loss_decrease(self) -> Optional[float]:
        if self.start_loss is not None and self.end_loss is not None:
            return self.start_loss - self.end_loss
        return None


def validate_dataset(datamodule, remove_bad_files=False):
    """
    Validates dataset samples before training.

    Attempts to:
    - load pickle
    - run preprocessing

    Invalid samples are logged and optionally deleted.

    Returns:
        list[str]: paths of invalid files
    """

    logger = Logging().log(level="DEBUG")

    logger.info("Setting up datamodule for validation...")
    datamodule.setup()

    datasets = {
        "train": datamodule.train_dataset,
        "val": datamodule.val_dataset,
    }

    if hasattr(datamodule, "test_dataset"):
        datasets["test"] = datamodule.test_dataset

    bad_files = []

    for split_name, dataset in datasets.items():
        logger.info(f"Validating {split_name} dataset...")
        logger.info(f"Number of samples: {len(dataset)}")

        for idx in tqdm(range(len(dataset)), desc=f"Checking {split_name}"):
            path = dataset.raw_paths[idx]

            try:
                with open(path, "rb") as handle:
                    data = pickle.load(handle)

                # IMPORTANT:
                # create fresh processor instance each time
                # avoids hidden mutable-state bugs
                processor = dataset.token_processor.__class__(
                    dataset.token_size,
                    agent_token_path=str(dataset.agent_token_path),
                    map_token_path=str(dataset.map_token_path),
                    training=dataset.training,
                )

                processor.preprocess(data)

            except Exception as e:
                logger.error(f"[INVALID] split={split_name} idx={idx} path={path}")
                logger.exception(e)

                bad_files.append(path)

                if remove_bad_files:
                    try:
                        os.remove(path)
                        logger.warning(f"Removed malformed file: {path}")
                    except Exception as remove_error:
                        logger.error(f"Failed to remove {path}: {remove_error}")

    logger.info(f"Validation complete. Found {len(bad_files)} bad files.")

    if len(bad_files) > 0:
        blacklist_path = "bad_samples.txt"

        with open(blacklist_path, "w") as f:
            for path in bad_files:
                f.write(path + "\n")

        logger.warning(f"Saved bad file list to {blacklist_path}")

    return bad_files


_SCRIPT_DIR = Path(__file__).resolve().parent


def maybe_build_agent_tokens_from_config(config) -> None:
    build_if_missing = getattr(config.Dataset, "build_agent_tokens_if_missing", False)
    token_size = config.Model.decoder.token_size
    agent_token_path = resolve_token_path(
        getattr(
            config.Model,
            "agent_token_path",
            getattr(config.Dataset, "agent_token_path", None),
        ),
        default_agent_token_path(token_size),
    )

    if build_if_missing:
        ensure_agent_tokens(
            raw_dirs=config.Dataset.train_raw_dir,
            output_path=agent_token_path,
            token_size=token_size,
            build_if_missing=True,
            max_samples_per_type=getattr(
                config.Dataset, "agent_token_build_max_samples", 1_000_000
            ),
            seed=getattr(config.Dataset, "agent_token_build_seed", 42),
            kdisks_max_attempts=getattr(
                config.Dataset,
                "agent_token_build_kdisks_max_attempts",
                1_000_000,
            ),
        )

    config.Dataset.agent_token_path = str(agent_token_path)
    config.Model.agent_token_path = str(agent_token_path)
    map_token_path = resolve_token_path(
        getattr(
            config.Model,
            "map_token_path",
            getattr(config.Dataset, "map_token_path", None),
        ),
        default_map_token_path(),
    )
    config.Dataset.map_token_path = str(map_token_path)
    config.Model.map_token_path = str(map_token_path)


def run_training(
    config: dict,
    return_metrics_path: str = "",
    validate_dataset_first: bool = False,
    remove_bad_files: bool = False,
    trial_number: int = 1000,  # When moving logging to top-level, do not forget to move this param as well
    wandb_logger=None,  # Allow passing in an optional logger instance
    training_profiler: Optional[
        TrainingProfiler
    ] = None,  # Optional training profiler for performance analysis
):
    """
    Unified SMART training entrypoint.

    Parameters
    ----------
    config : dict
        Unified top-level project config from train.py

    return_metrics_path : str
        Optional JSON output path for metrics.

    validate_dataset_first : bool
        Validate dataset before training.

    remove_bad_files : bool
        Remove malformed files during validation.

    trial_number : int
        Optional Optuna trial number.

    training_profiler : TrainingProfiler | None
        Optional profiler configured by the top-level trainer.
    """

    Predictor_hash = {
        "smart": SMART,
    }

    # =========================================================
    # Load SMART-native config
    # =========================================================

    smart_config = load_config_act(
        str(_SCRIPT_DIR / "configs/train/train_overfitting.yaml")
    )

    # =========================================================
    # Map unified config -> SMART config terminology
    # =========================================================

    # -------------------------
    # Dataset
    # -------------------------

    smart_config.Dataset.train_raw_dir = [config["data"]["train_path"]]

    smart_config.Dataset.val_raw_dir = [config["data"]["val_path"]]

    smart_config.Dataset.train_batch_size = config["data"]["batch_size"]

    smart_config.Dataset.val_batch_size = config["data"]["val_batch_size"]

    smart_config.Dataset.num_workers = config["data"]["data_workers"]

    if "shuffle_train" in config["data"]:
        smart_config.Dataset.shuffle = config["data"]["shuffle_train"]

    if "pin_memory" in config["data"]:
        smart_config.Dataset.pin_memory = config["data"]["pin_memory"]

    if "persistent_workers" in config["data"]:
        smart_config.Dataset.persistent_workers = config["data"]["persistent_workers"]

    if "token_processor_backend" in config["data"]:
        smart_config.Dataset.token_processor_backend = config["data"][
            "token_processor_backend"
        ]

    tokens_config = None
    if "smart" in config and "tokens" in config["smart"]:
        tokens_config = config["smart"]["tokens"]
    elif "tokens" in config:
        tokens_config = config["tokens"]

    if tokens_config is not None:
        smart_config.Dataset.agent_token_path = tokens_config.get(
            "agent_token_path",
            smart_config.Dataset.get("agent_token_path", None),
        )
        smart_config.Dataset.map_token_path = tokens_config.get(
            "map_token_path",
            smart_config.Dataset.get("map_token_path", None),
        )
        smart_config.Dataset.build_agent_tokens_if_missing = tokens_config.get(
            "build_agent_tokens_if_missing",
            False,
        )
        smart_config.Dataset.agent_token_build_max_samples = tokens_config.get(
            "agent_token_build_max_samples",
            smart_config.Dataset.get("agent_token_build_max_samples", 1_000_000),
        )
        smart_config.Dataset.agent_token_build_seed = tokens_config.get(
            "agent_token_build_seed",
            smart_config.Dataset.get("agent_token_build_seed", 42),
        )
        smart_config.Dataset.agent_token_build_kdisks_max_attempts = tokens_config.get(
            "agent_token_build_kdisks_max_attempts",
            smart_config.Dataset.get(
                "agent_token_build_kdisks_max_attempts",
                1_000_000,
            ),
        )
        smart_config.Model.agent_token_path = smart_config.Dataset.agent_token_path
        smart_config.Model.map_token_path = smart_config.Dataset.map_token_path

    loss_config = None
    if "smart" in config and "loss" in config["smart"]:
        loss_config = config["smart"]["loss"]
    elif "loss" in config:
        loss_config = config["loss"]

    if loss_config is not None:
        smart_config.Model.loss = loss_config

    maybe_build_agent_tokens_from_config(smart_config)

    # -------------------------
    # Training
    # -------------------------

    smart_config.Trainer.max_epochs = config["training"]["max_epochs"]

    if "accelerator" in config["training"]:
        smart_config.Trainer.accelerator = config["training"]["accelerator"]

    smart_config.Trainer.devices = config["training"]["devices"]

    if "accumulate_grad_batches" in config["training"]:
        smart_config.Trainer.accumulate_grad_batches = config["training"][
            "accumulate_grad_batches"
        ]

    if "precision" in config["training"]:
        smart_config.Trainer.precision = config["training"]["precision"]

    if "accumulate_grad_batches" in config["training"]:
        smart_config.Trainer.accumulate_grad_batches = config["training"][
            "accumulate_grad_batches"
        ]

    if "precision" in config["training"]:
        smart_config.Trainer.precision = config["training"]["precision"]

    # -------------------------
    # Optimization
    # -------------------------

    smart_config.Model.lr = config["optimization"]["learning_rate"]

    smart_config.Model.weight_decay = config["optimization"]["weight_decay"]

    # Optional overrides if present
    if "dropout" in config["model"]:
        smart_config.Model.dropout = config["model"]["dropout"]

    if "hidden_dim" in config["model"]:
        smart_config.Model.hidden_dim = config["model"]["hidden_dim"]

    # =========================================================
    # Predictor
    # =========================================================

    Predictor = Predictor_hash[smart_config.Model.predictor]

    # =========================================================
    # Datamodule
    # =========================================================

    datamodule = MultiDataModule(**vars(smart_config.Dataset))

    # =========================================================
    # Dataset Validation
    # =========================================================

    if validate_dataset_first:
        bad_files = validate_dataset(
            datamodule,
            remove_bad_files=remove_bad_files,
        )

        print(f"\nFound {len(bad_files)} malformed files.\n")

    # =========================================================
    # Model
    # =========================================================

    pretrain_ckpt = config["resume"].get(
        "pretrain_checkpoint",
        "",
    )

    if pretrain_ckpt == "":
        model = Predictor(smart_config.Model)

    else:
        logger = Logging().log(level="DEBUG")

        model = Predictor(smart_config.Model)

        model.load_params_from_file(
            filename=pretrain_ckpt,
            logger=logger,
        )

    # =========================================================
    # Strategy
    # =========================================================

    trainer_config = smart_config.Trainer

    if trainer_config.devices > 1:
        strategy = DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    else:
        strategy = "auto"

    # =========================================================
    # Logging
    # =========================================================

    if config["logging"]["wandb_logging"] and wandb_logger is not None:
        # SAFE CONFIG LOGGING
        #
        # Avoid logging raw objects/classes.
        # Only log primitive serializable values.

        flat_config = {
            "learning_rate": config["optimization"]["learning_rate"],
            "batch_size": config["data"]["batch_size"],
            "max_epochs": config["training"]["max_epochs"],
            "devices": config["training"]["devices"],
            "weight_decay": config["optimization"]["weight_decay"],
        }

        wandb_logger.log_hyperparams(flat_config)

    # =========================================================
    # Callbacks
    # =========================================================

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["checkpointing"]["save_dir"],
        filename="{epoch:02d}",
        monitor=config["checkpointing"]["checkpoint_metric"],
        save_top_k=config["checkpointing"]["top_k_checkpoints"],
        mode=config["checkpointing"]["mode"],
        every_n_epochs=1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    loss_tracker = EpochLossTracker(loss_key="train_loss")

    callbacks = [
        checkpoint_callback,
        lr_monitor,
        loss_tracker,
    ]

    # =========================================================
    # Early Stopping
    # =========================================================

    if config["early_stopping"]["enabled"]:
        from pytorch_lightning.callbacks import (
            EarlyStopping,
        )

        early_stop_callback = EarlyStopping(
            monitor=config["early_stopping"]["monitor"],
            patience=config["early_stopping"]["patience"],
            mode=config["early_stopping"]["mode"],
        )

        callbacks.append(early_stop_callback)

    # =========================================================
    # Trainer
    # =========================================================

    if training_profiler is None:
        training_profiler = TrainingProfiler.from_config(
            config.get("profiling", {}),
            default_profile_name=f"smart_profile_trial-{trial_number}",
        )

    profiler = training_profiler.build()

    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        strategy=strategy,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        num_nodes=trainer_config.num_nodes,
        callbacks=callbacks,  # Only early stopping is conditional, the rest are always included
        max_epochs=trainer_config.max_epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=config["training"].get("gradient_clip_val", 0.5),
        precision=trainer_config.precision,
        logger=wandb_logger,
        profiler=profiler,
    )

    # =========================================================
    # Resume Checkpoint
    # =========================================================

    ckpt_path = config["resume"].get(
        "checkpoint_path",
        "",
    )

    if ckpt_path == "":
        trainer.fit(model, datamodule)

    else:
        trainer.fit(
            model,
            datamodule,
            ckpt_path=ckpt_path,
        )

    training_profiler.finalize(
        wandb_logger=wandb_logger,
        is_global_zero=trainer.is_global_zero,
    )

    # =========================================================
    # Metrics Export
    # =========================================================

    metrics = get_training_metrics(
        model_checkpoint=checkpoint_callback,
        trainer=trainer,
        loss_tracker=loss_tracker,
        metric_name="val_cls_acc",
    )

    if return_metrics_path:
        try:
            Path(return_metrics_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(return_metrics_path, "w") as fh:
                json.dump(metrics, fh, indent=2)

            print(f"Wrote metrics to {return_metrics_path}")

        except Exception as e:
            print(f"Failed to write metrics file: {e}")

    # =========================================================
    # Cleanup
    # =========================================================

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    Predictor_hash = {"smart": SMART}
    parser.add_argument(
        "--config",
        type=str,
        default=str(_SCRIPT_DIR / "configs/train/train_overfitting.yaml"),
    )

    # Add arguments to override config values from the command line. This is helpful for top-level hyperparameter search scripts that want to sweep over a few parameters without needing to create a separate config file for each run.
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)

    parser.add_argument("--pretrain_ckpt", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--save_ckpt_path", type=str, default="")
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    parser.add_argument(
        "--return_metrics",
        type=str,
        default="",
        help="Path to write JSON with best-model accuracy and loss metrics",
    )
    parser.add_argument(
        "--loss_key",
        type=str,
        default="train_loss",
        help="Name of the loss metric logged by the model (default: train_loss)",
    )

    # Also allow for epochs, batch size, and other trainer parameters to be overridden from the command line if needed
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)

    parser.add_argument(
        "--train_path",
        type=str,
        nargs="+",
        default=None,
        help="One or more training data directories; overrides config if provided.",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        nargs="+",
        default=None,
        help="One or more validation data directories; overrides config if provided.",
    )
    parser.add_argument("--agent_token_path", type=str, default=None)
    parser.add_argument("--map_token_path", type=str, default=None)
    parser.add_argument(
        "--build_agent_tokens_if_missing",
        action="store_true",
        help="Explicitly build SMART agent tokens from train_path if the file is missing.",
    )

    # Optuna passes a trial number to the training script for each run, which can be used for logging and tracking purposes. We'll include it as an optional argument in case it's needed, but it won't affect the training logic directly.
    parser.add_argument(
        "--trial_number",
        type=int,
        default=0,
        help="Optional trial number for hyperparameter search runs; used for logging purposes",
    )

    parser.add_argument(
        "--validate_dataset",
        action="store_true",
        help="Validate dataset before training",
    )

    parser.add_argument(
        "--remove_bad_files",
        action="store_true",
        help="Delete malformed dataset files during validation",
    )

    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of GPU devices to use for training",
    )

    parser.add_argument(
        "--num_cpu_workers",
        type=int,
        default=None,
        help="Number of CPU workers for data loading; if not provided, will use value from config",
    )
    parser.add_argument(
        "--token_processor_backend",
        choices=["rust", "python", "auto"],
        default=None,
        help="Token processor backend for scalable dataset loading.",
    )
    add_profiler_args(parser)

    args = parser.parse_args()

    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("WANDB_API_KEY not set — running without wandb logging")

    config = load_config_act(args.config)

    # Override config values from CLI if provided
    if args.lr is not None:
        config.Model.lr = args.lr

    if args.dropout is not None:
        config.Model.dropout = args.dropout

    if args.hidden_dim is not None:
        config.Model.hidden_dim = args.hidden_dim

    if args.max_epochs is not None:
        config.Trainer.max_epochs = args.max_epochs

    if args.train_batch_size is not None:
        config.Dataset.train_batch_size = args.train_batch_size

    if args.val_batch_size is not None:
        config.Dataset.val_batch_size = args.val_batch_size

    if args.train_path is not None:
        config.Dataset.train_raw_dir = args.train_path

    if args.val_path is not None:
        config.Dataset.val_raw_dir = args.val_path

    if args.agent_token_path is not None:
        config.Dataset.agent_token_path = args.agent_token_path
        config.Model.agent_token_path = args.agent_token_path

    if args.map_token_path is not None:
        config.Dataset.map_token_path = args.map_token_path
        config.Model.map_token_path = args.map_token_path

    if args.build_agent_tokens_if_missing:
        config.Dataset.build_agent_tokens_if_missing = True

    if args.devices is not None:
        config.Trainer.devices = args.devices

    if args.num_cpu_workers is not None:
        config.Dataset.num_workers = args.num_cpu_workers

    if args.token_processor_backend is not None:
        config.Dataset.token_processor_backend = args.token_processor_backend

    maybe_build_agent_tokens_from_config(config)

    Predictor = Predictor_hash[config.Model.predictor]

    # This may be causing issues with Optuna parallelization, so we'll switch to the default strategy for now and can re-enable it later if needed.
    # strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)

    Data_config = config.Dataset
    datamodule = MultiDataModule(**vars(Data_config))

    # Validate dataset before training to catch any malformed files early and avoid wasting time during training. This is especially important when working with large datasets where a single bad file can cause the entire training run to fail after hours of computation.
    if args.validate_dataset:
        bad_files = validate_dataset(
            datamodule,
            remove_bad_files=args.remove_bad_files,
        )

        print(f"\nFound {len(bad_files)} malformed files.\n")

        if len(bad_files) > 0:
            print("Malformed files written to bad_samples.txt")

        # Exit after validation if desired
        if args.remove_bad_files:
            print("Malformed files removed.")

    if args.pretrain_ckpt == "":
        model = Predictor(config.Model)
    else:
        logger = Logging().log(level="DEBUG")
        model = Predictor(config.Model)
        model.load_params_from_file(filename=args.pretrain_ckpt, logger=logger)
    trainer_config = config.Trainer

    # Dynamically adjust strategy based on number of devices requested, since DDP doesn't work with a single device
    if trainer_config.devices > 1:
        strategy = DDPStrategy(
            find_unused_parameters=True, gradient_as_bucket_view=True
        )
    else:
        strategy = "auto"

    if args.cpu:
        trainer_config.accelerator = "cpu"

    training_profiler = TrainingProfiler.from_args(
        args,
        default_profile_name=f"smart_profile_trial-{args.trial_number}",
    )
    profiler = training_profiler.build()

    # model_checkpoint = ModelCheckpoint(
    #     dirpath=args.save_ckpt_path,
    #     filename="{epoch:02d}",
    #     monitor='val_cls_acc',
    #     every_n_epochs=1,
    #     save_top_k=5,
    #     mode='max',
    # )
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # loss_tracker = EpochLossTracker(loss_key=args.loss_key)

    # trainer = pl.Trainer(
    #     accelerator=trainer_config.accelerator,
    #     devices=trainer_config.devices,
    #     strategy=strategy,
    #     accumulate_grad_batches=trainer_config.accumulate_grad_batches,
    #     num_nodes=trainer_config.num_nodes,
    #     callbacks=[model_checkpoint, lr_monitor, loss_tracker],
    #     max_epochs=trainer_config.max_epochs,
    #     num_sanity_val_steps=0,
    #     gradient_clip_val=0.5,
    # )

    wandb_api_key = load_dotenv() and os.getenv("WANDB_API_KEY")

    if wandb_api_key:
        wandb_logger = WandbLogger(
            project="smart-train",
            name=f"op_trial-{args.trial_number}",
        )
        # Log the config values to wandb so they show up in the UI and are saved with the run.
        # wandb_logger.experiment.config.update(
        #    {
        #        "model": vars(config.Model),
        #        "trainer": vars(config.Trainer),
        #        "dataset": vars(config.Dataset),
        #    }
        # )
    else:
        wandb_logger = None

    model_checkpoint = ModelCheckpoint(
        dirpath=args.save_ckpt_path,
        filename="{epoch:02d}",
        monitor="train_loss_epoch",
        every_n_epochs=1,
        save_top_k=5,
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    loss_tracker = EpochLossTracker(loss_key=args.loss_key)
    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        strategy=strategy,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        num_nodes=trainer_config.num_nodes,
        callbacks=[model_checkpoint, lr_monitor, loss_tracker],
        max_epochs=trainer_config.max_epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        logger=wandb_logger,
        profiler=profiler,
    )

    if args.ckpt_path == "":
        trainer.fit(model, datamodule)
    else:
        trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)

    training_profiler.finalize(
        wandb_logger=wandb_logger,
        is_global_zero=trainer.is_global_zero,
    )

    if args.return_metrics:
        metrics = get_training_metrics(
            model_checkpoint=model_checkpoint,
            trainer=trainer,
            loss_tracker=loss_tracker,
            metric_name="val_cls_acc",
        )

        accuracy = metrics.get("accuracy")
        start_loss = metrics.get("start_loss")
        end_loss = metrics.get("end_loss")
        loss_decrease = metrics.get("loss_decrease")

        print("\n── Training summary ──────────────────────")
        print(
            f"  Best val accuracy : {accuracy:.4f}"
            if accuracy is not None
            else "  Best val accuracy : n/a"
        )
        print(
            f"  Start loss        : {start_loss:.4f}"
            if start_loss is not None
            else "  Start loss        : n/a"
        )
        print(
            f"  End loss          : {end_loss:.4f}"
            if end_loss is not None
            else "  End loss          : n/a"
        )
        if loss_decrease is not None:
            direction = "↓" if loss_decrease > 0 else "↑ (increased!)"
            print(f"  Loss change       : {abs(loss_decrease):.4f} {direction}")
        else:
            print("  Loss change       : n/a")
        print("──────────────────────────────────────────\n")

        try:
            Path(args.return_metrics).parent.mkdir(parents=True, exist_ok=True)
            with open(args.return_metrics, "w") as fh:
                json.dump(metrics, fh, indent=2)
            print(f"Wrote metrics to {args.return_metrics}")
        except Exception as e:
            print(f"Failed to write metrics file {args.return_metrics}: {e}")

    if wandb_api_key:
        wandb.finish()
