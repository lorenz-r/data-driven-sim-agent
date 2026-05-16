from argparse import ArgumentParser
from typing import Optional
import json
import os
import pytorch_lightning as pl
from dotenv import load_dotenv
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb
from smart.utils.config import load_config_act
from smart.datamodules import MultiDataModule
from smart.model import SMART
from smart.utils.log import Logging
from smart.metrics.utils import get_training_metrics
import pickle
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

<<<<<<< HEAD
=======
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
                processor = dataset.token_processor.__class__(2048)

                processor.preprocess(data)

            except Exception as e:

                logger.error(
                    f"[INVALID] split={split_name} idx={idx} path={path}"
                )
                logger.exception(e)

                bad_files.append(path)

                if remove_bad_files:
                    try:
                        os.remove(path)
                        logger.warning(f"Removed malformed file: {path}")
                    except Exception as remove_error:
                        logger.error(
                            f"Failed to remove {path}: {remove_error}"
                        )

    logger.info(f"Validation complete. Found {len(bad_files)} bad files.")

    if len(bad_files) > 0:

        blacklist_path = "bad_samples.txt"

        with open(blacklist_path, "w") as f:
            for path in bad_files:
                f.write(path + "\n")

        logger.warning(f"Saved bad file list to {blacklist_path}")

    return bad_files

>>>>>>> a2265b0 (added data validation)
load_dotenv()

# Set up the logging by creating a wandb account and getting the API key from the environment variable
# After creating an account, share your email/username with the team to get access to the project and its resources; this way the logs can be stored in the shared project and everyone can access them.
# If you wish to run without logging, simply comment out the wandb.login() line and the WandbLogger in the trainer configuration.
wandb.login(key=os.getenv("WANDB_API_KEY"))

if __name__ == "__main__":
    parser = ArgumentParser()
    Predictor_hash = {"smart": SMART}
    parser.add_argument(
        "--config", type=str, default="configs/train/train_overfitting.yaml"
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
        default=None,
        help="Optional override for training data path; if not provided, will use path from config",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Optional override for validation data path; if not provided, will use path from config",
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

    parser.add_argument("--devices", type=int, default=None, help="Number of GPU devices to use for training")

    args = parser.parse_args()
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

    if args.devices is not None:
        config.Trainer.devices = args.devices

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

    wandb_logger = WandbLogger(
        project="smart-train",
        name=f"op_trial-{args.trial_number}",
    )

    # Log the config values to wandb so they show up in the UI and are saved with the run. This is helpful for tracking what hyperparameters were used for each run, especially when doing hyperparameter sweeps.
    # This causes issues if runs are not properly finished (e.g. due to early stopping or errors), so we'll comment it out for now and can re-enable it later if needed.
    #wandb_logger.experiment.config.update(
    #    {
    #        "model": vars(config.Model),
    #        "trainer": vars(config.Trainer),
    #        "dataset": vars(config.Dataset),
    #    }
    #)

    model_checkpoint = ModelCheckpoint(
        dirpath=args.save_ckpt_path,
        filename="{epoch:02d}",
        monitor="val_cls_acc",
        every_n_epochs=1,
        save_top_k=5,
        mode="max",
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
    )

    if args.ckpt_path == "":
        trainer.fit(model, datamodule)
    else:
        trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)

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
            with open(args.return_metrics, "w") as fh:
                json.dump(metrics, fh, indent=2)
            print(f"Wrote metrics to {args.return_metrics}")
        except Exception as e:
            print(f"Failed to write metrics file {args.return_metrics}: {e}")

    wandb.finish()
