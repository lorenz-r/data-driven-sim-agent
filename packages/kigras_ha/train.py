import json
from typing import Optional

from kigras.dataloader.datamodule import ArgoDataModule
from kigras_ha.model.kigras_ha import KiGRAS_HA
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path


class _EpochLossTracker(pl.Callback):
    """Records train_loss_epoch after each epoch to compute loss_decrease."""

    def __init__(self):
        self.epoch_losses: list[float] = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss_epoch")
        if loss is not None:
            self.epoch_losses.append(float(loss))

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


def run_training(cfg: DictConfig, wandb_logger=None, return_metrics_path=None) -> dict:
    """Unified training entrypoint for KiGRAS_HA.

    Parameters
    ----------
    config : dict
        Config dict in the unified format (see the Hydra config tree at
        'configs/' for the schema).
    wandb_logger : optional
        Pre-configured wandb logger from the top-level framework.
    return_metrics_path : optional
        Path to save the training metrics.

    Returns
    -------
    dict
        Training metrics (currently empty, reserved for future use).
    """
    dl_cfg = cfg.training.dataloader
    optim_cfg = cfg.training.optimizer
    ckpt_cfg = cfg.training.checkpointing
    es_cfg = cfg.training.early_stopping
    trainer_cfg = cfg.training.trainer
    model_cfg = cfg.model.architecture

    # ------------------------------------------------------------------
    # Datamodule
    # ------------------------------------------------------------------
    datamodule = ArgoDataModule(
        train_dir=dl_cfg.train_path,
        val_dir=dl_cfg.val_path,
        train_batch_size=dl_cfg.train_batch_size,
        val_batch_size=dl_cfg.val_batch_size,
        num_workers=dl_cfg.data_workers,
        mode=dl_cfg.mode,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = KiGRAS_HA(
        lr=optim_cfg["learning_rate"],
        weight_decay=optim_cfg["weight_decay"],
        num_accel_bins=model_cfg.get("num_accel_bins", 63),
        num_steer_bins=model_cfg.get("num_steer_bins", 63),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
        max_agents=model_cfg.get("max_agents", 10),
        require_complete_history=model_cfg.get("require_complete_history", False),
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    """checkpoint = ModelCheckpoint(
        dirpath=ckpt_cfg.get("save_dir", "checkpoints"),
        filename="{epoch:02d}",
        monitor="val_loss",  # TODO: Make variable metric for config
        save_top_k=ckpt_cfg.get("top_k_checkpoints", 5),
        mode=ckpt_cfg.get("mode", "min"),
        every_n_epochs=1,
        save_last=True,
    )"""
    callbacks = []

    checkpoint = ModelCheckpoint(**ckpt_cfg)
    callbacks.append(checkpoint)

    loss_tracker = _EpochLossTracker()
    callbacks.append(loss_tracker)

    if es_cfg.enabled:
        from pytorch_lightning.callbacks import EarlyStopping

        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.monitor,
                patience=es_cfg.patience,
                mode=es_cfg.mode,
            )
        )

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    strategy = (
        DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        if num_devices > 1
        else "auto"
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        devices=num_devices,
        strategy=strategy,
        callbacks=callbacks,
        logger=wandb_logger,
        **trainer_cfg,
    )

    # ------------------------------------------------------------------
    # Fit (with optional resume)
    # ------------------------------------------------------------------
    resume_path = cfg.get("resume", {}).get("checkpoint_path", "auto")
    if resume_path and resume_path != "auto":
        ckpt_path = resume_path
    elif resume_path == "auto":
        ckpt_dir = Path(ckpt_cfg.dirpath)
        ckpt_path = None
        if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
            ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
            if ckpt_files:
                ckpt_path = str(ckpt_files[-1])
    else:
        ckpt_path = None
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    metrics = {
        "start_loss": loss_tracker.start_loss,
        "end_loss": loss_tracker.end_loss,
        "loss_decrease": loss_tracker.loss_decrease,
    }

    if return_metrics_path:
        out = Path(return_metrics_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


@hydra.main(config_path="../kigras/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    run_training(cfg)


if __name__ == "__main__":
    main()
