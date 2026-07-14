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
    """Records per-epoch train/val loss for the metrics summary + sweep objective."""

    def __init__(self):
        self.epoch_losses: list[float] = []
        self.val_losses: list[float] = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss_epoch")
        if loss is not None:
            self.epoch_losses.append(float(loss))

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss_epoch")
        if loss is None:
            loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses.append(float(loss))

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

    @property
    def best_val_loss(self) -> Optional[float]:
        return min(self.val_losses) if self.val_losses else None


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
    # Reproducibility: seed everything so a config's result is deterministic and
    # hparam-search trials are not confounded by random-init variance.
    pl.seed_everything(int(cfg.get("seed", 0)), workers=True)

    # DataLoader workers share sample tensors via shared memory. The default
    # "file_descriptor" strategy opens one FD per shared tensor; this dataset's
    # samples are dicts of many arrays, so with several workers the process hits
    # "Too many open files". "file_system" shares by name instead of FD, and we
    # also raise the open-file soft limit toward the hard limit.
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:
        pass

    dl_cfg = cfg.training.dataloader
    optim_cfg = cfg.training.optimizer
    ckpt_cfg = cfg.training.checkpointing
    es_cfg = cfg.training.early_stopping
    # Mutable copy so we can pop `devices` before splatting into pl.Trainer.
    trainer_cfg = OmegaConf.to_container(cfg.training.trainer, resolve=True)
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
    kin_cfg = cfg.model.get("kinematics", None)
    sched_cfg = optim_cfg.get("lr_scheduler", None)
    scheduler_config = (
        OmegaConf.to_container(sched_cfg, resolve=True)
        if sched_cfg is not None
        else None
    )
    model = KiGRAS_HA(
        lr=optim_cfg["learning_rate"],
        weight_decay=optim_cfg["weight_decay"],
        num_accel_bins=model_cfg.get("num_action_bins", 63),
        num_steer_bins=model_cfg.get("num_action_bins", 63),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
        max_agents=model_cfg.get("max_agents", 0),
        require_complete_history=model_cfg.get("require_complete_history", False),
        grid_power=kin_cfg.get("grid_power", 1.0) if kin_cfg is not None else 1.0,
        k_neighbors=model_cfg.get("k_neighbors", 16),
        fusion_chunk_size=model_cfg.get("fusion_chunk_size", 15),
        embedding_dim=model_cfg.get("embedding_dim", 64),
        num_head=model_cfg.get("num_head", 4),
        encoder_num_layers=model_cfg.get("encoder_num_layers", 3),
        decoder_num_layers=model_cfg.get("decoder_num_layers", 3),
        dropout=model_cfg.get("dropout", 0.1),
        scheduler_config=scheduler_config,
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
    #
    # `training.trainer.devices` controls the device count: "auto" (default) uses
    # every visible GPU with DDP; an int pins the count. The hyperparameter search
    # sets devices=1 so each trial runs single-GPU — DDP's subprocess relaunch does
    # NOT preserve the programmatic Hydra/argparse architecture overrides, so under
    # DDP rank 1 would rebuild the model from the default config (different #layers)
    # and crash with "DDP expects same model across all ranks".
    # ------------------------------------------------------------------
    requested_devices = trainer_cfg.pop("devices", "auto")
    if requested_devices in (None, "auto", "-1", -1):
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    else:
        num_devices = int(requested_devices)
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
        "best_val_loss": loss_tracker.best_val_loss,
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
