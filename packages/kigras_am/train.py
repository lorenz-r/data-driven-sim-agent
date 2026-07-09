from kigras.dataloader.datamodule import ArgoDataModule
from kigras.model.kigras import KiGRAS
from dotenv import load_dotenv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
import torch
import hydra


def _set_reproducibility(cfg: DictConfig) -> None:
    repro_cfg = cfg.training.get("reproducibility", {})
    seed = repro_cfg.get("seed", None)
    if seed is not None:
        pl.seed_everything(int(seed), workers=bool(repro_cfg.get("seed_workers", True)))
    torch.backends.cudnn.benchmark = bool(repro_cfg.get("benchmark", False))


def _save_resolved_config(cfg: DictConfig) -> Path | None:
    repro_cfg = cfg.training.get("reproducibility", {})
    if not repro_cfg.get("save_resolved_config", True):
        return None

    output_path = Path(cfg.training.checkpointing.dirpath) / repro_cfg.get(
        "resolved_config_filename", "resolved_config.yaml"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(config=resolved, f=output_path)
    return output_path


def _create_wandb_logger(cfg: DictConfig) -> WandbLogger | None:
    logging_cfg = cfg.training.get("logging", {})
    wandb_cfg = logging_cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    load_dotenv()

    if wandb_cfg.get("offline", False):
        os.environ["WANDB_MODE"] = "offline"

    logger = WandbLogger(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=wandb_cfg.run_name,
        group=wandb_cfg.group,
        tags=list(wandb_cfg.tags),
        notes=wandb_cfg.notes,
        save_dir=wandb_cfg.save_dir,
        log_model=wandb_cfg.log_model,
    )
    logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
    )
    return logger


def run_training(cfg: DictConfig, wandb_logger=None, return_metrics_path=None) -> dict:
    dl_cfg = cfg.training.dataloader
    optim_cfg = OmegaConf.to_container(cfg.training.optimizer, resolve=True)
    scheduler_cfg = OmegaConf.to_container(cfg.training.scheduler, resolve=True)
    ckpt_cfg = OmegaConf.to_container(cfg.training.checkpointing, resolve=True)
    es_cfg = cfg.training.early_stopping
    trainer_cfg = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    model_cfg = cfg.model.architecture
    lightning_checkpoint = ckpt_cfg.pop("lightning_ckpt", None)
    model_checkpoint = ckpt_cfg.pop("model_ckpt", None)

    if lightning_checkpoint and model_checkpoint:
        raise ValueError(
            "Use only one of training.checkpointing.lightning_ckpt or "
            "training.checkpointing.model_ckpt."
        )

    _set_reproducibility(cfg)
    saved_config_path = _save_resolved_config(cfg)
    if wandb_logger is not None and saved_config_path is not None:
        wandb_logger.experiment.save(str(saved_config_path), policy="now")

    datamodule = ArgoDataModule(
        train_dir=dl_cfg.train_path,
        val_dir=dl_cfg.val_path,
        train_batch_size=dl_cfg.train_batch_size,
        val_batch_size=dl_cfg.val_batch_size,
        num_workers=dl_cfg.data_workers,
        pin_memory=dl_cfg.pin_memory,
        persistent_workers=dl_cfg.persistent_workers,
        shuffle=dl_cfg.shuffle_train,
        mode=dl_cfg.mode,
        # Load exactly the tokens that match the model's action grid.
        grid_power=cfg.model.kinematics.get("grid_power", 1.0),
        num_bins=model_cfg.num_action_bins,
    )

    model = KiGRAS(
        optimizer_config=optim_cfg,
        scheduler_config=scheduler_cfg,
        d_model=model_cfg.embedding_dim,
        nhead=model_cfg.num_head,
        encoder_num_layers=model_cfg.encoder_num_layers,
        encoder_ff_dim=model_cfg.encoder_ff_dim,
        decoder_num_layers=model_cfg.decoder_num_layers,
        decoder_ff_dim=model_cfg.decoder_ff_dim,
        dropout=model_cfg.dropout,
        agent_context_radius=model_cfg.agent_context_radius,
        map_context_radius=model_cfg.map_context_radius,
        vocab_size=model_cfg.num_action_bins**model_cfg.num_actions,
    )
    if model_checkpoint:
        checkpoint = torch.load(
            model_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    # callbacks during training
    callbacks = []

    checkpoint = ModelCheckpoint(**ckpt_cfg)
    callbacks.append(checkpoint)
    if wandb_logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    if es_cfg.enabled:
        from pytorch_lightning.callbacks import EarlyStopping

        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.monitor,
                patience=es_cfg.patience,
                mode=es_cfg.mode,
            )
        )

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    strategy = (
        DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        if num_devices > 1
        else "auto"
    )

    trainer = pl.Trainer(
        devices=num_devices,
        strategy=strategy,
        callbacks=callbacks,
        logger=wandb_logger,
        **trainer_cfg,
    )

    ckpt_path = str(Path(lightning_checkpoint)) if lightning_checkpoint else None
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    return {"end_loss": trainer.callback_metrics.get("val_loss", None)}


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    wandb_logger = _create_wandb_logger(cfg)
    status = "success"
    try:
        run_training(cfg, wandb_logger=wandb_logger)
    except Exception:
        status = "failed"
        raise
    finally:
        if wandb_logger is not None:
            wandb_logger.finalize(status)


if __name__ == "__main__":
    main()
