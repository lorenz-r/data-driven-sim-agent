from kigras_ha.dataloader.datamodule import ArgoDataModule
from kigras_ha.model.kigras_ha import KiGRAS_HA
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

import hydra
from omegaconf import DictConfig


def run_training(config: dict, wandb_logger=None, return_metrics_path=None) -> dict:
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
    data_cfg = config["data"]
    training_cfg = config["training"]
    optim_cfg = config["optimization"]
    ckpt_cfg = config.get("checkpointing", {})
    resume_cfg = config.get("resume", {})

    # ------------------------------------------------------------------
    # Datamodule
    # ------------------------------------------------------------------
    datamodule = ArgoDataModule(
        train_raw_dir=data_cfg["train_path"],
        val_raw_dir=data_cfg.get("val_path", data_cfg["train_path"]),
        train_batch_size=data_cfg["batch_size"],
        val_batch_size=data_cfg.get("val_batch_size", data_cfg["batch_size"]),
        num_workers=data_cfg.get("data_workers", 8),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = config.get("model", {})
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
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_cfg.get("save_dir", "checkpoints"),
        filename="{epoch:02d}",
        monitor="val_loss",  # TODO: Make variable metric for config
        save_top_k=ckpt_cfg.get("top_k_checkpoints", 5),
        mode=ckpt_cfg.get("mode", "min"),
        every_n_epochs=1,
        save_last=True,
    )
    callbacks = [checkpoint]

    if config.get("early_stopping", {}).get("enabled", False):
        from pytorch_lightning.callbacks import EarlyStopping

        es = config["early_stopping"]
        callbacks.append(
            EarlyStopping(
                monitor=es["monitor"],
                patience=es.get("patience", 5),
                mode=es.get("mode", "min"),
            )
        )

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------
    devices = training_cfg.get("devices", 1)
    strategy = (
        DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        if devices > 1
        else "auto"
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        max_epochs=training_cfg["max_epochs"],
        num_sanity_val_steps=0,
        gradient_clip_val=training_cfg.get("gradient_clip_val", 0.5),
        precision=training_cfg.get("precision", "32"),
        logger=wandb_logger,
    )

    # ------------------------------------------------------------------
    # Fit (with optional resume)
    # ------------------------------------------------------------------
    ckpt_path = resume_cfg.get("checkpoint_path", "")
    if ckpt_path:
        trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, datamodule)

    return {"end_loss": trainer.callback_metrics.get("val_loss", None)}


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
