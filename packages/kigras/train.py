from kigras.dataloader.datamodule import ArgoDataModule
from kigras.model.kigras import KiGRAS
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import hydra


def run_training(cfg: DictConfig, wandb_logger=None, return_metrics_path=None) -> dict:
    dl_cfg = cfg.training.dataloader
    optim_cfg = cfg.training.optimizer
    ckpt_cfg = cfg.training.checkpointing
    es_cfg = cfg.training.early_stopping
    trainer_cfg = cfg.training.trainer
    model_cfg = cfg.model.architecture

    datamodule = ArgoDataModule(
        train_dir=dl_cfg.train_path,
        val_dir=dl_cfg.val_path,
        train_batch_size=dl_cfg.train_batch_size,
        val_batch_size=dl_cfg.val_batch_size,
        num_workers=dl_cfg.data_workers,
        mode=dl_cfg.mode,
    )

    model = KiGRAS(
        lr=optim_cfg.learning_rate,
        weight_decay=optim_cfg.weight_decay,
        d_model=model_cfg.embedding_dim,
        nhead=model_cfg.num_head,
        encoder_num_layers=model_cfg.encoder_num_layers,
        decoder_num_layers=model_cfg.decoder_num_layers,
        dropout=model_cfg.dropout,
        vocab_size=model_cfg.num_action_bins**model_cfg.num_actions,
    )

    # callbacks during training
    callbacks = []

    checkpoint = ModelCheckpoint(**ckpt_cfg)
    callbacks.append(checkpoint)

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

    ckpt_dir = Path(ckpt_cfg.dirpath)
    ckpt_path = None
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
        if ckpt_files:
            ckpt_path = str(ckpt_files[-1])
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    return {"end_loss": trainer.callback_metrics.get("val_loss", None)}


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    run_training(cfg)


if __name__ == "__main__":
    main()
