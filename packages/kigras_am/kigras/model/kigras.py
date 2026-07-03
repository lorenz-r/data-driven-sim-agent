import torch
import torch.nn as nn
import pytorch_lightning as pl

# Import your separated modules
from .scene_encoder import SparseSceneEncoder
from .control_action_decoder import ActionDecoder
from ..dataloader.dataset import CURRENT_STEP, PRED_LEN
from ..utils.train_utils import process_sparse_batch


class KiGRAS(pl.LightningModule):
    def __init__(
        self,
        vocab_size=3969,
        num_semantic_classes=10,
        d_model=64,
        nhead=4,
        encoder_num_layers=3,
        decoder_num_layers=3,
        dropout=0.1,
        lr=2e-4,
        weight_decay=1e-4,
    ):
        vocab_size = vocab_size

        super().__init__()
        self.save_hyperparameters()

        self.encoder = SparseSceneEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=encoder_num_layers,
            dropout=dropout,
            num_semantic_classes=num_semantic_classes,
        )
        self.decoder = ActionDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=decoder_num_layers,
            dropout=dropout,
            num_action_bins=vocab_size,
        )

        # The Start-Of-Sequence (SOS) token index
        self.start_token_idx = vocab_size

        # Loss function (ignores -1 padding for missing/padded agents)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Run sparse observer-centric encoding and action decoding.

        Returns:
            logits: [O_total, 60, vocab_size]
            processed: model-facing sparse context dict from process_sparse_batch
        """
        processed = process_sparse_batch(batch)
        scene_tokens = self.encoder(processed)  # [O, T, d_model]

        action_window = processed["action_tokens"][
            :, CURRENT_STEP - 1 : CURRENT_STEP + PRED_LEN
        ]
        action_window_valid = processed["action_valid"][
            :, CURRENT_STEP - 1 : CURRENT_STEP + PRED_LEN
        ]
        processed["action_window"] = action_window
        processed["action_window_valid"] = action_window_valid

        logits = self.decoder(
            scene_tokens,
            action_window=action_window,
            action_window_valid=action_window_valid,
            scene_valid=processed["observer_valid"],
        )
        return logits, processed

    def _shared_step(self, batch, batch_idx, prefix):
        logits, processed = self(batch)

        targets = processed["action_window"][:, 1:]
        target_valid = processed["action_window_valid"][:, 1:]
        targets = targets.masked_fill(~target_valid, -1)
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=processed["num_observers"],
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @torch.no_grad()
    def rollout_simulator(self, batch, ik_module, steps=60):
        raise NotImplementedError("Sparse rollout is not implemented yet.")
