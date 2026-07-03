import torch
import torch.nn as nn
import pytorch_lightning as pl

# Import your separated modules
from .scene_encoder import SparseSceneEncoder
from .decoder import ActionDecoder
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
        decoder_ff_dim=256,
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
            ff_dim=decoder_ff_dim,
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
        processed["prev_actions"] = action_window[:, :-1]
        processed["prev_actions_valid"] = action_window_valid[:, :-1]
        processed["target_actions"] = action_window[:, 1:]
        processed["target_actions_valid"] = action_window_valid[:, 1:]

        logits = self.decoder(
            scene_tokens,
            prev_actions=processed["prev_actions"],
            prev_actions_valid=processed["prev_actions_valid"],
            scene_valid=processed["observer_valid"],
        )
        return logits, processed

    def _shared_step(self, batch, batch_idx, prefix):
        logits, processed = self(batch)

        targets = processed["target_actions"]
        target_valid = processed["target_actions_valid"]
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

    def _tokens_to_actions(self, tokens: torch.Tensor, kinematic_model) -> torch.Tensor:
        num_yaw = int(kinematic_model.NUM_YAW)
        accel_bins = kinematic_model.accel_bins.to(device=tokens.device)
        yaw_rate_bins = kinematic_model.yaw_rate_bins.to(device=tokens.device)

        accel_idx = tokens // num_yaw
        yaw_idx = tokens % num_yaw
        return torch.stack([accel_bins[accel_idx], yaw_rate_bins[yaw_idx]], dim=-1)

    @torch.no_grad()
    def rollout(self, batch: dict, kinematic_model) -> dict:
        self.eval()

        rollout_batch = dict(batch)
        rollout_batch["agents"] = dict(batch["agents"])
        rollout_batch["agents"]["states"] = batch["agents"]["states"].clone()

        observer_idx = batch["observers"]["agent_idx"]
        prev_actions = batch["observers"]["action_tokens"][
            :, CURRENT_STEP - 1 : CURRENT_STEP
        ]
        prev_actions_valid = batch["observers"]["action_valid"][
            :, CURRENT_STEP - 1 : CURRENT_STEP
        ]
        scene_tokens = None
        scene_valid = None
        pred_tokens = []
        pred_states = []

        for t in range(CURRENT_STEP, CURRENT_STEP + PRED_LEN):
            processed_t = process_sparse_batch(rollout_batch, start_t=t, horizon=1)
            scene_t = self.encoder(processed_t)

            if scene_tokens is None:
                scene_tokens = scene_t
                scene_valid = processed_t["observer_valid"]
            else:
                scene_tokens = torch.cat([scene_tokens, scene_t], dim=1)
                scene_valid = torch.cat(
                    [scene_valid, processed_t["observer_valid"]], dim=1
                )

            logits = self.decoder(
                scene_tokens,
                prev_actions=prev_actions,
                prev_actions_valid=prev_actions_valid,
                scene_valid=scene_valid,
            )
            pred_token = logits[:, -1].argmax(dim=-1)
            pred_tokens.append(pred_token)

            current_state = rollout_batch["agents"]["states"][observer_idx, t]
            speed = torch.linalg.norm(current_state[:, 4:6], dim=-1)
            kinematic_state = torch.stack(
                [
                    current_state[:, 0],
                    current_state[:, 1],
                    current_state[:, 2],
                    current_state[:, 3],
                    speed,
                ],
                dim=-1,
            )
            action = self._tokens_to_actions(pred_token, kinematic_model).unsqueeze(1)
            next_kinematic_state = kinematic_model.forward(kinematic_state, action)[
                :, -1
            ]

            next_state = current_state.new_zeros(current_state.shape)
            next_state[:, 0:4] = next_kinematic_state[:, 0:4]
            next_state[:, 4] = next_kinematic_state[:, 4] * next_kinematic_state[:, 2]
            next_state[:, 5] = next_kinematic_state[:, 4] * next_kinematic_state[:, 3]
            next_state[:, 6] = current_state[:, 6]
            rollout_batch["agents"]["states"][observer_idx, t + 1] = next_state
            pred_states.append(next_state)

            prev_actions = torch.cat([prev_actions, pred_token.unsqueeze(1)], dim=1)
            prev_actions_valid = torch.cat(
                [
                    prev_actions_valid,
                    torch.ones_like(pred_token, dtype=torch.bool).unsqueeze(1),
                ],
                dim=1,
            )

        states = torch.stack(pred_states, dim=1)
        return {
            "tokens": torch.stack(pred_tokens, dim=1),
            "states": states,
            "xy": states[..., :2],
        }
