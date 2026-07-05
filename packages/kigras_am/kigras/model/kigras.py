import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any

# Import your separated modules
from .scene_encoder import SparseSceneEncoder
from .decoder import ActionDecoder
from ..dataloader.dataset import CURRENT_STEP, PRED_LEN
from ..utils.train_utils import process_sparse_batch


OPTIMIZER_REGISTRY: dict[str, Any] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

SCHEDULER_REGISTRY: dict[str, Any] = {
    "onecycle": torch.optim.lr_scheduler.OneCycleLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}


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
        optimizer_config: dict[str, Any] | None = None,
        scheduler_config: dict[str, Any] | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
    ):
        vocab_size = vocab_size

        super().__init__()
        if optimizer_config is None:
            optimizer_config = {
                "name": "adamw",
                "params": {
                    "lr": 2e-4 if lr is None else lr,
                    "weight_decay": 1e-4 if weight_decay is None else weight_decay,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
            }
        if scheduler_config is None:
            scheduler_config = {
                "name": "onecycle",
                "interval": "step",
                "frequency": 1,
                "params": {
                    "max_lr": optimizer_config["params"]["lr"],
                    "pct_start": 0.3,
                    "anneal_strategy": "cos",
                },
            }

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
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

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        cfg = dict(self.optimizer_config)
        name = str(cfg.pop("name")).lower()
        params = dict(cfg.pop("params", {}))
        if cfg:
            raise ValueError(f"Unexpected optimizer config keys: {sorted(cfg)}")

        optimizer_cls = OPTIMIZER_REGISTRY.get(name)
        if optimizer_cls is None:
            supported = ", ".join(sorted(OPTIMIZER_REGISTRY))
            raise ValueError(f"Unsupported optimizer '{name}'. Supported: {supported}")
        return optimizer_cls(self.parameters(), **params)

    def _configure_scheduler(self, optimizer: torch.optim.Optimizer) -> dict | None:
        cfg = dict(self.scheduler_config)
        name = str(cfg.pop("name", "none")).lower()
        interval = str(cfg.pop("interval", "epoch"))
        frequency = int(cfg.pop("frequency", 1))
        monitor = cfg.pop("monitor", None)
        params = dict(cfg.pop("params", {}))
        if cfg:
            raise ValueError(f"Unexpected scheduler config keys: {sorted(cfg)}")

        if name == "none":
            return None

        scheduler_cls = SCHEDULER_REGISTRY.get(name)
        if scheduler_cls is None:
            supported = ", ".join(["none", *sorted(SCHEDULER_REGISTRY)])
            raise ValueError(f"Unsupported scheduler '{name}'. Supported: {supported}")

        if name == "onecycle":
            params.setdefault("total_steps", self.trainer.estimated_stepping_batches)
        elif name == "cosine":
            params.setdefault(
                "T_max",
                (
                    self.trainer.estimated_stepping_batches
                    if interval == "step"
                    else self.trainer.max_epochs
                ),
            )

        scheduler_config = {
            "scheduler": scheduler_cls(optimizer, **params),
            "interval": interval,
            "frequency": frequency,
        }
        if monitor is not None:
            scheduler_config["monitor"] = monitor
        return scheduler_config

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()
        scheduler = self._configure_scheduler(optimizer)
        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
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
