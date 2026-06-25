import torch
import torch.nn as nn
import pytorch_lightning as pl

# Import your separated modules
from .scene_encoder import SceneEncoder
from .control_action_decoder import ActionDecoder
from ..utils.train_utils import transform_batch


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

        # The separated modules
        self.encoder = SceneEncoder(
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

    def forward(
        self,
        semantics,
        local_map_pos,
        local_map_dir,
        local_states_seq,
        target_actions=None,
        agent_valid=None,
    ):
        """
        Encode a sequence of T world snapshots, then decode actions autoregressively.

        Args:
            semantics:       [B, N] agent type IDs.
            local_map_pos:   [B, N, T, M, 2] map node positions in each agent's frame.
            local_map_dir:   [B, N, T, M, 2] map node directions in each agent's frame.
            local_states_seq: [B, N, N, T, 7] agent states in each observer's frame per timestep.
            target_actions:  [B, N, T] ground-truth action tokens for teacher forcing.
            agent_valid:     [B, N, T] - True if agent is valid at that timestep.

        Returns:
            logits: [B, N, T, num_action_bins] predicted action logits.
        """
        B, N, _, T, _ = local_states_seq.shape

        # --- Encode all T timesteps independently in parallel ---
        # Fold T into the batch dimension so the encoder processes them independently.
        semantics_flat = semantics.unsqueeze(1).expand(-1, T, -1).reshape(B * T, N)
        map_pos_flat = local_map_pos.permute(0, 2, 1, 3, 4).reshape(B * T, N, -1, 2)
        map_dir_flat = local_map_dir.permute(0, 2, 1, 3, 4).reshape(B * T, N, -1, 2)
        states_flat = local_states_seq.permute(0, 3, 1, 2, 4).reshape(B * T, N, N, 7)

        # TODO: Add temporal self-attention across the T encoded tokens
        # (currently each timestep is encoded independently). Not sure
        # how the paper handles this.
        scene_tokens = self.encoder(
            semantics_flat, map_pos_flat, map_dir_flat, states_flat
        )
        # [B*T, N, d_model]

        scene_tokens = scene_tokens.reshape(B, T, N, -1)  # [B, T, N, d_model]

        # --- Autoregressive decode ---
        logits = self.decoder(
            scene_tokens, target_actions, agent_valid
        )  # [B, N, T, num_bins]

        return logits

    def _shared_step(self, batch, batch_idx, prefix):
        total_loss = 0.0
        B = len(batch["agents"])

        # TODO: current implementation loops over the batch.
        # This is because scenarios cannot be stacked into a nice dense tensor
        # due to mismatches in the number of agents/map nodes. Ideally we should fix this
        # so training can make full use of the vectorized 'transform_batch'
        for i in range(B):
            history = batch["agents"][i]["history"]  # [N, 50, 7]
            future = batch["agents"][i]["future"]  # [N, 60, 7]
            full_traj = torch.cat([history, future], dim=-2)  # [N, 110, 7]

            graph_pos = batch["graph"][i]["position"]
            graph_dir = batch["graph"][i]["direction_vector"]
            semantics = batch["agents"][i]["agent_type_ids"]
            target_actions = batch["agents"][i]["action_tokens"]  # [N, 60]

            # --- Transform all 110 timesteps at once (fully vectorized) ---
            local_states, local_map_pos, local_map_dir = transform_batch(
                full_traj.unsqueeze(0),  # [1, N, 110, 7]
                graph_pos.unsqueeze(0),  # [1, M, 2]
                graph_dir.unsqueeze(0),  # [1, M, 2]
            )
            # local_states:  [1, N, N, 110, 7]
            # local_map_pos: [1, N, 110, M, 2]
            # local_map_dir: [1, N, 110, M, 2]

            # Index the 60 future timesteps (global indices 50..109)
            # TODO: Incorporate 50-step history encoding here
            states_seq = local_states[:, :, :, 50:, :]  # [1, N, N, 60, 7]
            map_pos_seq = local_map_pos[:, :, 50:, :, :]  # [1, N, 60, M, 2]
            map_dir_seq = local_map_dir[:, :, 50:, :, :]  # [1, N, 60, M, 2]

            agent_valid = future[:, :, 6].bool()  # [N, 60] - per-step valid flags

            logits = self(
                semantics=semantics.unsqueeze(0),
                local_map_pos=map_pos_seq,
                local_map_dir=map_dir_seq,
                local_states_seq=states_seq,
                target_actions=target_actions.unsqueeze(0),
                agent_valid=agent_valid.unsqueeze(0),
            )
            # logits: [1, N, 60, 3969]

            # --- Loss (mask invalid agents via valid flag) ---
            masked_targets = target_actions.masked_fill(~agent_valid, -1)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                masked_targets.reshape(-1),
            )
            total_loss = total_loss + loss

        avg_loss = total_loss / B

        self.log(
            f"{prefix}_loss",
            avg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
        )

        return avg_loss

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
        """
        Type B: The Physical Autoregressive Simulator.
        Predicts 1 step, updates the physical world, and repeats.

        Expects batch in the same format as _shared_step (lists of nested dicts).
        Currently only supports batch_size=1.
        """
        B = len(batch["agents"])
        assert B == 1, "rollout_simulator currently only supports batch_size=1"

        i = 0
        history = batch["agents"][i]["history"].clone()  # [N, 50, 7]
        graph_pos = batch["graph"][i]["position"]  # [M, 2]
        graph_dir = batch["graph"][i]["direction_vector"]  # [M, 2]
        semantics = batch["agents"][i]["agent_type_ids"]  # [N]
        device = history.device

        trajectories = history.clone()  # grows over time: [N, T_current, 7]
        N = trajectories.size(0)
        SOS = self.start_token_idx

        predictions: list[torch.Tensor] = []

        for step in range(steps):
            # Use only the most recent timestep (encoder processes single snapshots)
            current_traj = trajectories[:, -1:, :]  # [N, 1, 7]

            # Transform into agent-centric frames
            local_states, local_map_pos, local_map_dir = transform_batch(
                current_traj.unsqueeze(0),  # [1, N, 1, 7]
                graph_pos.unsqueeze(0),  # [1, M, 2]
                graph_dir.unsqueeze(0),  # [1, M, 2]
            )
            # local_states: [1, N, N, 1, 7] → squeeze T dim
            # local_map_pos: [1, N, 1, M, 2]
            # local_map_dir: [1, N, 1, M, 2]

            # Encode the single-timestep scene
            scene_token = self.encoder(
                semantics.unsqueeze(0),  # [1, N]
                local_map_pos.squeeze(2),  # [1, N, M, 2]
                local_map_dir.squeeze(2),  # [1, N, M, 2]
                local_states.squeeze(3),  # [1, N, N, 7]
            )  # [1, N, D]

            scene_tokens_seq = scene_token.unsqueeze(1)  # [1, 1, N, D]

            # Build prev action tokens: SOS + previously predicted tokens
            if step == 0:
                target_actions = torch.full(
                    (1, N, 1), SOS, dtype=torch.long, device=device
                )
            else:
                prev = torch.stack(predictions, dim=-1)  # [N, step]
                target_actions = torch.cat(
                    [
                        torch.full((1, N, 1), SOS, dtype=torch.long, device=device),
                        prev.unsqueeze(0),
                    ],
                    dim=-1,
                )  # [1, N, step+1]

            # Valid mask from last observed timestep
            valid = trajectories[:, -1, 6].bool()  # [N]
            agent_valid = valid.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]

            logits = self.decoder(scene_tokens_seq, target_actions, agent_valid)
            next_action_tokens = logits[0, :, -1, :].argmax(dim=-1)  # [N]

            # Only update the physically simulated state for valid agents
            next_action_tokens = torch.where(
                valid,
                next_action_tokens,
                torch.tensor(ik_module.stationary_idx, device=device),
            )
            predictions.append(next_action_tokens)

            # ==========================================================
            # E. THE PHYSICS ENGINE: Update the World
            # ==========================================================

            local_dx, local_dy, d_yaw, d_vel = ik_module.decode(next_action_tokens)

            # Current absolute state
            last_state = trajectories[:, -1, :]  # [N, 7]
            curr_x, curr_y = last_state[:, 0], last_state[:, 1]
            curr_cos, curr_sin = last_state[:, 2], last_state[:, 3]
            curr_vx, curr_vy = last_state[:, 4], last_state[:, 5]
            valid_flag = last_state[:, 6]

            # Rotate local displacement → global
            global_dx = local_dx * curr_cos - local_dy * curr_sin
            global_dy = local_dx * curr_sin + local_dy * curr_cos

            new_x = curr_x + global_dx
            new_y = curr_y + global_dy

            current_yaw = torch.atan2(curr_sin, curr_cos)
            new_yaw = current_yaw + d_yaw
            new_cos = torch.cos(new_yaw)
            new_sin = torch.sin(new_yaw)

            speed = torch.hypot(curr_vx, curr_vy) + d_vel
            new_vx = speed * new_cos
            new_vy = speed * new_sin

            new_state = torch.stack(
                [new_x, new_y, new_cos, new_sin, new_vx, new_vy, valid_flag], dim=-1
            )  # [N, 7]

            trajectories = torch.cat([trajectories, new_state.unsqueeze(1)], dim=1)

        predicted_tokens = torch.stack(predictions, dim=-1)  # [N, 60]
        predicted_paths = trajectories[:, 50:, :]  # [N, 60, 7]

        return predicted_tokens, predicted_paths
