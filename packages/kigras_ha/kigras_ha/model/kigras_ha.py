import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .scene_encoder import InteractiveSceneEncoder
from .control_action_decoder import ActionDecoder
from ..utils.train_utils import (
    transform_batch,
    filter_agents,
    frame_invariant_features,
)
from ..kinematics.ctra import CTRAKinematics


class KiGRAS_HA(pl.LightningModule):
    def __init__(
        self,
        vocab_size=None,
        num_accel_bins=63,
        num_steer_bins=63,
        num_semantic_classes=10,
        lr=2e-4,
        weight_decay=1e-4,
        gradient_checkpointing=False,
        max_agents: int = 0,
        require_complete_history: bool = False,
        grid_power: float = 1.0,
        k_neighbors: int = 16,
    ):
        if vocab_size is None:
            vocab_size = num_accel_bins * num_steer_bins

        super().__init__()
        self.save_hyperparameters()

        self.encoder = InteractiveSceneEncoder(
            num_semantic_classes=num_semantic_classes,
            vocab_size=vocab_size,
        )
        # Causal temporal decoder over the 60-step future sequence
        self.decoder = ActionDecoder(num_action_bins=vocab_size, seq_len=60)

        # SOS token index — one past the last real action bin
        self.start_token_idx = vocab_size

        # -1 targets are ignored (padded agents and agents that left the scene)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # When True, recomputes encoder activations during backward instead of
        # storing all 60 of them — trades ~2x compute for much lower peak VRAM.
        self.gradient_checkpointing = gradient_checkpointing

        # max_agents=0 means disabled (keep all agents after padding).
        # Any positive value applies filter_agents() before encoding.
        self.max_agents = max_agents
        self.require_complete_history = require_complete_history

        # Sparse interaction: each observer pools only its k nearest neighbours
        self.k_neighbors = k_neighbors

        # Used to compute action tokens on-the-fly when the dataset was not
        # preprocessed.  max_iter=0 means analytic-only (no L-BFGS) for speed.
        self._ctra = CTRAKinematics(
            dt=0.1,
            max_iter=0,
            num_accel=num_accel_bins,
            num_yaw=num_steer_bins,
            grid_power=grid_power,
        )

    # ------------------------------------------------------------------
    # Single-step encode  (used directly during rollout)
    # ------------------------------------------------------------------
    def forward(
        self,
        semantics,
        local_map_pos,
        local_map_dir,
        history_embedding,
        neighbor_states,
        prev_actions,
        padding_mask=None,
    ):
        """
        Encodes ONE timestep's scene into per-agent tokens.

        Args:
            semantics:         (B, N)
            local_map_pos:     (B, N, M, 2)
            local_map_dir:     (B, N, M, 2)
            history_embedding: (B, N, d_model) — causal GRU readout for this step
                               (see encoder.encode_history)
            neighbor_states:   (B, N, k, 7)  — k nearest agents per observer
            prev_actions:      (B, N) long   — previous action token, or SOS
            padding_mask:      (B, N) bool   — True for invalid/padded agents
        Returns:
            (B, N, d_model) — interaction-aware scene token per agent
        """
        return self.encoder(
            semantics,
            local_map_pos,
            local_map_dir,
            history_embedding,
            neighbor_states,
            prev_actions,
            padding_mask=padding_mask,
        )

    # ------------------------------------------------------------------
    # Training / validation step
    # ------------------------------------------------------------------
    def _shared_step(self, batch, _batch_idx, prefix):
        """
        Autoregressive Sequence Training with Teacher Forcing.

        The history GRU runs ONCE over the full 110-step ground-truth
        trajectory (frame-invariant per-step features).  Because the GRU is a
        causal left-to-right recurrence, its hidden output at frame 49+t
        conditions only on frames <= 49+t — exactly the teacher-forcing
        window the old per-step re-encoding provided, at 1/60th of the cost.

        For each future timestep t in [0, 60):
          1. Take the GRU readout at frame 49+t  +  cheap per-step context
             (map, k nearest neighbours, GROUND-TRUTH action from t-1)
        Stack all 60 scene tokens  →  (B, N, 60, d_model)
        Run causal decoder          →  (B, N, 60, 3969) logits
        Cross-entropy loss against ground-truth action tokens.
        """
        # =====================================================================
        # 1. Unpack and pad the variable-length batch
        # =====================================================================
        hist_list = [self.to_tensor(item["history"]) for item in batch["agents"]]
        fut_list = [self.to_tensor(item["future"]) for item in batch["agents"]]
        sem_list = [
            self.to_tensor(item["agent_type_ids"], dtype=torch.long)
            for item in batch["agents"]
        ]

        action_list = []
        for item in batch["agents"]:
            if "action_tokens" in item:
                action_list.append(
                    self.to_tensor(item["action_tokens"], dtype=torch.long)
                )
            else:
                n_agents = len(item["history"])
                action_list.append(
                    torch.full((n_agents, 60), -1, dtype=torch.long, device=self.device)
                )

        pos_list = [self.to_tensor(g["position"]) for g in batch["graph"]]
        dir_list = [self.to_tensor(g["direction_vector"]) for g in batch["graph"]]

        history = pad_sequence(
            hist_list, batch_first=True, padding_value=0.0
        )  # (B, N, 50, 7)
        future = pad_sequence(
            fut_list, batch_first=True, padding_value=0.0
        )  # (B, N, 60, 7)
        semantics = pad_sequence(sem_list, batch_first=True, padding_value=0)  # (B, N)
        all_target_actions = pad_sequence(
            action_list, batch_first=True, padding_value=-1
        )  # (B, N, 60)
        graph_pos = pad_sequence(
            pos_list, batch_first=True, padding_value=0.0
        )  # (B, M, 2)
        graph_dir = pad_sequence(
            dir_list, batch_first=True, padding_value=0.0
        )  # (B, M, 2)

        full_trajectories = torch.cat([history, future], dim=2)  # (B, N, 110, 7)

        if self.max_agents > 0:
            full_trajectories, semantics, all_target_actions = filter_agents(
                full_trajectories,
                semantics,
                all_target_actions,
                max_agents=self.max_agents,
                require_complete_history=self.require_complete_history,
            )

        B, N = semantics.shape

        # =====================================================================
        # 1b. Compute action tokens on-the-fly if dataset was not preprocessed
        # =====================================================================
        if all_target_actions.eq(-1).all():
            all_target_actions = self._tokens_from_trajectories(full_trajectories)

        # =====================================================================
        # 2. Build teacher-forced U-embedding inputs
        #
        #    At timestep t the encoder receives the action chosen at t-1.
        #    At t=0 there is no prior action, so we use the SOS token.
        #    Padded agents have target=-1; clamp to 0 for the embedding lookup
        #    (their loss contribution is masked out anyway).
        # =====================================================================
        sos = torch.full(
            (B, N, 1), self.start_token_idx, dtype=torch.long, device=self.device
        )
        # prev_actions_seq[:, :, t] = ground-truth action at t-1  (SOS at t=0)
        prev_actions_seq = torch.cat([sos, all_target_actions[:, :, :-1]], dim=2).clamp(
            min=0
        )  # (B, N, 60)

        # =====================================================================
        # 3a. Encode ALL histories in ONE causal GRU pass
        #
        #     The readout at frame 49+t conditions only on frames <= 49+t
        #     (left-to-right recurrence), so this replaces the 60 sliding-
        #     window re-encodes without leaking future information.
        # =====================================================================
        feats = frame_invariant_features(full_trajectories)  # (B, N, 110, F)

        if self.gradient_checkpointing and self.training:
            hist_outputs = grad_checkpoint(
                lambda f: self.encoder.encode_history(f)[0],
                feats,
                use_reentrant=False,
            )
        else:
            hist_outputs, _ = self.encoder.encode_history(feats)  # (B, N, 110, d)

        # Readout for future step t is the hidden state after frame 49+t
        hist_readouts = hist_outputs[:, :, 49:109, :]  # (B, N, 60, d)

        # =====================================================================
        # 3b. Cheap per-step fusion + interaction for each future timestep
        # =====================================================================
        scene_tokens = []
        valid_flags = []

        for t in range(60):
            present = full_trajectories[:, :, t + 49, :]  # (B, N, 7)
            prev_act_t = prev_actions_seq[:, :, t]  # (B, N)
            e_h_t = hist_readouts[:, :, t, :]  # (B, N, d)

            if self.gradient_checkpointing and self.training:
                # Recompute the transform+encode on the backward pass instead of
                # caching all 60 sets of activations simultaneously.
                def _encode(p, gp, gd, sem, eh, pa):
                    nbr, lmp, lmd, pad = transform_batch(
                        p, gp, gd, k_neighbors=self.k_neighbors
                    )
                    return self.encoder(sem, lmp, lmd, eh, nbr, pa, padding_mask=pad)

                token = grad_checkpoint(
                    _encode,
                    present,
                    graph_pos,
                    graph_dir,
                    semantics,
                    e_h_t,
                    prev_act_t,
                    use_reentrant=False,
                )
            else:
                neighbor_states, local_map_pos, local_map_dir, padding_mask = (
                    transform_batch(
                        present, graph_pos, graph_dir, k_neighbors=self.k_neighbors
                    )
                )
                token = self.encoder(
                    semantics,
                    local_map_pos,
                    local_map_dir,
                    e_h_t,
                    neighbor_states,
                    prev_act_t,
                    padding_mask=padding_mask,
                )  # (B, N, d_model)

            scene_tokens.append(token)
            # Agent is valid at this step if it exists at the present frame
            valid_flags.append(present[:, :, 6].bool())  # (B, N)

        # (B, N, 60, d_model)
        scene_token_seq = torch.stack(scene_tokens, dim=2)
        # (B, N, 60)
        valid_mask = torch.stack(valid_flags, dim=2)

        # =====================================================================
        # 4. Causal decode over all 60 steps in one pass
        # =====================================================================
        logits = self.decoder(scene_token_seq)  # (B, N, 60, num_action_bins)

        # =====================================================================
        # 5. Loss — ignore padded agents and agents that left the scene
        # =====================================================================
        masked_targets = all_target_actions.masked_fill(~valid_mask, -1)  # (B, N, 60)

        logits_flat = logits.view(-1, logits.size(-1))  # (B*N*60, num_action_bins)
        targets_flat = masked_targets.view(-1)  # (B*N*60,)

        loss = self.criterion(logits_flat, targets_flat)

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    # ------------------------------------------------------------------
    # On-the-fly action token computation (fallback for raw datasets)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _tokens_from_trajectories(
        self, full_trajectories: torch.Tensor
    ) -> torch.Tensor:
        """Derive CTRA action tokens analytically from ground-truth states.

        Used when the dataset was loaded in raw mode (no preprocess_kinematics.py
        run), so action_tokens are absent.  Analytical finite-differences give
        the exact tokens for noise-free data; for real sensor data they're a
        good approximation (L-BFGS is skipped for speed).

        Args:
            full_trajectories: (B, N, 110, 7) — history + future concatenated
        Returns:
            token_ids: (B, N, 60) long — one discrete action per future step
        """
        B, N = full_trajectories.shape[:2]

        # 61-step window: last history frame (t=49) + 60 future frames (t=50..109)
        seg = full_trajectories[:, :, 49:110, :]  # (B, N, 61, 7)
        v = torch.hypot(seg[..., 4], seg[..., 5])  # (B, N, 61)
        states = torch.stack(
            [seg[..., 0], seg[..., 1], seg[..., 2], seg[..., 3], v], dim=-1
        )  # (B, N, 61, 5)
        valid = seg[..., 6].bool()  # (B, N, 61)

        actions = self._ctra.solve(
            states.view(B * N, 61, 5),
            quantize=True,
            valid_mask=valid.view(B * N, 61),
        )  # (B*N, 60, 2)
        tokens = self._ctra.actions_to_token_ids(actions)  # (B*N, 60)
        return tokens.view(B, N, 60).clamp(min=0)

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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ------------------------------------------------------------------
    # Autoregressive rollout (inference)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout_simulator(self, batch, ik_module, steps=60):
        """
        Autoregressive physical simulator.

        The GRU hidden state is warmed up ONCE on the observed history and then
        carried across rollout steps — each new predicted frame is folded in
        with a single GRU step instead of re-encoding the whole 50-frame
        window (O(1) history encoding per step).

        At each step:
          1. Fuse the carried history embedding with the current scene context
             into a scene token, using the model's own previous prediction as
             the U-embedding.
          2. Append the token to the growing sequence and run the causal decoder
             over the full sequence — the last position gives the current action.
          3. Decode the chosen token into physical deltas, advance the world,
             and update the GRU hidden state with the new frame.

        The causal decoder is re-run over the full (growing) sequence at each
        step, which is correct but O(T²). For long horizons, KV-caching can be
        added as an optimisation without changing the semantics.
        """
        trajectories = batch["trajectories"].clone()  # (B, N, 50, 7)
        graph_pos = batch["graph_position"]  # (B, M, 2)
        graph_dir = batch["graph_direction"]  # (B, M, 2)
        semantics = batch["agent_type_ids"]  # (B, N)

        B, N = semantics.shape

        # SOS as the U-embedding for step 0
        prev_action_tokens = torch.full(
            (B, N), self.start_token_idx, dtype=torch.long, device=self.device
        )

        # Warm up the causal GRU on the observed history — done exactly once
        hist_feats = frame_invariant_features(trajectories)  # (B, N, 50, F)
        hist_outputs, gru_state = self.encoder.encode_history(hist_feats)
        history_embedding = hist_outputs[:, :, -1, :]  # (B, N, d_model)

        scene_token_history = []
        predictions = []

        for _ in range(steps):
            # ----------------------------------------------------------------
            # A. Fuse the carried history embedding with current scene context
            # ----------------------------------------------------------------
            present = trajectories[:, :, -1, :]  # (B, N, 7)

            neighbor_states, local_map_pos, local_map_dir, padding_mask = (
                transform_batch(
                    present, graph_pos, graph_dir, k_neighbors=self.k_neighbors
                )
            )

            scene_token = self.encoder(
                semantics,
                local_map_pos,
                local_map_dir,
                history_embedding,
                neighbor_states,
                prev_action_tokens,
                padding_mask=padding_mask,
            )  # (B, N, d_model)
            scene_token_history.append(scene_token)

            # ----------------------------------------------------------------
            # B. Causal decode over the accumulated sequence; take the last step
            # ----------------------------------------------------------------
            scene_seq = torch.stack(
                scene_token_history, dim=2
            )  # (B, N, step+1, d_model)
            logits = self.decoder(scene_seq)  # (B, N, step+1, num_bins)

            next_action_tokens = logits[:, :, -1, :].argmax(dim=-1)  # (B, N)
            predictions.append(next_action_tokens)
            prev_action_tokens = next_action_tokens

            # ----------------------------------------------------------------
            # C. Physics engine — advance the world state
            # ----------------------------------------------------------------
            local_dx, local_dy, d_yaw, d_vel = ik_module.decode(next_action_tokens)

            last_state = trajectories[:, :, -1, :]
            curr_x, curr_y = last_state[..., 0], last_state[..., 1]
            curr_cos, curr_sin = last_state[..., 2], last_state[..., 3]
            curr_vx, curr_vy = last_state[..., 4], last_state[..., 5]
            valid = last_state[..., 6]

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
                [new_x, new_y, new_cos, new_sin, new_vx, new_vy, valid], dim=-1
            )  # (B, N, 7)

            trajectories = torch.cat([trajectories, new_state.unsqueeze(2)], dim=2)

            # ----------------------------------------------------------------
            # D. Fold the new frame into the carried GRU state (single step).
            #    The last two frames give the (t-1, t) pair the per-step
            #    features need; only the newest step is fed to the GRU.
            # ----------------------------------------------------------------
            step_feats = frame_invariant_features(trajectories[:, :, -2:, :])
            step_outputs, gru_state = self.encoder.encode_history(
                step_feats[:, :, -1:, :], gru_state
            )
            history_embedding = step_outputs[:, :, -1, :]  # (B, N, d_model)

        predicted_tokens = torch.stack(predictions, dim=-1)  # (B, N, 60)
        predicted_paths = trajectories[:, :, 50:, :]  # (B, N, 60, 7)
        return predicted_tokens, predicted_paths

    def to_tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)
