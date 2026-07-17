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
        fusion_chunk_size: int = 15,
        embedding_dim: int = 64,
        num_head: int = 4,
        encoder_num_layers: int = 3,
        decoder_num_layers: int = 3,
        dropout: float = 0.1,
        scheduler_config: dict | None = None,
    ):
        if vocab_size is None:
            vocab_size = num_accel_bins * num_steer_bins

        super().__init__()
        self.save_hyperparameters()

        self.encoder = InteractiveSceneEncoder(
            num_semantic_classes=num_semantic_classes,
            vocab_size=vocab_size,
            d_model=embedding_dim,
            nhead=num_head,
            num_layers=encoder_num_layers,
            dropout=dropout,
        )
        # Causal temporal decoder over the 60-step future sequence
        self.decoder = ActionDecoder(
            num_action_bins=vocab_size,
            seq_len=60,
            d_model=embedding_dim,
            nhead=num_head,
            num_layers=decoder_num_layers,
            dropout=dropout,
        )

        # Optional LR-scheduler override (see configure_optimizers); None keeps
        # the default OneCycleLR behaviour.
        self.scheduler_config = scheduler_config

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

        # Teacher forcing has no sequential dependency in the encoder, so the
        # 60 per-step fusions are batched (time folded into the batch dim) in
        # chunks of this size — one big kernel launch per chunk instead of 60
        # small ones.  Larger = faster but higher peak memory; 0 = all 60 at
        # once.  Combined with gradient_checkpointing, each chunk is one
        # checkpoint block.
        self.fusion_chunk_size = fusion_chunk_size

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
    # Time-vectorized per-step fusion (teacher forcing only)
    # ------------------------------------------------------------------
    def _fuse_chunk(
        self,
        presents,
        graph_pos,
        graph_dir,
        semantics,
        hist_embeddings,
        prev_actions,
    ):
        """Fuse+interact C timesteps at once by folding time into the batch.

        Under teacher forcing all inputs (ground-truth poses, neighbour sets,
        previous actions, GRU readouts) are known upfront, so the C fusion
        passes are independent and can run as one (B·C)-batch forward instead
        of C sequential ones.  The interaction transformer still attends only
        over the N agents WITHIN each timestep — identical math to the loop.

        Args:
            presents:        (B, C, N, 7) — ground-truth state at each step
            graph_pos:       (B, M, 2)
            graph_dir:       (B, M, 2)
            semantics:       (B, N)
            hist_embeddings: (B, C, N, d) — GRU readouts for these steps
            prev_actions:    (B, C, N) long
        Returns:
            (B, C, N, d) scene tokens
        """
        B, C, N, _ = presents.shape
        M = graph_pos.shape[1]

        p = presents.reshape(B * C, N, 7)
        gp = graph_pos.unsqueeze(1).expand(-1, C, -1, -1).reshape(B * C, M, 2)
        gd = graph_dir.unsqueeze(1).expand(-1, C, -1, -1).reshape(B * C, M, 2)
        sem = semantics.unsqueeze(1).expand(-1, C, -1).reshape(B * C, N)
        eh = hist_embeddings.reshape(B * C, N, -1)
        pa = prev_actions.reshape(B * C, N)

        neighbor_states, local_map_pos, local_map_dir, padding_mask = transform_batch(
            p, gp, gd, k_neighbors=self.k_neighbors
        )
        tokens = self.encoder(
            sem,
            local_map_pos,
            local_map_dir,
            eh,
            neighbor_states,
            pa,
            padding_mask=padding_mask,
        )  # (B*C, N, d)
        return tokens.view(B, C, N, -1)

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
        # 3b. Time-vectorized fusion + interaction over the 60 future steps
        #
        #     No sequential dependency exists here under teacher forcing, so
        #     time is folded into the batch dimension and processed in chunks
        #     of fusion_chunk_size (memory/speed trade-off).  With gradient
        #     checkpointing each chunk is one recompute block.
        # =====================================================================
        # Time-major views: (B, 60, N, ...)
        presents_all = full_trajectories[:, :, 49:109, :].transpose(1, 2)  # (B,60,N,7)
        hist_emb_all = hist_readouts.transpose(1, 2)  # (B, 60, N, d)
        prev_acts_all = prev_actions_seq.transpose(1, 2)  # (B, 60, N)

        chunk_size = self.fusion_chunk_size if self.fusion_chunk_size > 0 else 60

        scene_chunks = []
        for start in range(0, 60, chunk_size):
            sl = slice(start, min(start + chunk_size, 60))

            if self.gradient_checkpointing and self.training:
                # Recompute the transform+encode on the backward pass instead
                # of caching every chunk's activations simultaneously.
                token_chunk = grad_checkpoint(
                    self._fuse_chunk,
                    presents_all[:, sl],
                    graph_pos,
                    graph_dir,
                    semantics,
                    hist_emb_all[:, sl],
                    prev_acts_all[:, sl],
                    use_reentrant=False,
                )
            else:
                token_chunk = self._fuse_chunk(
                    presents_all[:, sl],
                    graph_pos,
                    graph_dir,
                    semantics,
                    hist_emb_all[:, sl],
                    prev_acts_all[:, sl],
                )  # (B, C, N, d)

            scene_chunks.append(token_chunk)

        # (B, N, 60, d_model) — contiguous for the decoder's .view()
        scene_token_seq = torch.cat(scene_chunks, dim=1).transpose(1, 2).contiguous()
        # Agent is valid at step t if it exists at the present frame — (B, N, 60)
        valid_mask = full_trajectories[:, :, 49:109, 6].bool()

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

        # Default: step-wise OneCycleLR (max_lr = lr, 30% warmup, cosine anneal).
        # A scheduler_config dict {name, interval, params} overrides it so the
        # schedule is tunable (mirrors kigras_am's registry approach).
        cfg = dict(self.scheduler_config or {})
        name = str(cfg.pop("name", "onecycle")).lower()
        interval = str(cfg.pop("interval", "step"))
        params = dict(cfg.pop("params", {}))

        if name == "none":
            return optimizer

        if name == "onecycle":
            params.setdefault("max_lr", self.hparams.lr)
            params.setdefault("total_steps", self.trainer.estimated_stepping_batches)
            params.setdefault("pct_start", 0.3)
            params.setdefault("anneal_strategy", "cos")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **params)
        elif name == "cosine":
            params.setdefault(
                "T_max",
                self.trainer.estimated_stepping_batches
                if interval == "step"
                else self.trainer.max_epochs,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
        else:
            raise ValueError(
                f"Unsupported scheduler '{name}'. Use onecycle|cosine|none."
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": interval},
        }

    # ------------------------------------------------------------------
    # Autoregressive rollout (inference)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout_simulator(
        self,
        history: torch.Tensor,  # (B, N, obs_len, 7)
        graph_pos: torch.Tensor,  # (B, M, 2)
        graph_dir: torch.Tensor,  # (B, M, 2)
        semantics: torch.Tensor,  # (B, N) long
        *,
        num_samples: int = 1,
        top_k: int = 5,
        sample_tokens: bool = True,
        steps: int = 60,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive physical rollout — the single inference path.

        The history GRU is warmed up ONCE on the observed window and then carried
        across rollout steps, each new frame folded in with a single GRU step
        (O(1) history encoding per step) — the incremental equivalent of the
        teacher-forced encoding in ``_shared_step``.

        At each step:
          1. Fuse the carried GRU readout with the current-frame scene context
             (k nearest neighbours + local map) and the previous action token.
          2. Append the token, run the causal decoder over the growing sequence,
             and take the last position's logits.
          3. Pick the action, matching SMART's scheme (``smart/modules/agent_decoder.py``):
             softmax at temperature 1, truncate to the ``top_k`` most likely tokens,
             then draw from the renormalized top-k — or take top-1 deterministically
             when ``sample_tokens`` is False.
          4. Decode the token on the model's own CTRA grid (``self._ctra``) and
             advance the world one step, then fold the new frame into the GRU.

        The decoder is re-run over the full growing sequence each step (O(T²));
        KV-caching could be added later without changing the semantics.

        Args:
            history: observed states per agent, (B, N, obs_len, 7).
            graph_pos, graph_dir: lane graph, (B, M, 2).
            semantics: agent type ids, (B, N).
            num_samples: independent rollouts per scene.
            top_k: keep only the k most likely tokens before sampling (SMART's
                   ``beam_size``, default 5). ``0`` disables truncation and
                   samples over the full vocabulary.
            sample_tokens: draw from the top-k distribution (True) or take the
                   deterministic top-1 token (False). Independent of
                   ``num_samples``: with False, every rollout is identical.
            steps: future horizon to roll out.

        Returns:
            pred_states: (num_samples, B, N, steps, 7) full future states.
            pred_tokens: (num_samples, B, N, steps) chosen action-token ids.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        ctra = self._ctra
        device = history.device
        num_yaw = ctra.NUM_YAW
        accel_bins = ctra.accel_bins.to(device)
        yaw_bins = ctra.yaw_rate_bins.to(device)
        B, N, obs_len, _ = history.shape

        state_samples: list[torch.Tensor] = []
        token_samples: list[torch.Tensor] = []

        for _ in range(num_samples):
            traj = history.clone()  # grows to (B, N, obs_len + t, 7)
            prev_actions = torch.full(
                (B, N), self.start_token_idx, dtype=torch.long, device=device
            )
            scene_tokens: list[torch.Tensor] = []
            step_tokens: list[torch.Tensor] = []

            # Warm up the causal history GRU once; carry the hidden state.
            hist_feats = frame_invariant_features(traj)  # (B, N, obs_len, F)
            hist_outputs, gru_state = self.encoder.encode_history(hist_feats)
            history_embedding = hist_outputs[:, :, -1, :]  # (B, N, d) — readout @ now

            for _step in range(steps):
                present = traj[:, :, -1, :]  # (B, N, 7)
                nbr, lmp, lmd, pmask = transform_batch(
                    present, graph_pos, graph_dir, k_neighbors=self.k_neighbors
                )
                token = self.encoder(
                    semantics,
                    lmp,
                    lmd,
                    history_embedding,
                    nbr,
                    prev_actions,
                    padding_mask=pmask,
                )
                scene_tokens.append(token)

                # Causal decode over the growing sequence; keep the last step.
                seq = torch.stack(scene_tokens, dim=2)  # (B, N, step+1, d)
                logits = self.decoder(seq)[:, :, -1, :]  # (B, N, vocab)

                if sample_tokens:
                    # SMART parity: softmax at T=1, keep the top-k tokens, then
                    # sample from them. multinomial renormalizes the retained
                    # weights internally, so no explicit rescale is needed.
                    probs = torch.softmax(logits, dim=-1)  # (B, N, vocab)
                    k = probs.size(-1) if top_k <= 0 else min(top_k, probs.size(-1))
                    topk_prob, topk_idx = torch.topk(probs, k=k, dim=-1)
                    choice = torch.multinomial(topk_prob.view(B * N, k), 1)  # (B*N, 1)
                    action_tokens = (
                        topk_idx.view(B * N, k).gather(1, choice).view(B, N)
                    )
                else:
                    action_tokens = logits.argmax(dim=-1)  # (B, N) — SMART top-1

                step_tokens.append(action_tokens)
                prev_actions = action_tokens

                # Token → (accel, yaw_rate) on the model's own CTRA grid, then
                # advance one physics step from the current state.
                a = accel_bins[action_tokens // num_yaw]  # (B, N)
                omega = yaw_bins[action_tokens % num_yaw]  # (B, N)
                last = traj[:, :, -1, :]  # (B, N, 7)
                v = torch.hypot(last[..., 4], last[..., 5])  # (B, N)
                init_state = torch.stack(
                    [last[..., 0], last[..., 1], last[..., 2], last[..., 3], v],
                    dim=-1,
                )  # (B, N, 5)
                next_state = ctra.forward(
                    init_state,
                    torch.stack([a, omega], dim=-1).unsqueeze(-2),  # (B, N, 1, 2)
                )[:, :, 1, :]  # (B, N, 5)

                cos_h, sin_h, v_new = (
                    next_state[..., 2],
                    next_state[..., 3],
                    next_state[..., 4],
                )
                new_row = torch.stack(
                    [
                        next_state[..., 0],
                        next_state[..., 1],
                        cos_h,
                        sin_h,
                        v_new * cos_h,
                        v_new * sin_h,
                        last[..., 6],  # carry the valid flag forward
                    ],
                    dim=-1,
                ).unsqueeze(2)  # (B, N, 1, 7)
                traj = torch.cat([traj, new_row], dim=2)

                # Fold the new frame into the carried GRU state (single step).
                step_feats = frame_invariant_features(traj[:, :, -2:, :])
                step_out, gru_state = self.encoder.encode_history(
                    step_feats[:, :, -1:, :], gru_state
                )
                history_embedding = step_out[:, :, -1, :]  # (B, N, d)

            state_samples.append(traj[:, :, obs_len:, :])  # (B, N, steps, 7)
            token_samples.append(torch.stack(step_tokens, dim=2))  # (B, N, steps)

        pred_states = torch.stack(state_samples, dim=0)  # (K, B, N, steps, 7)
        pred_tokens = torch.stack(token_samples, dim=0)  # (K, B, N, steps)
        return pred_states, pred_tokens

    def to_tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)
