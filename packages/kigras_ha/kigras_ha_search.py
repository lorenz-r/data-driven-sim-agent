#!/usr/bin/env python3
"""Two-stage Optuna hyperparameter search for KiGRAS_HA.

Stage 1 (coarse): sample hyperparameters, train each config via the top-level
``train.py`` (Hydra overrides), and minimise the per-epoch ``best_val_loss``
emitted in the training metrics JSON.

Stage 2 (re-rank): take the top-K configs by val_loss, run K-sample autoregressive
rollouts over the validation split, and pick the winner by mean focal ``minFDE``
(the true objective; token val_loss and rollout accuracy are known to diverge).

Structurally mirrors ``packages/smart/smart_search.py`` (the SMART equivalent),
retargeted to kigras_ha's Hydra-override launch path and CTRA rollout.

Examples
--------
    # tiny end-to-end smoke test on the 1-scenario 'single' split
    uv run python packages/kigras_ha/kigras_ha_search.py \\
        --n-trials 2 --max-epochs 2 --download-size single

    # real coarse search on 'small' (GPU recommended)
    uv run python packages/kigras_ha/kigras_ha_search.py \\
        --n-trials 30 --max-epochs 30 --download-size small --top-k 5
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

KIGRAS_HA_DIR = Path(__file__).resolve().parent
REPO_ROOT = KIGRAS_HA_DIR.parent.parent
RESULTS_DIR = REPO_ROOT / "optuna_results_kigras_ha"

if str(KIGRAS_HA_DIR) not in sys.path:
    sys.path.insert(0, str(KIGRAS_HA_DIR))


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess helper (streams the training log to file + stdout)
# ──────────────────────────────────────────────────────────────────────────────
def run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"Command failed (exit {returncode}). See {log_path}")


def checkpoint_for_trial(checkpoint_dir: Path) -> Path:
    last = checkpoint_dir / "last.ckpt"
    if last.exists():
        return last
    ckpts = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt in {checkpoint_dir}")
    return ckpts[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Search space
# ──────────────────────────────────────────────────────────────────────────────
def sample_params(trial: Any) -> dict[str, Any]:
    # All {64,128,256} are divisible by all {2,4,8}, so emb/head combos are always
    # valid — no need to constrain the categorical choices per-trial.
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [64, 128, 256]),
        "num_head": trial.suggest_categorical("num_head", [2, 4, 8]),
        "encoder_num_layers": trial.suggest_int("encoder_num_layers", 2, 4),
        "decoder_num_layers": trial.suggest_int("decoder_num_layers", 2, 4),
        "k_neighbors": trial.suggest_categorical("k_neighbors", [8, 16]),
    }


def build_overrides(
    params: dict[str, Any], trial_dir: Path, args: argparse.Namespace
) -> list[str]:
    """Post-`--` Hydra overrides for the kigras_ha training run."""
    ov = [
        f"training.optimizer.learning_rate={params['lr']}",
        f"training.optimizer.weight_decay={params['weight_decay']}",
        f"model.architecture.dropout={params['dropout']}",
        f"model.architecture.embedding_dim={params['embedding_dim']}",
        f"model.architecture.num_head={params['num_head']}",
        f"model.architecture.encoder_num_layers={params['encoder_num_layers']}",
        f"model.architecture.decoder_num_layers={params['decoder_num_layers']}",
        f"model.architecture.k_neighbors={params['k_neighbors']}",
        # Always keep all agents (no proximity filtering).
        "model.architecture.max_agents=0",
        # Grid fixed for the coarse pass (avoids per-trial re-preprocessing).
        f"model.architecture.num_action_bins={args.num_action_bins}",
        f"model.kinematics.grid_power={args.grid_power}",
        # Batch size + checkpointing are compute knobs, fixed (not searched):
        # LR absorbs the batch-scale effect, and checkpointing lets the largest
        # sampled config fit on 16 GB so big models aren't silently pruned by OOM.
        f"training.dataloader.train_batch_size={args.train_batch_size}",
        f"training.dataloader.val_batch_size={args.val_batch_size}",
        f"model.architecture.gradient_checkpointing={str(not args.no_grad_checkpointing).lower()}",
        f"training.trainer.max_epochs={args.max_epochs}",
        f"training.checkpointing.dirpath={trial_dir / 'ckpts'}",
        "resume.checkpoint_path=null",
        f"seed={args.seed}",
    ]
    if args.cpu:
        ov.append("training.trainer.accelerator=cpu")
    return ov


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: train + read val_loss
# ──────────────────────────────────────────────────────────────────────────────
def run_trial(trial: Any, args: argparse.Namespace) -> float:
    params = sample_params(trial)
    trial_dir = args.results_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = trial_dir / "train_metrics.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--model",
        "kigras_ha",
        "--download-first",
        "--download-size",
        args.download_size,
        "--return_metrics",
        str(metrics_path),
        "--",
        *build_overrides(params, trial_dir, args),
    ]

    print(f"\nTrial {trial.number}: training  {params}")
    run_command(cmd, trial_dir / "train.log")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    val_loss = metrics.get("best_val_loss")
    if val_loss is None:  # no validation logged → fall back to final train loss
        val_loss = metrics.get("end_loss")
    if val_loss is None:
        raise RuntimeError(f"No usable loss in {metrics_path}")

    trial.set_user_attr("checkpoint_dir", str(trial_dir / "ckpts"))
    trial.set_user_attr("params", params)
    trial.set_user_attr("train_metrics_path", str(metrics_path))
    print(f"Trial {trial.number}: best_val_loss={float(val_loss):.6f}")
    return float(val_loss)


def objective(trial: Any, args: argparse.Namespace) -> float:
    try:
        return run_trial(trial, args)
    except Exception as exc:  # noqa: BLE001 — record and prune, keep the study alive
        trial_dir = args.results_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "failure.json").write_text(
            json.dumps({"error": str(exc)}, indent=2)
        )
        print(f"Trial {trial.number}: FAILED — {exc}")
        import optuna

        raise optuna.TrialPruned(str(exc)) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: rollout minADE / minFDE over the validation split
# ──────────────────────────────────────────────────────────────────────────────
def val_split_npy(args: argparse.Namespace) -> Path:
    name = f"{args.download_size}_val_split_b{args.num_action_bins}_p{float(args.grid_power)}.npy"
    return REPO_ROOT / "av2_data" / "kigras_processed" / "val" / name


def evaluate_min_fde(
    checkpoint_dir: Path, args: argparse.Namespace
) -> tuple[float, float]:
    """Mean focal minADE@K / minFDE@K over the validation split for one checkpoint."""
    import numpy as np
    import torch

    from kigras_ha.dataloader.dataset import ArgoDataset
    from kigras_ha.model.kigras_ha import KiGRAS_HA
    from infer_utils import rollout_batch
    from eval2 import min_ade_fde

    device = torch.device(args.eval_device)
    # Seed the sampling RNG so a config's stochastic minFDE is reproducible
    # (sampled rollouts draw from torch's global RNG).
    torch.manual_seed(args.seed)
    ckpt = checkpoint_for_trial(checkpoint_dir)
    model = KiGRAS_HA.load_from_checkpoint(str(ckpt), map_location=device).eval()

    val_npy = val_split_npy(args)
    dataset = ArgoDataset(
        {"preprocess": True, "preprocess_path": str(val_npy)}, train=False
    )

    ades: list[float] = []
    fdes: list[float] = []
    n = len(dataset) if args.eval_limit <= 0 else min(args.eval_limit, len(dataset))
    bs = max(1, args.eval_batch_size)
    for start in range(0, n, bs):
        samples = [dataset[i] for i in range(start, min(start + bs, n))]
        # One batched rollout for the whole chunk (shared batched inference path).
        for res in rollout_batch(
            model,
            samples,
            num_samples=args.eval_samples,
            temperature=args.eval_temperature,
            device=device,
        ):
            fidx = res["focal_idx"]
            gt_future = res["gt_future"]
            gt = gt_future[fidx, :, :2].copy()
            gt[~(gt_future[fidx, :, 6] > 0)] = np.nan  # mask invalid GT steps
            if np.isnan(gt).all():
                continue
            ade, fde = min_ade_fde(res["pred_xy"][:, fidx], gt)
            if not (np.isnan(ade) or np.isnan(fde)):
                ades.append(ade)
                fdes.append(fde)

    if not fdes:
        return float("nan"), float("nan")
    return float(sum(ades) / len(ades)), float(sum(fdes) / len(fdes))


# ──────────────────────────────────────────────────────────────────────────────
# Summaries
# ──────────────────────────────────────────────────────────────────────────────
def write_study_summary(study: Any, path: Path) -> None:
    trials = [
        {
            "number": t.number,
            "state": t.state.name,
            "val_loss": t.value,
            "params": dict(t.params),
        }
        for t in study.trials
    ]
    path.write_text(
        json.dumps(
            {
                "objective": "best_val_loss (minimize)",
                "best_trial": {
                    "number": study.best_trial.number,
                    "val_loss": study.best_trial.value,
                    "params": dict(study.best_trial.params),
                },
                "trials": trials,
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage Optuna search for KiGRAS_HA.")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--download-size", type=str, default="small")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="Force CPU training.")
    # Compute knobs (fixed across trials, not part of the search space).
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=16)
    p.add_argument(
        "--no-grad-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (faster, higher VRAM; may OOM big configs).",
    )
    # Grid fixed for the coarse pass (changing it re-preprocesses per trial).
    p.add_argument("--num-action-bins", type=int, default=63)
    p.add_argument("--grid-power", type=float, default=1.0)
    # Stage-2 rollout eval
    p.add_argument("--top-k", type=int, default=5, help="Configs re-ranked by minFDE.")
    p.add_argument(
        "--eval-samples", type=int, default=6, help="Rollout samples (minFDE@K)."
    )
    p.add_argument("--eval-temperature", type=float, default=1.0)
    p.add_argument(
        "--eval-limit", type=int, default=0, help="Max val scenarios (0=all)."
    )
    p.add_argument(
        "--eval-batch-size", type=int, default=16, help="Val scenarios per batched rollout."
    )
    p.add_argument(
        "--eval-device",
        default=None,
        help="Device for stage-2 rollouts (default: cuda if available else cpu).",
    )
    p.add_argument(
        "--skip-stage2", action="store_true", help="Coarse val_loss search only."
    )
    args = p.parse_args()

    if args.eval_device is None:
        import torch

        args.eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cpu:
        args.eval_device = "cpu"
    args.results_dir = args.results_dir.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    import optuna

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("\nNo trials completed. See per-trial failure.json / train.log.")
        sys.exit(1)

    study_summary = args.results_dir / "study_summary.json"
    write_study_summary(study, study_summary)
    print(
        f"\nStage 1 done. Best val_loss={study.best_trial.value:.6f} "
        f"(trial {study.best_trial.number}). Summary: {study_summary}"
    )

    if args.skip_stage2:
        return

    # ── Stage 2: re-rank the top-K by rollout minFDE ────────────────────────────
    top = sorted(completed, key=lambda t: t.value)[: args.top_k]
    print(
        f"\nStage 2: re-ranking top {len(top)} configs by minFDE@{args.eval_samples} "
        f"on the '{args.download_size}' val split ({args.eval_device})…"
    )

    ranked: list[dict[str, Any]] = []
    for t in top:
        ckpt_dir = Path(t.user_attrs["checkpoint_dir"])
        try:
            ade, fde = evaluate_min_fde(ckpt_dir, args)
        except Exception as exc:  # noqa: BLE001
            print(f"  trial {t.number}: eval failed — {exc}")
            ade, fde = float("nan"), float("nan")
        print(
            f"  trial {t.number}: val_loss={t.value:.4f}  minADE={ade:.3f}  minFDE={fde:.3f}"
        )
        ranked.append(
            {
                "trial_number": t.number,
                "val_loss": t.value,
                "min_ade": ade,
                "min_fde": fde,
                "params": dict(t.params),
                "checkpoint_dir": str(ckpt_dir),
            }
        )

    valid = [r for r in ranked if r["min_fde"] == r["min_fde"]]  # drop NaN
    ranked_sorted = sorted(valid, key=lambda r: r["min_fde"]) or ranked
    best = ranked_sorted[0]
    final = args.results_dir / "final_summary.json"
    final.write_text(
        json.dumps(
            {
                "objective": "mean focal minFDE (minimize), re-ranked from top val_loss",
                "best": best,
                "ranked": ranked_sorted,
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "\n==============================\nBEST CONFIG (by minFDE)\n=============================="
    )
    print(
        f"  trial {best['trial_number']}  minFDE={best['min_fde']:.3f}  "
        f"minADE={best['min_ade']:.3f}  val_loss={best['val_loss']:.4f}"
    )
    for k, v in best["params"].items():
        print(f"    {k}: {v}")
    print(f"\nFinal summary: {final}")


if __name__ == "__main__":
    main()
