import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SMART_DIR = Path(__file__).resolve().parent
REPO_ROOT = SMART_DIR.parent.parent
RESULTS_DIR = REPO_ROOT / "optuna_results"
DEFAULT_CONFIG = REPO_ROOT / "configs/train_params.yaml"


def run_command(cmd: list[str], log_path: Path, *, stream: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if stream:
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
            raise RuntimeError(
                f"Command failed with exit code {returncode}. See {log_path}"
            )
        return

    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}. See {log_path}"
        )


def latest_batch_prediction(preds_dir: Path) -> Path:
    batch_files = sorted(
        preds_dir.glob("batch_predictions_*.npz"),
        key=lambda path: path.stat().st_mtime,
    )
    if not batch_files:
        raise FileNotFoundError(f"No batch_predictions_*.npz found in {preds_dir}")
    return batch_files[-1]


def checkpoint_for_trial(checkpoint_dir: Path) -> Path:
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.exists():
        return last_checkpoint

    checkpoints = sorted(
        checkpoint_dir.glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")
    return checkpoints[-1]


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


def write_trial_summary(
    trial: Any,
    trial_dir: Path,
    min_ade: float,
    trial_config_path: Path,
    checkpoint_path: Path,
    batch_prediction_path: Path,
    eval_metrics_path: Path,
) -> None:
    summary = {
        "trial_number": trial.number,
        "objective": "summary_metrics.forecasting.min_ade",
        "value": min_ade,
        "params": dict(trial.params),
        "artifacts": {
            "trial_config_path": str(trial_config_path),
            "checkpoint_path": str(checkpoint_path),
            "batch_prediction_path": str(batch_prediction_path),
            "eval_metrics_path": str(eval_metrics_path),
        },
    }
    with (trial_dir / "trial_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)


def write_study_summary(study: Any, output_path: Path) -> None:
    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": dict(trial.params),
            "user_attrs": dict(trial.user_attrs),
        }
        for trial in study.trials
    ]
    summary = {
        "direction": "minimize",
        "objective": "summary_metrics.forecasting.min_ade",
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": dict(study.best_trial.params),
            "user_attrs": dict(study.best_trial.user_attrs),
        },
        "trials": trials,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)


def write_mtp_trial_config(
    base_config: dict[str, Any],
    trial_dir: Path,
    *,
    w2: float,
    w3: float,
    mtp_use_transformer_head: bool,
) -> Path:
    import yaml

    trial_config = copy.deepcopy(base_config)
    loss_config = trial_config.setdefault("smart", {}).setdefault("loss", {})
    loss_config["mtp_enabled"] = True
    loss_config["mtp_offsets"] = [1, 2, 3]
    loss_config["mtp_loss_weights"] = [1.0, float(w2), float(w3)]
    loss_config["mtp_use_transformer_head"] = bool(mtp_use_transformer_head)

    trial_config_path = trial_dir / "config.yaml"
    with trial_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(trial_config, file, sort_keys=False)
    return trial_config_path


def run_trial(trial: Any, args: argparse.Namespace) -> float:
    if args.fixed_optimizer_hparams:
        lr = float(config_value(args.base_config, "optimization", "learning_rate"))
        dropout = float(config_value(args.base_config, "model", "dropout"))
        weight_decay = float(
            config_value(args.base_config, "optimization", "weight_decay")
        )
    else:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    trial_dir = args.results_dir / f"trial_{trial.number:04d}"
    checkpoint_dir = trial_dir / "checkpoints"
    preds_dir = trial_dir / "preds"
    train_metrics_path = trial_dir / "train_metrics.json"
    eval_metrics_path = trial_dir / "eval_metrics.json"
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_config_path = args.config
    if args.search_mtp_loss_weights:
        w2 = trial.suggest_float(
            "mtp_w2",
            args.mtp_w2_range[0],
            args.mtp_w2_range[1],
            log=True,
        )
        w3 = trial.suggest_float(
            "mtp_w3",
            args.mtp_w3_range[0],
            args.mtp_w3_range[1],
            log=True,
        )
        if args.search_mtp_transformer_head:
            mtp_use_transformer_head = trial.suggest_categorical(
                "mtp_use_transformer_head",
                [False, True],
            )
        else:
            mtp_use_transformer_head = args.mtp_use_transformer_head

        trial_config_path = write_mtp_trial_config(
            args.base_config,
            trial_dir,
            w2=w2,
            w3=w3,
            mtp_use_transformer_head=mtp_use_transformer_head,
        )

    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--config",
        str(trial_config_path),
        "--model",
        "smart",
        "--learning-rate",
        str(lr),
        "--dropout",
        str(dropout),
        "--weight-decay",
        str(weight_decay),
        "--save_ckpt_path",
        str(checkpoint_dir),
        "--return_metrics",
        str(train_metrics_path),
    ]
    if args.max_epochs is not None:
        train_cmd.extend(["--max-epochs", str(args.max_epochs)])
    if args.train_batch_size is not None:
        train_cmd.extend(["--batch-size", str(args.train_batch_size)])
    if args.val_batch_size is not None:
        train_cmd.extend(["--val-batch-size", str(args.val_batch_size)])
    if args.devices is not None:
        train_cmd.extend(["--devices", str(args.devices)])
    if args.train_path:
        train_cmd.extend(["--train-path", str(args.train_path)])
    if args.val_path:
        train_cmd.extend(["--val-path", str(args.val_path)])
    if args.cpu:
        train_cmd.append("--cpu")
    if args.no_wandb_logging:
        train_cmd.append("--no-wandb-logging")

    print(f"\nTrial {trial.number}: training")
    run_command(train_cmd, trial_dir / "train.log", stream=True)

    checkpoint_path = checkpoint_for_trial(checkpoint_dir)

    infer_cmd = [
        sys.executable,
        str(SMART_DIR / "infer2.py"),
        "--input",
        str(args.inference_input),
        "--output",
        str(preds_dir),
        "--checkpoint",
        str(checkpoint_path),
        "--device",
        args.inference_device,
        "--rollouts-per-scenario",
        str(args.rollouts_per_scenario),
        "--batch-visualization-limit",
        "0",
        "--no-visualize",
        "--format",
        "smart",
        "--no-setup-dialog",
    ]
    if not args.sample_tokens:
        infer_cmd.append("--no-sample-tokens")

    print(f"Trial {trial.number}: inference from {checkpoint_path.name}")
    run_command(infer_cmd, trial_dir / "infer.log")

    batch_prediction_path = latest_batch_prediction(preds_dir)
    eval_cmd = [
        sys.executable,
        str(SMART_DIR / "evaluate_rollout_npz.py"),
        str(batch_prediction_path),
        "--output",
        str(eval_metrics_path),
        "--no-interactive",
    ]

    print(f"Trial {trial.number}: evaluating minADE")
    run_command(eval_cmd, trial_dir / "eval.log")

    with eval_metrics_path.open("r", encoding="utf-8") as metrics_file:
        report = json.load(metrics_file)

    min_ade = report["summary_metrics"]["forecasting"]["min_ade"]
    if min_ade is None:
        raise RuntimeError(f"Evaluation did not produce min_ade: {eval_metrics_path}")

    trial.set_user_attr("checkpoint_path", str(checkpoint_path))
    trial.set_user_attr("batch_prediction_path", str(batch_prediction_path))
    trial.set_user_attr("eval_metrics_path", str(eval_metrics_path))
    trial.set_user_attr("train_metrics_path", str(train_metrics_path))
    trial.set_user_attr("trial_config_path", str(trial_config_path))
    write_trial_summary(
        trial=trial,
        trial_dir=trial_dir,
        min_ade=float(min_ade),
        trial_config_path=trial_config_path,
        checkpoint_path=checkpoint_path,
        batch_prediction_path=batch_prediction_path,
        eval_metrics_path=eval_metrics_path,
    )
    print(f"Trial {trial.number}: minADE={min_ade:.6f}")
    return float(min_ade)


def objective(trial: Any, args: argparse.Namespace) -> float:
    try:
        return run_trial(trial, args)
    except Exception as exc:
        trial_dir = args.results_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        failure_path = trial_dir / "failure.json"
        with failure_path.open("w", encoding="utf-8") as failure_file:
            json.dump({"error": str(exc)}, failure_file, indent=2)
        trial.set_user_attr("failure_path", str(failure_path))
        print(f"Trial {trial.number}: failed, see {failure_path}")

        import optuna

        raise optuna.TrialPruned(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna search for SMART hyperparameters using rollout minADE."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU.")
    parser.add_argument(
        "--fixed-optimizer-hparams",
        action="store_true",
        help=(
            "Keep learning rate, dropout, and weight decay from the config instead "
            "of sampling them. Useful when isolating loss-weight effects."
        ),
    )
    parser.add_argument(
        "--search-mtp-loss-weights",
        action="store_true",
        help="Enable MTP and sample mtp_loss_weights as [1.0, mtp_w2, mtp_w3].",
    )
    parser.add_argument(
        "--mtp-w2-range",
        type=float,
        nargs=2,
        default=(0.01, 1.0),
        metavar=("LOW", "HIGH"),
        help="Log-scale search range for the offset-2 MTP loss weight.",
    )
    parser.add_argument(
        "--mtp-w3-range",
        type=float,
        nargs=2,
        default=(0.003, 1.0),
        metavar=("LOW", "HIGH"),
        help="Log-scale search range for the offset-3 MTP loss weight.",
    )
    parser.add_argument(
        "--mtp-use-transformer-head",
        action="store_true",
        help="Use separate copied transformer head layers for auxiliary MTP heads.",
    )
    parser.add_argument(
        "--search-mtp-transformer-head",
        action="store_true",
        help="Also sample whether auxiliary MTP heads use copied transformer layers.",
    )
    parser.add_argument(
        "--no-wandb-logging",
        action="store_true",
        help="Disable wandb logging for each training trial.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=None,
        help="Optional training data directory; defaults to config.",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=None,
        help="Optional training-time validation directory; defaults to config.",
    )
    parser.add_argument(
        "--inference-input",
        type=Path,
        default=None,
        help="Processed validation .pkl file or directory used for rollout scoring.",
    )
    parser.add_argument(
        "--inference-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device used for rollout inference.",
    )
    parser.add_argument(
        "--rollouts-per-scenario",
        type=int,
        default=1,
        help="Number of rollout samples per scenario for minADE evaluation.",
    )
    parser.add_argument(
        "--sample-tokens",
        dest="sample_tokens",
        action="store_true",
        help="Sample rollout tokens during inference scoring.",
    )
    parser.add_argument(
        "--no-sample-tokens",
        dest="sample_tokens",
        action="store_false",
        help="Use deterministic top-1 rollout tokens.",
    )
    parser.set_defaults(sample_tokens=False)
    args = parser.parse_args()

    if args.rollouts_per_scenario < 1:
        raise ValueError("--rollouts-per-scenario must be at least 1")
    if args.search_mtp_loss_weights:
        for name, value_range in (
            ("--mtp-w2-range", args.mtp_w2_range),
            ("--mtp-w3-range", args.mtp_w3_range),
        ):
            low, high = value_range
            if low <= 0 or high <= 0 or low > high:
                raise ValueError(f"{name} must be two positive values: LOW <= HIGH")
    if args.cpu:
        args.inference_device = "cpu"

    args.config = args.config.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()
    config = load_config(args.config)
    args.base_config = config
    if args.inference_input is None:
        args.inference_input = args.val_path or config_value(
            config,
            "data",
            "val_path",
            default=SMART_DIR / "validation_split",
        )
    args.inference_input = resolve_repo_path(args.inference_input)
    if args.train_path:
        args.train_path = resolve_repo_path(args.train_path)
    if args.val_path:
        args.val_path = resolve_repo_path(args.val_path)
    return args


if __name__ == "__main__":
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    import optuna

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        print(
            "\nNo trials completed successfully. Check per-trial failure.json/log files."
        )
        sys.exit(1)

    study_summary_path = args.results_dir / "study_summary.json"
    write_study_summary(study, study_summary_path)

    print("\n==============================")
    print("BEST TRIAL")
    print("==============================")
    print("Value:", study.best_trial.value)

    print("\nParams:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    print("\nArtifacts:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value}")
    print(f"\nStudy summary: {study_summary_path}")
