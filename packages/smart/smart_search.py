import json
import subprocess
from pathlib import Path

import optuna


RESULTS_DIR = Path("optuna_results")
RESULTS_DIR.mkdir(exist_ok=True)


def objective(trial):

    # ==========================================
    # Sample hyperparameters
    # ==========================================

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

    # ==========================================
    # Output file for this run
    # ==========================================

    metrics_path = RESULTS_DIR / f"trial_{trial.number}.json"

    # ==========================================
    # Launch training
    # ==========================================

    cmd = [
        "python",
        "train.py",
        "--config",
        "configs/train/train_overfitting.yaml",
        "--lr",
        str(lr),
        "--dropout",
        str(dropout),
        "--hidden_dim",
        str(hidden_dim),
        "--return_metrics",
        str(metrics_path),
        "--trial_number",
        str(trial.number),
        "--max_epochs",
        "50",
        "--train_batch_size",
        "1",
        "--val_batch_size",
        "1",
        "--train_path",
        "tiny_train"
        "--val_path",
        "validation_tiny",
        "--devices",
        "1"
    ]

    print("\nRunning:")
    print(" ".join(cmd))

    # Subprocess is maybe not the best approach, but without severely modifying train.py, it's the easiest way to run multiple training runs with different hyperparameters from a single script. We can explore more integrated approaches later if needed.
    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        print(result.stderr)
        return -9999

    # ==========================================
    # Read metrics
    # ==========================================

    with open(metrics_path) as f:
        metrics = json.load(f)

    end_loss = metrics.get("end_loss", 9999)

    # ==========================================
    # Define optimization objective
    # ==========================================

    # score = accuracy - 0.01 * end_loss

    # Since we want to specifically overfit on the training data, we'll just optimize for minimizing the end loss. In a more realistic setting, we would want to optimize for validation performance instead.
    score = -end_loss

    return score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=20)

    print("\n==============================")
    print("BEST TRIAL")
    print("==============================")

    print("Value:", study.best_trial.value)

    print("\nParams:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
