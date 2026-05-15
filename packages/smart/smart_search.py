import argparse
import json
import subprocess
from pathlib import Path

import optuna


RESULTS_DIR = Path("optuna_results")
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================
# CLI arguments
# ==========================================

parser = argparse.ArgumentParser()

# Optuna settings
parser.add_argument("--n_trials", type=int, default=20)

# Training config
parser.add_argument(
    "--config",
    type=str,
    default="configs/train/train_overfitting.yaml"
)

parser.add_argument("--max_epochs", type=int, default=50)

parser.add_argument("--train_batch_size", type=int, default=100)

parser.add_argument("--val_batch_size", type=int, default=100)

parser.add_argument("--train_path", type=str, default="train_split")

parser.add_argument("--val_path", type=str, default="validation_split")

parser.add_argument("--devices", type=str, default="1")

args = parser.parse_args()


def objective(trial):

    # ==========================================
    # Sample hyperparameters
    # ==========================================

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    hidden_dim = trial.suggest_categorical(
        "hidden_dim",
        [64, 128, 256]
    )

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
        args.config,

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
        str(args.max_epochs),

        "--train_batch_size",
        str(args.train_batch_size),

        "--val_batch_size",
        str(args.val_batch_size),

        "--train_path",
        args.train_path,

        "--val_path",
        args.val_path,

        "--devices",
        args.devices,
    ]

    print("\nRunning:")
    print(" ".join(cmd))

    # Subprocess is maybe not the best approach, but without severely
    # modifying train.py, it's the easiest way to run multiple training
    # runs with different hyperparameters from a single script.
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        print(result.stderr)
        return -9999

    # ==========================================
    # Read metrics
    # ==========================================

    with open(metrics_path) as f:
        metrics = json.load(f)

    end_loss = metrics.get("end_loss", 9999)
    accuracy = metrics.get("accuracy", 0)

    score = accuracy - 0.01 * end_loss

    # ==========================================
    # Define optimization objective
    # ==========================================

    # Since we specifically want to overfit on training data,
    # we optimize for minimizing training loss.
    #score = -end_loss

    return score


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")

    study.optimize(
        objective,
        n_trials=args.n_trials
    )

    print("\n==============================")
    print("BEST TRIAL")
    print("==============================")

    print("Value:", study.best_trial.value)

    print("\nParams:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")