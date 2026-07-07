### data-driven-sim-agent

Monorepo for data-driven simulation agent integration.

#### Prerequisites

- [uv](https://docs.astral.sh/uv/) >= 0.6
- [Rust / cargo](https://rustup.rs/) >= 1.75 (required to build `smart-preprocess`; `test.sh` will warn if missing)
- NVIDIA GPU + driver (for GPU tests / Docker)

#### Quick start

```bash
# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install rustup (if not present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

uv sync --all-packages  # also installs the pinned Rust toolchain (rust-toolchain.toml) and builds smart-preprocess
uv run pre-commit install    # install pre-commit hooks
uv run pytest -m "not gpu"   # CPU-only tests (Rust regression tests auto-skip if not built)
uv run pytest                # all tests (needs GPU)
```

`test.sh` does the full thing — lint, type-check, build the Rust extension, then pytest:

```bash
bash test.sh
```

#### Structure

| Path | Description |
|---|---|
| `packages/sim-integration` | Core simulation integration library |
| `packages/smart` | SMART motion generation model and training |
| `packages/smart-preprocess` | Rust/PyO3 AV2 preprocessor (`smart_preprocess` Python extension) |
| `packages/trajectory-data` | Shared trajectory dataset download, cache, setup, and split utilities |
| `tests/` | Bootstrap integration & test package |

#### SMART data pipeline

##### 1. Full setup (download + preprocess + optional splits)

AV2 download/cache/split orchestration lives in `packages/trajectory-data`.
`packages/smart/scripts/setup_av2.py` is the SMART wrapper that calls the Rust
preprocessor and writes SMART `.pkl` files.

```bash
# Dev: stream 100 total scenarios, preprocess official train/val subsets
uv run python packages/smart/scripts/setup_av2.py --dev --scenarios 100

# Full dataset
uv run python packages/smart/scripts/setup_av2.py --full

# Full dataset + create train/val splits
uv run python packages/smart/scripts/setup_av2.py --full --create-splits \
    --train-size 16000 --val-size 4000
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--dev` / `--full` | `--dev` | Stream N scenarios or download complete dataset |
| `--scenarios N` | `5` | Total train+val scenario count for `--dev` mode |
| `--val-ratio R` | `0.2` | Validation fraction for `--dev` mode |
| `--raw-dir` | `packages/smart/data/raw` | Where extracted scenario folders are written |
| `--processed-dir` | `packages/smart/data/processed` | Root for content-addressed output dirs |
| `--downloads-dir` | `~/argoverse_cache` | Where the raw TARs are stored (see below) |
| `--skip-download` | off | Skip download, go straight to preprocessing |
| `--skip-preprocess` | off | Download only |
| `--create-splits` | off | Run the shared `trajectory-data` splitter after preprocessing |

##### 2. Download cache

In `--full` mode the two AV2 TARs (~57 GB combined) are stored in the first
writable directory from this list, so they survive repo re-clones and are shared
across projects:

1. `--downloads-dir` if passed explicitly
2. `~/argoverse_cache`
3. `~/.cache/argoverse`
4. `<raw-dir>/downloads` (project-local fallback)

After download the S3 multipart ETag is recomputed from the bytes on disk
(`md5(md5_p1 ‖ … ‖ md5_p759)-759`, 8 MB parts) and compared against a hardcoded
value. A mismatch means a corrupted or partial download and triggers a re-fetch.

In `--dev` mode no TAR is kept. By default, `--scenarios` is split into
official train/val subsets using `--val-ratio 0.2`; pass `--val-ratio 0` for a
train-only dev subset.

##### 3. Rust preprocessor (`packages/smart-preprocess`)

The preprocessor is a standalone Rust/PyO3 crate. It reads raw AV2 parquet +
JSON files directly (no `av2` Python package needed) and writes `.pkl` files
identical to the Python pipeline.

SMART preprocessing keeps vehicle and bus tracks observed at the current
timestep (`t=49`). For those included tracks, all AV2 forecasting categories
(`TRACK_FRAGMENT`, `UNSCORED_TRACK`, `SCORED_TRACK`, and `FOCAL_TRACK`) are
mapped to SMART prediction category `3`.

The crate is a uv workspace member, so `uv sync` builds it automatically via
maturin. No manual build step is needed.

Benchmark vs the Python pipeline (downloads 100 scenarios automatically):

```bash
uv run python packages/smart/scripts/benchmark_preprocess.py --scenarios 100
```

Results on 100 real scenarios: **38× speedup** (183 vs 4.8 scenarios/s),
100/100 field-level matches within tolerance.

##### 4. SMART token vocabularies

SMART uses discrete vocabularies for agent motion tokens and map tokens. The
default files live in `packages/smart/smart/tokens/` and are inherited from the
SMART codebase. For AV2-native experiments, build an agent motion vocabulary
from the AV2 training split only:

```bash
uv run python packages/smart/scripts/build_agent_tokens.py \
  --raw-dir data/processed/current/train \
  --output packages/smart/smart/tokens/cluster_frame_5_2048_av2.pkl \
  --token-size 2048
```

The builder reuses SMART's existing K-disks clustering implementation in
`packages/smart/scripts/traj_clstering.py`. Training does not create tokens
unless explicitly configured to do so.

With the top-level model-agnostic trainer, keep SMART-specific settings under
the `smart:` namespace in `configs/train_params.yaml`:

```yaml
smart:
  tokens:
    agent_token_path: "packages/smart/smart/tokens/cluster_frame_5_2048_av2.pkl"
    map_token_path: "packages/smart/smart/tokens/map_traj_token5.pkl"
    build_agent_tokens_if_missing: false
  loss:
    type: "spatial_aware_label_smoothing"
    label_smoothing: 0.1
    distance_epsilon: 0.001
    trajectory_dims: [0, 1]
```

TrajTok-style spatial-aware label smoothing is implemented as the shared
`core.losses.SpatialAwareLabelSmoothingLoss`, with
`core.losses.build_ntp_classification_loss` available for config-driven NTP
models. It keeps the standard ground-truth mass and distributes smoothing mass
toward trajectory tokens that are spatially close to the target token.

#### SMART inference

SMART inference currently runs through `packages/smart/infer.py`. The script does not read raw AV2 parquet directly as model input; it expects processed SMART scenario `.pkl` files, typically created by the preprocessing step above.
Raw AV2 scenario folders are only needed when writing AV2-format output or saving GIF visualizations, because those steps need the original scenario parquet and map JSON.

The inference flow is:

1. Load the validation config from
   `packages/smart/configs/validation/validation_scalable.yaml`, unless
   `--config` is provided.
2. Build the `SMART` model, resolve the agent/map token vocabularies from the
   config, and load the checkpoint from `--checkpoint`.
3. Load each processed SMART `.pkl` scenario and run the token preprocessor.
4. Run SMART's map matching, sampled point prediction, and autoregressive
   token rollout.
5. Save future-only predictions. Historical context is consumed by the model
   but is not repeated in the saved prediction arrays.

Default paths are set for the current package layout:

| Argument | Default | Purpose |
|---|---|---|
| `--processed-folder` | `packages/smart/data/processed/current/val` | Folder containing processed SMART `.pkl` scenarios |
| `--raw-data-folder` | `packages/smart/data/raw/val` | Raw AV2 scenario folders, needed for AV2 conversion and GIFs |
| `--output` | `packages/smart/data/preds` | Prediction, parquet, and visualization output root |
| `--checkpoint` | `smart-checkpoints/run_01/last.ckpt` | Lightning checkpoint to load |
| `--config` | `packages/smart/configs/validation/validation_scalable.yaml` | SMART validation/inference config |
| `--device` | `cpu` | Use `cpu` or `cuda` |
| `--format` | `both` | Save `smart`, `av2`, or `both` output formats |

Run batch inference over every `.pkl` in a processed validation folder:

```bash
uv run python packages/smart/infer.py \
  --mode batch \
  --processed-folder packages/smart/data/processed/current/val \
  --raw-data-folder packages/smart/data/raw/val \
  --checkpoint smart-checkpoints/run_01/last.ckpt \
  --output packages/smart/data/preds \
  --device cuda
```

Run a single processed scenario:

```bash
uv run python packages/smart/infer.py \
  --mode single \
  --input packages/smart/data/processed/current/val/<scenario_id>.pkl \
  --raw-data-folder packages/smart/data/raw/val \
  --checkpoint smart-checkpoints/run_01/last.ckpt \
  --output packages/smart/data/preds \
  --device cuda
```

For headless or faster runs, disable GIF generation:

```bash
uv run python packages/smart/infer.py \
  --mode batch \
  --processed-folder packages/smart/data/processed/current/val \
  --checkpoint smart-checkpoints/run_01/last.ckpt \
  --format smart \
  --no-visualize
```

The SMART output for each scenario is written to:

```text
<output>/<scenario_id>/smart/<scenario_id>.npz
```

The `.npz` contains:

| Key | Shape / meaning |
|---|---|
| `scenario_id` | Scenario identifier derived from the input filename |
| `pred_traj` | `(num_agents, future_timesteps, 2)` future xy positions |
| `pred_head` | `(num_agents, future_timesteps)` future headings |
| `pred_valid_mask` | `(num_agents, future_timesteps)` future validity mask |
| `last_pos` | Last observed xy position per agent |
| `num_agents` | Number of agents in the scenario |
| `num_timesteps` | Number of predicted future timesteps |

When `--format av2` or `--format both` is used, the script also writes:

```text
<output>/<scenario_id>/av2/scenario_<scenario_id>.parquet
```

By default inference uses the top-1 predicted token at each rollout step, which
keeps evaluation deterministic. Pass `--sample-tokens` to sample among the
top-k token candidates for generative rollouts. In batch mode, GIFs are saved
for the first 10 scenarios by default; use `--batch-visualization-limit 0` or
`--no-visualize` to skip them.

### Training

Training has been refactored into a top level script to make it easier to switch between models and orchestrate hyperparameter searches. A quick explanation of the new components:

- train.py: The general training script. Starts the training and calls model-specific trainers.
- configs/train_params.yaml: Global training parameters plus optional model-specific namespaces. The generic `train.py` only handles shared CLI arguments; model-specific settings are consumed by the selected trainer.
- trainers/: Directory for managing the different trainers. They bridge the variety of approaches and configurations different models may use and make the unififed top-level interface possible.

Example command for running the training:

```bash
uv run python train.py --model smart --train-path packages/smart/data/processed/current/train --val-path packages/smart/data/processed/current/val --project-name smart-training --run-name demo-ru n --max-epochs 5 --batch-size 10 --val-batch-size 10 --learning-rate 0.000019 --save-ckpt-path smart-checkpoints/run_01 --profile simple -- profile-dir profiles/run_01 --profile-name run_01_profile
```

Example command, when downloading data is also required (essentially a quick-start):

```bash
uv run python train.py --download-first --download-size small --process-smart --model smart --batch-size 8 --data-workers 1 --val-batch-size 8 --max-epochs 20 --devices 1 --dropout 0.1 --learning-rate 1e-4 --weight-decay 1e-4 --save-ckpt-path ./checkpoints --wandb-logging --project-name smart-train --run-name small-training
```

Note that after downloading data like this, the processed SMART files will be stored under ```bash 
av2_data/smart_processed/<generated_hash_code>/
```
This means that the file paths for training and validation will have to be adjusted accordingly in future runs.

Also note that it is possible to configure a variety of parameters for the wandb logging when running the top-level script. For example, runs can be classified into groups immediately, additional notes can be pasted at run creation & model artifacts (like checkpoints) can be uploaded automatically. See the configs/train_params.yaml file for details.

##### Training Command (on 2xA100 machine)

This is the legacy approach of accessing SMART directly. The general training is preferred, but in case of bugs, the commands below can be used.

```bash
# Data preprocessing
uv run python packages/smart/scripts/setup_av2.py \
  --dev \
  --scenarios 10000 \
  --create-splits \
  --train-size 8000 \
  --val-size 2000 \
  --split-seed 42 \
  --train-split-out datasets/train \
  --val-split-out datasets/val \
  --hash-file-contents

# Run the training
uv run packages/smart/train.py \
  --lr 1.1103e-3 \
  --dropout 0.1 \
  --hidden_dim 128 \
  --save_ckpt_path checkpoints/run_01 \
  --return_metrics metrics/run_01.json \
  --max_epochs 100 \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --train_path datasets/train \
  --val_path datasets/val \
  --devices 2 \
  --num_cpu_workers 8 \
  --validate_dataset
```

#### Docker

```bash
docker rmi -f sim-agent 2>/dev/null || true && \
docker build -t sim-agent . && \
docker run --rm --gpus all sim-agent
```

#### Technical stack

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 | Pinned via `.python-version` |
| PyTorch | 2.6 | CUDA 12 wheels |
| Rust | 1.91.1 | Pinned via `rust-toolchain.toml`; auto-installed by rustup |
| Flash Attention | 2.8.3 | Prebuilt wheel, bundles CUDA runtime |
| NVIDIA driver | CUDA 12.x | Host requirement for GPU tests |

#### Hermeticity

| Layer | Mechanism |
|---|---|
| Python version | `.python-version` — UV auto-installs exact CPython 3.11 |
| Python deps | `uv.lock` + `UV_FROZEN=1` in CI and Docker |
| UV itself | Pinned version in Dockerfile (0.6.6) |
| CUDA | Torch/flash-attn wheels bundle their own CUDA runtime |
| Rust version | `rust-toolchain.toml` — rustup auto-installs the pinned toolchain if missing |
| AV2 dataset | S3 multipart ETags hardcoded; recomputed on disk after download |

#### Model logging

SMART training uses wandb. Add your API key to a `.env` file in `packages/smart/`
(see comments in `packages/smart/train.py`). To run without logging, comment out
the wandb calls in that file.

#### Training profiling

Training profiling is opt-in from the command line. For Azure VM runs, start with
the lightweight Lightning timing log:

```bash
uv run python train.py \
  --model smart \
  --train-path datasets/train \
  --val-path datasets/val \
  --profile simple \
  --profile-dir profiles/azure_run_01
```

The run prints the profiler file paths at the end. If wandb is enabled, the
files are also attached to the training run as a `training-profile` artifact.
Use `--no-profile-wandb` to keep the files local only. The same settings are
also available in `configs/train_params.yaml` under `profiling`.

Available modes:

| Mode | Output | Use case |
|---|---|---|
| `simple` | compact text timing log | quick answer to what takes longest |
| `advanced` | detailed cProfile text log + one merged `.prof` file | deeper Python-side bottlenecks |
| `pytorch` | Chrome/TensorBoard trace files | GPU/operator-level inspection |

The merged `.prof` file from `--profile advanced` can be opened with SnakeViz:

```bash
uv tool run snakeviz profiles/azure_run_01/*.merged.prof
```

Lightning creates one raw `.prof` per profiled action internally. Training merges
those into the `*.merged.prof` file and removes the per-action files by default.
Pass `--keep-profile-action-files` if you want to inspect individual actions.

PyTorch traces can become large on full training runs, so prefer `simple` first
and use `pytorch` on a short profiling run.

### KiGRAS

Supports two kinematics backends: CTRA (Constant
Turn Rate and Acceleration) and KBM (Kinematic Bicycle Model).

#### Config schema

All model-specific configs live under `packages/kigras/configs/` and use Hydra
composition:

```
packages/kigras/configs/
├── config.yaml                                  # Root: composes all sub-configs
├── paths.yaml                                   # Data paths (raw & processed)
├── model/
│   ├── architecture/kigras_default_arch.yaml    # Transformer dimensions
│   └── kinematics/{ctra,kbm}.yaml               # Kinematics parameters
├── training/
│   ├── trainer/trainer_default.yaml             # Lightning trainer settings
│   ├── dataloader/dataloader_default.yaml       # Batch size, workers, mode
│   ├── optimizer/optim_default.yaml             # AdamW, LR, scheduler
│   ├── checkpointing/checkpoints_default.yaml
│   └── early_stopping/default_es_behaviour.yaml
└── setup/preprocessing/kigras_default.yaml      # Preprocessing paths
```

| Config key | Description | Default |
|---|---|---|
| `model.architecture.embedding_dim` | Transformer embedding size | 64 |
| `model.architecture.num_head` | Multi-head attention heads | 4 |
| `model.architecture.encoder_num_layers` | Scene encoder layers | 3 |
| `model.architecture.decoder_num_layers` | Action decoder layers | 3 |
| `model.architecture.dropout` | Dropout rate | 0.1 |
| `model.architecture.num_actions` | Control dimensions (accel + steer/yaw) | 2 |
| `model.architecture.num_action_bins` | Discrete bins per control dim | 63 |
| `model.kinematics.name` | Kinematics backend (ctra or kbm) | ctra |
| `model.kinematics.max_accel` | Max acceleration (m/s²) | 5.0 |
| `model.kinematics.max_yaw_rate` | Max yaw rate (rad/s) | 1.5 |
| `training.dataloader.mode` | `raw` reads parquets; `proc` reads preprocessed `.npy` | proc |
| `training.dataloader.train_batch_size` | Training batch size | 1 |
| `training.dataloader.val_batch_size` | Validation batch size | 1 |
| `training.trainer.max_epochs` | Max training epochs | 10 |

#### Training via the unified pipeline

```bash
# Train with defaults (uses kigras's internal Hydra config)
uv run train.py --model kigras

# With downloads and Hydra overrides
uv run train.py --download-first \
                --download-size single \
                --model kigras \
                -- \
                model.architecture.num_action_bins=127

# With data paths and Hydra overrides
uv run train.py --model kigras \
                -- \
                training.trainer.max_epochs=200 \
                model.architecture.embedding_dim=128

```

Everything after `--` is forwarded as raw Hydra overrides to kigras's config
composition, giving full access to all model parameters without modifying YAML
files. NOTE: currently, preprocessing is handled automatically by `trainers/kigras_trainer.py`.
This behaviour is expected to change with further integration into the pipeline.

#### Standalone usage

KiGRAS can also be run directly without the top-level pipeline:

```bash
uv run python packages/kigras/train.py training.trainer.max_epochs=200
uv run python packages/kigras/eval.py
```

#### Evaluation

After training, use `packages/kigras/eval.py` to evaluate and visualize the output.

#### Data modes

- **Raw mode** (`mode: raw`): reads AV2 parquet files on the fly via
  `ArgoDataset`. No preprocessing needed, but slower iteration.
- **Preprocessed mode** (`mode: proc`, default): reads pre-computed `.npy` files
  with cached kinematics solutions. Run the preprocessing script to generate:

```bash
uv run python packages/kigras/scripts/preprocess_kinematics.py
```
