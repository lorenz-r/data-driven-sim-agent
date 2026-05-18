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
| `tests/` | Bootstrap integration & test package |

#### SMART data pipeline

##### 1. Full setup (download + preprocess + optional splits)

```bash
# Dev: stream first 100 scenarios, preprocess, no splits
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
| `--scenarios N` | `5` | Scenario count for `--dev` mode |
| `--raw-dir` | `packages/smart/data/raw` | Where extracted scenario folders are written |
| `--processed-dir` | `packages/smart/data/processed` | Root for content-addressed output dirs |
| `--downloads-dir` | `~/argoverse_cache` | Where the raw TARs are stored (see below) |
| `--skip-download` | off | Skip download, go straight to preprocessing |
| `--skip-preprocess` | off | Download only |
| `--create-splits` | off | Run `create_datasets.py` after preprocessing |

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

In `--dev` mode no TAR is kept — only the streamed scenario folders.

##### 3. Rust preprocessor (`packages/smart-preprocess`)

The preprocessor is a standalone Rust/PyO3 crate. It reads raw AV2 parquet +
JSON files directly (no `av2` Python package needed) and writes `.pkl` files
identical to the Python pipeline.

The crate is a uv workspace member, so `uv sync` builds it automatically via
maturin. No manual build step is needed.

Benchmark vs the Python pipeline (downloads 100 scenarios automatically):

```bash
uv run python packages/smart/scripts/benchmark_preprocess.py --scenarios 100
```

Results on 100 real scenarios: **38× speedup** (183 vs 4.8 scenarios/s),
100/100 field-level matches within tolerance.

##### Training Command (on 2xA100 machine)

```bash
# Data preprocessing
uv run packages/smart/scripts/setup_av2.py \
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
