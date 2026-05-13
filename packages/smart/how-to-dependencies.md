## Quick start

This project uses **Python 3.10/3.11**, **PyTorch 2.5.x** with **CUDA 12.4**.

---

## Option A: uv sync (recommended)

```
uv sync
```

This reads `pyproject.toml` and installs all dependencies (except Waymo SDK).

---

## Option B: pip

```
# 1. Create a clean environment
conda create -n smart python=3.11 -y
conda activate smart

# 2. Install PyTorch (CUDA 12.4 build)
pip install "torch>=2.5.0,<2.6" "torchvision>=0.20.0,<0.21" --index-url https://download.pytorch.org/whl/cu124

# 3. Install PyTorch Geometric extensions (pre-compiled wheels)
pip install "torch-cluster>=1.6.3,<1.7" "torch-scatter>=2.1.2,<2.2" "torch-sparse>=0.6.18,<0.7" "torch-spline-conv>=1.2.2,<1.3" -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

# 4. Install torch-geometric
pip install torch-geometric==2.6.1

# 5. Install all other dependencies
pip install -r requirements.txt
```

---

## Optional: Waymo SDK

The Waymo Open Dataset SDK is only needed for:
- **Preprocessing** Waymo TFRecord data (`data_preprocess.py`)
- **Exporting** submissions in Waymo Sim Agents format (`model/smart.py`)

Install it separately after all other dependencies:

```
pip install waymo-open-dataset-tf-2-12-0
```

Without it, training and inference work normally; only preprocessing and submission export are unavailable.

---

## Key changes from the old setup

| Component | Old | New |
|-----------|-----|-----|
| Python | 3.9.x | >=3.10,<3.12 |
| PyTorch | 1.12.1 (CUDA 11.3) | 2.5.1 (CUDA 12.4) |
| torchvision | 0.13.1 | 0.20.1 |
| pytorch-lightning | 2.0.3 | 2.4.0 |
| torch-geometric | 2.6.1 | 2.6.1 (compatible) |
| torch-scatter/sparse/cluster | pt112 builds | pt250 builds |
| torchmetrics | 1.5.0 | >=1.6.0 |
| numpy | 1.23 | >=1.26 |
| pandas | 1.5.3 | >=2.2 |
| scipy | 1.9.1 | >=1.14 |
| matplotlib | 3.6.1 | >=3.9 |
| scikit-learn | 1.2.2 | >=1.6 |
| tensorflow | 2.12.0 | **removed** (optional via waymo) |
| waymo-open-dataset-tf-2-12-0 | 1.6.5 | **optional** (install separately) |

Removed unused deps: `jax`, `jaxlib`, `dm-tree`, `einops`, `visu3d`,
`tensorflow-addons`, `tensorflow-graphics`, `tensorflow-probability`,
`tensorboard`, `dask`, `aiohttp`, `lark`, `openexr`, etc.

### Code changes for PyTorch Lightning 2.x
- `on_validation_start` → `on_validation_epoch_start` in `model/smart.py`
- Removed `num_nodes` from `pl.Trainer()`; passed via `DDPStrategy` instead
- Waymo imports made optional with try/except guards
