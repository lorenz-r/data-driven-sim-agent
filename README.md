### data-driven-sim-agent

Monorepo for data-driven simulation agent integration.

#### Prerequisites

- [uv](https://docs.astral.sh/uv/) >= 0.6
- NVIDIA GPU + driver (for GPU tests / Docker)

#### Quick start

```bash
uv sync --all-packages
uv run pre-commit install       # install pre-commit hooks
uv run pytest -m "not gpu"       # CPU-only tests
uv run pytest                    # all tests (needs GPU)
```

#### Docker

Build and run tests in an isolated container with GPU support:

```bash
docker rmi -f sim-agent 2>/dev/null || true && docker build -t sim-agent . && docker run --rm --gpus all sim-agent
```

**Technical stack:**
- **Python 3.11** — Most compatible version for the current ecosystem
- **PyTorch 2.8** — Latest stable release with CUDA 12 support
- **Flash Attention 2.8.3** — Prebuilt wheel for CUDA 12 / PyTorch 2.4+ / Python 3.11
- **CUDA Runtime** — Bundled in PyTorch and Flash Attention wheels (no system CUDA install needed)
- **Host requirement** — NVIDIA driver supporting CUDA 12.x

#### Structure

| Path | Description |
|---|---|
| `packages/sim-integration` | Core simulation integration library |
| `tests/` | Bootstrap integration & test package |

#### Hermeticity

| Layer | Mechanism |
|---|---|
| Python version | `.python-version` — UV auto-installs exact CPython 3.11 |
| Dependencies | `uv.lock` + `UV_FROZEN=1` in CI and Docker |
| UV itself | Pinned version in Dockerfile (0.6.6) |
| CUDA | Torch/flash-attn wheels bundle their own CUDA runtime |
