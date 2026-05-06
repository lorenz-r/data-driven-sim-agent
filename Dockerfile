# syntax=docker/dockerfile:1
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# UV — pinned version for hermetic builds
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /usr/local/bin/

# ---------------------------------------------------------------------------
# Project files (lockfile first for layer caching)
# ---------------------------------------------------------------------------
WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
COPY packages/ packages/
COPY tests/ tests/

# ---------------------------------------------------------------------------
# Install — frozen lockfile enforces hermeticity
# CUDA toolkit + cuDNN are pulled as Python wheels via torch/flash-attn,
# no system CUDA install needed. Only the host NVIDIA driver is required.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/tmp/.uv-cache \
    UV_FROZEN=1 uv sync --all-packages --extra gpu

# ---------------------------------------------------------------------------
# Runtime — use with: docker run --gpus all ...
# ---------------------------------------------------------------------------
ENTRYPOINT ["uv", "run"]
CMD ["pytest", "--tb=short", "-q"]
