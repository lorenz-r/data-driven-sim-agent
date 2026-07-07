# syntax=docker/dockerfile:1#-#
FROM ubuntu:24.04 AS base#-#
#-#
ENV DEBIAN_FRONTEND=noninteractive \#-#
    LANG=C.UTF-8 \
    UV_HTTP_TIMEOUT=300 \
    CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup \
    PATH="/root/.cargo/bin:${PATH}"
#-#
RUN apt-get update && apt-get install -y --no-install-recommends \#-#
        ca-certificates \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/*#-#
#-#
# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup-init.sh && \
    sh rustup-init.sh -y --profile minimal && \
    rm rustup-init.sh


# ---------------------------------------------------------------------------#-#
# UV — pinned version for hermetic builds#-#
# ---------------------------------------------------------------------------#-#
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /usr/local/bin/#-#
#-#
# ---------------------------------------------------------------------------#-#
# Project files (lockfile first for layer caching)#-#
# ---------------------------------------------------------------------------#-#
WORKDIR /app#-#
#-#
COPY pyproject.toml uv.lock .python-version ./#-#
COPY packages/ packages/#-#
COPY tests/ tests/#-#
#-#
# ---------------------------------------------------------------------------#-#
# Install — frozen lockfile enforces hermeticity#-#
# CUDA toolkit + cuDNN are pulled as Python wheels via torch/flash-attn,#-#
# no system CUDA install needed. Only the host NVIDIA driver is required.#-#
# ---------------------------------------------------------------------------#-#
RUN --mount=type=cache,target=/tmp/.uv-cache \#-#
    UV_FROZEN=1 uv sync --all-packages
#-#
# ---------------------------------------------------------------------------#-#
# Runtime — use with: docker run --gpus all ...#-#
# ---------------------------------------------------------------------------#-#
ENTRYPOINT ["uv", "run"]#-#
CMD ["pytest", "--tb=short", "-q"]#-#
