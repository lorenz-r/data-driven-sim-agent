#!/usr/bin/env bash
set -euo pipefail

RUST_PKG="packages/smart-preprocess"

# Ensure the pinned Rust toolchain (rust-toolchain.toml) is installed before
# uv sync invokes maturin, so the build always uses the correct compiler.
if command -v rustup &>/dev/null; then
    rustup toolchain install
fi

uv sync --all-packages

# Run linters, type checks and tests using uv to ensure the correct virtual env
# Ruff: linting
uv run ruff check .

# Mypy: static type checking
uv run mypy

# Rust: format check + clippy (build is handled by uv sync above)
if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo not found — install Rust via https://rustup.rs/" >&2
    exit 1
fi

echo "Checking Rust formatting (cargo fmt --check)..."
cargo fmt --check --manifest-path "$RUST_PKG/Cargo.toml"

echo "Running clippy..."
cargo clippy --manifest-path "$RUST_PKG/Cargo.toml" -- -D warnings

# Pytest: run tests (rust regression tests auto-skip if smart_preprocess not built)
uv run pytest -q -m "not gpu"
