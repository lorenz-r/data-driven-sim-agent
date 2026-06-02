#!/usr/bin/env bash
set -euo pipefail

RUST_PKG="packages/smart-preprocess"

# Prefer running tooling from UV-managed environments so no external install is required.
if command -v uv >/dev/null 2>&1; then
  echo "Running ruff format via uv (ephemeral or workspace-provided)..."
  uv run --with ruff -- ruff format .
  echo "ruff formatting complete."

  # Rust: apply cargo fmt
  if command -v cargo &>/dev/null; then
      echo "Running cargo fmt..."
      cargo fmt --manifest-path "$RUST_PKG/Cargo.toml"
      echo "cargo fmt complete."
  else
      echo "cargo not found — skipping Rust format. Install via https://rustup.rs/" >&2
  fi

  exit 0
fi

# Fallback to locally-installed ruff
if ! command -v ruff >/dev/null 2>&1; then
  echo "ruff not found. Install ruff: pip install ruff or install uv and run 'uv run --with ruff -- ruff format .'" >&2
  exit 1
fi

echo "Running ruff format..."
ruff format .
echo "ruff formatting complete."

# Rust: apply cargo fmt
if command -v cargo &>/dev/null; then
    echo "Running cargo fmt..."
    cargo fmt --manifest-path "$RUST_PKG/Cargo.toml"
    echo "cargo fmt complete."
else
    echo "cargo not found — skipping Rust format. Install via https://rustup.rs/" >&2
fi

