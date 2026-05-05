#!/usr/bin/env bash
set -euo pipefail

# Prefer running tooling from UV-managed environments so no external install is required.
if command -v uv >/dev/null 2>&1; then
  echo "Running ruff format via uv (ephemeral or workspace-provided)..."
  # Use uv to provide ruff if available in the workspace or install ephemeral
  uv run --with ruff -- ruff format .
  echo "ruff formatting complete."
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
