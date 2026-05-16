#!/usr/bin/env bash
set -euo pipefail

uv sync --package data-driven-sim-agent --package argoverse-forecasting --package sim-integration --package integration-tests

# Run linters, type checks and tests using uv to ensure the correct virtual env
# Ruff: linting
uv run --with ruff -- ruff check .

# Mypy: static type checking
uv run --with mypy -- mypy

# Pytest: run tests
uv run pytest -q -m "not gpu"
