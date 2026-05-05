#!/usr/bin/env bash
set -euo pipefail

# Run linters, type checks and tests using uv to ensure the correct virtual env
# Ruff: linting
uv run --with ruff -- ruff check .

# Mypy: static type checking
uv run --with mypy -- mypy

# Pytest: run tests
PYTHONPATH=packages/sim-integration/src uv run --with pytest --with numpy --with pandas --with pyarrow -- pytest -q -m "not gpu"
