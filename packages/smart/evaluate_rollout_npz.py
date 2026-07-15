"""Compatibility launcher for the repository-level rollout evaluator."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_EVALUATOR = REPO_ROOT / "evaluate_rollout_npz.py"


if __name__ == "__main__":
    # Ensure future repository-level helper imports resolve even when this
    # compatibility path is invoked directly.
    sys.path.insert(0, str(REPO_ROOT))
    runpy.run_path(str(ROOT_EVALUATOR), run_name="__main__")
