import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from trainers.profiling import PROFILE_CHOICES  # noqa: E402
from trainers.profiling import TrainingProfiler  # noqa: E402
from trainers.profiling import add_profiler_args  # noqa: E402


__all__ = [
    "PROFILE_CHOICES",
    "TrainingProfiler",
    "add_profiler_args",
]
