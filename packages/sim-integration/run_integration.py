#!/usr/bin/env python3
"""Convenience wrapper to run the sim-integration pipeline from repo root.

Usage:
    ./packages/sim-integration/run_integration.py --parquet tests/test.parquet --methods const_vel
"""

import os
import sys

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _main() -> int:
    from sim_integration.runner import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
