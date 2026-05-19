#!/usr/bin/env python3
"""
SMART AV2 setup: download raw AV2, preprocess into SMART tensors, and split.

Shared AV2 download/cache/split orchestration lives in trajectory_data.av2.
This wrapper owns only the SMART-specific Rust preprocessor integration.

Usage
-----
    uv run python packages/smart/scripts/setup_av2.py --dev --scenarios 100
    uv run python packages/smart/scripts/setup_av2.py --full
    uv run python packages/smart/scripts/setup_av2.py --dev --scenarios 100 \\
        --skip-preprocess
    uv run python packages/smart/scripts/setup_av2.py --dev --scenarios 100 \\
        --create-splits --train-size 80 --val-size 20
"""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import sys
from pathlib import Path

from trajectory_data.av2.setup import add_setup_arguments, run_setup

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)  # keep output ordered when piped

SMART_ROOT = Path(__file__).parent.parent
REPO_ROOT = SMART_ROOT.parent.parent
RUST_PKG = REPO_ROOT / "packages" / "smart-preprocess"


def _hash_preprocessor() -> str:
    """sha256 of every .rs file in packages/smart-preprocess/src/, sorted."""
    src = RUST_PKG / "src"
    h = hashlib.sha256()
    for f in sorted(src.rglob("*.rs")):
        h.update(f.read_bytes())
    return h.hexdigest()


def _ensure_smart_preprocess() -> None:
    try:
        import smart_preprocess  # noqa: F401

        return
    except ImportError:
        pass

    import shutil as _shutil

    if not _shutil.which("cargo"):
        print(
            "ERROR: smart_preprocess is not installed and cargo is not found.\n"
            "Install Rust via https://rustup.rs/ then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Building smart-preprocess (maturin develop) ...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "maturin",
            "develop",
            "--release",
            "--manifest-path",
            str(RUST_PKG / "Cargo.toml"),
        ],
        check=False,
    )
    if result.returncode != 0:
        subprocess.run(
            [
                "uv",
                "run",
                "maturin",
                "develop",
                "--release",
                "--manifest-path",
                str(RUST_PKG / "Cargo.toml"),
            ],
            check=True,
        )

    try:
        import importlib

        importlib.invalidate_caches()
        import smart_preprocess  # noqa: F401
    except ImportError:
        print(
            "ERROR: build succeeded but smart_preprocess still not importable.",
            file=sys.stderr,
        )
        sys.exit(1)


def _preprocess_smart_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
) -> tuple[int, int]:
    _ensure_smart_preprocess()
    import smart_preprocess

    return smart_preprocess.process_directory(str(input_dir), str(output_dir))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SMART AV2 pipeline: download -> preprocess (Rust) -> splits."
    )
    add_setup_arguments(
        parser,
        raw_dir_default=SMART_ROOT / "data" / "raw",
        processed_dir_default=SMART_ROOT / "data" / "processed",
        train_split_out_default=SMART_ROOT / "train_split",
        val_split_out_default=SMART_ROOT / "validation_split",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_setup(
        args=args,
        preprocess_split=None if args.skip_preprocess else _preprocess_smart_split,
        preprocessor_fingerprint=None if args.skip_preprocess else _hash_preprocessor(),
    )


if __name__ == "__main__":
    main()
