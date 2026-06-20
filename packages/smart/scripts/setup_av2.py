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
from trajectory_data.av2.setup import (
    add_setup_arguments,
    read_cache_from_blob,
    run_setup,
    update_current_symlink,
    write_cache_to_blob,
)

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)  # keep output ordered when piped

SMART_ROOT = Path(__file__).parent.parent
REPO_ROOT = SMART_ROOT.parent.parent
RUST_PKG = REPO_ROOT / "packages" / "smart-preprocess"


def _warn_cache_recommended(context: str) -> None:
    print(
        "\n"
        "WARNING: Processed AV2/SMART data is not being cached.\n"
        f"Context: {context}\n"
        "This may waste time and compute because preprocessing can be expensive.\n"
        "Recommended usage:\n"
        "  - Use --read-cache with --blob-cache-name when a processed cache already exists.\n"
        "  - Use --write-cache with --blob-cache-name when creating a processed cache.\n",
        file=sys.stderr,
    )


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

    if (args.read_cache or args.write_cache) and not args.blob_cache_name:
        raise ValueError(
            "--blob-cache-name is required with --read-cache or --write-cache"
        )

    if args.read_cache:
        read_cache_from_blob(
            blob_cache_name=args.blob_cache_name,
            local_cache_dir=args.processed_dir,
        )
        update_current_symlink(args.processed_dir, args.blob_cache_name)

        print(f"Restored processed cache into: {args.processed_dir}")
        print(f"Updated current symlink: {args.processed_dir / 'current'}")
        print("Skipping AV2 setup because --read-cache succeeded.")
        return

    cache_dir = run_setup(
        args=args,
        preprocess_split=None if args.skip_preprocess else _preprocess_smart_split,
        preprocessor_fingerprint=None if args.skip_preprocess else _hash_preprocessor(),
    )

    if args.write_cache:
        if cache_dir is None:
            raise RuntimeError(
                "Cannot write cache because setup did not produce a processed cache directory."
            )

        write_cache_to_blob(
            local_cache_dir=cache_dir,
            blob_cache_name=args.blob_cache_name,
            args=args,
        )
    else:
        _warn_cache_recommended(
            "setup_av2.py is about to run setup/preprocessing without --write-cache."
        )


if __name__ == "__main__":
    main()
