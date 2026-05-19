#!/usr/bin/env python3
"""
setup_av2.py  –  Full AV2 pipeline: download → preprocess (Rust) → splits.

Caching
-------
Downloads
  A HEAD request is issued against the S3 ETag before every run.  If the
  ETag and all download parameters match the last run's cache file the
  download/extraction step is skipped entirely.

Preprocessing
  Output goes into a content-addressed directory:
    data/processed/<cache_key>/train/
    data/processed/<cache_key>/val/
  where cache_key = sha256(tar_etag : preprocessor_source_hash)[:16].
  A symlink  data/processed/current  always points to the latest cache dir.
  If the target directory already contains a .done marker with a matching
  key the preprocessing step is skipped.

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
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)  # keep output ordered when piped

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SMART_ROOT = Path(__file__).parent.parent
REPO_ROOT = SMART_ROOT.parent.parent
RUST_PKG = REPO_ROOT / "packages" / "smart-preprocess"

# ---------------------------------------------------------------------------
# Preprocessor identity  (cache invalidation when Rust source changes)
# ---------------------------------------------------------------------------


def _hash_preprocessor() -> str:
    """sha256 of every .rs file in packages/smart-preprocess/src/, sorted."""
    src = RUST_PKG / "src"
    h = hashlib.sha256()
    for f in sorted(src.rglob("*.rs")):
        h.update(f.read_bytes())
    return h.hexdigest()


def _preprocess_cache_key(tar_etag: str, preprocessor_hash: str) -> str:
    payload = f"{tar_etag}:{preprocessor_hash}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Ensure smart_preprocess is importable, building it if needed
# ---------------------------------------------------------------------------


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

    print("Building smart-preprocess (maturin develop) …")
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
        # maturin may not be on sys.executable; fall back to uv run
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

    # Re-check
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


# ---------------------------------------------------------------------------
# Preprocessing with hash-directory caching
# ---------------------------------------------------------------------------


def _done_path(cache_dir: Path) -> Path:
    return cache_dir / ".done.json"


def _is_preprocess_cached(cache_dir: Path, cache_key: str) -> bool:
    p = _done_path(cache_dir)
    if not p.exists():
        return False
    try:
        meta = json.loads(p.read_text())
        return meta.get("cache_key") == cache_key
    except Exception:
        return False


def _mark_done(
    cache_dir: Path, cache_key: str, tar_etag: str, preprocessor_hash: str
) -> None:
    _done_path(cache_dir).write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "tar_etag": tar_etag,
                "preprocessor_hash": preprocessor_hash,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )


def _update_current_symlink(processed_dir: Path, cache_key: str) -> None:
    link = processed_dir / "current"
    target = Path(cache_key)  # relative so the symlink is portable
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        link.rename(link.with_suffix(".bak"))
    link.symlink_to(target)


def run_preprocess(
    raw_dir: Path,
    processed_dir: Path,
    tar_etag: str,
    splits: tuple[str, ...] = ("train", "val"),
    use_cache: bool = False,
) -> Path:
    """
    Run the Rust preprocessor on each split under raw_dir.
    Outputs go to processed_dir/<cache_key>/<split>/.
    Returns the cache directory path.
    """
    preprocessor_hash = _hash_preprocessor()
    cache_key = _preprocess_cache_key(tar_etag, preprocessor_hash)
    cache_dir = processed_dir / cache_key

    if use_cache and _is_preprocess_cached(cache_dir, cache_key):
        print(f"Preprocessing cache hit ({cache_key}); skipping.")
        _update_current_symlink(processed_dir, cache_key)
        return cache_dir

    print(f"Preprocessing cache key: {cache_key}")
    print(f"  tar_etag          : {tar_etag}")
    print(f"  preprocessor_hash : {preprocessor_hash[:16]}")

    _ensure_smart_preprocess()
    import smart_preprocess

    for split in splits:
        input_dir = raw_dir / split
        if not input_dir.exists():
            print(f"  {split}/ not found in {raw_dir}, skipping.")
            continue
        output_dir = cache_dir / split
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {split} …")
        n_ok, n_total = smart_preprocess.process_directory(
            str(input_dir), str(output_dir)
        )
        print(f"  {split}: {n_ok}/{n_total} scenarios processed.")

    _mark_done(cache_dir, cache_key, tar_etag, preprocessor_hash)
    _update_current_symlink(processed_dir, cache_key)
    print(f"Preprocessing complete → {cache_dir}")
    return cache_dir


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def run_splits(
    cache_dir: Path,
    train_size: int,
    val_size: int,
    seed: int,
    train_out: Path,
    val_out: Path,
    hash_contents: bool,
) -> None:
    train_src = cache_dir / "train"
    val_src = cache_dir / "val"

    # In --dev mode there is no val split — fall back to the train pool.
    # The shared trajectory-data splitter detects equal source dirs and samples
    # without overlap.
    if not val_src.exists() or not any(val_src.iterdir()):
        print(f"  val/ not found or empty in {cache_dir}, carving val from train pool.")
        val_src = train_src

    from trajectory_data.splits import create_splits

    create_splits(
        source_dir=train_src,
        source_val_dir=val_src,
        train_out=train_out,
        val_out=val_out,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
        hash_contents=hash_contents,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full AV2 pipeline: download → preprocess (Rust) → splits."
    )

    # Download options
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dev",
        dest="mode",
        action="store_const",
        const="dev",
        default="dev",
        help="Stream first N scenarios only (default)",
    )
    mode.add_argument(
        "--full",
        dest="mode",
        action="store_const",
        const="full",
        help="Download the complete dataset",
    )
    p.add_argument(
        "--scenarios",
        type=int,
        default=5,
        help="Scenarios to download in dev mode (default: 5)",
    )

    # Pipeline control
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-preprocess", action="store_true")

    # Paths
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=SMART_ROOT / "data" / "raw",
        help="Raw scenario storage directory",
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=SMART_ROOT / "data" / "processed",
        help="Processed output root (cache dirs live here)",
    )
    p.add_argument(
        "--downloads-dir",
        type=Path,
        default=None,
        help="Override TAR cache directory (default: ~/argoverse_cache, "
        "then ~/.cache/argoverse, then <raw-dir>/downloads)",
    )

    # Splits
    p.add_argument("--create-splits", action="store_true")
    p.add_argument("--train-size", type=int, default=16_000)
    p.add_argument("--val-size", type=int, default=4_000)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-split-out", type=Path, default=SMART_ROOT / "train_split")
    p.add_argument(
        "--val-split-out", type=Path, default=SMART_ROOT / "validation_split"
    )
    p.add_argument("--hash-file-contents", action="store_true")

    return p


def main() -> None:
    args = _parser().parse_args()

    # ── 1. Download ──────────────────────────────────────────────────────────
    if args.skip_download:
        # In full mode the ETag can be recovered from the already-downloaded
        # TAR via a cheap HEAD request.  In dev mode we just use a placeholder
        # (dev scenarios carry no stable identity across re-streams).
        if args.mode == "full":
            from trajectory_data.av2.download import TRAIN_URL, _head

            url = args.train_url if hasattr(args, "train_url") else TRAIN_URL
            tar_etag, _ = _head(url)
        else:
            tar_etag = "dev-no-cache"
        print(f"Skipping download; ETag: {tar_etag}")
    else:
        from trajectory_data.av2.download import ensure_downloaded

        print("=" * 55)
        print("STEP 1 — Download")
        print("=" * 55)
        tar_etag = ensure_downloaded(
            args.raw_dir,
            args.mode,
            args.scenarios,
            downloads_dir=args.downloads_dir,
        )

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    if args.skip_preprocess:
        print("Skipping preprocessing.")
        cache_dir = None
    else:
        splits = ("train",) if args.mode == "dev" else ("train", "val")
        print("=" * 55)
        print("STEP 2 — Preprocess")
        print("=" * 55)
        cache_dir = run_preprocess(
            args.raw_dir,
            args.processed_dir,
            tar_etag,
            splits,
            use_cache=(args.mode == "full"),
        )

    # ── 3. Splits ────────────────────────────────────────────────────────────
    if args.create_splits:
        if cache_dir is None:
            # Resolve from current symlink
            current = args.processed_dir / "current"
            cache_dir = current.resolve() if current.exists() else args.processed_dir
        print("=" * 55)
        print("STEP 3 — Create splits")
        print("=" * 55)
        run_splits(
            cache_dir,
            args.train_size,
            args.val_size,
            args.split_seed,
            args.train_split_out.resolve(),
            args.val_split_out.resolve(),
            args.hash_file_contents,
        )

    print("=" * 55)
    print("AV2 setup complete.")
    if cache_dir:
        print(f"Processed data : {cache_dir}")
        print(f"Symlink        : {args.processed_dir / 'current'}")
    print("=" * 55)


if __name__ == "__main__":
    main()
