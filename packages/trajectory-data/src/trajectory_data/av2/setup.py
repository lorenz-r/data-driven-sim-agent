"""Reusable AV2 setup orchestration.

This module stops at the shared AV2 boundary: raw scenarios are downloaded,
optionally transformed by a caller-provided preprocessor, cached, and split.
Model packages own the callback that turns AV2 into their training format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os
import subprocess

from trajectory_data.av2.download import TRAIN_URL, VAL_URL, _head, ensure_downloaded
from trajectory_data.splits import create_splits

PreprocessSplitFn = Callable[[Path, Path, str], Any]
REPO_ROOT = Path(__file__).resolve().parents[5]


def preprocess_cache_key(dataset_etag: str, preprocessor_fingerprint: str) -> str:
    payload = f"{dataset_etag}:{preprocessor_fingerprint}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _done_path(cache_dir: Path) -> Path:
    return cache_dir / ".done.json"


def is_preprocess_cached(cache_dir: Path, cache_key: str) -> bool:
    p = _done_path(cache_dir)
    if not p.exists():
        return False
    try:
        meta = json.loads(p.read_text())
        return meta.get("cache_key") == cache_key
    except Exception:
        return False


def mark_done(
    cache_dir: Path,
    cache_key: str,
    dataset_etag: str,
    preprocessor_fingerprint: str,
) -> None:
    _done_path(cache_dir).write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "dataset_etag": dataset_etag,
                "preprocessor_fingerprint": preprocessor_fingerprint,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )


def update_current_symlink(processed_dir: Path, cache_key: str) -> None:
    link = processed_dir / "current"
    target = Path(cache_key)  # relative so the symlink is portable
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        link.rename(link.with_suffix(".bak"))
    link.symlink_to(target)


def _format_preprocess_result(result: Any) -> str | None:
    if isinstance(result, tuple) and len(result) == 2:
        n_ok, n_total = result
        return f"{n_ok}/{n_total} scenarios processed."
    if result is None:
        return None
    return str(result)


def run_preprocess(
    *,
    raw_dir: Path,
    processed_dir: Path,
    dataset_etag: str,
    preprocessor_fingerprint: str,
    preprocess_split: PreprocessSplitFn,
    splits: tuple[str, ...] = ("train", "val"),
    use_cache: bool = False,
) -> Path:
    """
    Run a model-provided preprocessor on each AV2 split under raw_dir.

    Outputs go to processed_dir/<cache_key>/<split>/ and a portable
    processed_dir/current symlink is updated to the latest cache directory.
    """
    cache_key = preprocess_cache_key(dataset_etag, preprocessor_fingerprint)
    cache_dir = processed_dir / cache_key

    if use_cache and is_preprocess_cached(cache_dir, cache_key):
        print(f"Preprocessing cache hit ({cache_key}); skipping.")
        update_current_symlink(processed_dir, cache_key)
        return cache_dir

    print(f"Preprocessing cache key: {cache_key}")
    print(f"  dataset_etag            : {dataset_etag}")
    print(f"  preprocessor_fingerprint: {preprocessor_fingerprint[:16]}")

    for split in splits:
        input_dir = raw_dir / split
        if not input_dir.exists():
            print(f"  {split}/ not found in {raw_dir}, skipping.")
            continue
        output_dir = cache_dir / split
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {split} ...")
        result = preprocess_split(input_dir, output_dir, split)
        message = _format_preprocess_result(result)
        if message:
            print(f"  {split}: {message}")

    mark_done(cache_dir, cache_key, dataset_etag, preprocessor_fingerprint)
    update_current_symlink(processed_dir, cache_key)
    print(f"Preprocessing complete -> {cache_dir}")
    return cache_dir


def run_splits(
    *,
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

    # In dev mode there is often no val split. The shared splitter detects
    # equal source dirs and samples train/val without overlap.
    if not val_src.exists() or not any(val_src.iterdir()):
        print(f"  val/ not found or empty in {cache_dir}, carving val from train pool.")
        val_src = train_src

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


def add_setup_arguments(
    parser: argparse.ArgumentParser,
    *,
    raw_dir_default: Path,
    processed_dir_default: Path,
    train_split_out_default: Path,
    val_split_out_default: Path,
) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dev",
        dest="mode",
        action="store_const",
        const="dev",
        default="dev",
        help="Stream N train+val scenarios (default)",
    )
    mode.add_argument(
        "--full",
        dest="mode",
        action="store_const",
        const="full",
        help="Download the complete dataset",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=5,
        help="Total train+val scenarios to download in dev mode (default: 5)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation fraction for dev mode partial downloads (default: 0.2)",
    )

    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")

    parser.add_argument(
        "--read-cache",
        action="store_true",
        help="Read processed SMART cache from Azure Blob before setup.",
    )
    parser.add_argument(
        "--write-cache",
        action="store_true",
        help="Write processed SMART cache to Azure Blob after successful setup.",
    )
    parser.add_argument(
        "--blob-cache-name",
        dest="blob_cache_name",
        default=None,
        help="Remote Azure Blob cache folder name.",
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=raw_dir_default,
        help="Raw scenario storage directory",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=processed_dir_default,
        help="Processed output root (cache dirs live here)",
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=None,
        help=(
            "Override TAR cache directory (default: ~/argoverse_cache, "
            "then ~/.cache/argoverse, then <raw-dir>/downloads)"
        ),
    )
    parser.add_argument("--train-url", default=TRAIN_URL)
    parser.add_argument("--val-url", default=VAL_URL)

    parser.add_argument("--create-splits", action="store_true")
    parser.add_argument("--train-size", type=int, default=16_000)
    parser.add_argument("--val-size", type=int, default=4_000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--train-split-out",
        type=Path,
        default=train_split_out_default,
    )
    parser.add_argument(
        "--val-split-out",
        type=Path,
        default=val_split_out_default,
    )
    parser.add_argument("--hash-file-contents", action="store_true")


def resolve_download_etag(args: argparse.Namespace) -> str:
    if args.skip_download:
        if args.mode == "full":
            dataset_etag, _ = _head(args.train_url)
        else:
            dataset_etag = "dev-no-cache"
        print(f"Skipping download; ETag: {dataset_etag}")
        return dataset_etag

    print("=" * 55)
    print("STEP 1 - Download")
    print("=" * 55)
    return ensure_downloaded(
        out_dir=args.raw_dir,
        mode=args.mode,
        train_scenarios=args.scenarios,
        train_url=args.train_url,
        val_url=args.val_url,
        downloads_dir=args.downloads_dir,
        val_ratio=args.val_ratio,
    )


def _require_env(name: str) -> str:
    value = os.environ.get(name)

    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


def _join_sas_url(base_url: str, sas_token: str) -> str:
    """
    Build a usable SAS URL.

    base_url:
        https://account.blob.core.windows.net/container/path/to/blob_cache_name

    sas_token:
        sp=rl&st=...&se=...&sig=...
        OR
        ?sp=rl&st=...&se=...&sig=...
    """
    sas_token = sas_token.lstrip("?")

    if "?" in base_url:
        return f"{base_url}&{sas_token}"

    return f"{base_url}?{sas_token}"


def _blob_cache_url(blob_cache_name: str, sas_token: str) -> str:
    """
    Builds the full remote blob cache URL.

    Required env:
        AV2_CACHE_CONTAINER_URL

    Example:
        AV2_CACHE_CONTAINER_URL=https://account.blob.core.windows.net/av2-preprocessed-files

    Then blob_cache_name='smart-av2-massive-v1' maps to:
        https://account.blob.core.windows.net/av2-preprocessed-files/smart-av2-massive-v1?<sas>
    """
    if not blob_cache_name:
        raise ValueError("blob_cache_name is required")

    container_url = _require_env("AV2_CACHE_CONTAINER_URL").rstrip("/")
    cache_url = f"{container_url}/{blob_cache_name.strip('/')}"

    return _join_sas_url(cache_url, sas_token)


def _blob_cache_contents_url(blob_cache_name: str, sas_token: str) -> str:
    """
    Builds a SAS URL for the contents of a remote blob cache folder.

    Example:
        cache-name/current
        cache-name/dc25f7d8f0549f40/

    should be copied into the local processed_dir directly, without creating
    a local cache-name/ directory.
    """
    if not blob_cache_name:
        raise ValueError("--cache-name / --blob-cache-name is required")

    container_url = _require_env("AV2_CACHE_CONTAINER_URL").rstrip("/")
    cache_url = f"{container_url}/{blob_cache_name.strip('/')}/*"

    return _join_sas_url(cache_url, sas_token)


def _validate_cache_dir(cache_dir: Path) -> None:
    """
    The trainer expects:

        cache_dir/
            train/
            val/
    """
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")

    if not (cache_dir / "train").exists():
        raise FileNotFoundError(f"Missing train directory: {cache_dir / 'train'}")

    if not (cache_dir / "val").exists():
        raise FileNotFoundError(f"Missing val directory: {cache_dir / 'val'}")


def read_cache_from_blob(blob_cache_name: str, local_cache_dir: str | Path) -> Path:
    """
    Download a processed SMART cache from Azure Blob into the normal local
    processed-data location.

    blob_cache_name is the name of the directory in the blob to be read

    Returns:
        Path to local cache folder containing train/ and val/.
    """
    read_sas = _require_env("AV2_CACHE_READ_SAS")
    remote_url = _blob_cache_contents_url(blob_cache_name, read_sas)

    local_cache_dir = Path(local_cache_dir).resolve()
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "azcopy",
            "copy",
            remote_url,
            str(local_cache_dir),
            "--recursive=true",
            "--overwrite=true",
        ],
        check=True,
    )

    _validate_cache_dir(local_cache_dir)

    return local_cache_dir


def write_cache_to_blob(
    local_cache_dir: str | Path, blob_cache_name: str, args: argparse.Namespace
) -> None:
    """
    Upload the local processed SMART cache folder to Azure Blob.

    local_cache_dir should be the folder returned by process_2_smart(...).
    It should contain train/ and val/.

    blob_cache_name is only the remote blob folder name.
    """
    write_sas = _require_env("AV2_CACHE_WRITE_SAS")
    remote_url = _blob_cache_url(blob_cache_name, write_sas)

    local_cache_dir = Path(local_cache_dir).resolve()
    _validate_cache_dir(local_cache_dir)

    subprocess.run(
        [
            "azcopy",
            "copy",
            str(local_cache_dir),
            remote_url,
            "--recursive=true",
            "--overwrite=false",
        ],
        check=True,
    )


def run_setup(
    *,
    args: argparse.Namespace,
    preprocess_split: PreprocessSplitFn | None,
    preprocessor_fingerprint: str | None,
) -> Path | None:
    dataset_etag = resolve_download_etag(args)

    if args.skip_preprocess:
        print("Skipping preprocessing.")
        cache_dir = None
    else:
        if preprocess_split is None or preprocessor_fingerprint is None:
            raise ValueError(
                "A preprocess callback and fingerprint are required unless "
                "--skip-preprocess is set."
            )
        splits = ("train", "val")
        print("=" * 55)
        print("STEP 2 - Preprocess")
        print("=" * 55)
        cache_dir = run_preprocess(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            dataset_etag=dataset_etag,
            preprocessor_fingerprint=preprocessor_fingerprint,
            preprocess_split=preprocess_split,
            splits=splits,
            use_cache=(args.mode == "full"),
        )

    if args.create_splits:
        if cache_dir is None:
            current = args.processed_dir / "current"
            cache_dir = current.resolve() if current.exists() else args.processed_dir
        print("=" * 55)
        print("STEP 3 - Create splits")
        print("=" * 55)
        run_splits(
            cache_dir=cache_dir,
            train_size=args.train_size,
            val_size=args.val_size,
            seed=args.split_seed,
            train_out=args.train_split_out.resolve(),
            val_out=args.val_split_out.resolve(),
            hash_contents=args.hash_file_contents,
        )

    print("=" * 55)
    print("AV2 setup complete.")
    if cache_dir:
        print(f"Processed data : {cache_dir}")
        print(f"Symlink        : {args.processed_dir / 'current'}")
    print("=" * 55)
    return cache_dir
