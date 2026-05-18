#!/usr/bin/env python3
"""
download_av2.py  –  Download AV2 motion-forecasting scenarios.

Modes
-----
dev   Stream the head of the remote TAR and extract the first --scenarios
      scenario folders.  Fast and stateless — no caching, always re-runs.

full  Download both complete TARs (resumable via Range requests), then
      extract.  After each download the file is verified by recomputing its
      S3 multipart ETag (md5(md5_p1 || … || md5_pN)-N, 8 MB parts) and
      comparing against the hardcoded value.  On subsequent runs the same
      check is run against the file on disk; if it passes the download is
      skipped.  Disk space is checked upfront via HEAD before anything is
      written.

TAR cache locations (full mode, tried in order)
-----------------------------------------------
  1. --downloads-dir   if explicitly passed on the CLI / ensure_downloaded()
  2. ~/argoverse_cache  user-level shared cache
  3. ~/.cache/argoverse XDG cache dir
  4. <out>/downloads   project-local fallback

The first directory that either already exists or can be created is used.

Usage
-----
    uv run python packages/smart/scripts/download_av2.py --dev --scenarios 100
    uv run python packages/smart/scripts/download_av2.py --full
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

TRAIN_URL = (
    "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/train.tar"
)
VAL_URL = (
    "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/val.tar"
)

# Hardcoded S3 multipart ETags — md5(md5_p1 || … || md5_pN)-N, 8 MB parts.
# Derived once via boto3 HeadObject and confirmed against the formula below.
# Changing either value means the dataset was replaced on S3.
KNOWN_ETAGS: dict[str, str] = {
    TRAIN_URL: "26bd34c7100b7248ad9dfe67088d0e8b-6065",
    VAL_URL: "4cc1a4320f93c334b888cb8f64cb841e-759",
}

S3_PART_SIZE = 8 * 1024 * 1024  # 8 MB — AWS CLI default, confirmed for AV2
CHUNK = 4 * 1024 * 1024  # 4 MiB I/O chunk
PER_SCENARIO_BYTES = 512 * 1024  # conservative bytes-per-scenario in TAR head

# ---------------------------------------------------------------------------
# TAR cache directory resolution
# ---------------------------------------------------------------------------


def _candidate_download_dirs(out_dir: Path) -> list[Path]:
    """
    Ordered list of candidate directories for storing downloaded TARs.
    The caller's explicit --downloads-dir is prepended before this list.
    """
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return [
        Path.home() / "argoverse_cache",
        xdg_cache / "argoverse",
        out_dir / "downloads",  # project-local fallback
    ]


def resolve_downloads_dir(out_dir: Path, override: Path | None = None) -> Path:
    """
    Return the first usable TAR cache directory.

    Tries override first, then the standard candidates in order.  A directory
    is "usable" if it already exists or can be created with a quick mkdir.
    Falls back to out_dir/downloads if nothing else works.
    """
    candidates = ([override] if override else []) + _candidate_download_dirs(out_dir)
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            return d
        except OSError:
            continue
    # Should never reach here (out_dir/downloads always creatable if out_dir is)
    fallback = out_dir / "downloads"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _head(url: str) -> tuple[str, int]:
    """Return (etag, content_length) from a HEAD request."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        etag = resp.headers.get("ETag", "").strip('"')
        content_length = int(resp.headers.get("Content-Length", 0))
    return etag, content_length


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------


def _check_disk_space(out_dir: Path, required_bytes: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_dir).free
    if free < required_bytes:
        raise SystemExit(
            f"Insufficient disk space in {out_dir}:\n"
            f"  available : {free / 1e9:.1f} GB\n"
            f"  required  : {required_bytes / 1e9:.1f} GB\n"
            "Free up space or point --out at a larger volume."
        )


# ---------------------------------------------------------------------------
# ETag verification
# ---------------------------------------------------------------------------


def _compute_etag(tar_path: Path) -> str:
    """
    Recompute the S3 multipart ETag from the file on disk.
    Formula: md5(md5(part_1) || … || md5(part_N))-N, S3_PART_SIZE boundaries.
    """
    part_digests: list[bytes] = []
    size = tar_path.stat().st_size
    done = 0
    with open(tar_path, "rb") as f:
        while True:
            block = f.read(S3_PART_SIZE)
            if not block:
                break
            part_digests.append(hashlib.md5(block).digest())
            done += len(block)
            print(
                f"\r  verifying {tar_path.name}: {done / size * 100:.1f}%",
                end="",
                flush=True,
            )
    print()
    if len(part_digests) == 1:
        return hashlib.md5(part_digests[0]).hexdigest()
    combined = b"".join(part_digests)
    return f"{hashlib.md5(combined).hexdigest()}-{len(part_digests)}"


def _verify_on_disk(tar_path: Path, url: str) -> bool:
    """
    Return True if the file exists and its recomputed ETag matches the
    hardcoded value.  Missing file or mismatch → False.
    """
    if not tar_path.exists():
        return False
    expected = KNOWN_ETAGS.get(url)
    if expected is None:
        print(f"  WARNING: no known ETag for {url}; skipping on-disk verification.")
        return True
    actual = _compute_etag(tar_path)
    if actual == expected:
        print(f"  ETag OK ({actual})")
        return True
    print(f"  ETag MISMATCH\n    expected : {expected}\n    got      : {actual}")
    return False


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _is_complete_scenario(d: Path) -> bool:
    return any(d.glob("scenario_*.parquet")) and any(d.glob("log_map_archive_*.json"))


def _prune_to(train_dir: Path, n: int) -> int:
    kept = 0
    for d in sorted(train_dir.iterdir()):
        if not d.is_dir():
            continue
        if _is_complete_scenario(d):
            kept += 1
            if kept > n:
                shutil.rmtree(d)
        else:
            shutil.rmtree(d)
    return min(kept, n)


def _extract_stream(data: bytes, out_dir: Path) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                rel = Path(*Path(member.name).parts[1:])
                dest = out_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                fobj = tf.extractfile(member)
                if fobj:
                    dest.write_bytes(fobj.read())
    except tarfile.ReadError:
        pass


def _extract_tar(tar_path: Path, split_name: str, out_dir: Path) -> None:
    dest = out_dir / split_name
    if dest.exists():
        print(f"  {split_name}/ already extracted, skipping.")
        return
    print(f"Extracting {split_name} …")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if member.name.startswith(split_name + "/"):
                tf.extract(member, out_dir)


# ---------------------------------------------------------------------------
# Dev mode: stream head of TAR — stateless, no caching
# ---------------------------------------------------------------------------


def download_dev(url: str, n_scenarios: int, out_dir: Path) -> str:
    """
    Stream just enough bytes to extract n_scenarios complete scenario folders.
    Always re-runs; no cache check.  Returns the S3 ETag (from HEAD only).
    """
    print(f"HEAD {url}")
    etag, content_length = _head(url)
    print(f"  ETag={etag}  total size={content_length / 1e9:.1f} GB")

    byte_budget = n_scenarios * PER_SCENARIO_BYTES
    print(f"Streaming ~{byte_budget // 1024} KiB for {n_scenarios} scenarios …")

    buf = io.BytesIO()
    with urllib.request.urlopen(url, timeout=60) as req:
        downloaded = 0
        while downloaded < byte_budget:
            chunk = req.read(min(CHUNK, byte_budget - downloaded))
            if not chunk:
                break
            buf.write(chunk)
            downloaded += len(chunk)

    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    _extract_stream(buf.getvalue(), train_dir)
    kept = _prune_to(train_dir, n_scenarios)
    print(f"Extracted and kept {kept} complete scenario directories.")
    return etag


# ---------------------------------------------------------------------------
# Full mode: resumable download + on-disk ETag verification
# ---------------------------------------------------------------------------


def _download_tar(url: str, tar_path: Path, expected_bytes: int) -> None:
    existing = tar_path.stat().st_size if tar_path.exists() else 0
    if existing == expected_bytes:
        print(f"  {tar_path.name}: already complete ({expected_bytes / 1e9:.1f} GB).")
        return

    req = urllib.request.Request(url)
    if existing:
        print(f"  {tar_path.name}: resuming from {existing / 1e9:.2f} GB …")
        req.add_header("Range", f"bytes={existing}-")
        open_mode = "ab"
    else:
        open_mode = "wb"

    with (
        urllib.request.urlopen(req, timeout=120) as resp,
        open(tar_path, open_mode) as f,
    ):
        done = existing
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            f.write(block)
            done += len(block)
            print(
                f"\r  {tar_path.name}: {done / 1e9:.2f} / "
                f"{expected_bytes / 1e9:.1f} GB  ({done / expected_bytes * 100:.1f}%)",
                end="",
                flush=True,
            )
    print()

    actual = tar_path.stat().st_size
    if actual != expected_bytes:
        raise RuntimeError(
            f"{tar_path.name}: expected {expected_bytes} bytes, got {actual}."
        )


def download_full(
    train_url: str,
    val_url: str,
    out_dir: Path,
    downloads_dir: Path | None = None,
) -> tuple[str, str]:
    """
    Download and extract both train and val TARs with resume support.
    downloads_dir overrides the default cache-dir resolution; if None the
    standard candidates are tried (~/argoverse_cache first).
    """
    downloads = resolve_downloads_dir(out_dir, downloads_dir)
    print(f"TAR cache dir: {downloads}")

    train_tar = downloads / train_url.rsplit("/", 1)[-1]
    val_tar = downloads / val_url.rsplit("/", 1)[-1]

    # ── Cache check: recompute ETag from bytes on disk ────────────────────
    if _verify_on_disk(train_tar, train_url) and _verify_on_disk(val_tar, val_url):
        print("Both TARs verified on disk; skipping download.")
    else:
        # ── Disk space preflight (HEAD both URLs) ─────────────────────────
        _, train_size = _head(train_url)
        _, val_size = _head(val_url)
        print(f"  train : {train_size / 1e9:.1f} GB")
        print(f"  val   : {val_size / 1e9:.1f} GB")
        _check_disk_space(out_dir, (train_size + val_size) * 2)

        _download_tar(train_url, train_tar, train_size)
        _download_tar(val_url, val_tar, val_size)

        # Verify freshly-downloaded files
        if not _verify_on_disk(train_tar, train_url):
            raise RuntimeError(f"ETag verification failed for {train_tar.name}")
        if not _verify_on_disk(val_tar, val_url):
            raise RuntimeError(f"ETag verification failed for {val_tar.name}")

    _extract_tar(train_tar, "train", out_dir)
    _extract_tar(val_tar, "val", out_dir)

    return KNOWN_ETAGS[train_url], KNOWN_ETAGS[val_url]


# ---------------------------------------------------------------------------
# Public entry point used by setup_av2.py
# ---------------------------------------------------------------------------


def ensure_downloaded(
    out_dir: Path,
    mode: str,
    n_scenarios: int,
    train_url: str = TRAIN_URL,
    val_url: str = VAL_URL,
    downloads_dir: Path | None = None,
) -> str:
    """
    Ensure raw scenario data is present in out_dir.
    downloads_dir overrides the TAR cache location (full mode only).
    Returns the train ETag (hardcoded for full mode, HEAD-derived for dev).
    """
    if mode == "dev":
        return download_dev(train_url, n_scenarios, out_dir)

    train_etag, _ = download_full(train_url, val_url, out_dir, downloads_dir)
    return train_etag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download AV2 motion-forecasting scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dev",
        dest="mode",
        action="store_const",
        const="dev",
        default="dev",
        help="Stream first N scenarios only — fast, stateless (default)",
    )
    mode.add_argument(
        "--full",
        dest="mode",
        action="store_const",
        const="full",
        help="Download complete dataset; verify on-disk ETag before skipping",
    )
    p.add_argument(
        "--scenarios",
        type=int,
        default=5,
        help="Number of scenarios to extract in dev mode (default: 5)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw",
        help="Output directory (default: packages/smart/data/raw)",
    )
    p.add_argument("--train-url", default=TRAIN_URL)
    p.add_argument("--val-url", default=VAL_URL)
    p.add_argument(
        "--downloads-dir",
        type=Path,
        default=None,
        help="Override TAR cache directory (default: ~/argoverse_cache, "
        "then ~/.cache/argoverse, then <out>/downloads)",
    )
    return p


def main() -> None:
    args = _parser().parse_args()
    etag = ensure_downloaded(
        args.out,
        args.mode,
        args.scenarios,
        args.train_url,
        args.val_url,
        args.downloads_dir,
    )
    print(f"\nDone. Train ETag: {etag}")


if __name__ == "__main__":
    main()
