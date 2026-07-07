#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
import shutil
from pathlib import Path

DEFAULT_URL = (#-#
    "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/train.tar"#-#
)#-#
DEFAULT_BYTES = 10 * 1024 * 1024  # 10 MiB
CHUNK_SIZE = 1024 * 1024
PER_SCENARIO_BYTES = 500 * 1024  # 500 KiB per scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download first N bytes of a TAR and extract readable entries."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Source TAR URL")
    parser.add_argument(
        "--bytes",
        dest="byte_count",
        type=int,
        default=DEFAULT_BYTES,
        help="Number of bytes to download from start (overridden by --scenarios)",
    )
    parser.add_argument(
        "--scenarios",
        dest="max_scenarios",
        type=int,
        default=0,
        help="Maximum number of scenario folders to keep after extraction",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./av2_partial_py"),
        help="Output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.byte_count <= 0:
        raise SystemExit("--bytes must be > 0")

    if args.max_scenarios < 0:
        raise SystemExit("--scenarios must be >= 0")

    # If user specified a scenario count, derive byte count conservatively.
    if args.max_scenarios:
        args.byte_count = args.max_scenarios * PER_SCENARIO_BYTES

    args.out.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        args.url,
        headers={"Range": f"bytes=0-{args.byte_count - 1}"},
    )
    with urllib.request.urlopen(request) as response:
        data = response.read()

    extracted = 0
    skipped = 0

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        while True:
            try:
                member = tar.next()
            except (tarfile.ReadError, EOFError):
                break

            if member is None:
                break

            rel_path = Path(member.name.lstrip("/"))
            if not rel_path.parts or ".." in rel_path.parts:
                skipped += 1
                continue

            target = args.out / rel_path

            if member.isdir():
                try:
                    target.mkdir(parents=True, exist_ok=True)
                    extracted += 1
                except OSError:
                    skipped += 1
                continue

            if not member.isfile():
                skipped += 1
                continue

            source = tar.extractfile(member)
            if source is None:
                skipped += 1
                continue

            ok = True
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as dst:
                    while True:
                        try:
                            chunk = source.read(CHUNK_SIZE)
                        except (tarfile.ReadError, EOFError):
                            ok = False
                            break

                        if not chunk:
                            break
                        dst.write(chunk)
            except OSError:
                ok = False

            if ok:
                extracted += 1
            else:
                target.unlink(missing_ok=True)
                skipped += 1

    print(f"downloaded_bytes={len(data)}")
    print(f"extracted_members={extracted}")
    print(f"skipped_members={skipped}")
    print(f"output_dir={args.out.resolve()}")

    # Post-process: only keep up to --scenarios scenario folders under train/
    if args.max_scenarios:
        train_dir = args.out / "train"
        kept = 0
        removed = 0
        if train_dir.exists() and train_dir.is_dir():
            # Find immediate subdirectories that look like scenarios: contain a .json and a .parquet
            scenarios = []
            for child in sorted(train_dir.iterdir()):
                if not child.is_dir():
                    continue
                has_json = False
                has_parquet = False
                for f in child.rglob("*"):
                    if f.suffix.lower() == ".json":
                        has_json = True
                    if f.suffix.lower() == ".parquet":
                        has_parquet = True
                    if has_json and has_parquet:
                        scenarios.append(child)
                        break

            # Keep the first N scenarios, remove the rest
            if len(scenarios) > args.max_scenarios:
                to_remove = scenarios[args.max_scenarios :]#-#
                for d in to_remove:
                    try:
                        shutil.rmtree(d)
                        removed += 1
                    except OSError:
                        pass
                kept = len(scenarios) - removed
            else:
                kept = len(scenarios)

        print(f"scenarios_requested={args.max_scenarios}")
        print(f"scenarios_kept={kept}")
        print(f"scenarios_removed={removed}")


if __name__ == "__main__":
    main()
