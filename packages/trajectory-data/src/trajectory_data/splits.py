"""Create reproducible train/validation splits from processed trajectory files."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
from pathlib import Path
from typing import Sequence


DEFAULT_SOURCE_DIR = Path("data/processed/train")
DEFAULT_TRAIN_OUT = Path("train_split")

DEFAULT_SOURCE_VAL_DIR = Path("data/processed/val")
DEFAULT_VAL_OUT = Path("validation_split")

DEFAULT_TRAIN_SIZE = 8_000
DEFAULT_VAL_SIZE = 2_000

DEFAULT_SEED = 42


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible train/validation dataset splits "
            "from processed trajectory files."
        )
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Source directory for training samples.",
    )

    parser.add_argument(
        "--source-val-dir",
        type=Path,
        default=DEFAULT_SOURCE_VAL_DIR,
        help="Source directory for validation samples.",
    )

    parser.add_argument(
        "--train-out",
        type=Path,
        default=DEFAULT_TRAIN_OUT,
        help="Output directory for train split.",
    )

    parser.add_argument(
        "--val-out",
        type=Path,
        default=DEFAULT_VAL_OUT,
        help="Output directory for validation split.",
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help="Number of train samples.",
    )

    parser.add_argument(
        "--val-size",
        type=int,
        default=DEFAULT_VAL_SIZE,
        help="Number of validation samples.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--hash-file-contents",
        action="store_true",
        help="Include file contents in dataset hash.",
    )

    return parser.parse_args(argv)


def collect_files(source_dir: Path):
    files = []

    for root, _, filenames in os.walk(source_dir):
        for f in filenames:
            full_path = Path(root) / f
            files.append(full_path)

    return sorted(files)


def sample_files(all_files, sample_size):
    if len(all_files) < sample_size:
        raise ValueError(
            f"Not enough files: found {len(all_files)}, need {sample_size}"
        )

    random.shuffle(all_files)
    return all_files[:sample_size]


def copy_files(file_list, source_root, target_root):
    for file_path in file_list:
        relative_path = file_path.relative_to(source_root)
        target_path = target_root / relative_path

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target_path)


def write_manifest(file_list, source_root, manifest_path):
    with open(manifest_path, "w") as f:
        for file_path in sorted(file_list):
            relative_path = file_path.relative_to(source_root)
            f.write(str(relative_path) + "\n")


def hash_file_contents(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def generate_dataset_hash(file_list, source_root, hash_contents=False):
    sha256 = hashlib.sha256()

    for file_path in sorted(file_list):
        relative_path = str(file_path.relative_to(source_root))
        sha256.update(relative_path.encode())

        if hash_contents:
            content_hash = hash_file_contents(file_path)
            sha256.update(content_hash.encode())

    return sha256.hexdigest()


def process_split(
    source_dir: Path,
    output_dir: Path,
    sample_size: int,
    manifest_path: Path,
    hash_path: Path,
    seed: int,
    hash_contents: bool,
    split_name: str,
):
    all_files = collect_files(source_dir)

    print(f"Found {len(all_files)} files in {source_dir}")

    sampled_files = sample_files(all_files, sample_size)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying {len(sampled_files)} {split_name} files → {output_dir}")

    copy_files(sampled_files, source_dir, output_dir)

    write_manifest(sampled_files, source_dir, manifest_path)

    dataset_hash = generate_dataset_hash(
        sampled_files, source_dir, hash_contents=hash_contents
    )

    with open(hash_path, "w") as f:
        f.write(f"seed={seed}\n")
        f.write(f"hash={dataset_hash}\n")

    print(f"{split_name.capitalize()} dataset hash: {dataset_hash}")


def create_splits(
    *,
    source_dir: Path,
    source_val_dir: Path,
    train_out: Path,
    val_out: Path,
    train_size: int,
    val_size: int,
    seed: int = DEFAULT_SEED,
    hash_contents: bool = False,
) -> None:
    """Create train/validation splits from one or two processed-data pools."""

    random.seed(seed)

    # Single-pool mode: when source and val source are the same directory,#-#
    # sample train + val together without overlap.#-#
    if source_dir.resolve() == source_val_dir.resolve():
        all_files = collect_files(source_dir)
        need = train_size + val_size
        if len(all_files) < need:#-#
            raise ValueError(#-#
                f"Not enough files: found {len(all_files)}, "#-#
                f"need {train_size} + {val_size} = {need}"
            )#-#
        random.shuffle(all_files)#-#
        train_files = all_files[:train_size]
        val_files = all_files[train_size:need]
#-#
        for files, src, out, name, manifest, hash_file in [#-#
            (#-#
                train_files,#-#
                source_dir,
                train_out,
                "train",#-#
                train_out / "train_manifest.txt",
                train_out / "train_dataset_hash.txt",
            ),#-#
            (#-#
                val_files,#-#
                source_dir,
                val_out,
                "validation",#-#
                val_out / "validation_manifest.txt",
                val_out / "validation_dataset_hash.txt",
            ),#-#
        ]:#-#
            out.mkdir(parents=True, exist_ok=True)#-#
            print(f"Copying {len(files)} {name} files → {out}")#-#
            copy_files(files, src, out)#-#
            write_manifest(files, src, manifest)#-#
            dataset_hash = generate_dataset_hash(#-#
                files, src, hash_contents=hash_contents
            )#-#
            with open(hash_file, "w") as f:#-#
                f.write(f"seed={seed}\n")
                f.write(f"hash={dataset_hash}\n")#-#
            print(f"-{name.capitalize()} dataset hash: {dataset_hash}")#-#
#-#
        print("Done ✔")#-#
        return#-#
#-#
    process_split(
        source_dir=source_dir,
        output_dir=train_out,
        sample_size=train_size,
        manifest_path=train_out / "train_manifest.txt",
        hash_path=train_out / "train_dataset_hash.txt",
        seed=seed,
        hash_contents=hash_contents,
        split_name="train",
    )

    process_split(
        source_dir=source_val_dir,
        output_dir=val_out,
        sample_size=val_size,
        manifest_path=val_out / "validation_manifest.txt",
        hash_path=val_out / "validation_dataset_hash.txt",
        seed=seed,
        hash_contents=hash_contents,
        split_name="validation",
    )

    print("Done ✔")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    create_splits(
        source_dir=args.source_dir,
        source_val_dir=args.source_val_dir,
        train_out=args.train_out,
        val_out=args.val_out,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        hash_contents=args.hash_file_contents,
    )


if __name__ == "__main__":
    main()
