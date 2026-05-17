import argparse
import hashlib
import os
import random
import shutil
from pathlib import Path


DEFAULT_SOURCE_DIR = Path("data/processed/train")
DEFAULT_TRAIN_OUT = Path("train_split")

DEFAULT_SOURCE_VAL_DIR = Path("data/processed/val")
DEFAULT_VAL_OUT = Path("validation_split")

DEFAULT_TRAIN_SIZE = 8_000
DEFAULT_VAL_SIZE = 2_000

DEFAULT_SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible train/validation dataset splits "
            "from preprocessed SMART AV2 data."
        )
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Source directory for training samples."
    )

    parser.add_argument(
        "--source-val-dir",
        type=Path,
        default=DEFAULT_SOURCE_VAL_DIR,
        help="Source directory for validation samples."
    )

    parser.add_argument(
        "--train-out",
        type=Path,
        default=DEFAULT_TRAIN_OUT,
        help="Output directory for train split."
    )

    parser.add_argument(
        "--val-out",
        type=Path,
        default=DEFAULT_VAL_OUT,
        help="Output directory for validation split."
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help="Number of train samples."
    )

    parser.add_argument(
        "--val-size",
        type=int,
        default=DEFAULT_VAL_SIZE,
        help="Number of validation samples."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--hash-file-contents",
        action="store_true",
        help="Include file contents in dataset hash."
    )

    return parser.parse_args()


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
            f"Not enough files: found {len(all_files)}, "
            f"need {sample_size}"
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


def generate_dataset_hash(
    file_list,
    source_root,
    hash_contents=False
):
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

    print(
        f"Found {len(all_files)} files in {source_dir}"
    )

    sampled_files = sample_files(all_files, sample_size)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Copying {len(sampled_files)} "
        f"{split_name} files → {output_dir}"
    )

    copy_files(
        sampled_files,
        source_dir,
        output_dir
    )

    write_manifest(
        sampled_files,
        source_dir,
        manifest_path
    )

    dataset_hash = generate_dataset_hash(
        sampled_files,
        source_dir,
        hash_contents=hash_contents
    )

    with open(hash_path, "w") as f:
        f.write(f"seed={seed}\n")
        f.write(f"hash={dataset_hash}\n")

    print(
        f"{split_name.capitalize()} dataset hash: "
        f"{dataset_hash}"
    )


def main():
    args = parse_args()

    random.seed(args.seed)

    process_split(
        source_dir=args.source_dir,
        output_dir=args.train_out,
        sample_size=args.train_size,
        manifest_path=SCRIPT_DIR / "train_manifest.txt",
        hash_path=SCRIPT_DIR / "train_dataset_hash.txt",
        seed=args.seed,
        hash_contents=args.hash_file_contents,
        split_name="train",
    )

    process_split(
        source_dir=args.source_val_dir,
        output_dir=args.val_out,
        sample_size=args.val_size,
        manifest_path=SCRIPT_DIR / "validation_manifest.txt",
        hash_path=SCRIPT_DIR / "validation_dataset_hash.txt",
        seed=args.seed,
        hash_contents=args.hash_file_contents,
        split_name="validation",
    )

    print("Done ✔")


if __name__ == "__main__":
    main()