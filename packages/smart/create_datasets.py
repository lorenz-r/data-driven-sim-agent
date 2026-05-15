import os
import random
import shutil
import hashlib
from pathlib import Path

# ---------------- CONFIG ----------------
SOURCE_DIR = Path("train")
TRAIN_OUT = Path("train_split")

SOURCE_VAL_DIR = Path("validation")
VAL_OUT = Path("validation_split")

TRAIN_SIZE = 16_000
VAL_SIZE = 4_000

SEED = 42

HASH_FILE_CONTENTS = False
# ----------------------------------------


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


def generate_dataset_hash(file_list, source_root):
    sha256 = hashlib.sha256()

    for file_path in sorted(file_list):
        relative_path = str(file_path.relative_to(source_root))
        sha256.update(relative_path.encode())

        if HASH_FILE_CONTENTS:
            content_hash = hash_file_contents(file_path)
            sha256.update(content_hash.encode())

    return sha256.hexdigest()


def main():
    random.seed(SEED)

    # -------- TRAIN --------
    train_all_files = collect_files(SOURCE_DIR)
    train_files = sample_files(train_all_files, TRAIN_SIZE)

    TRAIN_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Copying {len(train_files)} train files → {TRAIN_OUT}")
    copy_files(train_files, SOURCE_DIR, TRAIN_OUT)

    write_manifest(
        train_files,
        SOURCE_DIR,
        TRAIN_OUT / "train_manifest.txt"
    )

    train_hash = generate_dataset_hash(train_files, SOURCE_DIR)

    with open(TRAIN_OUT / "dataset_hash.txt", "w") as f:
        f.write(f"seed={SEED}\n")
        f.write(f"hash={train_hash}\n")

    # -------- VALIDATION --------
    val_all_files = collect_files(SOURCE_VAL_DIR)
    val_files = sample_files(val_all_files, VAL_SIZE)

    VAL_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Copying {len(val_files)} validation files → {VAL_OUT}")
    copy_files(val_files, SOURCE_VAL_DIR, VAL_OUT)

    write_manifest(
        val_files,
        SOURCE_VAL_DIR,
        VAL_OUT / "validation_manifest.txt"
    )

    val_hash = generate_dataset_hash(val_files, SOURCE_VAL_DIR)

    with open(VAL_OUT / "dataset_hash.txt", "w") as f:
        f.write(f"seed={SEED}\n")
        f.write(f"hash={val_hash}\n")

    print("Done ✔")


# This script creates reproducible train and validation datasets by sampling from the ALREADY PREPROCESSED source directories.
if __name__ == "__main__":
    main()