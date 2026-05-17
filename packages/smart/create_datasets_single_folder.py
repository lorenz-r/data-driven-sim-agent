import os
import random
import shutil
import hashlib
from pathlib import Path

# ---------------- CONFIG ----------------
SOURCE_DIR = Path("train")  # original large dataset

TRAIN_OUT = Path("train_split")
VAL_OUT = Path("validation_split")

TRAIN_SIZE = 16_000
VAL_SIZE = 4_000

SEED = 42

# If True:
# hash includes file contents + paths
# If False:
# hash includes only relative paths
HASH_FILE_CONTENTS = False
# ----------------------------------------


def collect_files(source_dir: Path):
    files = []

    for root, _, filenames in os.walk(source_dir):
        for f in filenames:
            full_path = Path(root) / f
            files.append(full_path)

    return sorted(files)


def sample_without_overlap(all_files, train_size, val_size):
    total_needed = train_size + val_size

    if len(all_files) < total_needed:
        raise ValueError(
            f"Not enough files: found {len(all_files)}, need {total_needed}"
        )

    random.shuffle(all_files)

    train_files = all_files[:train_size]
    val_files = all_files[train_size : train_size + val_size]

    return train_files, val_files


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


def write_metadata(
    output_dir,
    dataset_hash,
    dataset_name,
    sample_size,
):
    with open(output_dir / "dataset_info.txt", "w") as f:
        f.write(f"dataset={dataset_name}\n")
        f.write(f"seed={SEED}\n")
        f.write(f"sample_size={sample_size}\n")
        f.write(f"hash={dataset_hash}\n")
        f.write(f"hash_file_contents={HASH_FILE_CONTENTS}\n")


def main():
    random.seed(SEED)

    # -------- COLLECT --------
    all_files = collect_files(SOURCE_DIR)

    # -------- SAMPLE --------
    train_files, val_files = sample_without_overlap(all_files, TRAIN_SIZE, VAL_SIZE)

    # -------- CREATE OUTPUT DIRS --------
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    VAL_OUT.mkdir(parents=True, exist_ok=True)

    # -------- COPY TRAIN --------
    print(f"Copying {len(train_files)} train files → {TRAIN_OUT}")

    copy_files(train_files, SOURCE_DIR, TRAIN_OUT)

    # -------- COPY VALIDATION --------
    print(f"Copying {len(val_files)} validation files → {VAL_OUT}")

    copy_files(val_files, SOURCE_DIR, VAL_OUT)

    # -------- MANIFESTS --------
    write_manifest(train_files, SOURCE_DIR, TRAIN_OUT / "train_manifest.txt")

    write_manifest(val_files, SOURCE_DIR, VAL_OUT / "validation_manifest.txt")

    # -------- HASHES --------
    train_hash = generate_dataset_hash(train_files, SOURCE_DIR)

    val_hash = generate_dataset_hash(val_files, SOURCE_DIR)

    # -------- METADATA --------
    write_metadata(TRAIN_OUT, train_hash, "train", TRAIN_SIZE)

    write_metadata(VAL_OUT, val_hash, "validation", VAL_SIZE)

    print("Done ✔")
    print(f"Train hash: {train_hash}")
    print(f"Validation hash: {val_hash}")


if __name__ == "__main__":
    main()
