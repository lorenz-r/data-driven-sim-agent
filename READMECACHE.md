# Azure Blob Dataset Caching

This README explains how to configure Azure Blob access, inspect the remote dataset structure, and run training jobs with local caching.

The intended workflow is:

1. Add Azure Blob credentials as environment variables.
2. Check what exists in the Azure Blob before downloading.
3a. Run training once in download/write-cache mode.
3b. Run later training jobs in read-from-cache mode.

---

## 1. Set Azure Blob Environment Variables

The Azure Blob URL and SAS token should be stored as environment variables.

Look in Zulip for the current Azure Blob URLs and SAS tokens.

Add the following to `~/.bashrc`:

```bash
export AV2_CACHE_CONTAINER_URL=="<container-url>"
export AV2_CACHE_READ_SAS="<sas-token>"
export AV2_CACHE_WRITE_SAS="<sas-token>"
```

Example:

```bash
export AV2_CACHE_CONTAINER_URL="https://<storage-account>.blob.core.windows.net/<container-name>"
export AV2_CACHE_READ_SAS="sp=...&st=...&se=...&spr=https&sv=...&sr=c&sig=..."
export AV2_CACHE_WRITE_SAS="sp=...&st=...&se=...&spr=https&sv=...&sr=c&sig=..."
```

Reload your shell configuration:

```bash
source ~/.bashrc
```

Check that the variables are set:

```bash
echo "$AV2_CACHE_CONTAINER_URL"
```

Do not commit SAS tokens, full SAS URLs, `.env` files, or shell config files containing secrets.

---

## 2. Inspect Azure Blob Contents

Before downloading or training, inspect the Azure Blob structure to confirm what data is available.

Use the directory-listing helper script:

```bash
./list_azure_blob_dirs.sh
```

This script should print the first, second, and third-level blob paths so that users can understand the remote directory structure before running a job.


Make the script executable:

```bash
chmod +x ./list_azure_blob_dirs.sh
```

Example output:

```text
Level 1 directories:
raw
processed
cache

Level 2 directories:
raw/train
raw/val
processed/train
processed/val

Level 3 directories:
processed/train/scenario_files
processed/val/scenario_files
```

---

## 3a. First Training Run: Download and Write to Cache

Use this mode when the VM does not already have the dataset cached locally.

This run should download the required data and write reusable files to the Azure Blob.

```bash
uv run python train.py --download-first --download-size massive --process-smart --write-cache --blob-cache-name massive-16000-4000-v1 --model smart --batch-size 32 --data-workers 8 --val-batch-size 32 --max-epochs 100 --devices 2 --dropout 0.1 --learning-rate 1e-4 --weight-decay 1e-4 --save-ckpt-path ./checkpoints --wandb-logging --project-name smart-train --run-name massive-training
```
Important: --blob-cache-name is the directory that the files will be written to in the blob

Expected behavior:

- Downloads missing files to the VM.
- Writes reusable files into `blob-cache-name`.
- Starts training after the required data is available.

Before running, you may want to check available disk space:

```bash
df -h
```

---

## 3b. Later Training Runs: Read From Cache

Use this mode when the Azure Blob already has the needed files.

This should avoid unnecessary downloads and preprocessing.

```bash
# TODO: Fill in actual training command
uv run python train.py --read-cache --blob-cache-name massive-16000-4000-v1 --model smart --batch-size 32 --data-workers 8 --val-batch-size 32 --max-epochs 100 --devices 2 --dropout 0.1 --learning-rate 1e-4 --weight-decay 1e-4 --save-ckpt-path ./checkpoints --wandb-logging --project-name smart-train --run-name massive-training
```

Expected behavior:

- Reads dataset files from Azure Blob.
- Starts faster than the first cache-writing run.

Before running, you may want to check available disk space:

```bash
df -h
```

---


## 4. Notes

- Do not commit SAS tokens or full SAS URLs.
- Do not hardcode Azure credentials in Python scripts or config files.
- Prefer environment variables for blob access and cache paths.
- Use the Azure directory-listing script before training to confirm that the expected data exists.
- Use download/write-cache mode when setting up a new VM or new data disk.
- Use read-from-cache mode for normal repeated training.
- Put the cache on a mounted data disk when possible, not the OS disk.
