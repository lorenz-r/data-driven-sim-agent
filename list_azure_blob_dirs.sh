#!/usr/bin/env bash
set -euo pipefail

if [ -z "${AV2_CACHE_CONTAINER_URL:-}" ]; then
    echo "Error: AV2_CACHE_CONTAINER_URL is not set."
    echo "Look in Zulip for the current Azure Blob container URL."
    echo "Then add it to ~/.bashrc and run: source ~/.bashrc"
    exit 1
fi

if [ -z "${AV2_CACHE_READ_SAS:-}" ]; then
    echo "Error: AV2_CACHE_READ_SAS is not set."
    echo "Look in Zulip for the current Azure Blob read SAS token."
    echo "Then add it to ~/.bashrc and run: source ~/.bashrc"
    exit 1
fi

AZURE_BLOB_FULL_URL="${AV2_CACHE_CONTAINER_URL}?${AV2_CACHE_READ_SAS}"

echo "Level 1 directories:"
azcopy list "$AZURE_BLOB_FULL_URL" | awk -F/ '{print $1}' | sort -u

echo
echo "Level 2 directories:"
azcopy list "$AZURE_BLOB_FULL_URL" | awk -F/ 'NF >= 2 {print $1 "/" $2}' | sort -u

echo
echo "Level 3 directories:"
azcopy list "$AZURE_BLOB_FULL_URL" | awk -F/ 'NF >= 3 {print $1 "/" $2 "/" $3}' | sort -u