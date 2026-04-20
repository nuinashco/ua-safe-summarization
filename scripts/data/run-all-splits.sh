#!/usr/bin/env bash
# Run prepare-data.py for all three XL-Sum splits, then optionally push to Hub.
#
# Usage:
#   ./scripts/run-all-splits.sh
#   ./scripts/run-all-splits.sh --push nuinashco/xlsum-ua-processed
#   ./scripts/run-all-splits.sh --push nuinashco/xlsum-ua-processed output.format=jsonl

set -euo pipefail

HF_REPO=""
HYDRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push) HF_REPO="$2"; shift 2 ;;
        *)      HYDRA_ARGS+=("$1"); shift ;;
    esac
done

for split in train validation test; do
    echo "=== Split: $split ==="
    uv run python scripts/data/prepare-data.py "dataset.split=${split}" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"}
done

if [[ -n "$HF_REPO" ]]; then
    echo "=== Pushing to Hub: $HF_REPO ==="
    hf upload "$HF_REPO" data/processed/ . --repo-type dataset
fi
