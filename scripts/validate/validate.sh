#!/usr/bin/env bash
# Usage: ./scripts/validate/validate.sh <model_path> <results_dir>
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_path> <results_dir>" >&2
    exit 1
fi

MODEL="$1"
RESULTS_DIR="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

run() { echo ">>> $*"; "$@"; echo; }

# ROUGE on validation + test splits
run python "$SCRIPT_DIR/validate_sft.py" \
    --model "$MODEL" --split validation --results-dir "$RESULTS_DIR"

run python "$SCRIPT_DIR/validate_sft.py" \
    --model "$MODEL" --split test --results-dir "$RESULTS_DIR"

# Toxicity on GRPO + SFT validation splits
run python "$SCRIPT_DIR/validate_grpo.py" \
    --model "$MODEL" --grpo-split validation --sft-split validation --results-dir "$RESULTS_DIR"

# Toxicity on SFT test split (GRPO predictions reused from above)
run python "$SCRIPT_DIR/validate_grpo.py" \
    --model "$MODEL" --grpo-split test --sft-split test --results-dir "$RESULTS_DIR"

echo "=== All done. Results: $RESULTS_DIR/metrics.json ==="
