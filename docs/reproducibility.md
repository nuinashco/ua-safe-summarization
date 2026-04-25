# Reproducibility Checklist

## Prerequisites

- Linux x86_64, Python 3.12, CUDA GPU
- `uv` package manager
- HuggingFace account with read access (for `csebuetnlp/xlsum` and `ukr-detect/ukr-toxicity-dataset-seminatural`)
- WandB account (for training metrics; set `training.report_to: none` in configs to skip)

## 1. Install dependencies

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone and install
git clone https://github.com/nuinashco/ua-safe-summarization.git
cd ua-safe-summarization
uv sync

# Install unsloth (must be done after uv sync, pins to detected CUDA version)
source .venv/bin/activate
uv pip install unsloth --torch-backend=auto
# Use nightly if the above fails:
pip uninstall unsloth unsloth_zoo -y \
  && pip install --no-deps git+https://github.com/unslothai/unsloth_zoo.git \
  && pip install --no-deps git+https://github.com/unslothai/unsloth.git
```

## 2. Configure environment

Copy `.env.sample` to `.env` and fill in:

```bash
cp .env.sample .env
# Edit .env:
#   HF_TOKEN=<your HuggingFace token>
#   HUGGINGFACE_TOKEN=<same token, used by hf CLI>
#   WANDB_API_KEY=<your WandB key>
#   SFT_DATA_PATH=data/processed/sft
#   GRPO_DATA_PATH=data/processed/grpo
```

Log in to HuggingFace and WandB:

```bash
hf auth login --token $HF_TOKEN
wandb login
```

## 3. Prepare data

**SFT data** (XL-Sum Ukrainian → `nuinashco/xlsum-ua-processed`):

```bash
# Process all three splits and push to Hub
bash scripts/data/prepare-sft.sh --push nuinashco/xlsum-ua-processed

# Or just process locally (no Hub push):
bash scripts/data/prepare-sft.sh
# → data/processed/sft/train.parquet, validation.parquet, test.parquet
```

**GRPO data** (ukr-toxicity-seminatural → `nuinashco/ukr-toxicity-processed`):

```bash
bash scripts/data/prepare-grpo.sh --push nuinashco/ukr-toxicity-processed
# → data/processed/grpo/train.parquet, validation.parquet, test.parquet
```

If the Hub datasets already exist (they do, in the collection above) you can skip this step — the training scripts pull directly from HuggingFace.

## 4. SFT training

```bash
uv run python scripts/train/train_sft.py
# Config: configs/train_sft.yaml
# Checkpoints: outputs/gemma3-1b-sft/checkpoints/
# Best checkpoint selected automatically by eval_rougeL
```

To override any config value:

```bash
uv run python scripts/train/train_sft.py training.num_train_epochs=1 training.max_steps=100
```

Expected outputs:
- `outputs/gemma3-1b-sft/checkpoints/` — final model (HF format)
- `outputs/gemma3-1b-sft/checkpoints/checkpoint-*/` — intermediate checkpoints
- `outputs/gemma3-1b-sft/checkpoints/wandb_run_id.txt` — WandB run ID

## 5. Evaluate SFT model

```bash
# ROUGE on test split
uv run python scripts/validate/validate_sft.py \
    --model outputs/gemma3-1b-sft/checkpoints \
    --split test

# → outputs/gemma3-1b-sft/results/metrics.json
# → outputs/gemma3-1b-sft/results/samples.json
```

Full evaluation (ROUGE validation + test, toxicity validation + test):

```bash
bash scripts/validate/validate.sh \
    outputs/gemma3-1b-sft/checkpoints \
    outputs/gemma3-1b-sft/results
```

## 6. GRPO alignment training

```bash
uv run python scripts/train/train_grpo.py
# Config: configs/train_grpo.yaml
# Base model: nuinashco/gemma-3-1b-it-xlsum-ua-sft (pulled from Hub)
# Checkpoints: outputs/gemma3-1b-grpo/checkpoints/
```

Expected outputs:
- `outputs/gemma3-1b-grpo/checkpoints/` — merged LoRA model (HF format)
- `outputs/gemma3-1b-grpo/checkpoints/wandb_run_id.txt`

## 7. Evaluate GRPO model

```bash
bash scripts/validate/validate.sh \
    outputs/gemma3-1b-grpo/checkpoints \
    outputs/gemma3-1b-grpo/results
# → outputs/gemma3-1b-grpo/results/metrics.json
```

## Random seeds

| Stage | Seed |
|---|---|
| SFT | 3407 (`seed: 3407` in `configs/train_sft.yaml`) |
| GRPO | 3407 (`seed: 3407` in `configs/train_grpo.yaml`) |

## Expected output files & paths

| File | Description |
|---|---|
| `outputs/gemma3-1b-sft/results/metrics.json` | ROUGE + toxicity on SFT checkpoint |
| `outputs/gemma3-1b-sft/results/samples.json` | Per-sample predictions (SFT) |
| `outputs/gemma3-1b-grpo/results/metrics.json` | ROUGE + toxicity on GRPO checkpoint |
| `outputs/gemma3-1b-grpo/results/samples.json` | Per-sample predictions (GRPO) |
| `outputs/gemma3-1b-sft/checkpoints/` | SFT model weights |
| `outputs/gemma3-1b-grpo/checkpoints/` | GRPO model weights |
