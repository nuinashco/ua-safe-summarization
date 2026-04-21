# git
# git clone https://nuinashco:$GITHUB_TOKEN@github.com/nuinashco/ua-safe-summarization.git; cd ua-safe-summarization; source scripts/setup/setup_vast.sh

git config --global user.name "Ivan Havlytskyi"
git config --global user.email "ivan.havlytskyi@gmail.com"

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync

# login
huggingface-cli login --token $HUGGINGFACE_TOKEN
wandb login
