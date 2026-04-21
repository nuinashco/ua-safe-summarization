.PHONY: setup
setup:
	uv sync
	uv pip install unsloth vllm --torch-backend=auto