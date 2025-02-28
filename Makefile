install:
	uv sync

train:
	uv run --no-sync src/llama3_jax/train.py
