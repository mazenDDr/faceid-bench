.PHONY: check test env-check pull

check:
	python -m ruff check .
	python -m ruff format --check .
	PYTHONPATH=src python -m pytest

test:
	PYTHONPATH=src python -m pytest

pull:
	rsync -az --exclude-from=.pullignore gpu-box:$${GPU_REMOTE_DIR:-faceid-bench}/outputs/ ./outputs/

env-check:
	PYTHONPATH=src python scripts/env_check.py
