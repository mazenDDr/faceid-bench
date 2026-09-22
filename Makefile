.PHONY: check test env-check pull readme

check:
	python -m ruff check .
	python -m ruff format --check .
	PYTHONPATH=src python -m pytest

test:
	PYTHONPATH=src python -m pytest

readme:
	PYTHONPATH=src python scripts/build_readme.py

pull:
	rsync -az --exclude-from=.pullignore gpu-box:$${GPU_REMOTE_DIR:-faceid-bench}/outputs/ ./outputs/

env-check:
	PYTHONPATH=src python scripts/env_check.py
