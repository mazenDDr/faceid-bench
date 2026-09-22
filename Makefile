.PHONY: check test env-check pull readme demo

check:
	python -m ruff check .
	python -m ruff format --check .
	PYTHONPATH=src python -m pytest

test:
	PYTHONPATH=src python -m pytest

demo:
	PYTHONPATH=src python scripts/demo_webcam.py

readme:
	PYTHONPATH=src python scripts/build_readme.py
	PYTHONPATH=src python scripts/build_site.py

pull:
	rsync -az --exclude-from=.pullignore gpu-box:$${GPU_REMOTE_DIR:-faceid-bench}/outputs/ ./outputs/

env-check:
	PYTHONPATH=src python scripts/env_check.py
