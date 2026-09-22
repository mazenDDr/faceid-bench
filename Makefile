.PHONY: check test env-check

check:
	python -m ruff check .
	python -m ruff format --check .
	PYTHONPATH=src python -m pytest

test:
	PYTHONPATH=src python -m pytest

env-check:
	PYTHONPATH=src python scripts/env_check.py
