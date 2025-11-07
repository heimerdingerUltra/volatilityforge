.PHONY: help install install-dev test lint format clean train serve

PYTHON := python3
POETRY := poetry
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8

help:
	@echo "Volatility Forge - Available Commands"
	@echo "======================================"
	@echo "install          - Install production dependencies"
	@echo "install-dev      - Install with development dependencies"
	@echo "install-all      - Install all dependencies including optional"
	@echo "test             - Run tests"
	@echo "test-cov         - Run tests with coverage"
	@echo "lint             - Run linters"
	@echo "format           - Format code with black and isort"
	@echo "clean            - Clean build artifacts and cache"
	@echo "train            - Train models with sample data"
	@echo "serve            - Start API server"
	@echo "serve-dev        - Start API server in development mode"
	@echo "build            - Build package"
	@echo "docs             - Build documentation"

install:
	$(POETRY) install --only main

install-dev:
	$(POETRY) install --with dev

install-all:
	$(POETRY) install --with dev,docs

install-pip:
	$(PYTHON) -m pip install -e .

install-pip-dev:
	$(PYTHON) -m pip install -e ".[dev,api]"

test:
	$(POETRY) run $(PYTEST) tests/ -v

test-cov:
	$(POETRY) run $(PYTEST) tests/ -v --cov=src --cov=config --cov=core --cov-report=html --cov-report=term

test-watch:
	$(POETRY) run ptw tests/ -- -v

lint:
	$(POETRY) run $(FLAKE8) src/ config/ core/ --max-line-length=100
	$(POETRY) run $(BLACK) --check src/ config/ core/ scripts/
	$(POETRY) run $(ISORT) --check-only src/ config/ core/ scripts/

format:
	$(POETRY) run $(BLACK) src/ config/ core/ scripts/ tests/
	$(POETRY) run $(ISORT) src/ config/ core/ scripts/ tests/

type-check:
	$(POETRY) run mypy src/ config/ core/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete

clean-cache:
	rm -rf .cache/

clean-all: clean clean-cache
	rm -rf checkpoints/ outputs/ logs/
	rm -rf model_registry/

train-sample:
	$(POETRY) run python scripts/train_advanced.py \
		--data data/sample.xlsx \
		--models tabpfn \
		--strategy balanced \
		--epochs 50 \
		--output outputs/sample/

train-full:
	$(POETRY) run python scripts/train_advanced.py \
		--data data/options.xlsx \
		--models tabpfn mamba xlstm \
		--ensemble-type attention \
		--strategy accurate \
		--epochs 200 \
		--output outputs/full/

train-fast:
	$(POETRY) run python scripts/train_advanced.py \
		--data data/options.xlsx \
		--models tabpfn \
		--strategy fast \
		--epochs 100 \
		--batch-size 1024 \
		--output outputs/fast/

serve:
	$(POETRY) run uvicorn api.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--workers 4

serve-dev:
	$(POETRY) run uvicorn api.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

serve-prod:
	$(POETRY) run gunicorn api.main:app \
		--workers 4 \
		--worker-class uvicorn.workers.UvicornWorker \
		--bind 0.0.0.0:8000 \
		--timeout 120 \
		--access-logfile logs/access.log \
		--error-logfile logs/error.log

build:
	$(POETRY) build

build-wheel:
	$(POETRY) build -f wheel

docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && $(PYTHON) -m http.server 8080

docker-build:
	docker build -t volatility-forge:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/model_registry:/app/model_registry volatility-forge:latest

pre-commit-install:
	$(POETRY) run pre-commit install

pre-commit-run:
	$(POETRY) run pre-commit run --all-files

db-upgrade:
	$(POETRY) run alembic upgrade head

db-downgrade:
	$(POETRY) run alembic downgrade -1

notebook:
	$(POETRY) run jupyter notebook

info:
	$(POETRY) show
	$(POETRY) env info

update:
	$(POETRY) update

lock:
	$(POETRY) lock

export-requirements:
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes
	$(POETRY) export -f requirements.txt --output requirements-dev.txt --with dev --without-hashes

profile-train:
	$(POETRY) run python -m cProfile -o profile.stats scripts/train_advanced.py --data data/sample.xlsx
	$(POETRY) run python -m pstats profile.stats

security-check:
	$(POETRY) run pip-audit

outdated:
	$(POETRY) show --outdated