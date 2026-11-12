# Developer entry points. Run `make help` for the list.
# PYTORCH_ENABLE_MPS_FALLBACK is for local Apple-silicon runs only; Docker and CI do not set it.

PY       ?= .venv/bin/python
PIP      ?= .venv/bin/pip
RUFF     ?= .venv/bin/ruff
PART     ?= part_B
RUN      ?=
LOCAL_ENV = PYTORCH_ENABLE_MPS_FALLBACK=1

.DEFAULT_GOAL := help
.PHONY: help setup env-check verify-data preprocess train eval calibrate api web test test-ml test-api test-web \
        lint format download-weights docker-build docker-up docker-down

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  %-18s %s\n", $$1, $$2}'

setup:  ## Create .venv, install pinned deps, the ml package and web deps
	python3.11 -m venv .venv
	$(PIP) install -r requirements-lock.txt
	$(PIP) install --no-deps -e ml
	cd web && npm ci

env-check:  ## Print versions and run a backward pass on the selected device
	$(LOCAL_ENV) $(PY) -m crowd.env_check

verify-data:  ## Compare data/ShanghaiTech against data/MANIFEST.json
	$(PY) -m crowd.data.manifest verify

preprocess:  ## Raw dataset -> data/processed/*.h5 density maps
	$(PY) -m crowd.data.build

train:  ## Train one part locally (PART=part_A|part_B). Full runs belong on Colab.
	$(LOCAL_ENV) $(PY) -m crowd.train --part $(PART)

eval:  ## Evaluate a finished run on the test split, once (RUN=runs/<run_id>)
	@test -n "$(RUN)" || (echo "usage: make eval RUN=runs/<run_id>"; exit 2)
	$(LOCAL_ENV) $(PY) -m crowd.evaluate $(RUN)

calibrate:  ## Recompute density-level bands from training annotations
	$(PY) -m crowd.calibrate

api:  ## Run the API on :8000 (needs models/ checkpoints, see api/.env.example)
	$(PY) -m uvicorn --factory app.main:create_app --app-dir api --reload --reload-dir api --port 8000

web:  ## Run the frontend dev server on :5173 (proxies /api to :8000)
	cd web && npm run dev

test: test-ml test-api test-web  ## All test suites

test-ml:
	cd ml && ../$(PY) -m pytest -q

test-api:
	cd api && ../$(PY) -m pytest -q

test-web:
	cd web && npm test

lint:  ## ruff lint + format check, eslint
	$(RUFF) check .
	$(RUFF) format --check .
	cd web && npm run lint

format:  ## Apply ruff formatting and safe fixes
	$(RUFF) check --fix .
	$(RUFF) format .

download-weights:  ## Fetch checkpoints listed in models/weights.json and verify SHA-256
	$(PY) scripts/download_weights.py

docker-build:  ## Build the api and web images
	docker compose build

docker-up: download-weights  ## Start api + web (http://localhost:8080)
	docker compose up -d --wait

docker-down:
	docker compose down
