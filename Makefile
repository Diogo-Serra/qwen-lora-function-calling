PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
HF := $(VENV)/bin/hf
DEFAULT_MODEL_REPO := Qwen/Qwen2.5-0.5B
DEFAULT_MODEL_DIR := src/model/Qwen2.5-0.5B

.PHONY: install run clean clean-cache

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt
	$(HF) download $(DEFAULT_MODEL_REPO) --local-dir $(DEFAULT_MODEL_DIR)

run:
	./run.sh

clean: clean-cache

clean-cache:
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
	find src -type f -name "*.py[cod]" -delete
