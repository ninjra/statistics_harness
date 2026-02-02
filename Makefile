SHELL := /bin/bash
PYTHON ?= python3

.PHONY: dev
dev:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && ./scripts/install_dev.sh
