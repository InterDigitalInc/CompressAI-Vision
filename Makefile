.DEFAULT_GOAL := help

PYTORCH_DOCKER_IMAGE = pytorch/pytorch:1.8.1-cuda11.1-cudnn8
PYTHON_DOCKER_IMAGE = python:3.8-buster

GIT_DESCRIBE = $(shell git describe --first-parent)
ARCHIVE = compressai_vision.tar.gz

src_dirs := compressai_vision scripts/metrics

.PHONY: help
help: ## Show this message
	@echo "Usage: make COMMAND\n\nCommands:"
	@grep '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' | cat


# Check style and linting
.PHONY: check-ruff-format check-ruff-organize-imports check-ruff-lint check-mypy static-analysis

check-ruff-format: ## Run ruff format checks
	@echo "--> Running ruff format checks"
	@ruff format --check $(src_dirs)

check-ruff-organize-imports: ## Run ruff organize imports checks
	@echo "--> Running ruff organize imports checks"
	@ruff check --ignore ALL --select I $(src_dirs)

check-ruff-lint: ## Run ruff lint checks
	@echo "--> Running ruff lint checks"
	@ruff check $(src_dirs)

# check-mypy: ## Run mypy checks
# 	@echo "--> Running mypy checks"
# 	@mypy

# code-format:
# 	@echo "--> Running black"
# 	@black $(src_dirs)
# 	@echo "--> Running isort"
# 	@isort $(src_dirs)

static-analysis: check-ruff-format check-ruff-organize-imports check-ruff-lint # check-mypy ## Run all static checks


# Apply styling
.PHONY: style

style: ## Apply style formating
	@echo "--> Running ruff format"
	@ruff format $(src_dirs)
	@echo "--> Running ruff check --ignore ALL --select I"
	@ruff check --ignore ALL --select I $(src_dirs)

