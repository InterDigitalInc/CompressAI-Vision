.DEFAULT_GOAL := help

PYTORCH_DOCKER_IMAGE = pytorch/pytorch:1.8.1-cuda11.1-cudnn8
PYTHON_DOCKER_IMAGE = python:3.8-buster

GIT_DESCRIBE = $(shell git describe --first-parent)
ARCHIVE = compressai_vision.tar.gz

src_dirs := compressai_vision

.PHONY: help
help: ## Show this message
	@echo "Usage: make COMMAND\n\nCommands:"
	@grep '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' | cat


# Check style and linting
.PHONY: check-black check-isort check-flake8 check-mypy static-analysis

check-black: ## Run black checks
	@echo "--> Running black checks"
	@black --check --verbose --diff $(src_dirs)

check-isort: ## Run isort checks
	@echo "--> Running isort checks"
	@isort --check-only $(src_dirs)

check-flake8: ## Run flake8 checks
	@echo "--> Running flake8 checks"
	@flake8 $(src_dirs)

# check-mypy: ## Run mypy checks
# 	@echo "--> Running mypy checks"
# 	@mypy

static-analysis: check-black check-isort check-flake8 # check-mypy ## Run all static checks


# Apply styling
.PHONY: style

style: ## Apply style formating
	@echo "--> Running black"
	@black $(src_dirs)
	@echo "--> Running isort"
	@isort $(src_dirs)
