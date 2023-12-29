#!make
include .env
.PHONY: install format run-kegg
.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate

all: install format run-kegg run-reactome run-analyze
install:
	mamba env update --prune -p ${ENV_FOLDER} -f environment.yml
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	pip install git+https://github.com/babelomics/ivae_scorer@develop
	pip install -e .
format:
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports isrobust
	nbqa autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports notebooks
	nbqa isort --profile black isrobust notebooks
	black isrobust notebooks
run-kegg: install format
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	mkdir -p results/logs/
	papermill notebooks/00-compute_scores.ipynb -p debug False -p model_kind ivae_kegg  > results/logs/ivae_kegg.out 2> results/logs/ivae_kegg.err
run-reactome: install format
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	mkdir -p results/logs/
	papermill notebooks/00-compute_scores.ipynb -p debug False -p model_kind ivae_reactome  > results/logs/ivae_reactome.out 2> results/logs/ivae_reactome.err
run-analyze: run-kegg run-reactome
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	papermill 01-analyze_results.ipynb > results/logs/01-analyze_results.out 2> results/logs/01-analyze_results.err
