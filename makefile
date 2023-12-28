#!make
include .env
.PHONY: install format run-kegg
.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate

all: install format run-kegg
install:
	mamba env update --prune -p ${ENV_FOLDER} -f environment.yml
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	pip install git+https://github.com/babelomics/ivae_scorer@develop
	pip install -e .
format:
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports isrobust
	isort --profile black isrobust notebooks
	black isrobust notebooks
run-kegg: install format
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	papermill notebooks/kegg.ipynb -p model_kind kegg  > results/kegg.out 2> results/kegg.err
