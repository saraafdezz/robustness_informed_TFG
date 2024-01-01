#!make
include .env
.PHONY: install format run-kegg
.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate

all: install format run-kegg run-reactome run-analyze
install:
	rm -rf ${ENV_FOLDER}
	mamba env create -p ${ENV_FOLDER} -f environment.yml
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	pip install git+https://github.com/babelomics/ivae_scorer@develop
	pip install -e .
format:
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports isrobust
	autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports notebooks
	nbqa autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports notebooks
	nbqa isort --profile black isrobust notebooks
	isort --profile black isrobust notebooks
	black isrobust notebooks
run-kegg: install format
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	rm -rf results/ivae_kegg
	mkdir -p results/ivae_kegg/logs/
	seq 0 99 | parallel -j${N_GPU} CUDA_VISIBLE_DEVICES='$$(({%} - 1))' \
		python notebooks/00-train.py ivae_kegg 0 {} \
		">" results/ivae_kegg/logs/train_seed-{}.out \
		"2>" results/ivae_kegg/logs/train_seed-{}.err
run-reactome: install format
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	rm -rf results/ivae_reactome
	mkdir -p results/ivae_reactome/logs
	seq 0 99 | nohup parallel -j${N_GPU} CUDA_VISIBLE_DEVICES='$$(({%} - 1))' \
		python notebooks/00-train.py ivae_reactome 0 {} \
		">" results/ivae_reactome/logs/train_seed-{}.out \
		"2>" results/ivae_reactome/logs/train_seed-{}.err &
run-scoring:
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	papermill notebooks/01-compute_scores.ipynb -p model_kind ivae_scorer -p debug False \
		> results/ivae_kegg/logs/scoring.out \
		2> results/ivae_kegg/logs/scoring.err
	papermill notebooks/01-compute_scores.ipynb -p model_kind ivae_reactome -p debug False \
		> results/ivae_reactome/logs/scoring.out \
		2> results/ivae_reactome/logs/scoring.err
