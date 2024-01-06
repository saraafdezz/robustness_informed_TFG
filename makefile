#!make
include .env
.PHONY: install-ivae format run-kegg run-reactome run-random run-analyze
.ONESHELL:

SHELL := /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
FRACS=$$(seq ${FRAC_START} ${FRAC_STEP} ${FRAC_STOP})
SEEDS=$$(seq ${SEED_START} ${SEED_STEP} ${SEED_STOP})

all: install-ivae format run-kegg run-reactome run-random run-analyze
install-ivae:
	rm -rf ${IVAE_ENV_FOLDER}
	mamba env create -p ${IVAE_ENV_FOLDER} -f environment-ivae.yml
	$(CONDA_ACTIVATE) ${IVAE_ENV_FOLDER}
	pip install git+https://github.com/babelomics/ivae_scorer@develop
	pip install -e .
# install-binn:
# 	rm -rf ${BINN_ENV_FOLDER}
# 	mamba env create -p ${BINN_ENV_FOLDER} -f environment-binn.yml
# 	$(CONDA_ACTIVATE) ${BINN_ENV_FOLDER}
# 	pip install mygene binn==0.0.3 --extra-index-url https://download.pytorch.org/whl/cu118
# 	pip install -e .
format:
	$(CONDA_ACTIVATE) ${IVAE_ENV_FOLDER}
	autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports isrobust
	autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports notebooks
	nbqa autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports notebooks
	nbqa isort --profile black isrobust notebooks
	isort --profile black isrobust notebooks
	black isrobust notebooks
run-kegg: 
	$(CONDA_ACTIVATE) ${IVAE_ENV_FOLDER}
	rm -rf results/ivae_kegg
	mkdir -p results/ivae_kegg/logs/
	parallel -j${N_GPU} CUDA_VISIBLE_DEVICES='$$(({%} - 1))' \
		python notebooks/00-train.py ivae_kegg ${DEBUG} {} \
		">" results/ivae_kegg/logs/train_seed-{}.out \
		"2>" results/ivae_kegg/logs/train_seed-{}.err \
		::: $(SEEDS)
run-reactome: install-ivae format
	$(CONDA_ACTIVATE) ${IVAE_ENV_FOLDER}
	rm -rf results/ivae_reactome
	mkdir -p results/ivae_reactome/logs
	parallel -j${N_GPU} CUDA_VISIBLE_DEVICES='$$(({%} - 1))' \
		python notebooks/00-train.py ivae_reactome ${DEBUG} {} \
		">" results/ivae_reactome/logs/train_seed-{}.out \
		"2>" results/ivae_reactome/logs/train_seed-{}.err \
		::: $(SEEDS)
run-random: install-ivae format
	$(CONDA_ACTIVATE) ${IVAE_ENV_FOLDER}
	rm -rf $$(printf "results/ivae_random-%s " $(FRACS))
	mkdir -p $$(printf "results/ivae_random-%s/logs " $(FRACS))

	parallel -j${N_GPU} CUDA_VISIBLE_DEVICES='$$(({%} - 1))' \
		python notebooks/00-train.py ivae_random-{2} ${DEBUG} {2} {1} \
		">" results/ivae_random-{2}/logs/train_seed-{1}.out \
		"2>" results/ivae_random-{2}/logs/train_seed-{1}.err \
		::: $(SEEDS) ::: $(FRACS)
run-scoring: run-kegg run-reactome run-random
	$(CONDA_ACTIVATE) ${IVAE_ENV_FOLDER}
	papermill notebooks/01-compute_scores.ipynb -p model_kind ivae_scorer -p debug False \
		> results/ivae_kegg/logs/scoring.out \
		2> results/ivae_kegg/logs/scoring.err
	papermill notebooks/01-compute_scores.ipynb -p model_kind ivae_reactome -p debug False \
		> results/ivae_reactome/logs/scoring.out \
		2> results/ivae_reactome/logs/scoring.err
