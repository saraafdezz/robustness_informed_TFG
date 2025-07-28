# Robustness-Informed

This repository contains the code for a project that investigates the use of pathway-informed Variational Autoencoders (VAEs) for analyzing single-cell RNA-sequencing data. The project compares the performance of VAEs informed by biological pathways (KEGG and Reactome) against VAEs with random connections and the PathSingle method.

The project goes along the paper "A Framework for Evaluating the Stability of Learned Representations in Biologically-Constrained Models in Single-Cell". To support fair, consistent, and reproducible experimentation, a modular and automated pipeline was developed using Pixi dependency manager, Prefect workflow manager and Ray computational distributed framework.

## Project Overview

The project is structured as a Prefect workflow that automates the following steps:

1.  **Data Loading and Preprocessing:** The workflow uses the Kang et al. (2018) dataset of peripheral blood mononuclear cells (PBMCs). The data is normalized and preprocessed for training.
2.  **Model Training:** The workflow trains several VAE models with different configurations:
    *   `ivae_kegg`: A VAE informed by the KEGG pathway database.
    *   `ivae_reactome`: A VAE informed by the Reactome pathway database.
    *   `ivae_random`: A VAE with randomly connected layers, used as a baseline.
3.  **Scoring and Evaluation:** The models are evaluated based on two main criteria:
    *   **Clustering performance:** The ability of the model's latent space to separate different cell types, measured by the Adjusted Mutual Information (AMI) score.
    *   **Model consistency:** The stability of the model's learned feature importances across different random initializations, measured by the weighted tau correlation.
4.  **Comparison with PathSingle:** The project also runs the PathSingle method on the same data and compares its performance against the VAE models.
5.  **Result Analysis and Visualization:** The workflow generates plots and tables to compare the performance of the different models.

## Installation

The project uses `pixi` for managing dependencies. To install the required packages, run the following command:

```bash
pixi install
```

## Workflow Execution

The main workflow is defined in `workflow.py` and can be executed using the following command:

```bash
python workflow.py [OPTIONS]
```

### Arguments

*   `--debug`: Run in debug mode (fewer models/epochs).
*   `--results_folder`: Path to the main folder where IVAE results will be saved.
*   `--results_folder_ps`: Path to the main folder where PathSingle results will be saved.
*   `--data_path`: Path to the folder containing or to download input data.
*   `--n_seeds`: Number of repeated holdout procedures.
*   `--frac_start`, `--frac_step`, `--frac_stop`: Parameters for the density of random layers.
*   `--n_gpus`: Number of GPUs used for training.
*   `--n_cpus`: Maximum number of CPUs used for non-GPU tasks.

## Project Structure

*   `workflow.py`: The main Prefect workflow definition.
*   `compare_models.py`: A script to compare the results of the different models.
*   `ivae/`: A Python package containing the implementation of the VAE models and related utilities.
*   `pathsingle/`: A Python package containing the implementation of the PathSingle method.
*   `notebooks/`: A collection of Jupyter notebooks for exploratory analysis and visualization.
*   `pixi.toml`: The `pixi` configuration file for managing dependencies.
*   `prefect.toml`: The Prefect configuration file.
*   `data/`: The directory where the input data is stored.
*   `results/`: The directory where the results of the experiments are saved.