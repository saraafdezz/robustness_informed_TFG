import numpy as np
import os
import subprocess

configfile: "config.yaml"

SEED_START = config.get("seed_start", 0)
SEED_STEP = config.get("seed_step", 1)
SEED_STOP = config.get("seed_stop", 50)

FRAC_START = config.get("frac_start", 0.05)
FRAC_STEP = config.get("frac_step", 0.2)
FRAC_STOP = config.get("frac_stop", 0.85)

import os
import subprocess

# Función para obtener las GPUs disponibles
def get_available_gpus():
    gpu_status = subprocess.check_output("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", shell=True).decode().strip().split("\n")
    available_gpus = [line.split(",")[0] for line in gpu_status if int(line.split(",")[1]) > 1000]  # Ajusta el umbral según lo necesario
    return available_gpus

# Obtener GPUs disponibles en cada momento
available_gpus = get_available_gpus()


rule all:
    input:
        "path/done_combination.txt"

rule install_ivae:
	output:
		"path/install_done.txt"
	shell:
		"""
		pixi install -e ivaecuda
		echo "Installation completed" > {output}
		"""

rule train_model_kegg:
    input:
        "path/install_done.txt"
    output:
        "path/ivae_kegg/done_{seed}.txt"
    resources:
        gpu=1
    params:
        seed = lambda wildcards: wildcards.seed,
        gpu_id = lambda wildcards: available_gpus[(wildcards.seed % len(available_gpus))]  # Asegúrate que sea un valor válido
    shell:
        """
		echo "Using GPU: {params.gpu_id}"  # Para depurar y ver qué GPU se usa
        export CUDA_VISIBLE_DEVICES={params.gpu_id} && pixi run --environment ivaecuda python notebooks/00-train.py --model_kind ivae_kegg --seed {params.seed}
        echo "Training completed for seed {params.seed} and model ivae_kegg" > {output}
        """

rule train_model_reactome:
    input:
        "path/install_done.txt"
    output:
        "path/ivae_reactome/done_{seed}.txt"
    resources:
        gpu=1
    params:
        seed = lambda wildcards: wildcards.seed,
        gpu_id = lambda wildcards: available_gpus[(wildcards.seed % len(available_gpus))]  # Asegúrate que sea un valor válido
    shell:
        """
		echo "Using GPU: {params.gpu_id}"  # Para depurar y ver qué GPU se usa
        export CUDA_VISIBLE_DEVICES={params.gpu_id} && pixi run --environment ivaecuda python notebooks/00-train.py --model_kind ivae_reactome --seed {params.seed}
        echo "Training completed for seed {params.seed} and model ivae_reactome" > {output}
        """

rule train_model_random:
    input:
        "path/install_done.txt"
    output:
        "path/ivae_random-{frac}/done_{seed}.txt"
    resources:
        gpu=1
    params:
        seed = lambda wildcards: wildcards.seed,
        frac = lambda wildcards: wildcards.frac,
        gpu_id = lambda wildcards: available_gpus[(wildcards.seed % len(available_gpus))]  # Asegúrate que sea un valor válido
    shell:
        """
        echo "Using GPU: {params.gpu_id}"  # Para depurar y ver qué GPU se usa
        export CUDA_VISIBLE_DEVICES={params.gpu_id} && pixi run --environment ivaecuda python notebooks/00-train.py --model_kind ivae_random --seed {params.seed} --frac {params.frac}
        echo "Training completed for seed {params.seed} and model ivae_random" > {output}
        """


rule scoring_kegg:                                                                                                          
	input:                                                                                                                      
	    expand("path/ivae_kegg/done_{seed}.txt", seed=range(SEED_START, SEED_STOP + 1, SEED_STEP))                                             
	output:                                                                                                                     
	    "path/ivae_kegg/scoring_done.txt"
	shell:                                                                                                                      
	    """                                                                                                                     
	    pixi run --environment ivaecuda python notebooks/01-scoring.py --model_kind ivae_kegg --seed_start {SEED_START} --seed_step {SEED_STEP} --seed_stop {SEED_STOP}                         
	    echo "Scoring completed for ivae_kegg" > {output}                                                                       
	    """

rule scoring_reactome:
	input:
		expand("path/ivae_reactome/done_{seed}.txt", seed=range(SEED_START, SEED_STOP + 1, SEED_STEP))
	output:
		"path/ivae_reactome/scoring_done.txt"
	shell:
		"""
		pixi run --environment ivaecuda python notebooks/01-scoring.py --model_kind ivae_reactome --seed_start {SEED_START} --seed_step {SEED_STEP} --seed_stop {SEED_STOP}
		echo "Scoring completed for ivae_reactome" > {output}
		"""

rule scoring_random:
    input:
        lambda wildcards: expand("path/ivae_random-{frac}/done_{seed}.txt",
                                 seed=range(SEED_START, SEED_STOP + 1, SEED_STEP),
                                 frac=[wildcards.frac])
    output:
        "path/ivae_random-{frac}/scoring_done.txt"
    params:
        frac = lambda wildcards: wildcards.frac
    shell:
        """
        pixi run --environment ivaecuda python notebooks/01-scoring.py --model_kind ivae_random --seed_start {SEED_START} --seed_step {SEED_STEP} --seed_stop {SEED_STOP} --frac {params.frac}
        echo "Scoring completed for ivae_random" > {output}
        """


rule combine_models:
    input:
        expand("path/ivae_random-{frac}/scoring_done.txt", frac = np.arange(FRAC_START, FRAC_STOP, FRAC_STEP)),
        "path/ivae_kegg/scoring_done.txt",
        "path/ivae_reactome/scoring_done.txt"
    output:
        "path/done_combination.txt"
    shell:
        """
		pixi run --environment ivaecuda python notebooks/02-analyze_results.py --frac_start {FRAC_START} --frac_stop {FRAC_STOP} --frac_step {FRAC_STEP}
        echo "Model training and scoring completed" > {output}
        """
