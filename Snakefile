import numpy as np

configfile: "config.yaml"

SEED_START = config["seed_start"]
SEED_STEP = config["seed_step"]
SEED_STOP = config["seed_stop"]

FRAC_START = config["frac_start"]
FRAC_STEP = config["frac_step"]
FRAC_STOP = config["frac_stop"]


rule all:
    input:
        "path/done_combination.txt"

rule install_ivae:
	output:
		"path/install_done.txt"
	shell:
		"""
		pixi install
		echo "Installation completed" > {output}
		"""

rule train_model_kegg:
	input:
	    "path/install_done.txt"
	output:
	    "path/ivae_kegg/train_done.txt"
	shell:
	    """
		for seed in $(seq {SEED_START} {SEED_STEP} {SEED_STOP}); do
			pixi run --environment cuda python notebooks/00-train-copy.py --model_kind ivae_kegg --seed $seed
			echo "Training completed for seed $seed and model ivae_kegg" > {output}
		done
	    """

rule train_model_reactome:
	input:
	    "path/install_done.txt"
	output:
	    "path/ivae_reactome/train_done.txt"
	shell:
	    """
		for seed in $(seq {SEED_START} {SEED_STEP} {SEED_STOP}); do
			pixi run --environment cuda python notebooks/00-train-copy.py --model_kind ivae_reactome --seed $seed
			echo "Training completed for seed $seed and model ivae_reactome" > {output}
		done
	    """


rule train_model_random:
    input:
        "path/install_done.txt"
    output:
        "path/ivae_random-{frac}/train_done.txt"
    params:
        frac = lambda wildcards: wildcards.frac
    shell:
        """
        for seed in $(seq {SEED_START} {SEED_STEP} {SEED_STOP}); do
            pixi run --environment cuda python notebooks/00-train-copy.py --model_kind ivae_random --seed $seed --frac {params.frac}
            echo "Training completed for seed $seed and model ivae_random" > {output}
        done
        """



rule scoring_kegg:                                                                                                          
	input:                                                                                                                      
	    "path/ivae_kegg/train_done.txt"                                            
	output:                                                                                                                     
	    "path/ivae_kegg/scoring_done.txt"
	shell:                                                                                                                      
	    """                                                                                                                     
	    pixi run --environment cuda python notebooks/01-scoring.py --model_kind ivae_kegg --seed_start {SEED_START} --seed_step {SEED_STEP} --seed_stop {SEED_STOP}                         
	    echo "Scoring completed for ivae_kegg" > {output}                                                                       
	    """

rule scoring_reactome:
	input:
		"path/ivae_reactome/train_done.txt"
	output:
		"path/ivae_reactome/scoring_done.txt"
	shell:
		"""
		pixi run --environment cuda python notebooks/01-scoring.py --model_kind ivae_reactome --seed_start {SEED_START} --seed_step {SEED_STEP} --seed_stop {SEED_STOP}
		echo "Scoring completed for ivae_reactome" > {output}
		"""

rule scoring_random:
	input:
		"path/ivae_random-{frac}/train_done.txt"
	output:
		"path/ivae_random-{frac}/scoring_done.txt"
	params:
		frac = lambda wildcards: wildcards.frac
	shell:
		"""
		pixi run --environment cuda python notebooks/01-scoring.py --model_kind ivae_random --seed_start {SEED_START} --seed_step {SEED_STEP} --seed_stop {SEED_STOP} --frac {params.frac}
		echo "Scoring completed for ivae_reactome" > {output}
		"""


rule combine_models:
    input:
        expand("path/ivae_random-{frac}/scoring_done.txt", frac = np.round(np.arange(FRAC_START, FRAC_STOP, FRAC_STEP), 2)),
        "path/ivae_kegg/scoring_done.txt",
        "path/ivae_reactome/scoring_done.txt"
    output:
        "path/done_combination.txt"
    shell:
        """
		pixi run --environment cuda python notebooks/02-analyze_results.py --frac_start {FRAC_START} --frac_stop {FRAC_STOP} --frac_step {FRAC_STEP}
        echo "Model training and scoring completed" > {output}
        """