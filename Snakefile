import numpy as np

configfile: "config.yaml"

SEED_START = config.get("seed_start", 0)
SEED_STEP = config.get("seed_step", 1)
SEED_STOP = config.get("seed_stop", 50)

FRAC_START = config.get("frac_start", 0.05)
FRAC_STEP = config.get("frac_step", 0.2)
FRAC_STOP = config.get("frac_stop", 0.85)


rule all:
    input:
        "path/done_combination.txt"

rule install_ivae:
	output:
		"path/install_done.txt"
	shell:
		"""
		pixi install -e cuda
		echo "Installation completed" > {output}
		"""

rule train_model_kegg:
	input:
	    "path/install_done.txt"
	output:
	    "path/ivae_kegg/done_{seed}.txt"
	params:
	    seed = lambda wildcards: wildcards.seed
	shell:
	    """
	    pixi run --environment cuda python notebooks/00-train.py --model_kind ivae_kegg --seed {params.seed}
	    echo "Training completed for seed {params.seed} and model ivae_kegg" > {output}
	    """

rule train_model_reactome:
	input:
	    "path/install_done.txt"
	output:
	    "path/ivae_reactome/done_{seed}.txt"
	params:
	    seed = lambda wildcards: wildcards.seed
	shell:
	    """
	    pixi run --environment cuda python notebooks/00-train.py --model_kind ivae_reactome --seed {params.seed}
	    echo "Training completed for seed {params.seed} and model ivae_reactome" > {output}
	    """

rule train_model_random:
	input:
	    "path/install_done.txt"
	output:
	    "path/ivae_random-{frac}/done_{seed}.txt"
	params:
	    seed = lambda wildcards: wildcards.seed,
		frac = lambda wildcards: wildcards.frac
	shell:
	    """
	    pixi run --environment cuda python notebooks/00-train.py --model_kind ivae_random --seed {params.seed} --frac {params.frac}
	    echo "Training completed for seed {params.seed} and model ivae_random" > {output}
	    """

rule scoring_kegg:                                                                                                          
	input:                                                                                                                      
	    expand("path/ivae_kegg/done_{seed}.txt", seed=range(SEED_START, SEED_STOP + 1, SEED_STEP))                                             
	output:                                                                                                                     
	    "path/ivae_kegg/scoring_done.txt"
	shell:                                                                                                                      
	    """                                                                                                                     
	    pixi run --environment cuda python notebooks/01-scoring.py --model_kind ivae_kegg --seed_start {SEED_START} --seed_step {SEED_STEP} --seed_stop {SEED_STOP}                         
	    echo "Scoring completed for ivae_kegg" > {output}                                                                       
	    """

rule scoring_reactome:
	input:
		expand("path/ivae_reactome/done_{seed}.txt", seed=range(SEED_START, SEED_STOP + 1, SEED_STEP))
	output:
		"path/ivae_reactome/scoring_done.txt"
	shell:
		"""
		pixi run --environment cuda python notebooks/01-scoring.py --model_kind ivae_reactome --seed_start {SEED_START} --seed_step {SEED_STEP} --seed_stop {SEED_STOP}
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
        pixi run --environment cuda python notebooks/01-scoring.py --model_kind ivae_random --seed_start {SEED_START} --seed_step {SEED_STEP} --seed_stop {SEED_STOP} --frac {params.frac}
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
		pixi run --environment cuda python notebooks/02-analyze_results.py --frac_start {FRAC_START} --frac_stop {FRAC_STOP} --frac_step {FRAC_STEP}
        echo "Model training and scoring completed" > {output}
        """
