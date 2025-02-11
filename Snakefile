configfile: "config.yaml"

SEED_START = config.get("seed_start", 0)
SEED_STEP = config.get("seed_step", 1)
SEED_STOP = config.get("seed_stop", 50)


rule all:
    input:
	    "path/done_scoring.txt"		

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
	    "path/ivae_kegg/done_{seed}.txt"
	params:
	    seed = lambda wildcards: wildcards.seed
	shell:
	    """
	    pixi run --environment cuda python notebooks/00-train-copy.py --model_kind ivae_kegg --seed_stop {params.seed}
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
	    pixi run --environment cuda python notebooks/00-train-copy.py --model_kind ivae_reactome --seed_stop {params.seed}
	    echo "Training completed for seed {params.seed} and model ivae_reactome" > {output}
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

rule combine_models:
    input:
        "path/ivae_kegg/scoring_done.txt",
        "path/ivae_reactome/scoring_done.txt",
    output:
        "path/done_scoring.txt"
    shell:
        """
        echo "Model training and scoring completed" > {output}
        """
