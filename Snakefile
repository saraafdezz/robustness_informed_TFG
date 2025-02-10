configfile: "config.yaml"

SEED_MAX = config.get("seed_max", 50)

rule all:
    input:
        expand("path/{model_kind}/done.txt", 
		model_kind = config["models"].values(), 
		seed = range(SEED_MAX + 1)),
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
        "notebooks/00-train-copy.py"
	"path/install_done.txt"
    output:
        "path/{params.model_kind}/done.txt"
    params:
        model_kind = "ivae_kegg",
        seed=lambda wildcards: wildcards.seed
    shell:
        """
        pixi run --environment cuda python {input} --model_kind {params.model_kind} --seed_stop {params.seed}
        echo "Training completed for seed {params.seed} and model {params.model_kind}" > {output}
        """

rule train_model_reactome:
	input:
	    "notebook/00-train-copy.py"
	    "path/install_done.txt"
	output:
	    "path/{params.model_kind}/done.txt"
	params:
	    model_kind = "ivae_reactome",
	    seed = lambda wildcards: wildcards.seed
	shell:
	    """
	    pixi run --environment cuda python {input} --model_kind {params.model_kind} --seed_stop {params.seed}
	    echo "Training completed for seed {params.seed} and model {params.model_kind}" > {output}
	    """

rule scoring_kegg:                                                                                                          
	input:                                                                                                                      
	    "path/ivae_kegg/done.txt"                                             
	output:                                                                                                                     
	    "path/ivae_kegg/scoring_done.txt"
	params:
	    model_kind = "ivae_kegg"
	    seed = lambda wildcards: wildcards.seed                                                                                   
	shell:                                                                                                                      
	    """                                                                                                                     
	    pixi run --environment cuda python notebooks/01-scoring.py --model_kind {params.model_kind} --seed_start 0 --seed_stop 2 --seed_step 1                         
	    echo "Scoring completed for ivae_kegg" > {output}                                                                       
	    """

rule scoring_reactome:
    input:
	"path/ivae_reactome/done.txt"
    output:
        "path/ivae_reactome/scoring_done.txt"
    params:
	model_kind = "ivae_reactome"
	seed = lambda wildcards: wildcards.seed
    shell:
        """
        pixi run --environment cuda python notebooks/01-scoring.py --model_kind {params.model_kind} --seed_start 0 --seed_step 1 --seed_stop 2
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
