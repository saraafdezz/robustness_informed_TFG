SEEDS = [0,1]

rule all:
	input:
        "path/ivae_kegg/done.txt"

rule train_model_kegg:
	input:
		"path/seed_0/install_done.txt" 
	output:
	    "path/ivae_kegg/done.txt"
	shell:
	    """
	    pixi run --environment cuda python notebooks/00-train-copy.py --model_kind ivae_kegg --seed 0
	    echo "Training completed for seed 0 and model ivae_kegg" > {output}
	    """