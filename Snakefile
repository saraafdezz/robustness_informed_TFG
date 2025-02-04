configfile: "config.yaml"

SEED_MAX = config.get("seed_max", 50)

rule all:
    input:
        expand("path/{model_kind}/seed_{seed}/done.txt", 
               model_kind=config["models"].values(), 
               seed=range(SEED_MAX + 1))

rule train_model:
    input:
        "notebooks/00-train-copy.py"
    output:
        "path/{model_kind}/seed_{seed}/done.txt"
    params:
        model_kind=lambda wildcards: wildcards.model_kind,
        seed=lambda wildcards: wildcards.seed
    shell:
        """
        SEED={params.seed} pixi run --environment cuda python {input} --model_kind {params.model_kind}
        echo "Training completed for seed {params.seed} and model {params.model_kind}" > {output}
        """

rule combine_models:
    input:
        expand("path/{model_kind}/seed_{seed}/done.txt", 
               model_kind=config["models"].values(), 
               seed=range(SEED_MAX + 1))
    output:
        "path/done.txt"
    shell:
        "echo 'Model training completed for: {input}' > {output}"

