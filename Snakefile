import os

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

# Variables de entorno
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "results")
DEBUG = os.getenv("DEBUG", "")
N_GPU = int(os.getenv("N_GPU", "1"))
N_CPU = int(os.getenv("N_CPU", "1"))
FRAC_START = int(os.getenv("FRAC_START", "1"))
FRAC_STEP = int(os.getenv("FRAC_STEP", "1"))
FRAC_STOP = int(os.getenv("FRAC_STOP", "5"))
SEED_START = int(os.getenv("SEED_START", "1"))
SEED_STEP = int(os.getenv("SEED_STEP", "1"))
SEED_STOP = int(os.getenv("SEED_STOP", "5"))

# Definir rangos
FRACS = range(FRAC_START, FRAC_STOP + 1, FRAC_STEP)
SEEDS = range(SEED_START, SEED_STOP + 1, SEED_STEP)

# Regla principal
rule all:
    input:
        expand(f"{RESULTS_FOLDER}/ivae_kegg/logs/train_seed-{seed}.out", seed=SEEDS),
        expand(f"{RESULTS_FOLDER}/ivae_reactome/logs/train_seed-{seed}.out", seed=SEEDS),
        expand(f"{RESULTS_FOLDER}/ivae_random-{frac}/logs/train_seed-{seed}.out", seed=SEEDS, frac=FRACS),
        f"{RESULTS_FOLDER}/ivae_kegg/logs/scoring.out",
        f"{RESULTS_FOLDER}/ivae_reactome/logs/scoring.out",
        expand(f"{RESULTS_FOLDER}/ivae_random-{frac}/logs/scoring.out", frac=FRACS)

# Instalación
rule install_ivae:
    shell:
        "conda install"

# Entrenamiento para KEGG
rule run_kegg:
    input:
        rules.install_ivae
    output:
        expand(f"{RESULTS_FOLDER}/ivae_kegg/logs/train_seed-{seed}.out", seed=SEEDS)
    shell:
        """
        rm -rf {RESULTS_FOLDER}/ivae_kegg
        mkdir -p {RESULTS_FOLDER}/ivae_kegg/logs/
        parallel -j{N_GPU} CUDA_VISIBLE_DEVICES='{{%}} - 1' \
            conda python notebooks/00-train.py ivae_kegg {DEBUG} {{}} \
            ">" {RESULTS_FOLDER}/ivae_kegg/logs/train_seed-{{}}.out \
            "2>" {RESULTS_FOLDER}/ivae_kegg/logs/train_seed-{{}}.err \
            ::: {SEEDS}
        """

# Entrenamiento para Reactome
rule run_reactome:
    input:
        rules.install_ivae
    output:
        expand(f"{RESULTS_FOLDER}/ivae_reactome/logs/train_seed-{seed}.out", seed=SEEDS)
    shell:
        """
        rm -rf {RESULTS_FOLDER}/ivae_reactome
        mkdir -p {RESULTS_FOLDER}/ivae_reactome/logs/
        parallel -j{N_GPU} CUDA_VISIBLE_DEVICES='{{%}} - 1' \
            conda python notebooks/00-train.py ivae_reactome {DEBUG} {{}} \
            ">" {RESULTS_FOLDER}/ivae_reactome/logs/train_seed-{{}}.out \
            "2>" {RESULTS_FOLDER}/ivae_reactome/logs/train_seed-{{}}.err \
            ::: {SEEDS}
        """

# Entrenamiento para Random
rule run_random:
    input:
        rules.install_ivae
    output:
        expand(f"{RESULTS_FOLDER}/ivae_random-{frac}/logs/train_seed-{seed}.out", seed=SEEDS, frac=FRACS)
    shell:
        """
        rm -rf $(printf "{RESULTS_FOLDER}/ivae_random-%s " {FRACS})
        mkdir -p $(printf "{RESULTS_FOLDER}/ivae_random-%s/logs " {FRACS})
        parallel -j{N_GPU} CUDA_VISIBLE_DEVICES='{{%}} - 1' \
            conda python notebooks/00-train.py ivae_random-{{2}} {DEBUG} {{2}} {{1}} \
            ">" {RESULTS_FOLDER}/ivae_random-{{2}}/logs/train_seed-{{1}}.out \
            "2>" {RESULTS_FOLDER}/ivae_random-{{2}}/logs/train_seed-{{1}}.err \
            ::: {SEEDS} ::: {FRACS}
        """

# Scoring para KEGG
rule run_scoring_kegg:
    input:
        expand(f"{RESULTS_FOLDER}/ivae_kegg/logs/train_seed-{seed}.out", seed=SEEDS)
    output:
        f"{RESULTS_FOLDER}/ivae_kegg/logs/scoring.out"
    shell:
        """
        conda papermill notebooks/01-compute_scores.ipynb - \
            -p model_kind ivae_kegg \
            > {output} \
            2> {RESULTS_FOLDER}/ivae_kegg/logs/scoring.err
        """

# Scoring para Reactome
rule run_scoring_reactome:
    input:
        expand(f"{RESULTS_FOLDER}/ivae_reactome/logs/train_seed-{seed}.out", seed=SEEDS)
    output:
        f"{RESULTS_FOLDER}/ivae_reactome/logs/scoring.out"
    shell:
        """
        conda papermill notebooks/01-compute_scores.ipynb - \
            -p model_kind ivae_reactome \
            > {output} \
            2> {RESULTS_FOLDER}/ivae_reactome/logs/scoring.err
        """

# Scoring para Random
rule run_scoring_random:
    input:
        expand(f"{RESULTS_FOLDER}/ivae_random-{frac}/logs/train_seed-{seed}.out", seed=SEEDS, frac=FRACS)
    output:
        expand(f"{RESULTS_FOLDER}/ivae_random-{frac}/logs/scoring.out", frac=FRACS)
    shell:
        """
        parallel -j{N_GPU} \
            conda papermill notebooks/01-compute_scores.ipynb - \
            -p model_kind ivae_random-{{}} -p frac {{}} \
            ">" {RESULTS_FOLDER}/ivae_random-{{}}/logs/scoring.out \
            "2>" {RESULTS_FOLDER}/ivae_random-{{}}/logs/scoring.err \
            ::: {FRACS}
        """
