from dotenv import load_dotenv
import os
import numpy as np

# Cargar el archivo .env
load_dotenv()

# Leer las variables de entorno
IVAE_ENV_FOLDER = os.getenv("IVAE_ENV_FOLDER")
BINN_ENV_FOLDER = os.getenv("BINN_ENV_FOLDER")
N_GPU = int(os.getenv("N_GPU"))
N_CPU = int(os.getenv("N_CPU"))
SEED = int(os.getenv("SEED"))
SEED_START = int(os.getenv("SEED_START"))
SEED_STOP = int(os.getenv("SEED_STOP"))
SEED_STEP = int(os.getenv("SEED_STEP"))
DEBUG = int(os.getenv("DEBUG"))
FRAC_START = float(os.getenv("FRAC_START"))
FRAC_STOP = float(os.getenv("FRAC_STOP"))
FRAC_STEP = float(os.getenv("FRAC_STEP"))
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER")

# Reemplazar la variable PY_FILES por la lista de archivos Python
PY_FILES = ["isrobust_TFG/" + f for f in os.listdir("isrobust_TFG") if f.endswith(".py")]

# Imprimir las variables para verificar si hay algún valor incorrecto
print(f"FRAC_STEP: {FRAC_STEP}, SEED_STEP: {SEED_STEP}")

# Creamos la lista de fracciones y semillas
FRACS = np.arange(FRAC_START, FRAC_STOP + 1, FRAC_STEP)
SEEDS = range(SEED_START, SEED_STOP + 1, SEED_STEP)

# Reglas

# Regla para instalar el entorno ivae
rule install_ivae:
    input:
        "environment-ivae.yml"
    output:
        directory(IVAE_ENV_FOLDER)
    shell:
        "conda env create -p {IVAE_ENV_FOLDER} -f {input}"

# Regla para formatear el código
rule format:
    input:
        IVAE_ENV_FOLDER
    output:
        temp("format_done.txt")  # Usamos un archivo temporal para indicar que se ha terminado
    conda:
        "{IVAE_ENV_FOLDER}/envs/ivae_env.yml"  # Usamos un entorno conda definido
    shell:
        """
        source $(conda info --base)/etc/profile.d/conda.sh && conda activate {IVAE_ENV_FOLDER}
        autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports isrobust_TFG
        autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports notebooks
        nbqa autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports notebooks
        nbqa isort --profile black isrobust_TFG
        isort --profile black isrobust_TFG
        black isrobust_TFG
        black notebooks
        """

# Añadir un 'wait' entre las reglas como en el Makefile. Esto se maneja con el orden de ejecución en Snakemake.
rule all:
    input:
        "format_done.txt"  # La regla all depende de la ejecución de la regla 'format'
