import itertools
import os
import subprocess
import time
import numpy as np
import hashlib
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import wait
from prefect.task_runners import ThreadPoolTaskRunner
from prefect_shell import ShellOperation


def check_cli_arg_is_bool(arg):
    """Check if argument is a boolean.

    Parameters
    ----------
    arg : str
        Argument.

    Returns
    -------
    bool
        Argument.
    """
    print("*" * 20, arg)
    if arg in ["true", "True", "TRUE", "1"]:
        arg = True
    elif arg in ["false", "False", "FALSE", "0"]:
        arg = False
    else:
        raise ValueError(f"argument {arg} must be a boolean")

    return arg


# --- Load Environment Variables ---
print("*"*20, find_dotenv())
load_dotenv()

FRAC_START = os.getenv("FRAC_START", "0.1")
FRAC_STEP = os.getenv("FRAC_STEP", "0.1")
FRAC_STOP = os.getenv("FRAC_STOP", "1.0")
SEED_START = os.getenv("SEED_START", "1")
SEED_STEP = os.getenv("SEED_STEP", "1")
SEED_STOP = os.getenv("SEED_STOP", "3")
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "results")
N_GPU = int(os.getenv("N_GPU", "3"))
N_CPU = int(os.getenv("N_CPU", "4"))
DEBUG = check_cli_arg_is_bool(os.getenv("DEBUG", "1"))
DATA_PATH = os.getenv("DATA_PATH", "data/data_pathsingle")  # add data path

N_DEVICES = N_GPU if N_GPU > 0 else (N_CPU -1)

print("*" * 20, N_CPU, os.getenv("DEBUG"), DEBUG)
print("*" * 20, os.getenv("PREFECT_RESULTS_PERSIST_BY_DEFAULT"))


# As in the makefile
fracsAux = np.arange(float(FRAC_START), float(FRAC_STOP), float(FRAC_STEP))
FRACS = [str(round(x, 10)) for x in fracsAux]

seedsAux = range(int(SEED_START), int(SEED_STOP) + 1, int(SEED_STEP))
SEEDS = [str(x) for x in seedsAux]

# --- For launching only necessary tasks ---

def file_hash(filepath):
    """Devuelve un hash del contenido del archivo para detectar cambios."""
    if not os.path.exists(filepath):
        return None
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


# --- Tasks ---


@task(cache_policy=TASK_SOURCE + INPUTS)
def install_ivae(results_folder: str = RESULTS_FOLDER):
    """Installs dependencies using pixi."""
    os.makedirs(f"{results_folder}/logs", exist_ok=True)
    ShellOperation(commands=["pixi install -e ivaecuda"], working_dir=".").run()

    return


@task(cache_policy=TASK_SOURCE + INPUTS)
def create_folders(model_type: str, frac: str = None):
    """Creates folders for a given model type, seed, and optionally fraction."""
    results_folder = os.path.join(RESULTS_FOLDER, model_type)
    if frac:
        results_folder = os.path.join(RESULTS_FOLDER, f"{model_type}-{frac}")

    ShellOperation(commands=[f"mkdir -p {results_folder}/logs"]).run()

    return



@task(cache_policy=TASK_SOURCE + (INPUTS - "gpu_id"), retries=3, retry_delay_seconds=2)
def run_training(model_type: str, seed: str, frac: str = None, gpu_id=None):
    """Ejecuta el entrenamiento y genera un archivo de salida."""
    results_folder = os.path.join(RESULTS_FOLDER, model_type)
    if frac:
        results_folder = os.path.join(RESULTS_FOLDER, f"{model_type}-{frac}")

    output_file = os.path.join(results_folder, f"metrics-seed-{int(seed):02d}.pkl")
    command = [
        "pixi",
        "run",
        "--environment",
        "ivaecuda",
        "python",
        "notebooks/00-train.py",
        "--model_kind", model_type,
        "--seed", str(seed),
        "--results_path_model", results_folder,
        "--data_path",
        DATA_PATH,
    ]

    if DEBUG:
        command.extend(["--debug", str(DEBUG)])
    if frac:
        command.extend(["--frac", str(frac)])

    ShellOperation(
        commands=[" ".join(command)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
    ).run()

    print(f"Tarea completada: {model_type} - seed {seed}")

    return output_file  # Devuelve la ruta del archivo generado



# Task for scoring
@task(
    cache_policy=TASK_SOURCE + (INPUTS - "gpu_id"), retries=3, retry_delay_seconds=2
) 
def score_training(model_type: str, seed_start: str, seed_step: str, seed_stop: str, frac: str = None, gpu_id=None):
    """Runs the training script for a given model type, seed, and optionally fraction.
    Which GPU to use is also passed as an argument."""
    # TODO: adapt to only CPUs scenarios

    results_folder = os.path.join(RESULTS_FOLDER)

    command = [
        "pixi",
        "run",
        "--environment",
        "ivaecuda",
        "python",
        "notebooks/01-scoring.py",
        "--model_kind",
        model_type,
        "--seed_start",
        str(seed_start),
        "--seed_step",
        str(seed_step),
        "--seed_stop",
        str(seed_stop),
        "--results_path",
        results_folder,
        "--data_path",
        DATA_PATH,
    ]

    print("*" * 20, DEBUG)

    if DEBUG:
        command.extend(["--debug", str(DEBUG)])
    if frac:
        command.extend(["--frac", str(frac)])

    log_file_out = os.path.join(results_folder, "logs", f"scoring-model-{model_type}.out")
    log_file_err = os.path.join(results_folder, "logs", f"scoring-model-{model_type}.err")

    ShellOperation(
        commands=[
            " ".join(command),  # Join command and redirect.
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
        },  # Set CUDA_VISIBLE_DEVICES
    ).run()

    # create outputs
    fname_informed = f"scores_informed.pkl"
    if "random" in model_type:
        return os.path.join(RESULTS_FOLDER, f"{model_type}-{frac}", fname_informed)
    else:
        return os.path.join(RESULTS_FOLDER, model_type, fname_informed)

    fname_clustering = f"scores_clustering.pkl"
    if "random" in model_type:
        return os.path.join(RESULTS_FOLDER, f"{model_type}-{frac}", fname_clustering)
    else:
        return os.path.join(RESULTS_FOLDER, model_type, fname_clustering)

    fname_metrics = f"scores_metrics.pkl"
    if "random" in model_type:
        return os.path.join(RESULTS_FOLDER, f"{model_type}-{frac}", fname_metrics)
    else:
        return os.path.join(RESULTS_FOLDER, model_type, fname_metrics)


# Task for analyze

@task(
    cache_policy=TASK_SOURCE + (INPUTS - "gpu_id"), retries=3, retry_delay_seconds=2
)
def analyze_results(frac_start: str, frac_step: str, frac_stop: str, gpu_id=None):
    """Runs the training script for a given model type, seed, and optionally fraction.
    Which GPU to use is also passed as an argument."""
    # TODO: adapt to only CPUs scenarios

    results_folder = os.path.join(RESULTS_FOLDER)

    command = [
        "pixi",
        "run",
        "--environment",
        "ivaecuda",
        "python",
        "notebooks/02-analyze_results.py",
        "--frac_start",
        str(frac_start),
        "--frac_step",
        str(frac_step),
        "--frac_stop",
        str(frac_stop),
        "--results_path",
        results_folder,
        "--data_path",
        DATA_PATH,
    ]

    print("*" * 20, DEBUG)

    log_file_out = os.path.join(results_folder, "logs", f"analyze-results.out")
    log_file_err = os.path.join(results_folder, "logs", f"analyze-results.err")

    ShellOperation(
        commands=[
            " ".join(command),  # Join command and redirect.
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
        },  # Set CUDA_VISIBLE_DEVICES
    ).run()

    # create outputs
    fname_informed = f"informed.txt"
    return os.path.join(RESULTS_FOLDER, fname_informed)

    fname_clustering = f"clustering.txt"
    return os.path.join(RESULTS_FOLDER, fname_clustering)



# --- Funciones auxiliares

def execute_if_file_missing(task, *args, output_files=None, **kwargs):
    """
    Ejecuta la tarea si los archivos de salida no existen.
    """
    if output_files is None:
        output_files = []

    # Verificar si faltan archivos
    missing_files = [file for file in output_files if not os.path.exists(file)]
    
    if missing_files:
        print(f"Archivos faltantes detectados: {missing_files}. Ejecutando la tarea...")
        task_future = task.submit(*args, **kwargs)  # Esto debería crear la tarea y ejecutarla
        return task_future
    else:
        print("Todos los archivos existen. No se ejecuta la tarea.")
        return None




# --- Flows ---


@flow(
    name="IVAE Training Workflow", task_runner=ThreadPoolTaskRunner(max_workers=N_DEVICES)
)  # Give the flow a name
def main_flow(results_folder: str = RESULTS_FOLDER):
    """Main workflow to install dependencies and run training for different models."""

    install_ivae(results_folder=results_folder)

    models = [f"ivae_random-{frac}" for frac in FRACS] + ["ivae_kegg", "ivae_reactome"]

    folders = []
    for model in models:
        if "random" in model:
            model_, frac = model.split("-")
        else:
            frac = None
            model_ = model
        create_folders.submit(model_, frac)
    wait(folders)

    seeds_run = []
    for index, (model, seed) in enumerate(itertools.product(models, SEEDS)):
        if "random" in model:
            model_, frac = model.split("-")
        else:
            frac = None
            model_ = model

        # Definir la ruta esperada del archivo de salida
        results_folder = os.path.join(RESULTS_FOLDER, model_)
        if frac:
            results_folder = os.path.join(RESULTS_FOLDER, f"{model_}-{frac}")
        output_file = os.path.join(results_folder, f"metrics-seed-{int(seed):02d}.pkl")

        # Solo ejecuta la tarea si falta el archivo
        task_future = execute_if_file_missing(
            run_training, model_, seed, frac, gpu_id=index % N_GPU, output_files=[output_file]
        )

        if task_future:  # Asegurar que solo añadimos tareas que realmente se ejecutaron
            seeds_run.append(task_future)

    wait(seeds_run)  # Espera a todas las tareas enviadas


    # Scoring
    models_scoring = []
    for index, model in enumerate(models):
        if "random" in model:
            model_, frac = model.split("-")
        else:
            frac = None
            model_ = model
        models_scoring.append(score_training.submit(model_, SEED_START, SEED_STEP, SEED_STOP, frac, gpu_id=index % N_GPU))
    wait(models_scoring)

    # Analyze results
    analyze_results.submit(FRAC_START, FRAC_STEP, FRAC_STOP, gpu_id=index % N_GPU)
    
    # Give time to shutdown connections
    time.sleep(2) 



if __name__ == "__main__":
    main_flow()
