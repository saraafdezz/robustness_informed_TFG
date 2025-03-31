import itertools
import os
import subprocess
import time
import numpy as np
import hashlib
from datetime import datetime
import json

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
    print(f"[DEBUG] Tarea run_training en proceso... {model_type} - seed {seed}")
    output_files = []
    results_folder = os.path.join(RESULTS_FOLDER, model_type)
    if frac:
        results_folder = os.path.join(RESULTS_FOLDER, f"{model_type}-{frac}")

    output_1 = os.path.join(results_folder, f"metrics-seed-{int(seed):02d}.pkl")
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
    if ("random") in model_type:
        output_2 = os.path.join(results_folder,
                f"encodings_layer-01_seed-{int(seed):02d}.pkl"
            )
        output_3 = os.path.join(results_folder,
            f"encodings_layer-04_seed-{int(seed):02d}.pkl"
        )
        output_files = [output_1, output_2, output_3]

    elif ("reactome") in model_type:
        output_2 = os.path.join(results_folder,
                f"encodings_layer-01_seed-{int(seed):02d}.pkl"
            )
        output_3 = os.path.join(results_folder,
            f"encodings_layer-04_seed-{int(seed):02d}.pkl"
        )
        output_files = [output_1, output_2, output_3]

    elif ("kegg") in model_type:
        output_2 = os.path.join(results_folder,
                f"encodings_layer-01_seed-{int(seed):02d}.pkl"
            )
        output_3 = os.path.join(results_folder,
            f"encodings_layer-02_seed-{int(seed):02d}.pkl"
        )
        output_4 = os.path.join(results_folder,
            f"encodings_layer-05_seed-{int(seed):02d}.pkl"
        )
        output_files = [output_1, output_2, output_3, output_4]

    print(f"[DEBUG] Tarea run_training completada: {model_type} - seed {seed}")

    return output_files  # Devuelve la ruta del archivo generado



# Task for scoring
@task(cache_policy=TASK_SOURCE + (INPUTS - "gpu_id"), retries=3, retry_delay_seconds=2)
def score_training(model_type: str, seed_start: str, seed_step: str, seed_stop: str, frac: str = None, gpu_id=None):
    """Runs the training script for a given model type, seed, and optionally fraction.
    Which GPU to use is also passed as an argument."""
    # TODO: adapt to only CPUs scenarios

    print(f"Ejecutando scoring para el modelo {model_type}...")
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

    ShellOperation(
        commands=[
            " ".join(command),  # Join command and redirect.
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
        },  # Set CUDA_VISIBLE_DEVICES
    ).run()

    # Archivos de salida esperados.
    fname_informed = f"scores_informed.pkl"
    fname_clustering = f"scores_clustering.pkl"
    fname_metrics = f"scores_metrics.pkl"

    # Rutas finales de los archivos generados.
    if "random" in model_type:
        base_path = os.path.join(RESULTS_FOLDER, f"{model_type}-{frac}")
    else:
        base_path = os.path.join(RESULTS_FOLDER, model_type)

    print(f"Tarea scoring_training completada: {model_type}")

    output_files = [
        os.path.join(base_path, fname_informed),
        os.path.join(base_path, fname_clustering),
        os.path.join(base_path, fname_metrics),
    ]

    return output_files

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
    fname_informed = f"informed.tex"
    fname_clustering = f"clustering.tex"
    model_mse = f"model_mse.pdf"
    layer_scores = f"layer_scores.pdf"
    mse = f"mse.tex"
    output_files = [fname_informed, fname_clustering, model_mse,
                layer_scores, mse]
    return output_files



# --- Funciones auxiliares

def execute_if_file_missing(task, task_name, *args, output_files=None, **kwargs):
    """
    Ejecuta la tarea si los archivos de salida no existen.
    """
    if output_files is None:
        output_files = []

    # Verificar si los scripts han sido modificados
    script_path = TASK_SCRIPT_MAP.get(task_name)
    script_hashes = load_script_hashes()
    script_changed = False

    if script_path:
        current_hash = calculate_file_hash(script_path)
        previous_hash = script_hashes.get(script_path)

        if current_hash != previous_hash:
            print(f"[DEBUG] Script cambiado: {script_path}. Se ejecutará la tarea.")
            script_hashes[script_path] = current_hash
            script_changed = True
        else:
            print(f"[DEBUG] Script sin cambios: {script_path}")


    # Verificar si faltan archivos
    missing_files = [file for file in output_files if not os.path.exists(file)]
    
    # Si hay archivos faltantes o scripts modificados, ejecutamos la tarea
    if missing_files or script_changed:
        print(f"Archivos faltantes detectados: {missing_files}. Ejecutando la tarea...")
        
        # Llamamos a submit() para lanzar la tarea
        task_future = task.submit(*args, **kwargs)
        print(f"Estado de la tarea después de enviar: {task_future.state}")
        task_future.result()  # Esto puede dar más información sobre la ejecución

        # Guardar nuevos hashes si se ejecutó la tarea
        save_script_hashes(script_hashes)

        # Asegurarnos de que la tarea se ha lanzado correctamente
        if task_future:
            print(f"Tarea enviada para el modelo {args[0]}")
        return task_future
    else:
        print(f"Todos los archivos existen y el script no cambió: {output_files}. No se ejecuta la tarea.")
        return None

# --- For checking if a script changes ---

HASH_FILE = "script_hashes.json"

# Calculate hash
def calculate_file_hash(filepath):
    """Devuelve un hash del contenido del archivo para detectar cambios."""
    if not os.path.exists(filepath):
        return None
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

# Load hash
def load_script_hashes():
    """Carga los hashes previos desde un archivo JSON."""
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}

# Save hash
def save_script_hashes(hashes):
    """Guarda los hashes actualizados en un archivo JSON."""
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=4)

# Mapping tasks and scripts
TASK_SCRIPT_MAP = {
    "run_training": "notebooks/00-train.py",
    "score_training": "notebooks/01-scoring.py",
    "analyze_results": "notebooks/02-analyze_results.py",
}


# --- Flows ---


@flow(
    name="IVAE Training Workflow", task_runner=ThreadPoolTaskRunner(max_workers=N_DEVICES)
)  # Give the flow a name
def main_flow(results_folder: str = RESULTS_FOLDER):
    """Main workflow to install dependencies and run training for different models."""

    install_ivae(results_folder=results_folder)

    models = [f"ivae_random-{frac}" for frac in FRACS] + ["ivae_kegg", "ivae_reactome"]
    print("[DEBUG] Modelos generados:", models) 

    folders = []
    for model in models:
        if "random" in model:
            model_, frac = model.split("-")
        else:
            frac = None
            model_ = model
        create_folders.submit(model_, frac)
    wait(folders)


    # Training
    tasks_to_run = []
    for index, (model, seed) in enumerate(itertools.product(models, SEEDS)):
        if "random" in model:
            model_, frac = model.split("-")
        else:
            frac = None
            model_ = model

        # Definir la ruta esperada del archivo de salida
        output_files = []
        results_folder = os.path.join(RESULTS_FOLDER, model_)
        if frac:
            results_folder = os.path.join(RESULTS_FOLDER, f"{model_}-{frac}")
        output_1 = os.path.join(results_folder, f"metrics-seed-{int(seed):02d}.pkl")
        if ("random") in model_:
            output_2 = os.path.join(results_folder,
                    f"encodings_layer-01_seed-{int(seed):02d}.pkl"
                )
            output_3 = os.path.join(results_folder,
                f"encodings_layer-04_seed-{int(seed):02d}.pkl"
            )
            output_files = [output_1, output_2, output_3]

        elif ("reactome") in model_:
            output_2 = os.path.join(results_folder,
                    f"encodings_layer-01_seed-{int(seed):02d}.pkl"
                )
            output_3 = os.path.join(results_folder,
                f"encodings_layer-04_seed-{int(seed):02d}.pkl"
            )
            output_files = [output_1, output_2, output_3]

        elif ("kegg") in model_:
            output_2 = os.path.join(results_folder,
                    f"encodings_layer-01_seed-{int(seed):02d}.pkl"
                )
            output_3 = os.path.join(results_folder,
                f"encodings_layer-02_seed-{int(seed):02d}.pkl"
            )
            output_4 = os.path.join(results_folder,
                f"encodings_layer-05_seed-{int(seed):02d}.pkl"
            )
            output_files = [output_1, output_2, output_3, output_4]


        # Solo ejecuta la tarea si falta el archivo
        print(f"[DEBUG] Comprobando archivos esperados para {model_} con frac={frac}: {output_files}")
        task_future = execute_if_file_missing(
            run_training, "run_training", model_, seed, frac, gpu_id=index % N_GPU, output_files=output_files
        )

        if task_future:  # Solo añadimos tareas que se ejecutan
            tasks_to_run.append(task_future)



    # Scoring
    for index, model in enumerate(models):
        if "random" in model:
            model_, frac = model.split("-")
            base_path = os.path.join(RESULTS_FOLDER, f"{model_}-{frac}")
        else:
            frac = None
            model_ = model
            base_path = os.path.join(RESULTS_FOLDER, model_)

        # Definir los archivos de salida esperados
        fname_informed = f"scores_informed.pkl"
        fname_clustering = f"scores_clustering.pkl"
        fname_metrics = f"scores_metrics.pkl"
        output_files =  [
            os.path.join(base_path, fname_informed),
            os.path.join(base_path, fname_clustering),
            os.path.join(base_path, fname_metrics),
        ]

        # Solo ejecuta la tarea si faltan los archivos de scoring
        task_future = execute_if_file_missing(
            score_training, "score_training", model_, SEED_START, SEED_STEP, SEED_STOP, frac, gpu_id=index % N_GPU, output_files=output_files
        )

        if task_future:  # Solo añadimos tareas que se ejecutan
            tasks_to_run.append(task_future)


    # Analyze results
    results_folder = os.path.join(RESULTS_FOLDER)
    fname_informed = f"informed.tex"
    fname_clustering = f"clustering.tex"
    model_mse = f"model_mse.pdf"
    layer_scores = f"layer_scores.pdf"
    mse = f"mse.tex"
    output_files = [fname_informed, fname_clustering, model_mse,
                    layer_scores, mse]
    # Solo ejecuta la tarea si faltan los archivos de scoring
    task_future = execute_if_file_missing(
        analyze_results, "analyze_results", FRAC_START, FRAC_STEP, FRAC_STOP, gpu_id=index % N_GPU, output_files=output_files
    )
    if task_future:  # Solo añadimos tareas que se ejecutan
        tasks_to_run.append(task_future)

    # Esperamos que todas las tareas de scoring se completen
    wait(tasks_to_run)
    
    # Give time to shutdown connections
    time.sleep(2) 



if __name__ == "__main__":
    main_flow()
