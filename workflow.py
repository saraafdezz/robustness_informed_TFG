import itertools
import os
import time
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from dotenv import find_dotenv, load_dotenv
from keras import callbacks
from keras.models import Model
from prefect import flow, get_run_logger, task, unmapped
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import wait
from prefect_ray import RayTaskRunner
from prefect_ray.context import remote_options
from prefect_shell import ShellOperation
from scipy.stats import weightedtau
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import minmax_scale
import ray

from ivae.bio import (
    InformedModelConfig,
    IvaeResults,
    ModelFamilyResults,
    build_model_config,
    get_activations,
    get_importances,
    train_val_test_split,
)
from ivae.datasets import load_kang
from ivae.models import (
    InformedVAE,
)


def sort_categories_by_pattern(category_list: list[str], pattern: str) -> list[str]:
    """
    Sorts a list of category strings, placing items matching the pattern last.

    The function separates the list into two groups: those that contain the
    pattern (case-insensitive, treated as a whole word) and those that don't.
    It then sorts each group alphabetically and concatenates them, with the
    non-matching group coming first.

    Args:
        category_list: The list of category strings to sort.
        pattern: The string pattern to search for within category names. Items
                 containing this pattern (as a whole word, case-insensitive)
                 will be placed at the end of the sorted list. Handles regex
                 special characters in the pattern safely.

    Returns:
        A new list with categories sorted according to the rule.
                 Returns an empty list if the input list is empty.
    """
    import re

    if not category_list:
        return []

    # Use re.escape to handle potential special characters in the pattern safely
    # Use \b for whole word matching (e.g., matches 'random' but not 'randomized')
    # Use re.IGNORECASE for case-insensitivity
    try:
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        print(f"Warning: Invalid pattern provided '{pattern}'. Error: {e}. Pattern matching disabled for this call.")
         # Fallback: sort alphabetically if pattern is invalid regex
        sorted_list = sorted(category_list)
        return sorted_list


    matches = []
    non_matches = []

    for cat in category_list:
        # Check if cat is None or not a string, handle gracefully
        if not isinstance(cat, str):
             print(f"Warning: Item '{cat}' in category_list is not a string. Skipping.")
             continue # Skip non-string items

        if compiled_pattern.search(cat):
            matches.append(cat)
        else:
            non_matches.append(cat)

    # Sort each group alphabetically
    non_matches.sort()
    matches.sort()

    # Combine the lists: non-matches first, then matches
    return non_matches + matches


@task(cache_policy=TASK_SOURCE + INPUTS)
def install_ivae(results_folder: str = RESULTS_FOLDER):
    """Installs dependencies using pixi."""
    os.makedirs(f"{results_folder}/logs", exist_ok=True)
    ShellOperation(commands=["pixi install -a"], working_dir=".").run()

    return


@task(cache_policy=TASK_SOURCE + INPUTS)
def create_folders(model_type: str, frac: str = None):
    """Creates folders for a given model type, seed, and optionally fraction."""
    results_folder = os.path.join(RESULTS_FOLDER, model_type)
    if frac:
        results_folder = os.path.join(RESULTS_FOLDER, f"{model_type}-{frac}")

    ShellOperation(commands=[f"mkdir -p {results_folder}/logs"]).run()

    return


@task(cache_policy=TASK_SOURCE + INPUTS)
def download_data(data_path: str = DATA_PATH) -> sc.AnnData:
    """Downloads data from a given path."""

    path = load_kang(data_folder=data_path, normalize=True, n_genes=None, return_path=True)
    return path

@task(cache_policy=TASK_SOURCE + INPUTS)
def get_genes(data_path: str) -> list:
    data = sc.read(data_path, cache=True)
    return data.to_df().columns.to_list()

@task(cache_policy=TASK_SOURCE + INPUTS)
def build_ivae_config(model_kind, genes) -> InformedModelConfig:
    if "random" in model_kind:
        model, frac = model_kind.split("-")
        frac = float(frac)
    else:
        frac = None
        model = model_kind

    model_config = build_model_config(
        genes,
        model_kind=model,
        frac=frac,
    )
    return model_config


def split_data(
    seed: int,
    path: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits data into train, validation, and test sets."""

    data = sc.read(path, cache=True)

    obs = data.obs.copy()  # Ignorar
    x_trans = data.to_df()  # Ignorar

    # Separa en train, val y test los datos de x_trans
    x_train, x_val, x_test = train_val_test_split(
        x_trans.apply(minmax_scale),  # Para que los datos esten en un rango similar
        val_size=0.20,
        test_size=0.20,
        stratify=obs["cell_type"].astype(str) + obs["condition"].astype(str),
        seed=seed,
    )

    obs_train = obs.loc[x_train.index, :]
    obs_test = obs.loc[x_test.index, :]
    obs_val = obs.loc[x_val.index, :]

    return x_train, x_val, x_test, obs_train, obs_val, obs_test


def gen_fit_key(context, parameters) -> str:
    """
    Generates a cache key incorporating fields from model_config and the rest of the signature.
    """
    model_config = parameters["model_config"]
    seed = parameters["seed"]

    model_kind = model_config.model_kind
    if "random" in model_kind:
        model_kind = f"{model_kind}_{model_config.frac}"

    seed_part = f"seed={seed}"

    return f"model_fit_{model_kind}_{seed_part}"


@task(cache_key_fn=gen_fit_key, cache_policy=TASK_SOURCE + INPUTS)
def fit_model(model_config, seed, path) -> IvaeResults:
    """Fits the model to the data."""

    split = split_data(seed, path)

    N_EPOCHS = 3 if DEBUG else 100

    x_train, x_val, x_test, obs_train, obs_val, obs_test = split
    x_train = x_train.loc[:, model_config.input_genes]
    x_val = x_val.loc[:, model_config.input_genes]
    x_test = x_test.loc[:, model_config.input_genes]

    # Build and train the VAE
    model = InformedVAE(
        adjacency_matrices=model_config.model_layer,
        adjacency_names=model_config.adj_name,
        adjacency_activation=model_config.adj_activ,
        seed=0,
    )

    model._build_vae()
    batch_size = 32

    callback = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-1,
        patience=100,
        verbose=0,
    )

    history = model.fit(
        x_train,
        x_train,
        shuffle=True,
        verbose=1,
        epochs=N_EPOCHS,
        batch_size=batch_size,
        callbacks=[callback],
        validation_data=(x_val, x_val),
    )

    evals = evaluate_model(model, model_config, split, seed)
    encodings = predict_encodings(model, model_config, split, seed)

    results = IvaeResults(
        config=model_config,
        history=history.history,
        eval=evals,
        encodings=encodings,
    )

    return results


def evaluate_model(model, model_config, split, seed):
    """Evaluates the model."""
    x_train, x_val, x_test, obs_train, obs_val, obs_test = split
    x_train = x_train.loc[:, model_config.input_genes]
    x_val = x_val.loc[:, model_config.input_genes]
    x_test = x_test.loc[:, model_config.input_genes]

    evaluation = {}
    evaluation["train"] = model.evaluate(
        x_train, model.predict(x_train), verbose=0, return_dict=True
    )
    evaluation["val"] = model.evaluate(
        x_val, model.predict(x_val), verbose=0, return_dict=True
    )
    evaluation["test"] = model.evaluate(
        x_test, model.predict(x_test), verbose=0, return_dict=True
    )

    eval_results = (
        pd.DataFrame.from_dict(evaluation)
        .reset_index(names="metric")
        .assign(seed=seed)
        .melt(
            id_vars=["seed", "metric"],
            value_vars=["train", "val", "test"],
            var_name="split",
            value_name="score",
        )
        .assign(model=model_config.model_kind)
    )

    return eval_results


def predict_encodings(model, model_config, split, seed):
    """Gets the activations of the model."""

    x_train, x_val, x_test, obs_train, obs_val, obs_test = split
    obs = pd.concat((obs_train, obs_val, obs_test), axis=0, ignore_index=False)
    x_train = x_train.loc[:, model_config.input_genes]
    x_val = x_val.loc[:, model_config.input_genes]
    x_test = x_test.loc[:, model_config.input_genes]
    x_trans = pd.concat([x_train, x_val, x_test], axis=0)

    layer_outputs = [layer.output for layer in model.encoder.layers]
    activation_model = Model(inputs=model.encoder.input, outputs=layer_outputs)

    encodings = []

    for layer_id in range(1, len(layer_outputs)):
        if layer_id == (len(layer_outputs) - 1):
            n_latents = len(model_config.layer_entity_names[-1]) // 2
            colnames = [f"latent_{i}" for i in range(n_latents)]
            layer = "funnel"
        elif "kegg" in model_config.model_kind:
            if layer_id in [1, 2]:
                colnames = model_config.layer_entity_names[layer_id - 1]
                layer = model_config.adj_name[layer_id - 1]
            else:
                continue
        elif (
            "reactome" in model_config.model_kind or "random" in model_config.model_kind
        ):
            if layer_id in [1]:
                colnames = model_config.layer_entity_names[layer_id - 1]
                layer = model_config.adj_name[layer_id - 1]
            else:
                continue
        else:
            colnames = model_config.layer_entity_names[layer_id - 1]
            layer = model_config.adj_name[layer_id - 1]

        encodings_i = get_activations(
            act_model=activation_model,
            layer_id=layer_id,
            data=x_trans,
        )

        encodings_i = pd.DataFrame(encodings_i, index=x_trans.index, columns=colnames)

        encodings_i["split"] = "train"
        encodings_i.loc[x_val.index, "split"] = "val"
        encodings_i.loc[x_test.index, "split"] = "test"
        encodings_i["layer"] = layer
        encodings_i["seed"] = seed
        encodings_i["model"] = model_config.model_kind

        encodings_i = encodings_i.merge(
            obs[["cell_type", "condition"]],
            how="left",
            left_index=True,
            right_index=True,
        )

        encodings.append(encodings_i)

    return encodings


@task(cache_policy=TASK_SOURCE + INPUTS)
def evalute_clustering(results, seed) -> pd.DataFrame:
    logger = get_run_logger()  # Moved inside
    logger.info(f"Evaluating clustering for (seed={seed})...")

    non_layer_names = ["split", "layer", "seed", "cell_type", "condition", "model"]
    clust_scores = {}

    batch_size = 256 * cpu_count() + 1

    model_kind = results.config.model_kind

    for results_layer in results.encodings:
        layer = results_layer["layer"].iloc[0]

        train_embeddings = results_layer.loc[
            (results_layer["split"] == "train")
            & (results_layer["condition"] == "control")
        ]
        val_embeddings = results_layer.loc[
            (results_layer["split"] == "val")
            & (results_layer["condition"] == "control")
        ]
        test_embeddings = results_layer.loc[
            (results_layer["split"] == "test")
            & (results_layer["condition"] == "control")
        ]

        y_train = train_embeddings["cell_type"]
        y_val = val_embeddings["cell_type"]
        y_test = test_embeddings["cell_type"]

        train_embeddings = train_embeddings.drop(columns=non_layer_names)
        val_embeddings = val_embeddings.drop(columns=non_layer_names)
        test_embeddings = test_embeddings.drop(columns=non_layer_names)

        clust_scores[layer] = {}
        clust_scores[layer]["train"] = []
        clust_scores[layer]["val"] = []
        clust_scores[layer]["test"] = []

        model = MiniBatchKMeans(n_clusters=y_train.nunique(), batch_size=batch_size)
        model.fit(train_embeddings)
        clust_scores[layer]["train"].append(
            adjusted_mutual_info_score(y_train, model.labels_)
        )
        clust_scores[layer]["val"].append(
            adjusted_mutual_info_score(y_val, model.predict(val_embeddings))
        )
        clust_scores[layer]["test"].append(
            adjusted_mutual_info_score(y_test, model.predict(test_embeddings))
        )

    clust_scores = (
        pd.DataFrame.from_dict(clust_scores)
        .melt(var_name="layer", value_name="score", ignore_index=False)
        .reset_index(names=["split"])
        .explode("score")
    )
    clust_scores["score"] = clust_scores["score"].astype("float")
    clust_scores["model"] = model_kind

    return clust_scores


@task(cache_policy=TASK_SOURCE + INPUTS)
def gather_results(model_config, results_lst):
    """Gathers all result futures for each model_config."""

    model_kind = model_config.model_kind
    family_results = []
    for result in results_lst:
        if model_kind == result.config.model_kind:
            family_results.append(result)
    results_by_model = ModelFamilyResults(model_kind, family_results)

    return results_by_model


@task(cache_policy=TASK_SOURCE + INPUTS)
def compute_consistedness(family_results):
    """Compute consistency of the results."""
    # Implement the logic to compute consistency
    non_layer_names = ["split", "layer", "seed", "cell_type", "condition", "model"]
    scores = {}
    model_kind = family_results.results[0].config.model_kind
    model_config = family_results.results[0].config
    model_kind = model_config.model_kind

    n_results = len(family_results)
    n_encoding_layers = len(family_results.results[0].encodings)
    for i in range(n_encoding_layers):
        layer_id = family_results.results[0].encodings[i]["layer"].iloc[0]
        scores[layer_id] = {}

        for split in ["train", "test", "val"]:
            # encodings_i = [x.loc[x["split"] == split].drop(non_layer_names, axis=1) for r in results for x in r.encodings[i]]
            scores[layer_id][split] = []
            for j in range(n_results):
                r_j = family_results.results[j].encodings[i]
                encodings_j = r_j.loc[r_j["split"] == split].drop(
                    non_layer_names, axis=1
                )
                importances_j = get_importances(data=encodings_j, abs=True)
                for k in range(j + 1, n_results):
                    r_k = family_results.results[k].encodings[i]
                    encodings_k = r_k.loc[r_k["split"] == split].drop(
                        non_layer_names, axis=1
                    )
                    importances_k = get_importances(data=encodings_k, abs=True)
                    scores[layer_id][split].append(
                        weightedtau(importances_j, importances_k)[0]
                    )

    scores_df = (
        pd.DataFrame.from_dict(scores)
        .melt(var_name="layer", value_name="score", ignore_index=False)
        .reset_index(names=["split"])
        .explode("score")
    )
    scores_df["score"] = scores_df["score"].astype("float")
    scores_df["model"] = model_kind

    return scores_df


@task(cache_policy=TASK_SOURCE + INPUTS)
def build_eval_df(results_by_model):
    """Builds a dataframe with the evaluation metrics."""
    df = pd.concat(
        [i.eval.drop(columns=["seed"]) for r in results_by_model for i in r.results],
        axis=0,
        ignore_index=True,
    )

    return df


@task(cache_policy=TASK_SOURCE + INPUTS)
def save_eval(df, output_path):
    import seaborn as sns

    fname = "eval_mse"

    df.to_csv(Path(output_path).joinpath("eval.tsv"), index=False, sep="\t")

    (
        df.query("metric=='mse'")
        .groupby(["model", "metric", "split"])["score"]
        .describe()
        .drop(["count", "min", "max"], axis=1)
        .to_latex(
            Path(output_path).joinpath(f"{fname}_summary.tex"),
            bold_rows=True,
            escape=True,
        )
    )

    metric_scores_to_plot = df.copy().query("split=='test'").query("metric=='mse'")
    metric_scores_to_plot["metric"] = r"$-\mathrm{log}(\mathrm{MSE})$"
    metric_scores_to_plot["score"] = -np.log(metric_scores_to_plot["score"])
    metric_scores_to_plot = metric_scores_to_plot.rename(
        columns={"score": "Score", "model": "Model"}
    )

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context="paper", font_scale=2, style="ticks", rc=custom_params)
    fac = 0.7

    model_list = metric_scores_to_plot["Model"].to_list()
    y_order = sort_categories_by_pattern(model_list, "random")

    g = sns.catplot(
        data=metric_scores_to_plot,
        kind="violin",
        col="metric",
        height=9 * fac,
        aspect=16 / 9 * fac,
        sharey=True,
        sharex=False,
        y="Model",
        x="Score",
        split=False,
        cut=0,
        fill=False,
        density_norm="count",
        inner="quart",
        linewidth=2,
        legend_out=False,
        col_wrap=4,
        order=y_order
    )

    g.savefig(
        Path(output_path).joinpath(f"{fname}_violin_test.pdf"), bbox_inches="tight"
    )
    g.savefig(
        Path(output_path).joinpath(f"{fname}_violin_test.png"),
        dpi=300,
        bbox_inches="tight",
    )


@task(cache_policy=TASK_SOURCE + INPUTS)
def build_clustering_df(clustering_results):
    """Builds a dataframe with the evaluation metrics."""
    df = pd.concat(clustering_results, axis=0, ignore_index=True)
    df["metric"] = "AdjustedMutualInfo"

    return df


@task(cache_policy=TASK_SOURCE + INPUTS)
def save_clustering(df, output_path):
    import seaborn as sns

    fname = "clustering"

    df.to_csv(Path(output_path).joinpath("clustering.tsv"), index=False, sep="\t")

    (
        df.groupby(["model", "layer", "metric", "split"])["score"]
        .describe()
        .drop(["count", "min", "max"], axis=1)
        .to_latex(
            Path(output_path).joinpath(f"{fname}_summary.tex"),
            bold_rows=True,
            escape=True,
        )
    )

    metric_scores_to_plot = df.copy().query("split=='test'")
    metric_scores_to_plot = metric_scores_to_plot.rename(
        columns={"score": "Score", "model": "Model Layer", "metric": "Metric"}
    )
    metric_scores_to_plot["Model Layer"] = (
        metric_scores_to_plot["Model Layer"] + " " + metric_scores_to_plot["layer"]
    )
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context="paper", font_scale=2, style="ticks", rc=custom_params)
    fac = 0.7

    model_list = metric_scores_to_plot["Model Layer"].to_list()
    y_order = sort_categories_by_pattern(model_list, "random")

    g = sns.catplot(
        data=metric_scores_to_plot,
        kind="violin",
        col="Metric",
        height=9 * fac,
        aspect=16 / 9 * fac,
        sharey=True,
        sharex=False,
        y="Model Layer",
        x="Score",
        split=False,
        cut=0,
        fill=False,
        density_norm="count",
        inner="quart",
        linewidth=2,
        legend_out=False,
        col_wrap=4,
        order=y_order
    )

    g.savefig(
        Path(output_path).joinpath(f"{fname}_violin_test.pdf"), bbox_inches="tight"
    )
    g.savefig(
        Path(output_path).joinpath(f"{fname}_violin_test.png"),
        dpi=300,
        bbox_inches="tight",
    )


@task(cache_policy=TASK_SOURCE + INPUTS)
def build_consistedness_df(consistedness_results):
    """Builds a dataframe with the evaluation metrics."""
    df = pd.concat(consistedness_results, axis=0, ignore_index=True)
    df["metric"] = r"$w_{\tau}$"

    return df


@task(cache_policy=TASK_SOURCE + INPUTS)
def save_consistedness(df, output_path):
    import seaborn as sns

    fname = "consistedness"

    df.to_csv(Path(output_path).joinpath(f"{fname}.tsv"), index=False, sep="\t")

    (
        df.groupby(["model", "layer", "metric", "split"])["score"]
        .describe()
        .drop(["count", "min", "max"], axis=1)
        .to_latex(
            Path(output_path).joinpath(f"{fname}_summary.tex"),
            bold_rows=True,
            escape=True,
        )
    )

    metric_scores_to_plot = df.copy().query("split=='test'")
    metric_scores_to_plot = metric_scores_to_plot.rename(
        columns={"score": "Score", "model": "Model Layer", "metric": "Metric"}
    )
    metric_scores_to_plot["Model Layer"] = (
        metric_scores_to_plot["Model Layer"] + " " + metric_scores_to_plot["layer"]
    )
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context="paper", font_scale=2, style="ticks", rc=custom_params)
    fac = 0.7

    model_list = metric_scores_to_plot["Model Layer"].to_list()
    y_order = sort_categories_by_pattern(model_list, "random")

    g = sns.catplot(
        data=metric_scores_to_plot,
        kind="violin",
        col="Metric",
        height=9 * fac,
        aspect=16 / 9 * fac,
        sharey=True,
        sharex=False,
        y="Model Layer",
        x="Score",
        split=False,
        cut=0,
        fill=False,
        density_norm="count",
        inner="quart",
        linewidth=2,
        legend_out=False,
        col_wrap=4,
        order=y_order
    )

    g.savefig(
        Path(output_path).joinpath(f"{fname}_violin_test.pdf"), bbox_inches="tight"
    )
    g.savefig(
        Path(output_path).joinpath(f"{fname}_violin_test.png"),
        dpi=300,
        bbox_inches="tight",
    )


@task(cache_policy=TASK_SOURCE + INPUTS)
def save_combined(consistedness_df, clustering_df, output_path):
    import seaborn as sns

    fname = "combined_scores"
    scores = (
        pd.concat((consistedness_df, clustering_df), axis=0, ignore_index=True)
        .query("split=='test'")
        .query("layer != 'funnel'")
        .drop(["split"], axis=1)
        .rename(columns={"kind": "metric"})
    )

    scores.head()

    scores_to_plot = scores.copy()

    scores_to_plot["Model"] = scores_to_plot["model"] + " - " + scores_to_plot["layer"]

    scores_to_plot = scores_to_plot.rename(
        columns={"score": "Score", "metric": "Metric"}
    )

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context="paper", font_scale=2, style="ticks", rc=custom_params)
    fac = 0.7

    scores_to_plot["Model Layer"] = scores_to_plot["Model"]
    model_list = scores_to_plot["Model Layer"].to_list()
    y_order = sort_categories_by_pattern(model_list, "random")

    g = sns.catplot(
        data=scores_to_plot,
        kind="violin",
        col="Metric",
        height=9 * fac,
        aspect=16 / 9 * fac,
        sharey=True,
        sharex=False,
        y="Model Layer",
        x="Score",
        split=False,
        cut=0,
        fill=False,
        density_norm="count",
        inner="quart",
        linewidth=2,
        legend_out=False,
        col_wrap=4,
        order=y_order
    )

    g.savefig(
        Path(output_path).joinpath(f"{fname}_violin_test.pdf"), bbox_inches="tight"
    )
    g.savefig(
        Path(output_path).joinpath(f"{fname}_violin_test.png"),
        dpi=300,
        bbox_inches="tight",
    )


@flow(
    name="IVAE Training Workflow",
    task_runner=RayTaskRunner(init_kwargs={"log_to_driver":False, "logging_config":ray.LoggingConfig(encoding="JSON", log_level="INFO")}),
)
def main(results_folder: str = RESULTS_FOLDER, models=MODELS, seeds=SEEDS):
    """Main workflow to install dependencies and run training for different models."""

    install_ivae(results_folder=results_folder)

    folders = []
    for model in models:
        if "random" in model:
            model_, frac = model.split("-")
        else:
            frac = None
            model_ = model
        create_folders.submit(model_, frac)
    wait(folders)

    # Download data
    data_path = download_data.submit(data_path=DATA_PATH).result()
    genes = get_genes.submit(data_path).result()
    ivae_config_futures = build_ivae_config.map(models, unmapped(genes))

    all_combinations = list(itertools.product(ivae_config_futures, seeds))
    all_models = [x[0] for x in all_combinations]
    all_seeds = [x[1] for x in all_combinations]

    with remote_options(num_cpus=1, num_gpus=1):
        result_futures = fit_model.map(all_models, all_seeds, unmapped(data_path))

    results = result_futures.result()

    with remote_options(num_cpus=N_CPUS_CLUSTERING, num_gpus=0):
        clustering_metrics_futures = evalute_clustering.map(results, all_seeds)

    ivae_config_lst = ivae_config_futures.result()

    with remote_options(num_cpus=1, num_gpus=0):
        results_by_model_futures = gather_results.map(
            ivae_config_lst, unmapped(results)
        )

    with remote_options(num_cpus=1, num_gpus=0):
        consistedness_futures = compute_consistedness.map(results_by_model_futures)

    # Wait for all tasks to finish
    results_by_model = results_by_model_futures.result()
    clustering_metrics = clustering_metrics_futures.result()
    consistedness = consistedness_futures.result()

    eval_df = build_eval_df.submit(results_by_model)
    save_eval.submit(eval_df, results_folder).result()

    clustering_df = build_clustering_df(clustering_metrics)
    save_clustering.submit(clustering_df, results_folder)

    consistedness_df = build_consistedness_df(consistedness)
    save_consistedness.submit(consistedness_df, results_folder)

    save_combined.submit(consistedness_df, clustering_df, results_folder)

    time.sleep(2)


if __name__ == "__main__":
    main()
