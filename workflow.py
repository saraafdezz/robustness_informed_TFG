import argparse
import gc
import os
import time
from multiprocessing import cpu_count
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import ray
import scanpy as sc
import scanpy.external as sce
from keras import callbacks
from keras.models import Model
from prefect import flow, get_run_logger, task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.futures import wait
from prefect_ray import RayTaskRunner
from prefect_ray.context import remote_options
from prefect_shell import ShellOperation
from scipy.stats import weightedtau
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import minmax_scale

from ivae.bio import (
    InformedModelConfig,
    IvaeResults,
    ModelFamilyResults,
    build_model_config,
    get_activations,
    get_importances,
    get_reactome_adj,
    train_val_test_split,
)
from ivae.datasets import load_kang
from ivae.models import (
    InformedVAE,
)
from pathsingle.activity import calc_activity

N_CPUS = os.cpu_count()


# IVAE
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
        print(
            f"Warning: Invalid pattern provided '{pattern}'. Error: {e}. Pattern matching disabled for this call."
        )
        # Fallback: sort alphabetically if pattern is invalid regex
        sorted_list = sorted(category_list)
        return sorted_list

    matches = []
    non_matches = []

    for cat in category_list:
        # Check if cat is None or not a string, handle gracefully
        if not isinstance(cat, str):
            print(f"Warning: Item '{cat}' in category_list is not a string. Skipping.")
            continue  # Skip non-string items

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
def install_ivae():
    """Installs dependencies using pixi."""
    ShellOperation(commands=["pixi install -a"], working_dir=".").run()
    return


@task(cache_policy=TASK_SOURCE + INPUTS)
def create_folders(base_results_folder: str, model_type: str, frac: str = None):
    """Creates folders for a given model type, seed, and optionally fraction."""
    results_folder = os.path.join(base_results_folder, model_type)  # Use passed arg
    if frac:
        results_folder = os.path.join(
            base_results_folder, f"{model_type}-{frac}"
        )  # Use passed arg

    # Use os.makedirs instead of ShellOperation for simplicity and reliability
    os.makedirs(os.path.join(results_folder, "logs"), exist_ok=True)
    # ShellOperation(commands=[f"mkdir -p {results_folder}/logs"]).run()


@task(cache_policy=TASK_SOURCE + INPUTS)
def download_data(data_path: str) -> str:
    """Downloads data from a given path."""

    path = load_kang(
        data_folder=data_path, normalize=True, n_genes=None, return_path=True
    )
    return path


@task(cache_policy=TASK_SOURCE + INPUTS)
def get_genes(data_path: str) -> list:
    data = sc.read(data_path, cache=True)
    return data.var_names.to_list()


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
    seed: int, path: str, model_config: InformedModelConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits data into train, validation, and test sets."""

    data = sc.read(path, cache=True)

    obs = data.obs.copy()  # Ignorar
    x_trans = data.to_df().loc[:, model_config.input_genes]  # Ignorar

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

    print(obs_val["condition"].value_counts())
    print(obs_test["condition"].value_counts())

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
def fit_model_ivae(
    model_config: InformedModelConfig, seed: int, path: str, debug: bool
) -> IvaeResults:
    """Fits the model to the data."""

    split = split_data(seed, path, model_config)

    N_EPOCHS = 10 if debug else 1000

    x_train, x_val, x_test, obs_train, obs_val, obs_test = split

    # Build and train the VAE
    model = InformedVAE(
        adjacency_matrices=model_config.model_layer,
        adjacency_names=model_config.adj_name,
        adjacency_activation=model_config.adj_activ,
        seed=seed,
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
        history={},
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


@task(cache_policy=TASK_SOURCE + INPUTS, cache_result_in_memory=False)
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
def gather_evals(results_lst):
    """Gathers all result futures for each model_config."""

    df = pd.concat([result.eval for result in results_lst], axis=0, ignore_index=True)

    return df


@task(cache_policy=TASK_SOURCE + INPUTS)
def compute_consistedness(results_list: list[IvaeResults]) -> pd.DataFrame:
    """
    Compute consistency (pairwise weighted tau of importances) across multiple runs (seeds)
    for a single model configuration.

    Args:
        results_list: A list of IvaeResults objects, typically one per seed for the same model.
                      Prefect resolves the futures passed via .submit() to provide this list.

    Returns:
        A pandas DataFrame with consistency scores per layer and data split.
    """
    logger = get_run_logger()
    logger.info(f"Computing consistency for {len(results_list)} results.")

    if not results_list:
        logger.warning(
            "Received empty list for consistency computation. Returning empty DataFrame."
        )
        return pd.DataFrame(columns=["layer", "split", "score", "model"])

    # --- Configuration and Setup ---
    try:
        # Assume all results in the list share the same config
        model_config = results_list[0].config
        model_kind = model_config.model_kind
        # Check if the first result actually has encodings
        if not results_list[0].encodings:
            logger.warning(
                f"First result for model '{model_kind}' has no encodings. Cannot compute consistency."
            )
            return pd.DataFrame(columns=["layer", "split", "score", "model"])
        n_encoding_layers = len(results_list[0].encodings)
    except (AttributeError, IndexError) as e:
        logger.error(
            f"Could not extract config or encodings from first result: {e}. Returning empty DataFrame."
        )
        return pd.DataFrame(columns=["layer", "split", "score", "model"])

    non_layer_names = ["split", "layer", "seed", "cell_type", "condition", "model"]
    scores = {}
    n_results = len(results_list)
    logger.info(
        f"Model: {model_kind}, Seeds: {n_results}, Encoding Layers: {n_encoding_layers}"
    )

    # --- Calculate Pairwise Consistency ---
    for i in range(n_encoding_layers):
        # Get layer ID from the first result, assuming structure is consistent
        try:
            # Check if encodings[i] exists and is not empty before accessing iloc[0]
            if (
                i >= len(results_list[0].encodings)
                or results_list[0].encodings[i].empty
            ):
                logger.warning(
                    f"Skipping layer index {i} for model '{model_kind}' as it's missing or empty in the first result."
                )
                continue
            layer_id = results_list[0].encodings[i]["layer"].iloc[0]
            scores[layer_id] = {"train": [], "test": [], "val": []}
            logger.debug(f"Processing layer: {layer_id}")
        except (IndexError, KeyError, AttributeError) as e:
            logger.warning(
                f"Could not get layer_id for layer index {i} from first result: {e}. Skipping layer."
            )
            continue

        for split in ["train", "test", "val"]:
            # Pre-calculate importances for all valid results for this layer/split
            all_importances = {}  # Dictionary {result_index: importances}
            for res_idx in range(n_results):
                try:
                    # Check if encodings exist for this result and layer index i
                    if (
                        i < len(results_list[res_idx].encodings)
                        and not results_list[res_idx].encodings[i].empty
                    ):
                        r_res = results_list[res_idx].encodings[i]
                        encodings_split = r_res.loc[r_res["split"] == split].drop(
                            non_layer_names, axis=1, errors="ignore"
                        )

                        if not encodings_split.empty:
                            # Calculate importances (ensure get_importances is robust)
                            imp = get_importances(data=encodings_split, abs=True)
                            if (
                                imp is not None
                            ):  # Check if get_importances returned valid data
                                all_importances[res_idx] = imp
                            else:
                                logger.warning(
                                    f"get_importances returned None for result {res_idx}, layer '{layer_id}', split '{split}'."
                                )
                        # else: logger.debug(f"No data for result {res_idx}, layer '{layer_id}', split '{split}' after filtering.")
                    # else: logger.debug(f"Encodings missing/empty at index {i} for result {res_idx}.")
                except Exception as e:
                    logger.warning(
                        f"Error processing importances for result {res_idx}, layer '{layer_id}', split '{split}': {e}"
                    )

            # Compute pairwise weighted tau using pre-calculated importances
            valid_indices = sorted(all_importances.keys())
            num_valid = len(valid_indices)
            logger.debug(
                f"  Split '{split}': Found valid importances for {num_valid}/{n_results} results."
            )

            for j_outer_idx in range(num_valid):
                idx_j = valid_indices[j_outer_idx]
                importances_j = all_importances[idx_j]

                for k_outer_idx in range(j_outer_idx + 1, num_valid):
                    idx_k = valid_indices[k_outer_idx]
                    importances_k = all_importances[idx_k]

                    try:
                        # Ensure vectors are not identical or problematic for weightedtau
                        if (
                            len(importances_j) == len(importances_k)
                            and len(importances_j) > 0
                        ):
                            tau_score, _ = weightedtau(importances_j, importances_k)
                            scores[layer_id][split].append(tau_score)
                        # else: logger.debug(f"Skipping tau calc due to length mismatch or zero length between {idx_j} and {idx_k}")
                    except Exception as e:
                        logger.warning(
                            f"Could not compute weightedtau between results {idx_j} and {idx_k} for layer '{layer_id}', split '{split}': {e}"
                        )

    # --- Format Results ---
    if not scores:
        logger.warning(
            f"No consistency scores calculated for model '{model_kind}'. Returning empty DataFrame."
        )
        return pd.DataFrame(columns=["layer", "split", "score", "model"])

    try:
        scores_df = (
            pd.DataFrame.from_dict(scores)
            .melt(var_name="layer", value_name="score", ignore_index=False)
            .reset_index(names=["split"])
            .explode("score")  # Handle potential empty lists gracefully
        )
        scores_df = scores_df.dropna(
            subset=["score"]
        )  # Remove rows where score is NaN/None
        if not scores_df.empty:
            scores_df["score"] = scores_df["score"].astype("float")
        scores_df["model"] = model_kind
    except Exception as e:
        logger.error(
            f"Failed to create DataFrame from scores dict for model '{model_kind}': {e}"
        )
        return pd.DataFrame(
            columns=["layer", "split", "score", "model"]
        )  # Return empty on formatting error

    logger.info(
        f"Finished consistency computation for model '{model_kind}'. Result shape: {scores_df.shape}"
    )
    return scores_df


@task(cache_policy=TASK_SOURCE + INPUTS)
def build_eval_df(eval_lst):
    """Builds a dataframe with the evaluation metrics."""
    df = pd.concat(
        [df.drop(columns=["seed"]) for df in eval_lst],
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
        order=y_order,
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
    print("DEBUG: build_clustering_df got metrics:", clustering_results)
    df = pd.concat(clustering_results, axis=0, ignore_index=True)
    df["metric"] = "AdjustedMutualInfo"

    return df


@task(cache_policy=TASK_SOURCE + INPUTS)
def save_clustering(df, output_path):
    print("*"*20, df.columns)
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

    metric_scores_to_plot = metric_scores_to_plot.dropna(subset=["Score"])

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
        order=y_order,
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
        order=y_order,
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

    scores_to_plot = scores_to_plot.dropna(subset=["Score"])

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
        order=y_order,
    )

    g.savefig(
        Path(output_path).joinpath(f"{fname}_violin_test.pdf"), bbox_inches="tight"
    )
    g.savefig(
        Path(output_path).joinpath(f"{fname}_violin_test.png"),
        dpi=300,
        bbox_inches="tight",
    )


# Task for PathSingle
@task(cache_policy=TASK_SOURCE + INPUTS)
def fit_model_pathsingle(
    model_config,
    seed: int,
    data_path,
    debug: bool,
    results_folder_ps=".",
    n_jobs=-1,
):
    """
    Run PathSingle on split data, return annotated embeddings
    """
    results_folder_ps = Path(results_folder_ps)
    x_train, x_val, x_test, obs_train, obs_val, obs_test = split_data(
        seed, data_path, model_config
    )

    x_train = x_train.sample(n=100, random_state=seed)
    x_val = x_val.sample(n=100, random_state=seed)
    obs_train = obs_train.loc[x_train.index]
    obs_val = obs_val.loc[x_val.index]

    x = pd.concat((x_train, x_val, x_test), axis=0)
    obs = pd.concat((obs_train, obs_val, obs_test), axis=0)

    adata = sc.AnnData(X=x, obs=obs)
    adata.obs_names = x.index
    adata.var_names = x.columns
    adata.obs["split"] = "train"
    adata.obs.loc[x_val.index, "split"] = "val"
    adata.obs.loc[x_test.index, "split"] = "test"

    # MAGIC para reducir ruido
    if debug:
        adata = sc.pp.subsample(adata, n_obs=500, random_state=0, copy=True)
    nonzero_genes = np.array((adata.X != 0).sum(axis=0)).flatten() > 0
    adata = adata[:, nonzero_genes].copy()

    sce.pp.magic(
        adata,
        name_list="all_genes",
        solver="approximate",
        n_pca=30,
        knn=3,
        random_state=seed,
        n_jobs=n_jobs
    )

    # DEBUG
    if np.isnan(adata.X).any():
        raise ValueError(
            "NaNs found in adata.X after MAGIC - possible preprocessing issue."
        )

    # Run PathSingle
    print("Starting calc_activity...")

    activity_path = results_folder_ps.joinpath(f"output_activity_{seed}.csv")
    interaction_path = results_folder_ps.joinpath(
        f"output_interaction_activity_{seed}.csv"
    )
    calc_activity(
        adata, n_jobs=n_jobs, output_path=activity_path, interaction_path=interaction_path
    )  # Guarda el resultado a CSV
    print("Finished calc_activity")

    # Reading output_activity.csv
    activity = pd.read_csv(activity_path, index_col=0)

    # DEBUG
    if activity.empty:
        raise ValueError("Activity DataFrame is empty. PathSingle likely failed.")
    missing = set(activity.columns) - set(adata.obs_names)
    if missing:
        raise ValueError(f"Activity file contains unknown cell IDs: {missing}")

    # Readjusting the df
    activity.index.name = None
    activity = activity.T  # Cada fila es una celula

    # DEBUG
    if activity.shape[0] != adata.shape[0]:
        raise ValueError("Mismatch between activity and AnnData cells after transpose.")

    activity["split"] = adata.obs.loc[activity.index, "split"].values
    activity["cell_type"] = adata.obs.loc[activity.index, "cell_type"].values
    activity["condition"] = adata.obs.loc[activity.index, "condition"].values
    activity["model"] = "pathsingle"
    activity["seed"] = seed

    print(
        f"[DEBUG] PathSingle finished with {activity.shape[0]} cells, {activity.shape[1]} features"
    )
    print(f"[DEBUG] Unique cell types: {activity['cell_type'].nunique()}")

    return SimpleNamespace(eval=activity)


@task(cache_policy=TASK_SOURCE + INPUTS)
def compute_pathsingle_clustering(pathsingle_namespace, seed):

    embedding_df = pathsingle_namespace.eval

    clust_scores = {"PathSingle": {"train": [], "val": [], "test": []}}
    batch_size = 256 * os.cpu_count() + 1

    print("[DEBUG] Distribution of cell types per split/condition:")
    print(embedding_df.groupby(["split", "condition"])["cell_type"].value_counts())
    print(embedding_df.groupby(["split", "condition"]).size())

    # Filter control only once
    embedding_df = embedding_df[embedding_df["condition"] == "control"]

    # Fit on train
    train_df = embedding_df[embedding_df["split"] == "train"]
    if train_df.empty or train_df["cell_type"].nunique() < 2:
        print("[ERROR] Cannot train model due to insufficient data in 'train'")
        return pd.DataFrame()  # Return empty DataFrame to prevent crash

    y_train = train_df["cell_type"]
    X_train = train_df.drop(columns=["split", "cell_type", "condition", "model"])
    model = MiniBatchKMeans(n_clusters=y_train.nunique(), batch_size=batch_size)
    model.fit(X_train)

    print("[DEBUG] Split counts:")
    print(embedding_df["split"].value_counts())

    print("[DEBUG] Unique cell types per split and condition:")
    print(embedding_df.groupby(["split", "condition"])["cell_type"].nunique())

    for split in ["train", "val", "test"]:
        df = embedding_df[embedding_df["split"] == split]
        if df.empty or df["cell_type"].nunique() < 2:
            print(f"[WARN] Skipping {split} due to insufficient data")
            continue

        y_true = df["cell_type"]
        X = df.drop(columns=["split", "cell_type", "condition", "model"])

        preds = model.labels_ if split == "train" else model.predict(X)
        score = adjusted_mutual_info_score(y_true, preds)
        clust_scores["PathSingle"][split].append(score)

    df = (
        pd.DataFrame.from_dict(clust_scores)
        .melt(var_name="layer", value_name="score", ignore_index=False)
        .reset_index(names=["split"])
        .explode("score")
    )

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["model"] = "pathsingle"

    return df


@task(cache_policy=TASK_SOURCE + INPUTS)
def compute_pathsingle_consistency(pathsingle_namespace_lst):
    """
    Compute pairwise weighted tau consistency for PathSingle across seeds
    """

    scores = {"PathSingle": {"train": [], "val": [], "test": []}}
    drop_cols = ["split", "cell_type", "condition", "model"]

    for split in ["train", "val", "test"]:
        importances = []
        for ns in pathsingle_namespace_lst:
            df = ns.eval
            print("[DEBUG] Split counts:")
            print(df["split"].value_counts())

            print("[DEBUG] Unique cell types per split and condition:")
            print(df.groupby(["split", "condition"])["cell_type"].nunique())
            subset = df[df["split"] == split].drop(columns=drop_cols, errors="ignore")
            if subset.empty:
                continue
            imp = get_importances(subset, abs=True)
            if imp is not None:
                importances.append(imp)

        for i in range(len(importances)):
            for j in range(i + 1, len(importances)):
                tau, _ = weightedtau(importances[i], importances[j])
                scores["PathSingle"][split].append(tau)

    df = (
        pd.DataFrame.from_dict(scores)
        .melt(var_name="layer", value_name="score", ignore_index=False)
        .reset_index(names=["split"])
        .explode("score")
    )
    df["score"] = df["score"].astype(float)
    df["model"] = "pathsingle"
    return df


# Flow
@flow(
    name="m-IVAE-PathSingle",
    task_runner=RayTaskRunner(
        init_kwargs={
            "log_to_driver": False,
            "logging_config": ray.LoggingConfig(encoding="JSON", log_level="ERROR"),
        }
    ),
)
def main(
    results_folder: str = "results/IVAE",
    results_folder_ps: str = "results/PathSingle",
    data_path: str = "data",
    frac_start: float = 0.1,
    frac_step: float = 0.1,
    frac_stop: float = 1.0,
    n_seeds: int = 3,
    n_gpu: int = 3,
    n_cpu: int = 4,
    debug: bool = True,
):
    """Main workflow to run IVAE experiments and PathSingle benchmark on Kang dataset"""

    print("*" * 20, debug)

    # --- IVAE ---
    # Initial parameters
    if debug:
        frac_start = 0.1
        frac_stop = 0.2
        frac_step = 0.1

    fracsAux = np.arange(frac_start, frac_stop + frac_step / 2, frac_step)
    fracs = [f"{x:.2f}" for x in fracsAux]

    seeds = list(range(n_seeds))

    # Models for training
    models = [f"ivae_random-{frac}" for frac in fracs] + ["ivae_kegg", "ivae_reactome"]

    # Use n_cpu and n_gpu arguments
    # Ensure at least 1 CPU
    n_cpus_clustering = max(1, n_cpu - 2 * n_gpu if n_gpu > 0 else n_cpu - 1)
    n_cpus_clustering = max(1, n_cpu//3)

    # Installation and folders preparation
    install_ivae.submit().result()

    folders = []
    for model in models:
        if "random" in model:
            model_, frac = model.split("-")
        else:
            frac = None
            model_ = model
        create_folders.submit(results_folder, model_, frac)
        create_folders.submit(results_folder_ps, model_, frac)
    wait(folders)

    # Download data
    data_path_future = download_data.submit(data_path=data_path)
    data_path = data_path_future.result()
    genes_future = get_genes.submit(data_path)
    genes = genes_future.result()

    ps_model_config = InformedModelConfig(
            model_kind= "pathsingle",
            frac = None,
            n_encoding_layers = 1,
            adj_name = ["Reactome"],
            input_genes = get_reactome_adj().index.intersection(genes).unique().to_list()
        )
    # ivae_config_futures = build_ivae_config.map(models, unmapped(genes))

    # all_combinations = list(itertools.product(ivae_config_futures, seeds))
    # all_models = [x[0] for x in all_combinations]
    # all_seeds = [x[1] for x in all_combinations]

    evals = []
    # PathSingle no tiene MSE -> NO uso eval
    consistedness = []
    clustering_metrics = []
    consistedness_ps = []
    clustering_metrics_ps = []

    # For each model...
    for model in models:
        if "random" in model:
            model_, frac = model.split("-")
        else:
            frac = None
            model_ = model
        create_folders.submit(results_folder, model_, frac).result()
        create_folders.submit(results_folder_ps, model_, frac).result()
        ivae_config = build_ivae_config.submit(
            model, genes
        ).result()  # Build IVAE config

        results = []
        results_ps = []

        # For each seed...
        for seed in seeds:
            with remote_options(num_cpus=1, num_gpus=1):
                result_future = fit_model_ivae.submit(
                    ivae_config, seed, data_path, debug
                )  # Train IVAE for that seed (and model)

            with remote_options(num_cpus=n_cpus_clustering, num_gpus=0):

                result_future_ps = fit_model_pathsingle.submit(
                    ps_model_config,
                    seed,
                    data_path,
                    debug,
                    results_folder_ps,
                    n_jobs=n_cpus_clustering,
                )

            with remote_options(num_cpus=n_cpus_clustering, num_gpus=0):
                clustering_metrics_futures = evalute_clustering.submit(  # Clustering metrics for that seed and model (AMI)
                    result_future, seed
                )
                clustering_metrics_futures_ps = compute_pathsingle_clustering.submit(
                    result_future_ps, seed
                )

            results.append(result_future)
            results_ps.append(result_future_ps)
            clustering_metrics.append(clustering_metrics_futures)
            clustering_metrics_ps.append(clustering_metrics_futures_ps)

        # Waiting and saving results
        wait(results)
        wait(results_ps)

        with remote_options(num_cpus=1, num_gpus=0):
            eval_futures = gather_evals.submit(results)
            evals.append(eval_futures)
            consistedness_futures = compute_consistedness.submit(
                results
            )  # Calculating consistency
            consistedness_futures_ps = compute_pathsingle_consistency.submit(
                results_ps
            )
            consistedness.append(consistedness_futures)
            consistedness_ps.append(consistedness_futures_ps)

        del results  # Free some memory
        del results_ps
        gc.collect()  # Garbage collection

    # Wait for all tasks to finish
    wait(clustering_metrics)
    wait(evals)
    wait(consistedness)
    wait(consistedness_ps)
    wait(clustering_metrics_ps)

    # Saving metrics and results
    # IVAE
    eval_df = build_eval_df.submit(evals)
    save_eval.submit(eval_df, results_folder).result()

    clustering_df = build_clustering_df(clustering_metrics)
    save_clustering.submit(clustering_df, results_folder).result()

    consistedness_df = build_consistedness_df(consistedness)
    save_consistedness.submit(consistedness_df, results_folder).result()

    save_combined.submit(consistedness_df, clustering_df, results_folder).result()

    # PathSingle
    #evals_ps_resolved = [f for f in results_ps]
    #clustering_metrics_ps_resolved = [f.result() for f in clustering_metrics_ps]
    #consistedness_ps_resolved = [f.result() for f in consistedness_ps]

    #print("DEBUG: clustering_metrics_ps_resolved =", clustering_metrics_ps_resolved)
    #print("DEBUG: types =", [type(m) for m in clustering_metrics_ps_resolved])

    clustering_df_ps = build_clustering_df(clustering_metrics_ps)
    #print("DEBUG: clustering_df_ps =", clustering_df_ps)
    consistedness_df_ps = build_consistedness_df(consistedness_ps)

    save_clustering.submit(clustering_df_ps, results_folder_ps).result()
    save_consistedness.submit(consistedness_df_ps, results_folder_ps).result()
    save_combined.submit(
        consistedness_df_ps, clustering_df_ps, results_folder_ps
    ).result()

    # Waiting before finishing
    time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the IVAE and PathSIngle Benchmark Workflow."
    )

    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run in debug mode (fewer models/epochs). Use --no-debug to disable.",
    )

    parser.add_argument(
        "--results_folder",
        type=str,
        default="results/debug/IVAE",
        help="Path to the main folder where IVAE results will be saved.",
    )

    parser.add_argument(
        "--results_folder_ps",
        type=str,
        default="results/debug/PathSingle",
        help="Path to the main folder where PathSingle results will be saved.",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to the folder containing or to download input data.",
    )

    parser.add_argument(
        "--n_seeds",
        type=int,
        default=40,
        help="Number of repeated holdout procedures.",
    )

    parser.add_argument(
        "--frac_start",
        type=float,
        default=0.05,
        help="Start point for density level used when bulding random layers. Start point.",
    )

    parser.add_argument(
        "--frac_step",
        type=float,
        default=0.05,
        help="Step point for density level used when bulding random layers. Start point.",
    )

    parser.add_argument(
        "--frac_stop",
        type=float,
        default=1.0,
        help="Final point for density level used when bulding random layers. Start point.",
    )

    parser.add_argument(
        "--n_gpus",
        type=int,
        default=3,
        help="Number of GPUs used for training. Max one model per GPU.",
    )

    parser.add_argument(
        "--n_cpus",
        type=int,
        default=N_CPUS,
        help="Max number of CPUs used for no  GPU tasks.",
    )

    args = parser.parse_args()

    main(
        debug=args.debug,
        results_folder=args.results_folder,
        results_folder_ps=args.results_folder_ps,
        data_path=args.data_path,
        n_seeds=args.n_seeds,
        frac_start=args.frac_start,
        frac_step=args.frac_step,
        frac_stop=args.frac_stop,
        n_gpu=args.n_gpus,
        n_cpu=args.n_cpus,
    )
