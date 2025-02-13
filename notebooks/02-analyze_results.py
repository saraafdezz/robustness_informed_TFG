from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from isrobust_TFG.utils import (set_all_seeds, print_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the results")
    parser.add_argument("--frac_start", type=float, default=0.05, help="Frac start")
    parser.add_argument("--frac_stop", type=float, default=0.85, help="Frac step")
    parser.add_argument("--frac_step", type=float, default=0.2, help="Frac stop")
#     parser.add_argument("--seed_start", type=int, default=0, help="Seed start")
#     parser.add_argument("--seed_step", type=int, default=1, help="Seed step")
#     parser.add_argument("--seed_stop", type=int, default=50, help="Seed stop")
#     parser.add_argument("--n_genes", type=int, default=None, help="Number of genes")
    args = parser.parse_args()
    model_kind = args.model_kind
    start = args.frac_start
    step = args.frac_step
    stop = args.frac_stop
#     seed_start = args.seed_start
#     seed_step = args.seed_step
#     seed_stop = args.seed_stop
#     n_genes = args.n_genes
    frac_list = np.arange(start, stop, step)
    

    project_path = Path(dotenv.find_dotenv()).parent
    data_path = project_path.joinpath("data")

    tf.config.experimental.enable_op_determinism()
    config = dotenv.dotenv_values()

    frac_list = np.arange(start, stop, step)

    results_path = Path(config["RESULTS_FOLDER"]).resolve()
    figs_path = results_path.joinpath("figs")
    tables_path = results_path.joinpath("tables")

    results_path

    models = ["ivae_kegg", "ivae_reactome"] + [
        f"ivae_random-{frac:.2f}" for frac in frac_list
    ]
    models

    # Metrics

    metric_scores = [
        pd.read_pickle(results_path.joinpath(m, "scores_metrics.pkl")) for m in models
    ]
    metric_scores = pd.concat(metric_scores, axis=0, ignore_index=True)
    metric_scores = metric_scores
    metric_scores.head()

    # clustering

    clustering_scores = [
        pd.read_pickle(results_path.joinpath(m, "scores_clustering.pkl")) for m in models
    ]
    clustering_scores = pd.concat(clustering_scores, axis=0, ignore_index=True)
    clustering_scores.head()

    clustering_scores.groupby(["model", "layer"]).size()

    # informed

    informed_scores = [
        pd.read_pickle(results_path.joinpath(m, "scores_informed.pkl")) for m in models
    ]
    informed_scores = pd.concat(informed_scores, axis=0, ignore_index=True)
    informed_scores.head()

    informed_scores.groupby(["model", "layer"]).size()

    clustering_scores["kind"] = "clustering"
    informed_scores["kind"] = "informed"

    (metric_scores
     .query("metric=='mse'")
     .drop(["seed"], axis=1)
     .groupby(["model", "metric", "split"])["score"]
     .describe()
     .drop(["count", "min", "max"], axis=1)
     .to_latex(
       "mse.tex",
        bold_rows=True,
        escape=True,
    ))

    metric_scores_to_plot = metric_scores.copy().query("split=='test'").query("metric=='mse'").rename(columns={"model": "Modelo", "score":"Valor"})
    metric_scores_to_plot["Valor"] = -np.log(metric_scores_to_plot["Valor"])

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context="paper", font_scale=2, style="ticks", rc=custom_params)
    fac = 0.7

    g = sns.catplot(
        data=metric_scores_to_plot,
        kind="violin",
        col="metric",
        height=9 * fac,
        aspect=16 / 9 * fac,
        sharey=True,
        sharex=False,
        y="Modelo",
        x="Valor",
        split=False,
        cut=0,
        fill=False,
        density_norm="count",
        inner="quart",
        linewidth=2,
        legend_out=False,
        col_wrap=4,
    )

    g.savefig("model_mse.pdf")

    scores = (
        pd.concat((clustering_scores, informed_scores), axis=0, ignore_index=True)
        .query("split=='test'")
        .query("layer <= 2")
        .drop(["split"], axis=1)
        .rename(columns={"kind": "metric"})
    )

    scores.head()

    scores_to_plot = scores.copy()
    scores_to_plot["layer_name"] = "Pathways"

    scores_to_plot = scores_to_plot.replace("clustering", "AMI")
    scores_to_plot = scores_to_plot.replace("informed", r"$\tau$")

    mask = (scores_to_plot["layer"] == 1) & (scores_to_plot["model"] == "ivae_kegg")
    scores_to_plot.loc[mask, "layer_name"] = "Circuitos"

    mask = (scores_to_plot["layer"] == 1) & (scores_to_plot["model"] == "ivae_random")
    scores_to_plot.loc[mask, "layer_name"] = "RndInf"

    scores_to_plot["Modelo"] = scores_to_plot["model"] + " (" + scores_to_plot["layer_name"] + ")"

    scores_to_plot = scores_to_plot.rename(columns={"score": "Valor", "metric": "Métrica"})
    scores_to_plot.head()

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context="paper", font_scale=2, style="ticks", rc=custom_params)
    fac = 0.7

    g = sns.catplot(
        data=scores_to_plot,
        kind="violin",
        col="Métrica",
        height=9 * fac,
        aspect=16 / 9 * fac,
        sharey=True,
        sharex=False,
        y="Modelo",
        x="Valor",
        split=False,
        cut=0,
        fill=False,
        density_norm="count",
        inner="quart",
        linewidth=2,
        legend_out=False,
        col_wrap=4,
    )

    g.savefig("layer_scores.pdf")

    print_scores(informed_scores, "informed.tex")
    print_scores(clustering_scores, "clustering.tex")
