# https://www.sc-best-practices.org/conditions/gsea_pathway.html#id380
# Kang HM, Subramaniam M, Targ S, et al. Multiplexed droplet single-cell RNA-sequencing using natural genetic variation
#   Nat Biotechnol. 2020 Nov;38(11):1356]. Nat Biotechnol. 2018;36(1):89-94. doi:10.1038/nbt.4042

from pathlib import Path

import glob
import pickle



import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import tensorflow as tf
import argparse
from scipy.stats import weightedtau
from sklearn.model_selection import train_test_split

from ivae.bio import (
    build_hipathia_renamers,
    get_adj_matrices,
    get_random_adj,
    get_reactome_adj,
    sync_gexp_adj,
    get_importances,
    get_activations,
    train_val_test_split,
)

from ivae.datasets import load_kang
from ivae.utils import set_all_seeds


from multiprocessing import cpu_count

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring with a specific model")
    parser.add_argument("--model_kind", type=str, help="Type of model: ivae_kegg, ivae_reactome or ivae_random")
    parser.add_argument("--frac", type=float, default=1, help="Distribution of random layer (if needed)")
    parser.add_argument("--seed_start", type=int, default=0, help="Seed start")
    parser.add_argument("--seed_step", type=int, default=1, help="Seed step")
    parser.add_argument("--seed_stop", type=int, default=50, help="Seed stop")
    parser.add_argument("--n_genes", type=int, default=None, help="Number of genes")
    parser.add_argument(
        "--results_path", type=str, default=".", help="Output folder"
    )
    parser.add_argument("--data_path", type=str, default=".", help="Data folder")
    # parser.add_argument("--figs_path", type=str, default=".", help="Figures folder")
    # parser.add_argument("--tables_path", type=str, default=".", help="Tables folder")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")
    args = parser.parse_args()
    model_kind = args.model_kind
    frac = args.frac
    seed_start = args.seed_start
    seed_step = args.seed_step
    seed_stop = args.seed_stop
    n_genes = args.n_genes
    results_path = Path(args.results_path)
    data_path = Path(args.data_path)
    # figs_path = Path(args.figs_path)
    # tables_path = Path(args.tables_path)
    debug = args.debug

    print("+"*20, debug)

    print(model_kind, frac, n_genes)
    
    config = dotenv.dotenv_values()
 #   debug = bool(int(config["DEBUG"])) 
#     set_all_seeds(seed=seed)
    batch_size = 256 * cpu_count() + 1


    # Rutas del proyecto
    # project_path = Path(dotenv.find_dotenv()).parent
    # data_path = project_path.joinpath("data")
    # data_path.mkdir(exist_ok=True, parents=True)
    # results_path = Path(config["RESULTS_FOLDER"])
    results_path.mkdir(exist_ok=True, parents=True)
    figs_path = results_path.joinpath("figs")
    figs_path.mkdir(exist_ok=True, parents=True)
    tables_path = results_path.joinpath("tables")
    tables_path.mkdir(exist_ok=True, parents=True)
    if("ivae_random" in model_kind):
        results_path_model = results_path.joinpath(model_kind + f"-{frac}")
        results_path_model.mkdir(exist_ok=True, parents=True)
    else:
        results_path_model = results_path.joinpath(model_kind)
        results_path_model.mkdir(exist_ok=True, parents=True)
    print(f"{results_path_model}")

    
    adata = load_kang(data_folder=data_path, normalize=True, n_genes=n_genes)
    obs = adata.obs.copy()
    x_trans = adata.to_df()
    
    tf.config.experimental.enable_op_determinism()

    sc.set_figure_params(dpi=300, color_map="inferno")
    sc.settings.verbosity = 1
    sc.logging.print_header()


    seeds = list(
        range(
            seed_start,
            seed_stop + 1,
            seed_step,
        )
    )
    N_ITERS = len(seeds)

    if debug:
        N_EPOCHS = 2
    else:
        N_EPOCHS = 300
        

    if model_kind == "ivae_kegg":
        n_encoding_layers = 3
        circuit_adj, circuit_to_pathway_adj = get_adj_matrices(
        gene_list=x_trans.columns.to_list()
        )
        circuit_renamer, pathway_renamer, circuit_to_effector = build_hipathia_renamers()
        kegg_circuit_names = circuit_adj.rename(columns=circuit_renamer).columns
        kegg_pathway_names = circuit_to_pathway_adj.rename(columns=pathway_renamer).columns
        circuit_adj.head()
        x_trans, circuit_adj = sync_gexp_adj(gexp=x_trans, adj=circuit_adj)
        layer_ids = [1, 2, 3]
        
    elif model_kind == "ivae_reactome":
        n_encoding_layers = 2
        reactome = get_reactome_adj()
        reactome_pathway_names = reactome.columns
        x_trans, reactome = sync_gexp_adj(x_trans, reactome)
        layer_ids = [1, 2]
        
    elif "ivae_random" in model_kind:
        reactome = get_reactome_adj()
        n_encoding_layers = 2
        n_genes = 3000 # En los otros models no se pone porque el default ya es None
        state = np.random.get_state()
        random_layer, random_layer_names = get_random_adj(
        frac, shape=reactome.shape, size=reactome.size, index=reactome.index, seed=0)
        np.random.set_state(state)
        x_trans, random_layer = sync_gexp_adj(x_trans, random_layer)
        layer_ids = [1, 2]
        
    else:
        raise NotImplementedError(f"{model_kind} not implemented yet.")

    print(f"{debug=} {model_kind=}")
    
    
    non_layer_names = ["split", "layer", "seed", "cell_type", "condition", "model"]
    

    scores_metrics = [
        pd.read_pickle(results_path_model.joinpath(f"metrics-seed-{seed:02d}.pkl")) 
        for seed in seeds
    ]
    scores_metrics = pd.concat(scores_metrics, axis=0, ignore_index=True)
    scores_metrics.to_pickle(results_path_model.joinpath("scores_metrics.pkl"))

#     scores_metrics.head()

#     custom_params = {"axes.spines.right": False, "axes.spines.top": False}
#     sns.set_theme(context="paper", font_scale=0.5, style="ticks", rc=custom_params)

#     g = sns.catplot(
#         data=scores_metrics,
#         kind="violin",
#         col="metric",
#         height=2,
#         aspect=0.9,
#         sharey=False,
#         x="model",
#         y="score",
#         hue="split",
#         split=False,
#         cut=0,
#         fill=False,
#         density_norm="count",
#         inner="quart",
#         linewidth=0.5,
#     )

    
    
    scores_informed = {}

    for layer_id in layer_ids:
        if results_path_model.joinpath(
            f"encodings_layer-{layer_id:02d}_seed-00.pkl"
        ).exists():
            results_layer = [
                pd.read_pickle(
                    results_path_model.joinpath(
                        f"encodings_layer-{layer_id:02d}_seed-{seed:02d}.pkl"
                    )
                )
                for seed in seeds
            ]
        else:
            continue

        scores_informed[layer_id] = {}
        for split in ["train", "test", "val"]:
            results = [
                x.loc[x["split"] == split].drop(non_layer_names, axis=1)
                for x in results_layer
            ]
            scores_informed[layer_id][split] = []
            for seed_i in seeds:
                for seed_j in range(seed_i + 1, N_ITERS):
                    scores_informed[layer_id][split].append(
                        weightedtau(
                            get_importances(data=results[seed_i], abs=True),
                            get_importances(data=results[seed_j], abs=True),
                        )[0]
                    )

    scores_informed = (
        pd.DataFrame.from_dict(scores_informed)
        .melt(var_name="layer", value_name="score", ignore_index=False)
        .reset_index(names=["split"])
        .explode("score")
    )
    scores_informed["score"] = scores_informed["score"].astype("float")
    scores_informed["model"] = model_kind
    scores_informed.to_pickle(results_path_model.joinpath("scores_informed.pkl"))

    results_path_model.joinpath("scores_informed.pkl")

#     scores_informed.head()

#     custom_params = {"axes.spines.right": False, "axes.spines.top": False}
#     sns.set_theme(context="paper", font_scale=0.5, style="ticks", rc=custom_params)
#     plt.figure(figsize=(2, 2))
#     sns.violinplot(
#         data=scores_informed,
#         x="layer",
#         y="score",
#         hue="split",
#         split=False,
#         cut=0,
#         fill=False,
#         density_norm="count",
#         inner="quart",
#         linewidth=0.5,
#     )
#     sns.despine()

    clust_scores = {}

    for layer_id in layer_ids:
        if results_path_model.joinpath(
            f"encodings_layer-{layer_id:02d}_seed-00.pkl"
        ).exists():
            results_layer = [
                pd.read_pickle(
                    results_path_model.joinpath(
                        f"encodings_layer-{layer_id:02d}_seed-{seed:02d}.pkl"
                    )
                )
                for seed in range(N_ITERS)
            ]
        else:
            continue

        train_embeddings_lst = [
            x.loc[(x["split"] == "train") & (x["condition"] == "control")]
            for x in results_layer
        ]
        val_embeddings_lst = [
            x.loc[(x["split"] == "val") & (x["condition"] == "control")]
            for x in results_layer
        ]
        test_embeddings_lst = [
            x.loc[(x["split"] == "test") & (x["condition"] == "control")]
            for x in results_layer
        ]

        clust_scores[layer_id] = {}
        clust_scores[layer_id]["train"] = []
        clust_scores[layer_id]["val"] = []
        clust_scores[layer_id]["test"] = []

        for seed in range(N_ITERS):
            y_train = train_embeddings_lst[seed]["cell_type"]
            y_val = val_embeddings_lst[seed]["cell_type"]
            y_test = test_embeddings_lst[seed]["cell_type"]

            train_embeddings = train_embeddings_lst[seed].drop(non_layer_names, axis=1)
            val_embeddings = val_embeddings_lst[seed].drop(non_layer_names, axis=1)
            test_embeddings = test_embeddings_lst[seed].drop(non_layer_names, axis=1)

            model = MiniBatchKMeans(n_clusters=y_train.nunique(), batch_size=batch_size)
            model.fit(train_embeddings)
            clust_scores[layer_id]["train"].append(
                adjusted_mutual_info_score(y_train, model.labels_)
            )
            clust_scores[layer_id]["val"].append(
                adjusted_mutual_info_score(y_val, model.predict(val_embeddings))
            )
            clust_scores[layer_id]["test"].append(
               adjusted_mutual_info_score(y_test, model.predict(test_embeddings))
            )

    results_path_model.joinpath("scores_clustering.pkl")

    clust_scores = (
        pd.DataFrame.from_dict(clust_scores)
        .melt(var_name="layer", value_name="score", ignore_index=False)
        .reset_index(names=["split"])
        .explode("score")
    )
    clust_scores["score"] = clust_scores["score"].astype("float")
    clust_scores["model"] = model_kind
    clust_scores.to_pickle(results_path_model.joinpath("scores_clustering.pkl"))

#     clust_scores.head()

#     custom_params = {"axes.spines.right": False, "axes.spines.top": False}
#     sns.set_theme(context="paper", font_scale=0.5, style="ticks", rc=custom_params)
#     plt.figure(figsize=(2, 2))
#     sns.violinplot(
#         data=clust_scores,
#         x="layer",
#         y="score",
#         hue="split",
#         split=False,
#         cut=0,
#         fill=False,
#         density_norm="count",
#         inner="quart",
#         linewidth=0.5,
#     )
#     sns.despine()



#    sns.save()
#    clust_scores.save()
#    scores_informed.save()
#    g.save()



