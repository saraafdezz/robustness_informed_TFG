# https://www.sc-best-practices.org/conditions/gsea_pathway.html#id380
# Kang HM, Subramaniam M, Targ S, et al. Multiplexed droplet single-cell RNA-sequencing using natural genetic variation
#   Nat Biotechnol. 2020 Nov;38(11):1356]. Nat Biotechnol. 2018;36(1):89-94. doi:10.1038/nbt.4042

from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import tensorflow as tf
from scipy.stats import weightedtau
from sklearn.model_selection import train_test_split

from isrobust_TFG.bio import (
    build_hipathia_renamers,
    get_adj_matrices,
    get_random_adj,
    get_reactome_adj,
    sync_gexp_adj,
    get_importances,
    get_activations,
    train_val_test_split,
)

from isrobust_TFG.datasets import load_kang
from isrobust_TFG.utils import set_all_seeds


from multiprocessing import cpu_count

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score


# Rutas del proyecto
project_path = Path(dotenv.find_dotenv()).parent
data_path = project_path.joinpath("data")
data_path.mkdir(exist_ok=True, parents=True)
results_path = Path(config["RESULTS_FOLDER"])
results_path.mkdir(exist_ok=True, parents=True)
figs_path = results_path.joinpath("figs")
figs_path.mkdir(exist_ok=True, parents=True)
tables_path = results_path.joinpath("tables")
tables_path.mkdir(exist_ok=True, parents=True)
results_path_model = results_path.joinpath(model_kind)
obs = adata.obs.copy()
results_path_model.mkdir(exist_ok=True, parents=True)
results_path_model_seed = results_path_model.joinpath("seed_" + str(seed))
obs = adata.obs.copy()
results_path_model_seed.mkdir(exist_ok=True, parents=True)


set_all_seeds(seed=42)

tf.config.experimental.enable_op_determinism()

sc.set_figure_params(dpi=300, color_map="inferno")
sc.settings.verbosity = 1
sc.logging.print_header()

config = dotenv.dotenv_values()

debug = bool(int(config["DEBUG"]))



seeds = list(
    range(
        int(config["SEED_START"]),
        int(config["SEED_STOP"]) + 1,
        int(config["SEED_STEP"]),
    )
)
N_ITERS = len(seeds)

if debug:
    N_EPOCHS = 2
else:
    N_EPOCHS = 300

if model_kind == "ivae_kegg":
    n_encoding_layers = 3
elif model_kind == "ivae_reactome":
    n_encoding_layers = 2
elif "ivae_random" in model_kind:
    n_encoding_layers = 2
else:
    raise NotImplementedError(f"{model_kind} not implemented yet.")

print(f"{debug=} {model_kind=}")

if "ivae_random" in model_kind:
    n_genes = 3000
else:
    n_genes = None
adata = load_kang(data_folder=data_path, normalize=True, n_genes=n_genes)

x_trans = adata.to_df()

circuit_adj, circuit_to_pathway_adj = get_adj_matrices(
    gene_list=x_trans.columns.to_list()
)

circuit_renamer, pathway_renamer, circuit_to_effector = build_hipathia_renamers()

kegg_circuit_names = circuit_adj.rename(columns=circuit_renamer).columns

kegg_pathway_names = circuit_to_pathway_adj.rename(columns=pathway_renamer).columns

circuit_adj.head()

reactome = get_reactome_adj()
reactome_pathway_names = reactome.columns

if model_kind == "ivae_kegg":
    x_trans, circuit_adj = sync_gexp_adj(gexp=x_trans, adj=circuit_adj)
elif model_kind == "ivae_reactome":
    x_trans, reactome = sync_gexp_adj(x_trans, reactome)
    
state = np.random.get_state()

# we ensure to presever the same sparsity structure for a given frac across all seeds
random_layer, random_layer_names = get_random_adj(
    frac, shape=reactome.shape, size=reactome.size, index=reactome.index, seed=0
)

np.random.set_state(state)

if model_kind == "ivae_kegg":
    x_trans, circuit_adj = sync_gexp_adj(gexp=x_trans, adj=circuit_adj)
elif model_kind == "ivae_reactome":
    x_trans, reactome = sync_gexp_adj(x_trans, reactome)
elif "ivae_random" in model_kind:
    x_trans, random_layer = sync_gexp_adj(x_trans, random_layer)


if model_kind == "ivae_kegg":
    # vae, encoder, decoder = build_kegg_vae(
    #     circuits=circuit_adj, pathways=circuit_to_pathway_adj, seed=42
    # )
    layer_ids = [1, 2, 3]
elif model_kind == "ivae_reactome":
    # vae, encoder, decoder = build_reactome_vae(reactome, seed=42)
    layer_ids = [1, 2]
elif "ivae_random" in model_kind:
    # vae, encoder, decoder = build_reactome_vae(random_layer, seed=42)
    layer_ids = [1, 2]
else:
    raise NotImplementedError("Model not yet implemented.")
    
    
non_layer_names = ["split", "layer", "seed", "cell_type", "condition", "model"]

results_path_model

scores_metrics = [
    pd.read_pickle(results_path_model_seed.joinpath(f"metrics-seed-{seed:02d}.pkl"))
    for seed in seeds
]
scores_metrics = pd.concat(scores_metrics, axis=0, ignore_index=True)
scores_metrics.to_pickle(results_path_model_seed.joinpath("scores_metrics.pkl"))

scores_metrics.head()

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(context="paper", font_scale=0.5, style="ticks", rc=custom_params)

g = sns.catplot(
    data=scores_metrics,
    kind="violin",
    col="metric",
    height=2,
    aspect=0.9,
    sharey=False,
    x="model",
    y="score",
    hue="split",
    split=False,
    cut=0,
    fill=False,
    density_norm="count",
    inner="quart",
    linewidth=0.5,
)

scores_informed = {}

for layer_id in layer_ids:
    if results_path_model_seed.joinpath(
        f"encodings_layer-{layer_id:02d}_seed-00.pkl"
    ).exists():
        results_layer = [
            pd.read_pickle(
                results_path_model_seed.joinpath(
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
scores_informed.to_pickle(results_path_model_seed.joinpath("scores_informed.pkl"))

results_path_model_seed.joinpath("scores_informed.pkl")

scores_informed.head()

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(context="paper", font_scale=0.5, style="ticks", rc=custom_params)
plt.figure(figsize=(2, 2))
sns.violinplot(
    data=scores_informed,
    x="layer",
    y="score",
    hue="split",
    split=False,
    cut=0,
    fill=False,
    density_norm="count",
    inner="quart",
    linewidth=0.5,
)
sns.despine()

batch_size = 256 * cpu_count() + 1

clust_scores = {}

for layer_id in layer_ids:
    if results_path_model_seed.joinpath(
        f"encodings_layer-{layer_id:02d}_seed-00.pkl"
    ).exists():
        results_layer = [
            pd.read_pickle(
                results_path_model_seed.joinpath(
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
        
        
        
results_path_model_seed.joinpath("scores_clustering.pkl")

clust_scores = (
    pd.DataFrame.from_dict(clust_scores)
    .melt(var_name="layer", value_name="score", ignore_index=False)
    .reset_index(names=["split"])
    .explode("score")
)
clust_scores["score"] = clust_scores["score"].astype("float")
clust_scores["model"] = model_kind
clust_scores.to_pickle(results_path_model_seed.joinpath("scores_clustering.pkl"))

clust_scores.head()

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(context="paper", font_scale=0.5, style="ticks", rc=custom_params)
plt.figure(figsize=(2, 2))
sns.violinplot(
    data=clust_scores,
    x="layer",
    y="score",
    hue="split",
    split=False,
    cut=0,
    fill=False,
    density_norm="count",
    inner="quart",
    linewidth=0.5,
)
sns.despine()





    

