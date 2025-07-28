import os
from itertools import chain, repeat
from pathlib import Path

import decoupler
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
from activity import *
from metrics import *
from scipy import stats
from sklearn.preprocessing import (
    Normalizer,  # Unit norm, row wise. # StandardScaler # Normal distribution. MinMaxScaler # [0,1] range, column wise.
)
from tqdm.notebook import tqdm

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def gmt_to_decoupler() -> pd.DataFrame:
    pth = "./data/c2.cp.reactome.v7.5.1.symbols.gmt"
    """Parse a gmt file to a decoupler pathway dataframe."""
    pathways = {}
    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    return pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )


def run_method(method_name, method, adata):
    """Run a given method 30 times, calculate metrics and confidence intervals."""
    metrics = [[] for _ in range(6)]  # Initialize 6 empty lists.
    metric_names = [
        "Silhouette",
        "Calinski",
        "Special",
        "Completeness",
        "Homogeneity",
        "Adjusted",
    ]
    for i in tqdm(range(30)):
        pathway_activity_df = method(adata)
        # Perform KMeans clustering and plot UMAP.
        kmeans = cluster_with_kmeans(
            method_name, pathway_activity_df, adata, n_clusters=n_clusters
        )
        scores = calc_stats(
            pathway_activity_df, true_labels, kmeans.labels_, debug=True
        )
        print(i)
        for score, metric_list in zip(scores, metrics):
            metric_list.append(score)

    print(f"Results for {method_name}:")
    for name, metric in zip(metric_names, metrics):
        print(
            f"{name} - mean: {np.mean(metric)}, ci: {stats.t.interval(0.95, len(metric)-1, loc=np.mean(metric), scale=stats.sem(metric))}"
        )


# Running GSEA. We will use the python package [`decoupler`](https://decoupler-py.readthedocs.io/en/latest/) <cite>`badia2022decoupler`</cite> to perform GSEA enrichment tests on our data.
# We use the normalized scores from sc.pp.normalize_total(adata) as a proxy for differential expression (DE) scores, which will significantly speed up the process since we don't have to
# calculate DE scores for each cell individually.
def run_gsea(adata):
    reactome = gmt_to_decoupler()
    # Prepare the result matrix for GSEA scores.
    num_cells = adata.shape[0]
    num_gene_sets = len(reactome["geneset"].unique())
    gsea_results_matrix = np.zeros((num_cells, num_gene_sets))

    # Loop through each cell to run GSEA.
    for cell_index in range(num_cells):
        # Get normalized expression values for the specific cell.
        cell_expr = adata.X[cell_index]
        # Create a DataFrame to hold DE scores.
        de_scores = pd.DataFrame(cell_expr, index=adata.var_names, columns=["scores"])
        # Run GSEA using decoupler.
        _, norm, _ = decoupler.run_gsea(
            de_scores.T, reactome, source="geneset", target="genesymbol"
        )
        # Store the normalized enrichment scores (NES) in the result matrix.
        gsea_results_matrix[cell_index, :] = norm.iloc[:, 0].values
        print(cell_index, end="\r")
    return gsea_results_matrix


def run_progeny(adata):
    progeny = decoupler.get_progeny(organism="human", top=2000)
    decoupler.run_mlm(
        mat=adata,
        net=progeny,
        source="source",
        target="target",
        weight="weight",
        verbose=False,
        use_raw=False,
    )
    acts = decoupler.get_acts(adata, obsm_key="mlm_estimate")
    # Convert the pathway activity matrix to a DataFrame.
    return pd.DataFrame(
        acts.obsm["mlm_estimate"], index=adata.obs_names, columns=acts.var_names
    )


def run_aucell(adata):
    reactome = gmt_to_decoupler()
    decoupler.run_aucell(
        adata,
        reactome,
        source="geneset",
        target="genesymbol",
        use_raw=False,
        verbose=False,
    )
    return adata.obsm["aucell_estimate"]


def run_pathsingle(adata):
    from sklearn.decomposition import PCA

    activity_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    activity = sc.AnnData(activity_df)
    calc_activity(activity, sparsity)
    output_activity = pd.read_csv(
        "~/TFG/PathSingle/pathsingle/data/output_activity.csv", index_col=0
    )

    # Scale the data.
    scaler = Normalizer()
    output_activity = scaler.fit_transform(output_activity)
    PCA = PCA(n_components=30, svd_solver="arpack")
    output_activity = PCA.fit_transform(output_activity)
    return output_activity


if __name__ == "__main__":
    # Kang data
    adata = sc.read_h5ad("data/kang_counts_25k.h5ad")
    adata = sc.pp.subsample(
        adata, fraction=0.05, copy=True
    )  # Reduce para pruebas (ELIMINAR PARA EL LAUNCH FINAL)
    adata.obs["labels"] = (
        adata.obs["cell_type"].astype("category").cat.codes
    )  # convierte a enteros
    true_labels = adata.obs["labels"]
    n_clusters = adata.obs["labels"].nunique()
    sc.pp.normalize_total(adata)
    sc.pp.sqrt(adata)
    adata.raw = adata.copy()

    """
    # 68K PBMC data.
    adata = sc.datasets.pbmc68k_reduced()
    adata.obs['labels'] = adata.obs.bulk_labels.map({'CD14+ Monocyte':0, 'Dendritic':1, 'CD56+ NK':2, 'CD4+/CD25 T Reg':3, 'CD19+ B':4, 'CD8+ Cytotoxic T':5, 'CD4+/CD45RO+ Memory':6, 'CD8+/CD45RA+ Naive Cytotoxic':7, 'CD4+/CD45RA+/CD25- Naive T':8, 'CD34+':9})
    n_clusters=10
    true_labels = adata.obs.labels
    print(adata)

    
    # T-cells data.
    num_splits = 5
    split_files = [f'./data/sc_training_split_{i+1}.h5ad' for i in range(num_splits)]
    splits = [sc.read_h5ad(file) for file in split_files]
    # Concatenate the splits back into a single AnnData object.
    adata = splits[0].concatenate(*splits[1:], batch_key='batch', batch_categories=[f'batch_{i+1}' for i in range(num_splits)])
    adata = sc.pp.subsample(adata, fraction=0.1, copy=True) #28697 cells × 15077 genes.
    print(adata)
    true_labels = adata.obs.state.map({'cycling':0, 'effector':1, 'other':2, 'progenitor':3, 'terminal exhausted':4})
    n_clusters=5
    sc.pp.normalize_total(adata)  # Library size normalization (works on adata.X).
    sc.pp.sqrt(adata)             # Square root transformation (works on adata.X).
    adata.raw = adata.copy()      # Copy adata.X plus other objects to adata.raw.
    """

    sparsity = calculate_sparsity(adata)
    # Run Magic.
    sce.pp.magic(adata, name_list="all_genes")
    # Define list of method functions.
    methods = [run_pathsingle, run_gsea, run_progeny, run_aucell]

    # Loop through method functions.
    for method_func in methods:
        method_name = method_func.__name__.replace("run_", "")  # Remove 'run_' prefix
        print(f"\nRunning {method_name.upper()}...")
        run_method(method_name, method_func, adata)
