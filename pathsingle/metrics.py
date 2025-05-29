import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import seaborn as sns
import umap.umap_ as umap
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.stats import combine_pvalues
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    adjusted_mutual_info_score,
    calinski_harabasz_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))


# General metric functions used in the benchmarks.
# Dunn index calculation function.
def dunn_index(X, labels):
    # Compute pairwise distances.
    distances = squareform(pdist(X))

    # Find unique cluster labels.
    unique_labels = np.unique(labels)

    # Calculate inter-cluster distances (minimum distance between points in different clusters).
    inter_cluster_dists = np.inf
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            inter_cluster_dists = min(
                inter_cluster_dists,
                np.min(
                    distances[
                        np.ix_(labels == unique_labels[i], labels == unique_labels[j])
                    ]
                ),
            )

    # Calculate intra-cluster distances (maximum distance within a cluster).
    intra_cluster_dists = 0
    for label in unique_labels:
        cluster = X[labels == label]
        intra_cluster_dists = max(
            intra_cluster_dists, np.max(pdist(cluster)) if len(cluster) > 1 else 0
        )

    # Calculate Dunn index.
    return (
        inter_cluster_dists / intra_cluster_dists
        if intra_cluster_dists != 0
        else np.inf
    )


# Special accuracy function for clustering.
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. This acc function calculates clustering accuracy, which is a measure of how well a clustering algorithm assigns
    data points to clusters compared to the ground truth labels. It does so by finding the best matching between the predicted cluster assignments
    (y_pred) and the true labels (y_true), and then calculating the accuracy based on this matching. Here's how the function works:
    1. It first ensures that both y_true and y_pred are integer arrays.
    2. It asserts that the sizes of y_true and y_pred are equal.
    3. It initializes a confusion matrix w where each entry w[i, j] denotes the number of data points that belong to cluster i according to y_pred
       and to cluster j according to y_true.
    4. It computes the optimal one-to-one matching between the predicted clusters and the true clusters using the Hungarian algorithm, implemented
       in linear_assignment from scikit-learn.
    5. It calculates the accuracy by summing the values of the confusion matrix corresponding to the optimal matching and dividing by the total
       number of data points.
    6. This accuracy measure is specifically tailored for clustering algorithms, where there is no inherent ordering of cluster labels. It
       accounts for the fact that clusters may be assigned arbitrary labels by the algorithm, and it finds the best matching between these labels and the true labels to compute accuracy.

    Traditional accuracy metrics assume a one-to-one correspondence between predicted and true labels, which is not the case in clustering.
    Instead, the acc function finds the best matching between clusters and true labels, providing a more appropriate measure of clustering
    performance. We need to find the best matching between the predicted clusters and the true clusters and then evaluate the performance based on
    this matching. This is what the acc function does by finding the optimal matching using the Hungarian algorithm and computing the accuracy
    based on this matching.
    Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    Return
        accuracy, in [0,1]
    """
    # Ensure y_true and y_pred are NumPy arrays of integers.
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    assert y_pred.size == y_true.size

    # Determine the size of the confusion matrix and initialize it.
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    # Fill the confusion matrix.
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Perform the assignment using the Hungarian algorithm.
    ind = linear_sum_assignment(w.max() - w)

    # Return accuracy.
    return sum(w[ind[0], ind[1]]) * 1.0 / y_pred.size


def calc_stats(act_mat, true_labels, kmeans_labels, debug=False):
    # Silhouette score.
    Silhouette = silhouette_score(act_mat, kmeans_labels, metric="euclidean")

    # Calinski-Harabasz index.
    Calinski = calinski_harabasz_score(act_mat, kmeans_labels)

    # Special accuracy function.
    Special = acc(true_labels, kmeans_labels)

    # Completeness score.
    Completeness = completeness_score(true_labels, kmeans_labels)

    # Homogeneity_score.
    Homogeneity = homogeneity_score(true_labels, kmeans_labels)

    # Adjusted_mutual_info_scor
    Adjusted = adjusted_mutual_info_score(true_labels, kmeans_labels)

    if debug:
        print(f"Silhouette Score: {Silhouette}")
        print(f"Calinski-Harabasz Index: {Calinski}")
        print(f"Special accuracy: {Special}")
        print(f"completeness score: {Completeness}")
        print(f"homogeneity_score: {Homogeneity}")
        print(f"adjusted_mutual_info_score: {Adjusted}\n")

    return Silhouette, Calinski, Special, Completeness, Homogeneity, Adjusted


def decoupler_feature_importance(scores, reactome, pvals=None):
    print(f"Shape of data: {scores.shape}")
    num_genesets_returned = scores.shape[1]

    # Extract the processed gene sets from the scores DataFrame.
    processed_gene_sets = scores.columns

    # Store the normalized enrichment scores (NES) in the result matrix.
    gsea_results_matrix = scores.T.values

    if pvals is not None:
        # Combine p-values across all cells using Fisher's method.
        combined_pvals = np.zeros(num_genesets_returned)
        for j in range(num_genesets_returned):
            # Replace zero values with a very small number (e.g., 1e-10).
            pvals_col = np.where(pvals.iloc[:, j] == 0, 1e-10, pvals.iloc[:, j])
            combined_pvals[j] = combine_pvalues(pvals_col)[1]

        # Create a DataFrame to store the combined p-values and sort by p-value.
        gene_sets = reactome["geneset"].unique()
        combined_pvals_df = pd.DataFrame(
            {"geneset": processed_gene_sets, "combined_pval": combined_pvals}
        )
        combined_pvals_df = combined_pvals_df.sort_values("combined_pval")

        # Replace zero p-values with a very small number (e.g., 1e-10).
        combined_pvals_df["combined_pval"] = combined_pvals_df["combined_pval"].replace(
            0, 1e-10
        )
        # Replace p-values of 1 with a value slightly less than 1 (e.g., 1 - 1e-10).
        combined_pvals_df["combined_pval"] = combined_pvals_df["combined_pval"].replace(
            1, 1 - 1e-10
        )
        # Set a threshold for p-values.
        threshold = 1e-20
        combined_pvals_df["adjusted_pval"] = combined_pvals_df["combined_pval"].apply(
            lambda x: max(x, threshold)
        )
        combined_pvals_df["pval_rank"] = combined_pvals_df["combined_pval"].rank()

        # Plot the distribution of p-values for the gene sets.
        # plt.figure(figsize=(10, 6))
        # plt.hist(combined_pvals_df['combined_pval'], bins=50, edgecolor='k')
        # plt.yscale('log')
        # plt.xlabel('Gene sets p-value')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Gene Sets p-values')
        # plt.show()
    else:
        combined_pvals_df = pd.DataFrame(
            {"geneset": processed_gene_sets, "combined_pval": 0, "pval_rank": 0}
        )

    # Use additional metric.
    combined_pvals_df["enrichment_score"] = scores.mean().values
    # Rank by combined p-value and enrichment score.
    combined_pvals_df["enrichment_rank"] = combined_pvals_df["enrichment_score"].rank(
        ascending=False
    )
    # Create a composite score.
    combined_pvals_df["composite_score"] = (
        combined_pvals_df["pval_rank"] + combined_pvals_df["enrichment_rank"]
    )
    # Sort by composite score
    combined_pvals_df = combined_pvals_df.sort_values("composite_score")

    # Display the top 20 gene sets based on combined p-values.
    top_gene_sets = combined_pvals_df.head(20).copy()
    # top_gene_sets['-log10(combined_pval)'] = -np.log10(top_gene_sets['combined_pval'])
    # print(top_gene_sets.head())

    # Plot the top 20 gene sets with regards to combined ranks.
    plt.figure(figsize=(10, 8))
    sns.barplot(x="combined_pval", y="geneset", data=top_gene_sets)
    plt.xlabel("Gene sets p-value)")
    plt.ylabel("Gene Set")
    plt.title("Top 20 Gene Sets by p-value")
    plt.show()

    return top_gene_sets


def feature_importance(scores_df, true_labels):
    # Loop through the features, omitting one at a time and calculating the acc score.
    # Dictionary to store the accuracy results.
    acc_results = {}

    # Loop through each gene set.
    for pathway in tqdm(scores_df.columns):
        # Omit the current pathway.
        pathsingle_results_matrix_omitted = scores_df.drop(columns=[pathway]).values

        selector = VarianceThreshold(threshold=0.01)
        pathsingle_results_matrix_omitted = selector.fit_transform(
            pathsingle_results_matrix_omitted
        )

        # Scale the data.
        scaler = MinMaxScaler()
        pathsingle_results_matrix_omitted = scaler.fit_transform(
            pathsingle_results_matrix_omitted
        )

        # Perform KMeans clustering.
        num_clusters = len(np.unique(true_labels))
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(
            pathsingle_results_matrix_omitted
        )
        cluster_assignments = kmeans.labels_

        # Evaluate clustering results using acc metric.
        clustering_accuracy = acc(true_labels, cluster_assignments)

        # Store the accuracy result.
        acc_results[pathway] = clustering_accuracy

    # Sort the gene sets according to the acc result (the best gene set is the one that reduced the results the most).
    sorted_genesets = sorted(acc_results.items(), key=lambda x: x[1])

    # Display the sorted gene sets.
    print("Top 40 gene sets by clustering accuracy:")
    for pathway, accuracy in sorted_genesets[:40]:
        print(f"Pathway set: {pathway}")


def cluster_with_kmeans(
    method_name, results_matrix, adata, n_clusters=10, draw_umap=False
):
    # Perform KMeans clustering on the UMAP coordinates.
    kmeans = KMeans(n_clusters).fit(results_matrix)
    if draw_umap:
        # Perform UMAP on the results matrix.
        umap_model = umap.UMAP(n_neighbors=15)
        umap_coords = umap_model.fit_transform(results_matrix)

        # Add the UMAP coordinates and clustering results to the AnnData object.
        adata.obsm[f"umap_{method_name}"] = umap_coords
        adata.obs[f"kmeans_{method_name}"] = pd.Categorical(kmeans.labels_)

        # Plot the UMAP with KMeans clustering results using Scanpy.
        # sc.pl.umap(adata, color=['kmeans1'], title='KMeans Clustering on Simulated GSEA NES Values', obsm_key='umap_gsea')
        # sc.pl.umap(adata, color=["labels"], title="UMAP of True Labels", obsm_key='umap_gsea')
        sc.pl.embedding(
            adata,
            basis=f"umap_{method_name}",
            color=[f"kmeans_{method_name}"],
            title=f"KMeans Clustering on {method_name} Values",
        )
        sc.pl.embedding(
            adata,
            basis=f"umap_{method_name}",
            color=["labels"],
            title="UMAP of True Labels",
        )
    return kmeans


def gaussian_scaling(p, q, sigma=0.5):
    """Calculate the scaled activity of inputs (p) of an interaction by the outputs (q).
    1. Distance term: (p - q)**2 : Measures squared difference between p and q. Always positive. Larger when p and q are far apart.
    2. Scaling factor: exp(-(p-q)²/(2σ²)) : Returns 1.0 when p=q. Decreases exponentially as |p-q| increases. σ controls how quickly scaling drops off.
    3. Final value: p * scaling : When p=q: returns p (scaling=1). When p≠q: reduces p based on distance. Never increases above p.
    """
    scaling = np.exp(-((p - q) ** 2) / (2 * sigma**2))
    return p * scaling


def consistency_scaling(p, q):
    """Calculate the scaled activity of inputs (p) of an interaction by the outputs (q)."""
    return p * q - (1 - p) * (1 - q)


def proximity_scaling(p, q):
    """Calculate the scaled activity of inputs (p) of an interaction by the outputs (q)."""
    proximity = 1 - abs(p - q)  # How close p and q are.
    return p * proximity


def calculate_sparsity(adata):
    """Calculate data sparsity percentage."""
    if scipy.sparse.issparse(adata.X):
        sparsity2 = (1.0 - (adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]))) * 100
    else:
        sparsity2 = np.sum(adata.X == 0) / adata.X.size * 100

    print(f"nnz-based sparsity: {sparsity2:.2f}%.")
    return sparsity2


def choose_scaling_method(sparsity):
    """Choose scaling method based on data sparsity.
    adata.X.nnz gives number of non-zero elements.
    adata.X.shape[0] * adata.X.shape[1] gives total matrix size.
    """
    return proximity_scaling if sparsity >= 40 else gaussian_scaling
