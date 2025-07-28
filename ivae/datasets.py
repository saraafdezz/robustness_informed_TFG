"""
@author: Carlos Loucera
"""

from pathlib import Path

import scanpy as sc


def load_kang(data_folder=".", normalize=True, n_genes=None, return_path=False):
    data_folder = Path(data_folder)
    out_path = data_folder.joinpath("kang_counts_25k_for_splitting.h5ad")
    adata = sc.read(
        data_folder.joinpath("kang_counts_25k.h5ad"),
        backup_url="https://figshare.com/ndownloader/files/34464122",
        cache=True,
    )

    adata.obs["label"] = adata.obs["label"].replace(
        {"ctrl": "control", "stim": "stimulated"}
    )
    adata.obs = adata.obs.rename(columns={"label": "condition"})

    # Storing the counts for later use
    adata.layers["counts"] = adata.X.copy()

    # Normalizing
    if normalize:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    if n_genes is not None:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_genes, flavor="seurat_v3", subset=False, layer="counts"
        )

        adata = adata[:, adata.var.highly_variable]

    sc.write(filename=out_path, adata=adata)

    if return_path:
        return out_path
    else:
        return adata
