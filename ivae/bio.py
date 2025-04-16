""" "
@author: Carlos Loucera
"""

from collections import namedtuple
from dataclasses import dataclass
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split

from ivae.utils import get_resource_path


def get_reactome_adj(pth=None):
    """
    Parse a gmt file to a decoupler pathway dataframe.
    """
    # Adapted from Single-cell best practices book

    if pth is None:
        pth = get_resource_path("c2.cp.reactome.v7.5.1.symbols.gmt")

    pathways = {}

    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    reactome = pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )

    reactome = (
        reactome.drop_duplicates()
        .assign(belongs_to=1)
        .pivot(columns="geneset", index="genesymbol", values="belongs_to")
        .fillna(0)
    )

    return reactome


def read_circuit_names():
    path = get_resource_path("circuit_names.tsv.tar.xz")
    circuit_names = pd.read_csv(path, sep="\t")
    circuit_names.hipathia_id = circuit_names.hipathia_id.str.replace(" ", ".")
    circuit_names["effector"] = circuit_names.name.str.split(": ").str[-1]
    # circuit_names = circuit_names.set_index("hipathia_id")

    return circuit_names


def read_circuit_adj(with_effectors=False, gene_list=None):
    path = get_resource_path("pbk_circuit_hsa_sig.tar.xz")
    adj = pd.read_csv(path, sep=",", index_col=0)
    adj.index = adj.index.str.upper()
    if not with_effectors:
        adj = 1 * (adj > 0)

    if gene_list is not None:
        adj = adj.loc[adj.index.intersection(gene_list), :]

    adj.columns = adj.columns.str.replace(" ", ".")
    to_remove = adj.columns.str.contains("hsa04218")
    adj = adj.loc[:, ~to_remove]

    return adj


def build_pathway_adj_from_circuit_adj(circuit_adj):
    tmp_adj = circuit_adj.T
    tmp_adj.index.name = "circuit"
    tmp_adj = tmp_adj.reset_index()
    tmp_adj["pathway"] = tmp_adj.circuit.str.split("-").str[1]
    tmp_adj = tmp_adj.drop("circuit", axis=1)
    adj = 1 * tmp_adj.groupby("pathway").any()

    return adj


def build_circuit_pathway_adj(circuit_adj, pathway_adj):
    return (1 * (pathway_adj.dot(circuit_adj) > 0)).T


def get_adj_matrices(gene_list=None):
    circuit_adj = read_circuit_adj(with_effectors=False, gene_list=gene_list)
    pathway_adj = build_pathway_adj_from_circuit_adj(circuit_adj)
    circuit_to_pathway = build_circuit_pathway_adj(circuit_adj, pathway_adj)

    return circuit_adj, circuit_to_pathway


def get_random_adj(frac, shape, size, seed, index):
    np.random.RandomState(seed)
    random_layer = np.random.binomial(1, frac, size=size)
    random_layer = random_layer.reshape(shape)
    random_layer_names = [f"rand-{icol:02d}" for icol in range(random_layer.shape[1])]
    random_layer = pd.DataFrame(random_layer, index=index, columns=random_layer_names)
    random_layer = random_layer.loc[random_layer.any(axis=1), :]
    random_layer = random_layer.loc[:, random_layer.any(axis=0)]
    random_layer_names = random_layer.columns

    return random_layer, random_layer_names


def build_hipathia_renamers():
    circuit_names = read_circuit_names()
    circuit_names = circuit_names.rename(
        columns={"name": "circuit_name", "hipathia_id": "circuit_id"}
    )
    circuit_names["pathway_id"] = circuit_names["circuit_id"].str.split("-").str[1]
    circuit_names["pathway_name"] = circuit_names["circuit_name"].str.split(":").str[0]
    circuit_renamer = circuit_names.set_index("circuit_id")["circuit_name"].to_dict()
    pathway_renamer = circuit_names.set_index("pathway_id")["pathway_name"].to_dict()
    circuit_to_effector = (
        circuit_names.set_index("circuit_name")["effector"].str.strip().to_dict()
    )

    return circuit_renamer, pathway_renamer, circuit_to_effector


def sync_gexp_adj(gexp, adj):
    gene_list = adj.index.intersection(gexp.columns)
    gexp = gexp.loc[:, gene_list]
    adj = adj.loc[gene_list, :]

    return gexp, adj


########################################################################################################################


# Funciones auxiliares sara
def get_importances(
    data, abs=False
):  # Calcula la importancia de los datos, si no hay abs, lo pone a false
    if abs:  # Comprueba si debe tomar valor absoluto o no
        return np.abs(data).mean(axis=0)
    else:
        return data.mean(
            axis=0
        )  # Lo que devuelve es un vector de una fila unica con la media de cada columna


def get_activations(
    act_model, layer_id, data
):  # Obtiene las activaciones (?) de una capa específica de los datos
    data_encoded = act_model.predict(data)[
        layer_id
    ]  # act_model es el modelo de la rn que se esta usando para predecir
    return data_encoded  # Pasa los datos de entrada a través del modelo y obtiene lass predicciones de la capa indicada


# Divide el conjunto de datos en train, val y test
def train_val_test_split(features, val_size, test_size, stratify, seed):
    train_size = 1 - (val_size + test_size)  # Tamaño del train

    x_train, x_test, y_train, y_test = train_test_split(  # Divide train y tv est
        features,
        stratify,
        train_size=train_size,
        stratify=stratify,  # Mantiene las proporciones (?)
        random_state=seed,
    )

    x_val, x_test = train_test_split(  # Extrae del test el val
        x_test,
        test_size=test_size / (test_size + val_size),
        stratify=y_test,
        random_state=seed,
    )

    x_train = x_train.astype("float32")
    x_val = x_val.astype("float32")
    x_test = x_test.astype("float32")

    return x_train, x_val, x_test


@dataclass
class InformedModelConfig:
    model_kind: str
    frac: float
    n_encoding_layers: int
    adj_name: list
    adj_activ: list
    input_genes: list
    layer_entity_names: list
    model_layer: list


def build_model_config(data, model_kind, frac=None):
    if isinstance(data, sc.AnnData):
        x_trans = data.to_df()
    else:
        x_trans = data
    if model_kind == "ivae_kegg":
        circuit_adj, circuit_to_pathway_adj = get_adj_matrices(
            gene_list=x_trans.columns.to_list()
        )
        circuit_renamer, pathway_renamer, circuit_to_effector = (
            build_hipathia_renamers()
        )
        kegg_circuit_names = circuit_adj.rename(columns=circuit_renamer).columns
        kegg_pathway_names = circuit_to_pathway_adj.rename(
            columns=pathway_renamer
        ).columns
        circuit_adj.head()
        n_encoding_layers = 3
        x_trans, circuit_adj = sync_gexp_adj(gexp=x_trans, adj=circuit_adj)
        model_layer = [circuit_adj, circuit_to_pathway_adj]
        adj_name = ["circuits", "pathways"]
        layer_entity_names = [kegg_circuit_names, kegg_pathway_names]
        adj_activ = ["tanh", "tanh"]
        input_genes = x_trans.columns.to_list()

    elif model_kind == "ivae_reactome":
        reactome = get_reactome_adj()
        reactome_pathway_names = reactome.columns
        n_encoding_layers = 2
        x_trans, reactome = sync_gexp_adj(x_trans, reactome)
        model_layer = [reactome]
        layer_entity_names = [reactome_pathway_names]
        adj_name = ["pathways"]
        adj_activ = ["tanh"]
        input_genes = x_trans.columns.to_list()

    elif "ivae_random" in model_kind:
        reactome = get_reactome_adj()
        random_layer, random_layer_names = get_random_adj(
            frac, shape=reactome.shape, size=reactome.size, index=reactome.index, seed=0
        )
        n_encoding_layers = 2
        x_trans, random_layer = sync_gexp_adj(x_trans, random_layer)
        model_layer = [random_layer]
        layer_entity_names = [random_layer_names]
        adj_name = [f"density-{frac}"]
        adj_activ = ["tanh"]
        input_genes = x_trans.columns.to_list()

    else:
        raise NotImplementedError("Model not yet implemented.")

    model_config = InformedModelConfig(
        model_kind=model_kind,
        frac=frac,
        n_encoding_layers=n_encoding_layers,
        adj_name=adj_name,
        adj_activ=adj_activ,
        input_genes=input_genes,
        layer_entity_names=layer_entity_names,
        model_layer=model_layer,
    )

    return model_config


@dataclass
class IvaeResults:
    config: InformedModelConfig
    history: Dict[str, Any]
    eval: pd.DataFrame
    encodings: list


ModelFamilyResults = namedtuple(
    "ModelFamilyResults",
    [
        "model_kind",
        "results",
    ],
)


@dataclass
class ClusteringResults:
    config: InformedModelConfig
    clust_scores: list
