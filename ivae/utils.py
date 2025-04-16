import importlib.resources as pkg_resources
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def get_resource_path(fname):
    """Get path to pkg resources by filename.

    Returns
    -------
    pathlib.PosixPath
        Path to file.
    """
    with pkg_resources.path("ivae.resources", fname) as f:
        data_file_path = f

    return Path(data_file_path)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_scores(df, fname):
    df_to_print = df.copy().query("layer <= 2")

    df_to_print["Capa"] = "Pathways"

    mask = (df_to_print["layer"] == 1) & (df_to_print["model"] == "ivae_kegg")
    df_to_print.loc[mask, "Capa"] = "Circuitos"

    mask = (df_to_print["layer"] == 1) & (df_to_print["model"] == "ivae_random")
    df_to_print.loc[mask, "Capa"] = "RndInf"

    (
        df_to_print.rename(columns={"model": "Modelo", "split": "Partición"})
        .groupby(["Modelo", "Capa", "Partición"])["score"]
        .describe()
        .drop(["count", "min", "max"], axis=1)
        .to_latex(
            fname,
            bold_rows=True,
            escape=True,
        )
    )
