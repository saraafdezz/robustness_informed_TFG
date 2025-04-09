# %%
# https://www.sc-best-practices.org/conditions/gsea_pathway.html#id380
# Kang HM, Subramaniam M, Targ S, et al. Multiplexed droplet single-cell RNA-sequencing using natural genetic variation
#   Nat Biotechnol. 2020 Nov;38(11):1356]. Nat Biotechnol. 2018;36(1):89-94. doi:10.1038/nbt.4042

import argparse
from pathlib import Path
import time

import dotenv
import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
from keras import callbacks
from keras.models import Model
from sklearn.preprocessing import minmax_scale

from isrobust_TFG.bio import (
    build_hipathia_renamers,
    get_activations,
    get_adj_matrices,
    get_random_adj,
    get_reactome_adj,
    sync_gexp_adj,
    train_val_test_split,
)
from isrobust_TFG.CI_VAE_CLASS import InformedVAE
from isrobust_TFG.datasets import load_kang

# args = sys.argv

# import os
# import torch

# # Leer la GPU asignada por Snakemake
# gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


# # Asegurar que PyTorch usa la GPU correcta
# torch.cuda.set_device(int(gpu_id))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Usando GPU {gpu_id}")

if __name__ == "__main__":
    tf.keras.mixed_precision.set_global_policy('float32')

    parser = argparse.ArgumentParser(description="Train with a specific model")
    parser.add_argument(
        "--model_kind",
        type=str,
        help="Type of model: ivae_kegg, ivae_reactome or ivae_random",
    )
    #     parser.add_argument("--debug", type=int, help="True to debug with less epochs: Insert 0 or 1")
    parser.add_argument(
        "--frac", type=float, default=1, help="Distribution of random layer (if needed)"
    )
    parser.add_argument("--seed", type=int, default=50, help="Seed")
    parser.add_argument("--n_genes", type=int, default=None, help="Number of genes")
    parser.add_argument(
        "--results_path_model", type=str, default=".", help="Output folder"
    )
    parser.add_argument("--data_path", type=str, default=".", help="Data folder")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")
    args = parser.parse_args()
    model_kind = args.model_kind
    #     debug = args.debug
    frac = args.frac
    seed = args.seed
    n_genes = args.n_genes
    results_path_model = Path(args.results_path_model)
    data_path = Path(args.data_path)
    debug = args.debug

    print("+"*20, debug)
    print(model_kind, frac, n_genes)


    adata = load_kang(data_folder=data_path, normalize=True)  # Carga los datos de la db
    x_trans = adata.to_df()  # Pasa los datos a df para trabajar en Pandas

    tf.config.experimental.enable_op_determinism()  # Operaciones no dependan de la aleatoriedad -> reproducibilidad

    sc.set_figure_params(dpi=300, color_map="inferno")  # Parametros de visualización
    sc.settings.verbosity = 1  # Mensajes informativos básicos
    sc.logging.print_header()  # Imprime encabezado informativo al principio del script

    ####################################################################################
    # Define Mirrored Strategy
    ####################################################################################
    fis_gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in fis_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    gpus=tf.config.list_logical_devices('GPU')
    print("#"*50)
    print(f"[DEBUG][00-train.py] GPU's availables: {gpus}")
    
    strategy = tf.distribute.MirroredStrategy(
        gpus,
        cross_device_ops=tf.distribute.ReductionToOneDevice()
        )
    print("WAIT")
    time.sleep(5)
    print("#"*50)
    print(f"[INFO] Number of devices: {strategy.num_replicas_in_sync}")
    print("#"*50)
    time.sleep(5)

    if debug:
        N_EPOCHS = 2
    else:
        N_EPOCHS = 100

    # Construye matrices de adyacencia. Sincroniza las dimensiones. Interseccion entre los de nuestro dataset y los circuitos. Crea el VAE.
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
        adj_name = ["circuit_adj", "circuit_to_pathway_adj"]
        adj_activ = ["tanh", "tanh"]
    elif model_kind == "ivae_reactome":
        reactome = get_reactome_adj()
        reactome_pathway_names = reactome.columns
        n_encoding_layers = 2
        x_trans, reactome = sync_gexp_adj(x_trans, reactome)
        model_layer = reactome
        adj_name = "reactome"
        adj_activ = "tanh"

    elif "ivae_random" in model_kind:
        reactome = get_reactome_adj()
        state = np.random.get_state()
        random_layer, random_layer_names = get_random_adj(
            frac, shape=reactome.shape, size=reactome.size, index=reactome.index, seed=0
        )
        np.random.set_state(state)
        n_encoding_layers = 2
        x_trans, random_layer = sync_gexp_adj(x_trans, random_layer)
        model_layer = random_layer
        adj_name = "random"
        adj_activ = "tanh"

    else:
        raise NotImplementedError("Model not yet implemented.")


    results_path_model.mkdir(exist_ok=True, parents=True)
    print(f"{results_path_model}")
    #     results_path_model_seed = results_path_model.joinpath("seed_" + str(seed))
    #     results_path_model_seed.mkdir(exist_ok=True, parents=True)
    #     print(f"{results_path_model_seed}")

    obs = adata.obs.copy()  # Ignorar

    # Separa en train, val y test los datos de x_trans
    x_train, x_val, x_test = train_val_test_split(
        x_trans.apply(minmax_scale),  # Para que los datos esten en un rango similar
        val_size=0.20,
        test_size=0.20,
        stratify=obs["cell_type"].astype(str) + obs["condition"].astype(str),
        seed=seed,
    )

    print("#"*50)
    print(f"[INFO] Shape: {x_train.shape}")
    

    # Construye y entrena el vae
    with strategy.scope():
        vae = InformedVAE(
            adjacency_matrices=model_layer,
            adjacency_names=adj_name,
            adjacency_activation=adj_activ,
            seed=seed,
            learning_rate=1e-6
        )
        vae._build_vae()

    ###################################################################################
    n_gpus = len(gpus)
    batch_size = 150
    steps_per_epoch = x_train.shape[0]//batch_size
    validation_steps = x_test.shape[0]//batch_size
    ####################################################################################

    callback = callbacks.EarlyStopping(  # Detiene el entrenamiento si no hay mejora
        monitor="val_loss",
        min_delta=1e-1,
        patience=100,
        verbose=0,
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,x_train))
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).repeat()

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    tf.debugging.enable_check_numerics()
    history = vae.fit(  # Entrena
        train_dataset,
        # x_train,
        # x_train,
        shuffle=True,
        verbose=1,
        epochs=3,
        # batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        callbacks=[callback],
        validation_data=val_dataset,
        validation_steps=validation_steps
    )
    
    if False:
        encoder = vae.encoder
        decoder = vae.decoder

        # Evaluacion del modelo
        evaluation = {}
        evaluation["train"] = vae.evaluate(
            x_train, vae.predict(x_train), verbose=0, return_dict=True
        )
        evaluation["val"] = vae.evaluate(
            x_val, vae.predict(x_val), verbose=0, return_dict=True
        )
        evaluation["test"] = vae.evaluate(
            x_test, vae.predict(x_test), verbose=0, return_dict=True
        )

        # Formateo de la evaluacion
        pd.DataFrame.from_dict(evaluation).reset_index(names="metric").assign(
            seed=seed
        ).melt(
            id_vars=["seed", "metric"],
            value_vars=["train", "val", "test"],
            var_name="split",
            value_name="score",
        ).assign(model=model_kind).to_pickle(
            results_path_model.joinpath(f"metrics-seed-{seed:02d}.pkl")
        )

        layer_outputs = [layer.output for layer in encoder.layers]
        activation_model = Model(
            inputs=encoder.input, outputs=layer_outputs
        )  # Crea nuevo modelo de activacion

        # only analyze informed and funnel layers
        # Extrae procesa y guarda las activaciones de las vias
        for layer_id in range(1, len(layer_outputs)):
            # Nombre de las cols y los layers segun el modelo
            if model_kind == "ivae_kegg":
                if layer_id == 1:
                    colnames = kegg_circuit_names
                    layer_name = "circuits"
                elif layer_id == 2:
                    colnames = kegg_pathway_names
                    layer_name = "pathways"
                elif layer_id == (len(layer_outputs) - 1):
                    n_latents = len(kegg_pathway_names) // 2
                    colnames = [f"latent_{i:02d}" for i in range(n_latents)]
                    layer_name = "funnel"
                else:
                    continue
            elif model_kind == "ivae_reactome":
                if layer_id == 1:
                    colnames = reactome_pathway_names
                    layer_name = "pathways"
                elif layer_id == (len(layer_outputs) - 1):
                    n_latents = len(reactome_pathway_names) // 2
                    colnames = [f"latent_{i:02d}" for i in range(n_latents)]
                    layer_name == "funnel"
                else:
                    continue
            elif "ivae_random" in model_kind:
                if layer_id == 1:
                    colnames = random_layer_names
                    layer_name = "pathways"
                elif layer_id == (len(layer_outputs) - 1):
                    n_latents = len(random_layer_names) // 2
                    colnames = [f"latent_{i:02d}" for i in range(n_latents)]
                    layer_name == "funnel"
                else:
                    continue
            else:
                raise NotImplementedError("Model not yet implemented.")

            print(f"encoding layer {layer_id}")

            # Obtencion de las activaciones de la capa
            encodings = get_activations(
                act_model=activation_model,
                layer_id=layer_id,
                data=x_trans.apply(minmax_scale),
            )

            encodings = pd.DataFrame(encodings, index=x_trans.index, columns=colnames)

            # Nuevas columnas
            encodings["split"] = "train"
            encodings.loc[x_val.index, "split"] = "val"
            encodings.loc[x_test.index, "split"] = "test"
            encodings["layer"] = layer_name
            encodings["seed"] = seed
            encodings["model"] = model_kind

            # Fusiona con la muestra
            encodings = encodings.merge(
                obs[["cell_type", "condition"]],
                how="left",
                left_index=True,
                right_index=True,
            )
            # Guarda los resultados
            print(f"{results_path_model}")
            encodings.to_pickle(
                results_path_model.joinpath(
                    f"encodings_layer-{layer_id:02d}_seed-{seed:02d}.pkl"
                )
            )
            print(f" {results_path_model}")
