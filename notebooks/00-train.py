# %%
# https://www.sc-best-practices.org/conditions/gsea_pathway.html#id380
# Kang HM, Subramaniam M, Targ S, et al. Multiplexed droplet single-cell RNA-sequencing using natural genetic variation
#   Nat Biotechnol. 2020 Nov;38(11):1356]. Nat Biotechnol. 2018;36(1):89-94. doi:10.1038/nbt.4042

# %%


# import sys
# from pathlib import Path

# import dotenv
# import numpy as np
# import pandas as pd
# import scanpy as sc
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import minmax_scale
# from tensorflow.keras import callbacks
# from tensorflow.keras.models import Model

# from isrobust.bio import (
#     build_hipathia_renamers,
#     get_adj_matrices,
#     get_random_adj,
#     get_reactome_adj,
#     sync_gexp_adj,
# )
# from isrobust.datasets import load_kang
# from isrobust.models import build_kegg_vae, build_reactome_vae
# from isrobust.utils import set_all_seeds

import sys
import dotenv
import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf

from pathlib import Path

from keras import callbacks
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from isrobust_TFG.bio import (
    build_hipathia_renamers,
    get_adj_matrices,
    get_random_adj,
    get_reactome_adj,
    sync_gexp_adj,
)
from isrobust_TFG.CI_VAE_CLASS import InformedVAE
from isrobust_TFG.datasets import load_kang
from isrobust_TFG.utils import set_all_seeds


def get_importances(data, abs=False): # Calcula la importancia de los datos, si no hay abs, lo pone a false
    if abs:                              # Comprueba si debe tomar valor absoluto o no
        return np.abs(data).mean(axis=0) 
    else:
        return data.mean(axis=0) # Lo que devuelve es un vector de una fila unica con la media de cada columna


def get_activations(act_model, layer_id, data):  # Obtiene las activaciones (?) de una capa específica de los datos
    data_encoded = act_model.predict(data)[layer_id] # act_model es el modelo de la rn que se esta usando para predecir
    return data_encoded                              # Pasa los datos de entrada a través del modelo y obtiene lass predicciones de la capa indicada


# Divide el conjunto de datos en train, val y test
def train_val_test_split(features, val_size, test_size, stratify, seed): 
    train_size = 1 - (val_size + test_size) # Tamaño del train

    x_train, x_test, y_train, y_test = train_test_split( # Divide train y test
        features,
        stratify,   
        train_size=train_size,
        stratify=stratify,  # Mantiene las proporciones (?)
        random_state=seed,
    )

    x_val, x_test = train_test_split( # Extrae del test el val
        x_test,
        test_size=test_size / (test_size + val_size),
        stratify=y_test, 
        random_state=seed,
    )

    x_train = x_train.astype("float32")
    x_val = x_val.astype("float32")
    x_test = x_test.astype("float32")

    return x_train, x_val, x_test


args = sys.argv

# LLamarlo con argparse
if len(args) == 5:        # Donde se usa?????
    model_kind, debug, frac, seed = args[1:]
    frac = float(frac)
    print(frac)
else:
    model_kind, debug, seed = args[1:]
    frac = 1

model_kind = str(model_kind)
debug = bool(int(debug))
seed = int(seed)

print(model_kind, debug, seed)

# Ruta del proyecto
project_path = Path(dotenv.find_dotenv()).parent 
data_path = project_path.joinpath("data") 
data_path.mkdir(exist_ok=True, parents=True)

set_all_seeds(seed=seed)

tf.config.experimental.enable_op_determinism() # Operaciones no dependan de la aleatoriedad -> reproducibilidad

sc.set_figure_params(dpi=300, color_map="inferno") # Parametros de visualización 
sc.settings.verbosity = 1 # Mensajes informativos básicos
sc.logging.print_header() # Imprime encabezado informativo al principio del script

config = dotenv.dotenv_values()

print("Loaded config:", config)
debug = bool(int(config["DEBUG"])) 

# Para almacenar resultados del proyecto
results_path = Path(config["RESULTS_FOLDER"])
print("RESULTS_FOLDER")
results_path.mkdir(exist_ok=True, parents=True)
figs_path = results_path.joinpath("figs")
figs_path.mkdir(exist_ok=True, parents=True)
tables_path = results_path.joinpath("tables")
tables_path.mkdir(exist_ok=True, parents=True)


# Configura parámetros del modelo -> argparse. Se puede dejar
if debug:
    N_EPOCHS = 2
else:
    N_EPOCHS = 100

if model_kind == "ivae_kegg":
    n_encoding_layers = 3
elif model_kind == "ivae_reactome":
    n_encoding_layers = 2
elif "ivae_random" in model_kind:
    n_encoding_layers = 2
else:
    raise NotImplementedError(f"{model_kind} not implemented yet.")

# poner como parametro de entrada 
# %%
if "ivae_random" in model_kind:
    n_genes = None
else:
    n_genes = None
    
adata = load_kang(data_folder=data_path, normalize=True, n_genes=n_genes) # Carga los datos de la db

# %%
x_trans = adata.to_df() # Pasa los datos a df para trabajar en Pandas

# FIltrar por el tipo de modelo y crear una u otra segun el tipo
# Construcción de las matrices de adyacencia de KEGG
circuit_adj, circuit_to_pathway_adj = get_adj_matrices(
    gene_list=x_trans.columns.to_list()  
)

# Remane
circuit_renamer, pathway_renamer, circuit_to_effector = build_hipathia_renamers()

kegg_circuit_names = circuit_adj.rename(columns=circuit_renamer).columns
kegg_pathway_names = circuit_to_pathway_adj.rename(columns=pathway_renamer).columns

circuit_adj.head() # Muestra las primeras filas

# Obtener los nombres de las vias de reactome
reactome = get_reactome_adj()
reactome_pathway_names = reactome.columns

# Para mantener la consistencia establece la config de los random
state = np.random.get_state()

random_layer, random_layer_names = get_random_adj(
    frac, shape=reactome.shape, size=reactome.size, index=reactome.index, seed=0
)

np.random.set_state(state)

# Sincroniza las dimensiones. Interseccion entre los de nuestro dataset y los circuitos.
if model_kind == "ivae_kegg":
    x_trans, circuit_adj = sync_gexp_adj(gexp=x_trans, adj=circuit_adj)
elif model_kind == "ivae_reactome":
    x_trans, reactome = sync_gexp_adj(x_trans, reactome)
elif "ivae_random" in model_kind:
    x_trans, random_layer = sync_gexp_adj(x_trans, random_layer)


# Path para guardar los resultados
results_path_model = results_path.joinpath(model_kind)
results_path_model.mkdir(exist_ok=True, parents=True)

obs = adata.obs.copy() # Ignorar

# Separa en train, val y test los datos de x_trans 
x_train, x_val, x_test = train_val_test_split( 
    x_trans.apply(minmax_scale), # Para que los datos esten en un rango similar
    val_size=0.20,
    test_size=0.20,
    stratify=obs["cell_type"].astype(str) + obs["condition"].astype(str),
    seed=seed,
)

# Crea el vae segun el tipo de modelo -> añadir a los if
if model_kind == "ivae_kegg":
    #     vae, encoder, decoder = build_kegg_vae(
    #         circuits=circuit_adj, pathways=circuit_to_pathway_adj, seed=seed
    #     )
    vae = InformedVAE(
        adjacency_matrices=[circuit_adj, circuit_to_pathway_adj],
        adjacency_names=["circuit_adj", "circuit_to_pathway_adj"],
        adjacency_activation=["tanh", "tanh"],
        seed=seed,
    )

elif model_kind == "ivae_reactome":
    #     vae, encoder, decoder = build_reactome_vae(reactome, seed=seed)
    vae = InformedVAE(
        adjacency_matrices=reactome,
        adjacency_names="reactome",
        adjacency_activation="tanh",
        seed=seed,
    )
elif "ivae_random" in model_kind:
    #     vae, encoder, decoder = build_reactome_vae(random_layer, seed=seed)
    vae = InformedVAE(
        adjacency_matrices=random_layer,
        adjacency_names="random",
        adjacency_activation="tanh",
        seed=seed,
    )
else:
    raise NotImplementedError("Model not yet implemented.")

# Construye el vae
vae._build_vae()

# Entrenamiento del vae
batch_size = 32 

callback = callbacks.EarlyStopping( # Detiene el entrenamiento si no hay mejora
    monitor="val_loss",
    min_delta=1e-1,
    patience=100,
    verbose=0,
)

history = vae.fit( # Entrena
    x_train,
    x_train,
    shuffle=True,
    verbose=1,
    epochs=N_EPOCHS,
    batch_size=batch_size,
    callbacks=[callback],
    validation_data=(x_test, x_test),
)

encoder = vae.encoder
decoder = vae.decoder

# Evaluacion del modelo
evaluation = {}
evaluation["train"] = vae.evaluate(
    x_train, vae.predict(x_train), verbose=0, return_dict=True
)
evaluation["val"] = vae.evaluate(x_val, vae.predict(x_val), verbose=0, return_dict=True)
evaluation["test"] = vae.evaluate(
    x_test, vae.predict(x_test), verbose=0, return_dict=True
)

# Formateo de la evaluacion
pd.DataFrame.from_dict(evaluation).reset_index(names="metric").assign(seed=seed).melt(
    id_vars=["seed", "metric"],
    value_vars=["train", "val", "test"],
    var_name="split",
    value_name="score",
).assign(model=model_kind).to_pickle(
    results_path_model.joinpath(f"metrics-seed-{seed:02d}.pkl")
)

layer_outputs = [layer.output for layer in encoder.layers]
activation_model = Model(inputs=encoder.input, outputs=layer_outputs) # Crea nuevo modelo de activacion

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
    print("hola 1")
    
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