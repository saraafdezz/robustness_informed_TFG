import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources as impresources

import numpy as np
import pandas as pd

import pathsingle
from pathsingle.metrics import choose_scaling_method


# Function to determine if an interaction type is inhibitory.
def is_inhibitory(interaction_type):
    inhibitory_keywords = [
        "inhibition",
        "repression",
        "dissociation",
        "dephosphorylation",
        "ubiquitination",
    ]  #'missing interaction'
    return any(keyword in interaction_type for keyword in inhibitory_keywords)


def process_sample(args):
    """Process all pathways for a single sample"""
    (
        sample_idx,
        sample_data,
        sample_name,
        pathway_interactions,
        gene_to_index,
        scaling_func,
    ) = args
    # print(f"Processing sample {sample_idx}") # Debug
    pathway_activities = {}
    interaction_dict = {"sample_name": sample_name}

    # Process each pathway sequentially.
    for pathway, interactions in pathway_interactions.items():
        # print(f"[{sample_idx}] Procesando pathway: {pathway}")  # Debug
        try:
            pathway_activity, interaction_acts = process_pathway(
                (pathway, interactions, sample_data, gene_to_index, scaling_func)
            )
            pathway_activities[pathway] = pathway_activity
            interaction_dict.update(interaction_acts)
        except Exception as e:
            print(f"Error en pathway {pathway} (sample {sample_idx}): {e}")  # Debug
            continue

    return sample_idx, pathway_activities, interaction_dict


def process_pathway(args):
    """Calculate the activities of one pathway for a given sample."""
    pathway, interactions, gene_expression, gene_to_index, scaling_func = args
    pathway_activity = 0
    interactions_counter = 0
    interaction_activities = {}

    # Calculate activities for each interaction. interactions is a list of tuples: (['baiap2', 'wasf2', 'wasf3', 'wasf1'], 'activation', 'activation', ['cyp2b6', 'cyp2j2']).
    for interaction_idx, interaction in enumerate(interactions):
        interaction_activity = 0
        # Calculate input activity.
        for gene in interaction[0]:
            if gene in gene_to_index:
                interaction_activity += gene_expression[gene_to_index[gene]]

        # Calculate output activity.
        output_activity = 0
        for gene in interaction[2]:
            if gene in gene_to_index:
                output_activity += gene_expression[gene_to_index[gene]]
        output_activity = max(1e-10, output_activity)

        interaction_activity = scaling_func(interaction_activity, output_activity)
        if is_inhibitory(interaction[1]):
            interaction_activity = -interaction_activity

        pathway_activity += interaction_activity
        interactions_counter += 1

        # Add interaction activity to the dictionary.
        interaction_activities[f"interaction_{pathway}_{interaction_idx}"] = (
            interaction_activity
        )

    return pathway_activity / max(1, interactions_counter), interaction_activities


def calc_activity(
    adata,
    n_jobs=-1,
    sparsity=20,
    output_path="./activity.csv",
    interaction_path="./interaction_activity.csv",
):
    """Calculate the activity of pathways based on gene expression data."""
    # Load gene expression data.
    gene_expression_tensor = adata.X  # gene_expression_df.values (samples, genes)

    # Create a mapping of gene names to their indices in the gene_expression_tensor.
    gene_names = adata.var_names.str.lower()  # gene_expression_df.index
    gene_to_index = {gene: i for i, gene in enumerate(gene_names)}

    # Load and parse pathway relations.
    inp_file = (
        impresources.files(pathsingle)
        / "pathway_db"
        / "reactome"
        / "pathway_relations.zip"
    )
    # inp_file = impresources.files(pathsingle) / "data" / 'pathway_relations.csv'
    renamer = {
        "PATHWAY_NAMES": "pathway",
        "INTERACTION_TYPE": "interactiontype",
        "PARTICIPANT_A": "source",
        "PARTICIPANT_B": "target",
    }
    # pathway_relations = pd.read_csv(inp_file, sep=",")
    pathway_relations = pd.read_csv(inp_file, sep="\t")
    pathway_relations = pathway_relations.rename(columns=renamer)

    pathway_relations["source"] = (
        pathway_relations["source"].fillna("").astype(str).str.lower().str.split("*")
    )
    pathway_relations["target"] = (
        pathway_relations["target"].fillna("").astype(str).str.lower().str.split("*")
    )

    # Parse pathway relations to create a simplified data structure.
    pathway_interactions = {}
    for _, row in pathway_relations.iterrows():
        pathway = row["pathway"]
        sources = row["source"]
        targets = row["target"]
        inttype = row["interactiontype"]
        if pathway not in pathway_interactions:
            pathway_interactions[pathway] = []
        pathway_interactions[pathway].append(
            (sources, inttype, targets)
        )  # For example: (['baiap2', 'wasf2', 'wasf3', 'wasf1'], 'activation', 'activation', ['cyp2b6', 'cyp2j2']).

    # Initialize a dictionary to store pathway activities.
    pathway_activities = {pathway: [] for pathway in pathway_interactions.keys()}

    # Initialize a list to store dictionaries of interaction activities per sample.
    interaction_dicts = []
    n_samples = gene_expression_tensor.shape[0]
    ordered_results = [None] * n_samples  # Pre-allocate list for ordered results.
    scaling_func = choose_scaling_method(sparsity)

    # Process samples in parallel.
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Prepare arguments for all samples
        sample_args = [
            (
                idx,
                gene_expression_tensor[idx],
                adata.obs_names[idx],
                pathway_interactions,
                gene_to_index,
                scaling_func,
            )
            for idx in range(gene_expression_tensor.shape[0])
        ]

        # Submit all jobs
        futures = [executor.submit(process_sample, arg) for arg in sample_args]

        # Collect results maintaining order.
        for future in as_completed(futures):
            try:  # Debug
                idx, sample_pathway_activities, interaction_dict = future.result()
                ordered_results[idx] = (sample_pathway_activities, interaction_dict)
                print(
                    f"Processed sample {idx+1}/{gene_expression_tensor.shape[0]}",
                    end="\r",
                )
            except Exception as e:
                print(f"\n Error in future: {e}")
                traceback.print_exc()
    # Debug
    for i, result in enumerate(ordered_results):
        if result is None:
            print(f"Result at index {i} is None.")

    # Process results in correct order.
    print("Starting result aggregation...")  # Debug
    for idx, (sample_pathway_activities, interaction_dict) in enumerate(
        ordered_results
    ):
        # Store pathway activities.
        for pathway, activity in sample_pathway_activities.items():
            pathway_activities[pathway].append(activity)
        interaction_dicts.append(interaction_dict)
    print("Aggregated results.")  # Debug

    print("Building mean activity matrix...")  # Debug
    mean_activity_matrix = np.zeros(
        (gene_expression_tensor.shape[0], len(pathway_interactions))
    )  # (samples, pathways)

    for idx, (pathway_name, activities) in enumerate(pathway_activities.items()):
        if activities:
            mean_activity_matrix[:, idx] = activities
        else:
            print(f"No activities for pathway {pathway_name}")
    print("Built mean activity matrix.")  # Debug

    # Create the DataFrame for the activity matrix.
    activity_df = pd.DataFrame(
        mean_activity_matrix,
        index=adata.obs_names,
        columns=list(pathway_interactions.keys()),
    ).T

    # Save results to CSV.
    print("Saving activity CSV...")
    activity_df.to_csv(
        output_path
    )  # Not transposing to keep cells in rows and pathway names in columns
    print("Saved activity CSV.")

    # Convert the list of dictionaries to a DataFrame.
    interaction_activities = pd.DataFrame(interaction_dicts)
    # Set the sample name as the index.
    interaction_activities.set_index("sample_name", inplace=True)
    interaction_activities = interaction_activities.astype(np.float16)
    print("Saving interaction activity CSV...")
    interaction_activities.to_csv(interaction_path)
    print("Saved interaction activity CSV.")

