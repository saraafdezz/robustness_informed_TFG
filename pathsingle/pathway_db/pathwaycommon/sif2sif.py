# %%
import pandas as pd

# Define paths.
extended_sif_file = "./owl/PathwayCommons12.kegg.BIOPAX.sif"
output_file = "pathway_relations.csv"

# Read the extended SIF file.
with open(extended_sif_file, "r") as file:
    lines = file.readlines()

# Split the file into two sections.
interaction_data = []
participant_data = []
is_participant_section = False

for line in lines:
    if line.strip() == "":
        is_participant_section = True
        continue
    if is_participant_section:
        participant_data.append(line.strip())
    else:
        interaction_data.append(line.strip())

# Parse interaction data.
interaction_df = pd.DataFrame(
    [line.split("\t") for line in interaction_data],
    columns=[
        "PARTICIPANT_A",
        "INTERACTION_TYPE",
        "PARTICIPANT_B",
        "INTERACTION_DATA_SOURCE",
        "INTERACTION_PUBMED_ID",
        "PATHWAY_NAMES",
        "MEDIATOR_IDS",
    ],
)

# Parse participant data.
participant_df = pd.DataFrame(
    [line.split("\t") for line in participant_data],
    columns=[
        "PARTICIPANT",
        "PARTICIPANT_TYPE",
        "PARTICIPANT_NAME",
        "UNIFICATION_XREF",
        "RELATIONSHIP_XREF",
    ],
)

# Create a mapping from PARTICIPANT to PARTICIPANT_NAME.
participant_mapping = dict(
    zip(participant_df["PARTICIPANT"], participant_df["PARTICIPANT_NAME"])
)

# Map PARTICIPANT_A and PARTICIPANT_B to their names.
interaction_df["PARTICIPANT_A"] = interaction_df["PARTICIPANT_A"].map(
    participant_mapping
)
interaction_df["PARTICIPANT_B"] = interaction_df["PARTICIPANT_B"].map(
    participant_mapping
)

# Fill NaN values with empty strings.
interaction_df["PARTICIPANT_A"] = interaction_df["PARTICIPANT_A"].fillna("")
interaction_df["PARTICIPANT_B"] = interaction_df["PARTICIPANT_B"].fillna("")

# Take only the first part of the pathway names if they are separated by ';'.
interaction_df["PATHWAY_NAMES"] = interaction_df["PATHWAY_NAMES"].str.split(";").str[0]
# %%
# Filter interactions based on relevant INTERACTION_TYPEs.
"""
Distribution of INTERACTION_TYPEs:
INTERACTION_TYPE
catalysis-precedes           163239 Indicates that one reaction catalyzes another.
used-to-produce                8021 Indicates that one entity is used to produce another.
controls-production-of         5253 Denotes regulation of the transport of a substance.
consumption-controlled-by      5202 Suggests that the consumption of an entity is regulated by another.
reacts-with                    2793 Indicates a direct interaction between two entities.
neighbor-of                     611 Represents spatial or functional proximity between entities.
interacts-with                    4 A general term for any interaction between entities.

Relevant types:
used-to-produce: Represents a relationship where one entity is used to produce another.
controls-production-of: Represents a regulatory relationship where one entity controls the production of another.
consumption-controlled-by: Represents a relationship where the consumption of one entity is controlled by another.
"""
# relevant_interaction_types = ['catalysis-precedes', 'used-to-produce', 'controls-production-of', 'consumption-controlled-by']
# interaction_df = interaction_df[interaction_df['INTERACTION_TYPE'].isin(relevant_interaction_types)]

# Group interactions by a combination of columns to form biological interactions.
grouped_interactions = (
    interaction_df.groupby(["PATHWAY_NAMES", "INTERACTION_TYPE", "MEDIATOR_IDS"])
    .agg(
        {
            "PARTICIPANT_A": lambda x: "*".join(sorted(set(x))),
            "PARTICIPANT_B": lambda x: "*".join(sorted(set(x))),
        }
    )
    .reset_index()
)

# Remove the '_HUMAN' suffix from all participant names.
grouped_interactions["PARTICIPANT_A"] = grouped_interactions[
    "PARTICIPANT_A"
].str.replace("_HUMAN", "", regex=False)
grouped_interactions["PARTICIPANT_B"] = grouped_interactions[
    "PARTICIPANT_B"
].str.replace("_HUMAN", "", regex=False)

# Define keywords that suggest inhibition.
inhibitory_keywords = ["inhibition", "suppress", "block", "reduce", "degradation"]


# Identify potential inhibitory interactions.
def identify_and_update_inhibitory_relations(df):
    # Check if any of the keywords are in the PATHWAY_NAMES.
    mask = df["PATHWAY_NAMES"].str.contains(
        "|".join(inhibitory_keywords), case=False, na=False
    )

    # Update INTERACTION_TYPE to 'inhibition' for identified interactions.
    df.loc[mask, "INTERACTION_TYPE"] = "inhibition"

    return df


# Update the DataFrame.
grouped_interactions = identify_and_update_inhibitory_relations(grouped_interactions)

# Write to CSV.
grouped_interactions.to_csv(
    output_file,
    index=False,
    columns=["PATHWAY_NAMES", "PARTICIPANT_A", "INTERACTION_TYPE", "PARTICIPANT_B"],
)
# %%
