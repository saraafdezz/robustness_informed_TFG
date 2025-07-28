# %%
import pandas as pd
import pybiopax
from tqdm import tqdm

# Load the pathway commons file.
file_path = "./data/PathwayCommons12.Detailed.BIOPAX.owl.gz"

# Initialize the model.
model = pybiopax.model_from_owl_gz(file_path, encoding="utf8")
# %%
# Extract all pathways.
pathways = model.get_objects_by_type(pybiopax.biopax.Pathway)

# Initialize an empty list to hold the dataframe rows.
data = []


def get_molecule_info(molecule):
    if isinstance(molecule, pybiopax.biopax.Complex):
        # If it's a complex, recursively find the components.
        components = molecule.component
        for component in components:
            if component.member_entity_reference:
                for member in component.member_entity_reference:
                    member_type, member_name = get_molecule_info(member)
                    if member_type and member_name:
                        return member_type, member_name
    else:
        molecule_type = molecule.get_type().split(".")[-1]
        molecule_name = (
            molecule.display_name
            or molecule.standard_name
            or (molecule.name[0] if molecule.name else "Unknown")
        )
        return molecule_type, molecule_name


# Function to process a pathway.
def process_pathway(pathway):
    source = pathway.data_source[0].xref_id if pathway.data_source else "Unknown"
    relations = pathway.get_relations()

    for relation in relations:
        relation_type = relation.get_type().split(".")[-1]

        for left in relation.left:
            left_type, left_name = get_molecule_info(left)

            for right in relation.right:
                right_type, right_name = get_molecule_info(right)

                data.append(
                    [
                        source,
                        left_type,
                        left_name,
                        right_type,
                        right_name,
                        relation_type,
                    ]
                )


# Process each pathway.
for pathway in tqdm(pathways):
    process_pathway(pathway)

# Convert to dataframe.
columns = [
    "Pathway source",
    "Source molecule type",
    "Source molecule name",
    "Target molecule type",
    "Target molecule name",
    "Relation type",
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV.
df.to_csv("pathway_relations.csv", index=False)

# Print the first few rows of the dataframe.
print(df.head())
