from Bio.KEGG.KGML.KGML_parser import read as read_kgml
import os
import pandas as pd
import numpy as np


#Ensure output directory exists.
kgml_dir = './data/kgml_files'
output_file = './data/pathway_relations.csv'
gene2symbol_file = './data/gene2symbol.csv'

#Load the gene2symbol mapping.
gene2symbol_df = pd.read_csv(gene2symbol_file)
gene2symbol_dict = dict(zip(gene2symbol_df['ID'], gene2symbol_df['Symbol']))

#Function to get names from an entry, unwrapping groups recursively.
def get_entry_names(entry, pathway):
    names = []
    if entry.type in ['gene', 'compound']:
        names.extend(entry.name.split())
    elif entry.type == 'group':
        for component in entry.components:
            component_entry = pathway.entries.get(component.id)
            if component_entry:
                names.extend(get_entry_names(component_entry, pathway))
    return names

#Process all KGML files.
with open(output_file, 'w') as out:
    out.write("pathway,source,interactiontype,target\n")

    for kgml_file in os.listdir(kgml_dir):
        if kgml_file.endswith('.kgml'):
            pathway = read_kgml(open(os.path.join(kgml_dir, kgml_file)))

            for relation in pathway.relations:
                entry1_names = get_entry_names(relation.entry1, pathway)
                entry2_names = get_entry_names(relation.entry2, pathway)
                entry1_names = str(entry1_names).replace(",","*").replace(" ","").replace("[","").replace("]","").replace("'","")
                entry2_names = str(entry2_names).replace(",","*").replace(" ","").replace("[","").replace("]","").replace("'","")
                subtypes = '$'.join([subtype[0] for subtype in relation.subtypes])
                out.write(f"{kgml_file.replace('.kgml','')},{entry1_names},{subtypes},{entry2_names}\n")


#Replace gene names.
#Function to replace IDs with symbols.
def replace_ids_with_symbols(col):
    if col is np.nan:
        return ""
    #id_list = eval(id_list)  #Convert string representation of list to actual list.
    id_list = col.split('*')
    ret = [gene2symbol_dict.get(id, id) for id in id_list]
    return str(ret).replace(",","*").replace(" ","").replace("[","").replace("]","").replace("'","")


pathway_relations_df = pd.read_csv(output_file, dtype={'source': str, 'target': str})

#Replace IDs in the dataframe.
pathway_relations_df['source'] = pathway_relations_df['source'].apply(replace_ids_with_symbols)
pathway_relations_df['target'] = pathway_relations_df['target'].apply(replace_ids_with_symbols)

#Drop rows where both 'source' and 'target' are empty strings.
pathway_relations_df = pathway_relations_df[~((pathway_relations_df['source'] == "") & (pathway_relations_df['target'] == ""))]
pathway_relations_df = pathway_relations_df[(pathway_relations_df['interactiontype'] != "") & (~pathway_relations_df['interactiontype'].isnull())]

#Save the updated dataframe to a new CSV file.
pathway_relations_df.to_csv(output_file, index=False)
print("Processing completed and relations saved to pathway_relations.csv.")