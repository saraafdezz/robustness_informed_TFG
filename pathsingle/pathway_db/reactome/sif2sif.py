import pandas as pd
from tqdm import tqdm


input_sif = '../owl/Homo_sapiens.sif'
output_file = 'pathway_relations.csv'

# Initialize empty list to store chunk results.
chunk_results = []
# Read and process file in chunks.
chunk_size = 100000  # Adjust based on available memory.
# Count total rows efficiently,
total_rows = 1844995
total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)

# Read participant data first (comes after empty line in SIF).
participant_data = []
with open(input_sif, 'r') as file:
    past_empty_line = False
    for line in file:
        if line.strip() == "":
            past_empty_line = True
            continue
        if past_empty_line:
            participant_data.append(line.strip())

# Process participants.
participant_df = pd.DataFrame([line.split('\t') for line in participant_data[1:]],
                            columns=['PARTICIPANT', 'PARTICIPANT_TYPE', 
                                   'PARTICIPANT_NAME', 'UNIFICATION_XREF'])

# Load mapping file.
name_mapping = {}
with open('uniprot_to_gene.tsv', 'r') as f:
    for line in f:
        uniprot, gene = line.strip().split('\t')
        name_mapping[uniprot] = gene

print(f"\nNumber of mappings created: {len(name_mapping)}")
print("\nFirst 5 mappings:")
for k, v in list(name_mapping.items())[:5]:
    print(f"{k} -> {v}")

for chunk in tqdm(pd.read_csv(input_sif, sep='\t', 
                        usecols=['PARTICIPANT_A', 'INTERACTION_TYPE', 
                                'PARTICIPANT_B', 'PATHWAY_NAMES', 'MEDIATOR_IDS'],
                        chunksize=chunk_size), total=total_chunks):
    # Safer mapping function.
    def map_participant(x):
        if pd.isna(x):
            return ''
        parts = str(x).split('*')
        mapped = [name_mapping.get(p, p) for p in parts if p]  # Keep original if no mapping
        return '*'.join(mapped) if mapped else ''
    
    # Map participants while preserving original if no mapping exists.
    chunk['PARTICIPANT_A'] = chunk['PARTICIPANT_A'].apply(map_participant)
    chunk['PARTICIPANT_B'] = chunk['PARTICIPANT_B'].apply(map_participant)
    
    # Filter out empty rows.
    mask = ((chunk['PARTICIPANT_A'].str.len() > 0) | (chunk['PARTICIPANT_B'].str.len() > 0))
    chunk = chunk[mask]

    # Process each chunk.
    grouped_chunk = chunk.groupby(['PATHWAY_NAMES', 'INTERACTION_TYPE', 'MEDIATOR_IDS']).agg({
         'PARTICIPANT_A': lambda x: '*'.join(sorted(set(filter(None, x)))),
         'PARTICIPANT_B': lambda x: '*'.join(sorted(set(filter(None, x))))
    }).reset_index()
    
    chunk_results.append(grouped_chunk)

# Combine all chunks.
print('Combining chunks.')
final_df = pd.concat(chunk_results, ignore_index=True)

# Take only the first part of the pathway names if they are separated by ';'.
final_df['PATHWAY_NAMES'] = final_df['PATHWAY_NAMES'].str.split(';').str[0]

# Final grouping of combined results.
print('Grouping data.')
final_grouped = final_df.groupby(['PATHWAY_NAMES', 'INTERACTION_TYPE', 'MEDIATOR_IDS']).agg({
    'PARTICIPANT_A': lambda x: '*'.join(sorted(set(x.str.split('*').explode()))),
    'PARTICIPANT_B': lambda x: '*'.join(sorted(set(x.str.split('*').explode())))
}).reset_index().drop('MEDIATOR_IDS', axis=1)

# Define keywords that suggest inhibition.
inhibitory_keywords = ['inhibit', 'suppress', 'block', 'reduce', 'degrad', 'negative']

# Update interaction type to 'inhibition' for pathways containing inhibitory keywords
mask = final_grouped['PATHWAY_NAMES'].str.contains('|'.join(inhibitory_keywords), 
                                                        case=False, na=False)
final_grouped.loc[mask, 'INTERACTION_TYPE'] = 'inhibition'

# Write output. Columns: PATHWAY_NAMES,PARTICIPANT_A,INTERACTION_TYPE,PARTICIPANT_B (might need to adjust to PathSigle format).
print('Saving data.')
final_grouped.to_csv(output_file, sep='\t', index=False)