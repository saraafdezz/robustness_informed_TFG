import os

import pandas as pd
import requests

# Code for downloading KEGG pathways.
# Read the CSV file,
csv_file = "./data/kegg_pathways.csv"
df = pd.read_csv(csv_file, dtype={"ID": str})

# Ensure output directory exists.
output_dir = "./data/kgml_files"
os.makedirs(output_dir, exist_ok=True)

# Loop through each row in the dataframe.
for index, row in df.iterrows():
    pathway_id = row["ID"]
    pathway_name = row["Name"]
    url = f"https://rest.kegg.jp/get/hsa{pathway_id}/kgml"
    try:
        response = requests.get(url)
        if response.status_code == 200 and response.content.strip():
            # Save the content to a file named after the pathway name.
            file_path = os.path.join(output_dir, f"{pathway_name}.kgml")
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Successfully downloaded and saved {pathway_name}.kgml")
        else:
            print(f"Empty result for {pathway_name} (ID: {pathway_id})")
    except Exception as e:
        print(f"An error occurred for {pathway_name} (ID: {pathway_id}): {e}")

print("Download completed.")
