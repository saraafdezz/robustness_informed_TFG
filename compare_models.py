import pandas as pd
from pathlib import Path
from workflow import save_consistedness, save_clustering, save_combined
import argparse

def main(results_folder_IVAE, results_folder_PathSingle, output_path):
    # Aseguro que sean Path
    results_folder_IVAE = Path(results_folder_IVAE)
    results_folder_PathSingle = Path(results_folder_PathSingle)
    output_path = Path(output_path)

    # Cargo tsv de consistering y clustering
    ivae_consistedness = pd.read_csv(results_folder_IVAE / "IVAE_consistedness.tsv", sep="\t")
    pathsingle_consistedness = pd.read_csv(results_folder_PathSingle / "pathsingle_consistedness.tsv", sep="\t")
    ivae_clustering = pd.read_csv(results_folder_IVAE / "IVAE_clustering.tsv", sep="\t")
    pathsingle_clustering = pd.read_csv(results_folder_PathSingle / "pathsingle_clustering.tsv", sep="\t")

    # Filtro por test
    ivae_consistedness = ivae_consistedness.query("split == 'test'")
    pathsingle_consistedness = pathsingle_consistedness.query("split == 'test'")
    ivae_clustering = ivae_clustering.query("split == 'test'")
    pathsingle_clustering = pathsingle_clustering.query("split == 'test'")

    # Uno los df
    combined_consistedness = pd.concat([ivae_consistedness, pathsingle_consistedness], ignore_index=True)
    combined_clustering = pd.concat([ivae_clustering, pathsingle_clustering], ignore_index=True)

    # Crear carpeta de salida
    output_path.mkdir(parents=True, exist_ok=True)

    # Guardar y plotear
    save_consistedness.fn(combined_consistedness, output_path)
    save_clustering.fn(combined_clustering, output_path)
    save_combined.fn(combined_consistedness, combined_clustering, output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare IVAE and PathSingle metrics."
    )

    parser.add_argument(
        "--results_folder_IVAE",
        type=str,
        default="results/IVAE",
        help="Path to the main folder where IVAE results will be saved.",
    )

    parser.add_argument(
        "--results_folder_PathSingle",
        type=str,
        default="results/PathSingle",
        help="Path to the main folder where PathSingle results will be saved.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="results/comparison",
        help="Path to the main folder where comparison results will be saved.",
    )

    args = parser.parse_args()

    main(
        results_folder_IVAE=args.results_folder_IVAE,
        results_folder_PathSingle=args.results_folder_PathSingle,
        output_path=args.output_path,
    )

