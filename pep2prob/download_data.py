import os
from pathlib import Path
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# Script to download and prepare the Pep2Prob dataset locally.
# Places all files into a "pep2prob" folder alongside this script.

def main():
    # Define output directory
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "pep2prob"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "bandeiralab/Pep2Prob"
    repo_type = "dataset"
    revision = "main"

    # 1. Download and convert main dataset
    print("Downloading pep2prob_dataset.parquet...")
    data_parquet = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename="data/pep2prob_dataset.parquet",
        revision=revision,
    )
    print("Reading parquet into pandas...")
    df = pd.read_parquet(data_parquet)
    csv_path = out_dir / "pep2prob_dataset.csv"
    print(f"Saving main dataset to {csv_path}...")
    df.to_csv(csv_path, index=False)

    # 2. Download and convert metadata files
    for meta in ["X_columns.parquet", "Y_columns.parquet"]:
        print(f"Downloading {meta}...")
        meta_parquet = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=f"meta_data/{meta}",
            revision=revision,
        )
        print(f"Converting {meta} to CSV...")
        meta_df = pd.read_parquet(meta_parquet)
        csv_name = meta.replace(".parquet", ".csv")
        meta_df.to_csv(out_dir / csv_name, index=False)

    # 3. Download and save train/test split indices for 5 sets
    for i in range(1, 6):
        print(f"Processing split set {i}...")
        train_pq = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=f"data_split/train_test_split_set_{i}/train_indices.parquet",
            revision=revision,
        )
        test_pq = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=f"data_split/train_test_split_set_{i}/test_indices.parquet",
            revision=revision,
        )
        train_idx = pd.read_parquet(train_pq)[pd.read_parquet(train_pq).columns[0]].to_numpy()
        test_idx = pd.read_parquet(test_pq)[pd.read_parquet(test_pq).columns[0]].to_numpy()

        # Save a single .npy file per split containing a dict of both
        out_file = out_dir / f"train_test_split_set_{i}.npy"
        print(f"Saving indices to {out_file}...")
        np.save(out_file, {"train": train_idx, "test": test_idx})

    print("All files downloaded and converted successfully.")


if __name__ == "__main__":
    main()

