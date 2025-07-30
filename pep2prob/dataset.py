import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pep2prob.utils import *

class Pep2ProbDataSet:
    """ 
    Class to handle the training or test set in Pep2Prob Dataset.
    """
    
    def __init__(self, X, Y, Y_mask, X_info):
        """
        Initialize the dataset with features and labels.
        """
        self.X = X
        self.Y = Y
        self.Y_mask = Y_mask
        self.X_info = X_info

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Y_mask[idx], self.X_info.iloc[idx]

class Pep2ProbDataset:
    """
    Class to handle the Pep2Prob dataset.
    """
    def __init__(self, data_dir, split_idx=1, min_length_input=7, max_length_input=40, min_num_psm_for_statistics=10, skip_download_if_exists=True):
        """
        Initialize the dataset.
        """

        # Set parameters
        self.data_dir = data_dir
        self.split_idx = split_idx
        self.min_length_input = min_length_input
        self.max_length_input = max_length_input
        self.min_num_psm_for_statistics = min_num_psm_for_statistics

        # Download dataset if not already present
        Pep2ProbDataSet.download_dataset_from_HuggingFace(self.data_dir, skip_download_if_exists=True)

        # Load train and test indices
        train_indices_df, test_indices_df = self.load_train_test_X_Y()

        # construct training and testing data
        self.train = Pep2ProbDataSet(*self.get_X_Y_from_indices(train_indices_df))
        self.test = Pep2ProbDataSet(*self.get_X_Y_from_indices(test_indices_df))

    @staticmethod
    def download_dataset_from_HuggingFace(data_dir, skip_download_if_exists=True):
        """
        Downloads the pep2prob dataset from Hugging Face if not already present.
        """
        try:
            os.makedirs(data_dir, exist_ok=True)

            # download dataset
            dataset_url = "https://huggingface.co/datasets/bandeiralab/Pep2Prob/resolve/main/data/pep2prob_dataset.parquet"
            dataset_path = os.path.join(data_dir, "pep2prob_dataset.parquet")
            if not os.path.exists(dataset_path) or not skip_download_if_exists:
                print(f"Downloading dataset from {dataset_url} to {dataset_path}...")
                os.system(f"wget {dataset_url} -O {dataset_path}")
                print("Download complete.")
            else:
                print("Dataset already exists, skipping download.")

            # download data split
            root_split_url = "https://huggingface.co/datasets/bandeiralab/Pep2Prob/resolve/main/data_split"

            for split_idx in range(1, 6):
                test_url = f"{root_split_url}/train_test_split_set_{split_idx}/test_indices.parquet"
                train_url = f"{root_split_url}/train_test_split_set_{split_idx}/train_indices.parquet"
                test_path = os.path.join(data_dir, f"test_indices_{split_idx}.parquet")
                train_path = os.path.join(data_dir, f"train_indices_{split_idx}.parquet")
                if not os.path.exists(test_path) or not skip_download_if_exists:
                    print(f"Downloading test indices from {test_url} to {test_path}...")
                    os.system(f"wget {test_url} -O {test_path}")
                    print("Test indices download complete.")
                if not os.path.exists(train_path) or not skip_download_if_exists:
                    print(f"Downloading train indices from {train_url} to {train_path}...")
                    os.system(f"wget {train_url} -O {train_path}")
                    print("Train indices download complete.")
            print("Dataset and data split downloaded successfully.")
            return 1        
        except Exception as e:
            print(f"Error downloading dataset and data split: {e}")
            exit(1)

    def load_train_test_X_Y(self):
        """
        Loads the train and test indices for a specific split.
        """
        try:
            train_path = os.path.join(self.data_dir, f"train_indices_{self.split_idx}.parquet")
            test_path = os.path.join(self.data_dir, f"test_indices_{self.split_idx}.parquet")

            train_indices = pq.read_table(train_path).to_pandas()
            test_indices = pq.read_table(test_path).to_pandas()

            return train_indices, test_indices
        except Exception as e:
            print(f"Error loading train/test indices: {e}")
            exit(1)
    

    def get_X_Y_from_indices(self, data_indices_df):
        """
        Extracts X and Y from the given indices DataFrame.
        """
        try:
            # Filter based on length and number of PSMs
            filtered_df = data_indices_df[
                (data_indices_df['peptide_length'] >= self.min_length_input) &
                (data_indices_df['peptide_length'] <= self.max_length_input) &
                (data_indices_df['#PSM'] >= self.min_num_psm_for_statistics)
            ]

            X = filtered_df[['peptide', 'charge']].values
            X_info_df = filtered_df[['precursor_index', 'peptide', 'charge', '#PSM', 'peptide_length']]
            Y = filtered_df.values[:, 5:]

            # get the mask for Y
            Y_mask = np.zeros(Y.shape, dtype=bool)
            for i, (peptide, charge) in enumerate(X_info_df[['peptide', 'charge']].values):
                mask = get_ion_mask(len(peptide), charge, self.max_length_input)
                Y_mask[i, :] = mask

            # X = np.array([encode_sequence_and_charge(seq, charge) for seq, charge in X_info_df[['peptide', 'charge']].values])
            # # create one-hot version of X
            # X_1hot = np.zeros((X.shape[0], X.shape[1], len(char_to_int)), dtype=int)
            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         if X[i, j] != 0:
            #             X_1hot[i, j, X[i, j]-1] = 1
            # X_1hot = X_1hot.reshape(X.shape[0], -1)

            return X, Y, Y_mask, X_info_df
        except Exception as e:
            print(f"Error extracting X and Y: {e}")
            exit(1)
