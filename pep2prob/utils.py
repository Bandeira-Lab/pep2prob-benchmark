import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

char_to_int = {'G': 1, 'I': 2, 'S': 3, 'W': 4, 'E': 5, 'H': 6, 'M': 7, 'T': 8, 'Y': 9, 'F': 10, 'L': 11, 'V': 12, 'A': 13, 'P': 14, 'Q': 15, 'R': 16, 'D': 17, 'N': 18, 'K': 19, 'C': 20}
max_sequence_length = 40

frag_ion_names = [('a', 1, 2), ('b', 1, 1), ('b', 1, 2), ('b', 1, 3), ('b', 1, 4), ('b', 1, 5), ('b', 1, 6), ('b', 1, 7), ('b', 1, 8), ('b', 1, 9), ('b', 1, 10), ('b', 1, 11), ('b', 1, 12), ('b', 1, 13), ('b', 1, 14), ('b', 1, 15), ('b', 1, 16), ('b', 1, 17), ('b', 1, 18), ('b', 1, 19), ('b', 1, 20), ('b', 1, 21), ('b', 1, 22), ('b', 1, 23), ('b', 1, 24), ('b', 1, 25), ('b', 1, 26), ('b', 1, 27), ('b', 1, 28), ('b', 1, 29), ('b', 1, 30), ('b', 1, 31), ('b', 1, 32), ('b', 1, 33), ('b', 1, 34), ('b', 1, 35), ('b', 1, 36), ('b', 1, 37), ('b', 1, 38), ('b', 1, 39), ('b', 2, 1), ('b', 2, 2), ('b', 2, 3), ('b', 2, 4), ('b', 2, 5), ('b', 2, 6), ('b', 2, 7), ('b', 2, 8), ('b', 2, 9), ('b', 2, 10), ('b', 2, 11), ('b', 2, 12), ('b', 2, 13), ('b', 2, 14), ('b', 2, 15), ('b', 2, 16), ('b', 2, 17), ('b', 2, 18), ('b', 2, 19), ('b', 2, 20), ('b', 2, 21), ('b', 2, 22), ('b', 2, 23), ('b', 2, 24), ('b', 2, 25), ('b', 2, 26), ('b', 2, 27), ('b', 2, 28), ('b', 2, 29), ('b', 2, 30), ('b', 2, 31), ('b', 2, 32), ('b', 2, 33), ('b', 2, 34), ('b', 2, 35), ('b', 2, 36), ('b', 2, 37), ('b', 2, 38), ('b', 2, 39), ('b', 3, 1), ('b', 3, 2), ('b', 3, 3), ('b', 3, 4), ('b', 3, 5), ('b', 3, 6), ('b', 3, 7), ('b', 3, 8), ('b', 3, 9), ('b', 3, 10), ('b', 3, 11), ('b', 3, 12), ('b', 3, 13), ('b', 3, 14), ('b', 3, 15), ('b', 3, 16), ('b', 3, 17), ('b', 3, 18), ('b', 3, 19), ('b', 3, 20), ('b', 3, 21), ('b', 3, 22), ('b', 3, 23), ('b', 3, 24), ('b', 3, 25), ('b', 3, 26), ('b', 3, 27), ('b', 3, 28), ('b', 3, 29), ('b', 3, 30), ('b', 3, 31), ('b', 3, 32), ('b', 3, 33), ('b', 3, 34), ('b', 3, 35), ('b', 3, 36), ('b', 3, 37), ('b', 3, 38), ('b', 3, 39), ('y', 1, 1), ('y', 1, 2), ('y', 1, 3), ('y', 1, 4), ('y', 1, 5), ('y', 1, 6), ('y', 1, 7), ('y', 1, 8), ('y', 1, 9), ('y', 1, 10), ('y', 1, 11), ('y', 1, 12), ('y', 1, 13), ('y', 1, 14), ('y', 1, 15), ('y', 1, 16), ('y', 1, 17), ('y', 1, 18), ('y', 1, 19), ('y', 1, 20), ('y', 1, 21), ('y', 1, 22), ('y', 1, 23), ('y', 1, 24), ('y', 1, 25), ('y', 1, 26), ('y', 1, 27), ('y', 1, 28), ('y', 1, 29), ('y', 1, 30), ('y', 1, 31), ('y', 1, 32), ('y', 1, 33), ('y', 1, 34), ('y', 1, 35), ('y', 1, 36), ('y', 1, 37), ('y', 1, 38), ('y', 1, 39), ('y', 2, 1), ('y', 2, 2), ('y', 2, 3), ('y', 2, 4), ('y', 2, 5), ('y', 2, 6), ('y', 2, 7), ('y', 2, 8), ('y', 2, 9), ('y', 2, 10), ('y', 2, 11), ('y', 2, 12), ('y', 2, 13), ('y', 2, 14), ('y', 2, 15), ('y', 2, 16), ('y', 2, 17), ('y', 2, 18), ('y', 2, 19), ('y', 2, 20), ('y', 2, 21), ('y', 2, 22), ('y', 2, 23), ('y', 2, 24), ('y', 2, 25), ('y', 2, 26), ('y', 2, 27), ('y', 2, 28), ('y', 2, 29), ('y', 2, 30), ('y', 2, 31), ('y', 2, 32), ('y', 2, 33), ('y', 2, 34), ('y', 2, 35), ('y', 2, 36), ('y', 2, 37), ('y', 2, 38), ('y', 2, 39), ('y', 3, 1), ('y', 3, 2), ('y', 3, 3), ('y', 3, 4), ('y', 3, 5), ('y', 3, 6), ('y', 3, 7), ('y', 3, 8), ('y', 3, 9), ('y', 3, 10), ('y', 3, 11), ('y', 3, 12), ('y', 3, 13), ('y', 3, 14), ('y', 3, 15), ('y', 3, 16), ('y', 3, 17), ('y', 3, 18), ('y', 3, 19), ('y', 3, 20), ('y', 3, 21), ('y', 3, 22), ('y', 3, 23), ('y', 3, 24), ('y', 3, 25), ('y', 3, 26), ('y', 3, 27), ('y', 3, 28), ('y', 3, 29), ('y', 3, 30), ('y', 3, 31), ('y', 3, 32), ('y', 3, 33), ('y', 3, 34), ('y', 3, 35), ('y', 3, 36), ('y', 3, 37), ('y', 3, 38), ('y', 3, 39)]

def get_ion_mask(seq_len, charge, max_seq_len):
    # a2+ ions
    mask = [True]
    if charge > 3:
        charge = 3
    # b/y ions with charge 1/2/3
    for ion in range(1, 3):
        for chr in range(1, 4):
            if chr > charge:
                for seq_idx in range(1, max_seq_len):
                    mask.append(False)
            else:
                for seq_idx in range(1, max_seq_len):
                    if seq_idx < seq_len:
                        mask.append(True)
                    else:
                        mask.append(False)
    return mask

def encode_sequence_and_charge(seq, charge):
    # charge + sequence (int encoding)
    seq_encoded = np.zeros(max_sequence_length+1, dtype=int)
    seq_encoded[0] = charge
    for i, char in enumerate(seq):
        seq_encoded[i+1] = char_to_int[char]
    return seq_encoded

def convert_int_to_onehot_encode(X):
    # create one-hot version of X
    X_1hot = np.zeros((X.shape[0], X.shape[1], len(char_to_int)), dtype=int)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] != 0:
                X_1hot[i, j, X[i, j]-1] = 1
    X_1hot = X_1hot.reshape(X.shape[0], -1)

def get_standard_error(p, N):
    """
    return the standard error of the probability p with sample size N
    """
    if N == 0:
        return 0
    else:
        return np.sqrt(p * (1 - p) / N)

def get_min_distance_two_sequence(seq1, seq2):
    """
    return the minimum number of steps to convert seq1 to seq2 by adding, removing or replacing one amino acid.
    """
    # convert seq1 and seq2 to list of amino acids
    seq1 = list(seq1)
    seq2 = list(seq2)

    # create a distance matrix
    distance_matrix = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]

    # initialize the distance matrix
    for i in range(len(seq1) + 1):
        distance_matrix[i][0] = i
    for j in range(len(seq2) + 1):
        distance_matrix[0][j] = j

    # calculate the distance matrix
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                distance_matrix[i][j] = distance_matrix[i - 1][j - 1]
            else:
                distance_matrix[i][j] = min(distance_matrix[i - 1][j] + 1, distance_matrix[i][j - 1] + 1, distance_matrix[i - 1][j - 1] + 1)

    return distance_matrix[-1][-1]

def generate_prefix_suffix_low_dist_set_dict(sequence_list, max_low_dist, save_path):
    num_seq = len(sequence_list)
    prefix_set, suffix_set = set(), set()
    low_dist_set_dict = {i: set() for i in range(max_low_dist + 1)}
    # we know sequences are sorted by length ascending
    for i in tqdm(range(num_seq)):
        for j in range(i + 1, num_seq):
            if len(sequence_list[i]) == len(sequence_list[j]):
                if sequence_list[i] == sequence_list[j]:
                    low_dist_set_dict[0].add((i, j))
            else:
                if sequence_list[j].startswith(sequence_list[i]):
                    prefix_set.add((i, j))
                elif sequence_list[j].endswith(sequence_list[i]):
                    suffix_set.add((i, j))
                elif len(sequence_list[j]) - len(sequence_list[i]) <= max_low_dist:
                    distance = get_min_distance_two_sequence(sequence_list[i], sequence_list[j])
                    if distance <= max_low_dist:
                        low_dist_set_dict[distance].add((i, j))
    low_dist_set_dict['prefix'] = prefix_set
    low_dist_set_dict['suffix'] = suffix_set
    with open(save_path, "wb") as f:
        pickle.dump(low_dist_set_dict, f)
    
    return 1

def load_train_test_prob(precursor_info_path, indices_path, matrix_path, return_X=True, use_peak_mask=True):
    precursor_df = pd.read_csv(precursor_info_path, sep='\t')
    # each row is a precursor, which is a pair of (peptide_sequence, charge)
    # columns: [
    #   'precursor_index',          # corresponds to the index of the first order of the matrix
    #   'sequence',                 # peptide sequence, no modification by now
    #   'charge',                   # charge of the precursor
    #   'num_PSMs'                  # number of spectra that associated with this precursor
    # ]

    matrix = np.load(matrix_path)
    # matrix is a 3D numpy array with shape (num_precursors, num_tokens_to_predict, 4)
    #   num_precursors: the ith precursor corresponds to the ith row in precursor_df
    #   num_tokens_to_predict: the number of tokens (peak) to predict in the ith precursor
    #   4: for each token, there are 4 values: [
    #       "m/z",                  # the m/z value of the token (peak)
    #       "probability",          # the probability of the token (peak) to be observed
    #       "mean_intensity",       # the mean intensity of the token (peak)
    #       "var_intensity"         # the variance of the intensity of the token (peak)
    # if the shape show it is 2d array then convert it to 3d
    if len(matrix.shape) == 2:
        matrix = matrix[:, :, np.newaxis]
        print("matrix shape is 2d, convert it to 3d, shepe is: ", matrix.shape)


    loaded_data = np.load(indices_path, allow_pickle=True).reshape(1)[0]

    # loaded_data is a dictionary with the following keys:
    #   'train_indices': the indices of the training set
    #   'test_indices': the indices of the test set
    #   'train_indices' and 'test_indices' are numpy arrays of shape (80% num_samples,) and (20% num_samples,) respectively
    train_indices = loaded_data['train_indices']
    test_indices = loaded_data['test_indices']

    # create peak mask: probabilities outside mask shouldn't have value and are set to -1
    if matrix.shape[2] == 1:
        probabilities = matrix[:, :, 0].copy()
    else:
        probabilities = matrix[:, :, 1].copy()
    if use_peak_mask:
        peak_mask = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=bool)
        for i in range(matrix.shape[0]):
            seq_len = len(precursor_df['sequence'][i])
            charge = precursor_df['charge'][i]
            mask = get_ion_mask(seq_len, charge, 40)
            peak_mask[i, :] = mask
        assert matrix[:, :, 1][~peak_mask].max() < 1e-8
        probabilities[~peak_mask] = -1

    # create filter mask: only precursors satisfying the following conditions are kept
    min_num_psms = 30
    max_charge = 100
    max_seq_length = 100
    filtered_indices = precursor_df[
        (precursor_df['num_PSMs'] >= min_num_psms) &
        (precursor_df['charge'] <= max_charge) &
        (precursor_df['sequence'].str.len() <= max_seq_length)
    ].index
    filter_mask = np.zeros((matrix.shape[0],), dtype=bool)
    filter_mask[filtered_indices] = True

    # train test split
    train_mask = np.zeros(matrix.shape[0], dtype=bool)
    train_mask[train_indices] = True
    train_mask = np.logical_and(train_mask, filter_mask)

    test_mask = np.zeros(matrix.shape[0], dtype=bool)
    test_mask[test_indices] = True
    test_mask = np.logical_and(test_mask, filter_mask)

    Y = probabilities
    Y_train = Y[train_mask]
    Y_test = Y[test_mask]

    precursor_train_df = precursor_df[train_mask]
    precursor_test_df = precursor_df[test_mask]

    if return_X:
        # encode sequences and charges into integers. 0 is reserved for padding
        sequences = precursor_df['sequence'].values
        charges = precursor_df['charge'].values
        num_PSMs = precursor_df['num_PSMs'].values

        X = np.array([encode_sequence_and_charge(seq, charge) for seq, charge in zip(sequences, charges)])
        X_1hot = convert_int_to_onehot_encode(X)

        X_train = X_1hot[train_mask]
        X_test = X_1hot[test_mask]

        return X_train, Y_train, X_test, Y_test, precursor_train_df, precursor_test_df
    else:
            

        return Y_train, Y_test, precursor_train_df, precursor_test_df

def compute_l1_distance(a, b):
    """
    Compute the L1 distance between two arrays.
    """
    return np.mean(np.abs(a - b))

def compute_mse(a, b):
    """
    Compute the mean squared error between two arrays.
    """
    return np.mean((a - b) ** 2)

def compute_cosine(a, b):
    """
    Compute the cosine similarity between two arrays.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0
    else:
        return np.dot(a, b) / (a_norm * b_norm)

def get_confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix for the given true and predicted labels.
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    label = 1
    TP += np.sum((y_true == label) & (y_pred == label))
    TN += np.sum((y_true != label) & (y_pred != label))
    FP += np.sum((y_true != label) & (y_pred == label))
    FN += np.sum((y_true == label) & (y_pred != label))
    return TP, TN, FP, FN

def get_accuracy_sensitivity_specificity(y_true, y_pred, threshold):
    """
    Compute precision, recall, and specificity for the given true and predicted labels.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    TP, TN, FP, FN = get_confusion_matrix(y_true, y_pred_binary)
    
    accuracy = TP / (TP + FP) if (TP + FP) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    return accuracy, sensitivity, specificity

def get_masked_spectral_distance(cosine):
    arccosine = np.arccos(cosine)
    return 1- 2 * arccosine / np.pi

# def get_masked_spectral_distance(y_true, y_pred, epsilon=1e-7):
#     import numpy as np

#     y_true = np.array(y_true).reshape(1, -1)
#     y_pred = np.array(y_pred).reshape(1, -1)

#     pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + 1e-7)
#     true_masked = ((y_true + 1) * y_true) / (y_true + 1 + 1e-7)

#     true_norm = (
#         true_masked * (1 / np.sqrt(np.sum(np.square(true_masked), axis=1)))[:, None]
#     )

#     pred_norm = (
#         pred_masked * (1 / np.sqrt(np.sum(np.square(pred_masked), axis=1)))[:, None]
#     )

#     product = np.sum(true_norm * pred_norm, axis=1)

#     arccosine = np.arccos(product)

#     return 2 * arccosine / np.pi