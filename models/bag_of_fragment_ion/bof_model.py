import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

import os, sys
project_root = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir)
)
sys.path.insert(0, project_root)
from data.utils import *

print("Starting Bag-of-Fragment-ion model script.")

seed = 42

set_idx = 4

# load raw data
precursor_info_path = 'data/pep2prob/pep2prob_dataset.csv'

indices_path = f'data/pep2prob/train_test_split_set_{set_idx}.npy'
predictions_folder = f'./predictions/{set_idx}_'

precursor_df = pd.read_csv(precursor_info_path, sep=',')
# each row is a precursor, which is a pair of (peptide_sequence, charge)
# columns: [
#   'precursor_index',          # corresponds to the index of the first order of the matrix
#   'peptide',                 # peptide sequence, no modification by now
#   'charge',                   # charge of the precursor
#   '#PSM'                  # number of spectra that associated with this precursor
#   'peptide_length',                 # length of the peptide sequence
#   '('a', '1', '2')',                 # the probability of appearing fragment ion type ('a', '1', '2') 
#    .........
#   up to 235 fragment ion types
# ]

cols = precursor_df.columns[5:]
matrix = precursor_df[cols].to_numpy()
# matrix is a 2by2 numpy array with shape (num_precursors,probability)
#   num_precursors: the ith precursor corresponds to the ith row in precursor_df
#   "probability",          # the probability of the token (peak) to be observed



loaded_data = np.load(indices_path, allow_pickle=True).item()

# loaded_data is a dictionary with the following keys:
#   'train_indices': the indices of the training set
#   'test_indices': the indices of the test set
#   'train_indices' and 'test_indices' are numpy arrays of shape (80% num_samples,) and (20% num_samples,) respectively
train_indices = loaded_data['train']
test_indices = loaded_data['test']

np.concatenate((train_indices, test_indices), axis=0).max()
matrix.shape[0]

# number of rows
# print(precursor_df.index.max())
# print(precursor_df.shape[0])
# print(precursor_df['precursor_index'].max())

# create peak mask: probabilities outside mask shouldn't have value and are set to -1

peak_mask = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=bool)

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

for i in range(matrix.shape[0]):
    seq_len = len(precursor_df['peptide'][i])
    charge = precursor_df['charge'][i]
    max_seq_len = matrix.shape[1]
    mask = get_ion_mask(seq_len, charge, 40)
    peak_mask[i, :] = mask

assert matrix[:, :][~peak_mask].max() < 1e-8


probabilities = matrix[:, :].copy()
probabilities[~peak_mask] = -1

# create filter mask: only precursors satisfying the following conditions are kept
min_num_psms = 30
max_charge = 100
max_seq_length = 100


filtered_indices = precursor_df[
    (precursor_df['#PSM'] >= min_num_psms) &
    (precursor_df['charge'] <= max_charge) &
    (precursor_df['peptide'].str.len() <= max_seq_length)
].index

filter_mask = np.zeros((matrix.shape[0],), dtype=bool)
filter_mask[filtered_indices] = True

# filtered_sequences = precursor_df['sequence'].iloc[filtered_indices].values
# filtered_charges = precursor_df['charge'].iloc[filtered_indices].values
# filtered_num_PSMs = precursor_df['num_PSMs'].iloc[filtered_indices].values

# filtered_probabilities = probabilities[filtered_indices]

# encode sequences and charges into integers. 0 is reserved for padding
sequences = precursor_df['peptide'].values
charges = precursor_df['charge'].values
num_PSMs = precursor_df['#PSM'].values


unique_chars = set(''.join(sequences))
char_to_int = {char: i+1 for i, char in enumerate(unique_chars)}
# reserve 0 for padding

unique_charges = set(charges)
charge_to_int = {charge: i+1 for i, charge in enumerate(unique_charges)}
# reserve 0 for padding

max_sequence_length = max(len(seq) for seq in sequences)
def encode_sequence_and_charge(seq, charge):
    seq_encoded = np.zeros(max_sequence_length+1, dtype=int)
    seq_encoded[0] = charge_to_int[charge]
    for i, char in enumerate(seq):
        seq_encoded[i+1] = char_to_int[char]
    return seq_encoded



X = np.array([encode_sequence_and_charge(seq, charge) for seq, charge in zip(sequences, charges)])
Y = probabilities


# idx = 1234
# print(f"Sequence: {sequences[idx]}")
# print(f"Charge: {charges[idx]}")
# print(f"Encoded Sequence: {X[idx, :]}")
# print(f"Probabilities (first 20): {Y[idx, :20]}")

# create one-hot version of X
X_1hot = np.zeros((X.shape[0], X.shape[1], len(char_to_int)), dtype=int)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if X[i, j] != 0:
            X_1hot[i, j, X[i, j]-1] = 1
X_1hot = X_1hot.reshape(X.shape[0], -1)

# idx = 1234
# print(f"One-hot Encoded Sequence: {X_1hot[idx, :]}")

# train test split

train_mask = np.zeros(matrix.shape[0], dtype=bool)
train_mask[train_indices] = True
train_mask = np.logical_and(train_mask, filter_mask)

test_mask = np.zeros(matrix.shape[0], dtype=bool)
test_mask[test_indices] = True
test_mask = np.logical_and(test_mask, filter_mask)

X_train = X_1hot[train_mask]
Y_train = Y[train_mask]
X_test = X_1hot[test_mask]
Y_test = Y[test_mask]

# Y_total = Y
# X_total = X_1hot

sequences_train = sequences[train_mask]
sequences_test = sequences[test_mask]
charges_train = charges[train_mask]
charges_test = charges[test_mask]
PSMs_train = num_PSMs[train_mask]
PSMs_test = num_PSMs[test_mask]

# construct estimators


# estimator B conditions on ion and charge and pre/suffix
estimB = {}

for idx, seq in enumerate(sequences_train):
    num_PSMs = PSMs_train[idx]
    for pos in range(Y.shape[1]):

        if Y_train[idx, pos] < -0.1:
            continue

        ion_type, ion_charge, cut_pos = frag_ion_names[pos]
        if ion_type == 'a' or ion_type == 'b':
            pre_suf_fix = seq[:cut_pos]
        else:
            pre_suf_fix = seq[-cut_pos:]
        keyB = (ion_type, ion_charge, pre_suf_fix)
        prob = Y_train[idx, pos]
        
        if keyB not in estimB:
            estimB[keyB] = (prob*num_PSMs, num_PSMs)
        else:
            estimB[keyB] = (estimB[keyB][0] + prob*num_PSMs, estimB[keyB][1] + num_PSMs)

# calculate mean
estimB0 = {x: estimB[x][0]/estimB[x][1] for x in estimB}

# evaluate estimators on the whole dataset
predB = np.zeros((Y.shape[0], Y.shape[1]))

for idx, seq in enumerate(sequences):
    for pos in range(Y.shape[1]):

        if Y[idx, pos] < -0.1:
            predB[idx, pos] = -1
            continue

        ion_type, ion_charge, cut_pos = frag_ion_names[pos]
        if ion_type == 'a' or ion_type == 'b':
            pre_suf_fix = seq[:cut_pos]
        else:
            pre_suf_fix = seq[-cut_pos:]

        keyB = (ion_type, ion_charge, pre_suf_fix)

        if keyB in estimB0:
            predB[idx, pos] = estimB0[keyB]

# save the whole results for later evaluatio
predictions_folder = './predictions/'    
# ensure the directory exists
os.makedirs(predictions_folder, exist_ok=True)
np.save(predictions_folder+'estimator_B'+'.npy', predB)
# predictions_folder
print("Estimator B predictions saved to: ", predictions_folder+'estimator_B'+'.npy')

# mean l1 and MSE loss on test set
tmp_mask = (Y_test >= -.1)
lossB = np.abs(predB[test_mask][tmp_mask] - Y_test[tmp_mask]).mean()
mseB = ((predB[test_mask][tmp_mask] - Y_test[tmp_mask])**2).mean()

print(f"Mean L1 loss of estimator B: {lossB}")
print(f"MSE loss of estimator B: {mseB}")