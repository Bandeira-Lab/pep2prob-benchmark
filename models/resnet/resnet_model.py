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
print(precursor_df.index.max())
print(precursor_df.shape[0])
print(precursor_df['precursor_index'].max())

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


idx = 1234
print(f"Sequence: {sequences[idx]}")
print(f"Charge: {charges[idx]}")
print(f"Encoded Sequence: {X[idx, :]}")
print(f"Probabilities (first 20): {Y[idx, :20]}")

# create one-hot version of X
X_1hot = np.zeros((X.shape[0], X.shape[1], len(char_to_int)), dtype=int)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if X[i, j] != 0:
            X_1hot[i, j, X[i, j]-1] = 1
X_1hot = X_1hot.reshape(X.shape[0], -1)

idx = 1234
print(f"One-hot Encoded Sequence: {X_1hot[idx, :]}")

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

import torch
import torch.nn as nn
import torch.nn.functional as F

# fix random seed for torch
torch.manual_seed(seed)

# Create DataLoader
batch_size = 512

from torch.utils.data import DataLoader, TensorDataset
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(SimpleNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.fc2(x))/2 + x/2
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.fc3(x)/2) + x/2
        x = self.fc4(x)
        x = F.sigmoid(x)
        return x

# Initialize the model, and optimizer
input_size = X_train.shape[1]
output_size = Y_train.shape[1]

hidden_size = 512
dropout_rate = 0.15
num_epochs = 200

model = SimpleNN(input_size, hidden_size, output_size, dropout_rate).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()


# training

train_loss_list = []
test_loss_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        mask = targets > -.1
        loss = loss_fn(outputs[mask], targets[mask])
        # loss = -F.cosine_similarity(outputs, targets).mean()

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        train_loss_list.append(loss.item())
        
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # eval on test set
    model.eval()
    total_loss = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            mask = targets > -.1
            loss = loss_fn(outputs[mask], targets[mask])
            total_loss.append(loss.item())
    total_loss = np.mean(total_loss)
    test_loss_list.append(total_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {total_loss:.4f}")
    

    x = np.arange(len(train_loss_list)) / (len(train_loss_list)-1)
plt.plot(x, train_loss_list)

x = np.arange(len(test_loss_list)) / (len(test_loss_list)-1)
plt.plot(x, test_loss_list)

# save train and test loss
np.save(predictions_folder+'nn_train_loss.npy', np.array(train_loss_list))
np.save(predictions_folder+'nn_test_loss.npy', np.array(test_loss_list))
print("Train loss saved to: ", predictions_folder+'nn_train_loss.npy') 
print("Test loss saved to: ", predictions_folder+'nn_test_loss.npy')

# predict on the whole dataset
Y_pred = np.zeros((X.shape[0], Y.shape[1]))
model.eval()
with torch.no_grad():
    for i in range(0, X.shape[0], batch_size):
        inputs = X_1hot[i:i+batch_size]
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        outputs = model(inputs)
        Y_pred[i:i+batch_size] = outputs.cpu().numpy()

# save the whole results for later evaluation
predictions_folder = './predictions/'    
# ensure the directory exists
os.makedirs(predictions_folder, exist_ok=True)
np.save(predictions_folder+'nn'+'.npy', Y_pred)
print("NN predictions saved to: ", predictions_folder+'nn'+'.npy')

# compute l1 and MSE loss on test set
tmp_mask = (Y_test >= -.1)
l1_nn = np.abs(Y_pred[test_mask][tmp_mask] - Y_test[tmp_mask]).mean()
mse_nn = ((Y_pred[test_mask][tmp_mask] - Y_test[tmp_mask])**2).mean()
print(f"Mean L1 loss of NN: {l1_nn}")
print(f"MSE loss of NN: {mse_nn}")