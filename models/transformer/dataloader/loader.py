import torch
from torch.utils.data import Dataset
from .tokenizer import build_token_vocab, encode_input


def get_ion_mask(seq_len, charge, max_seq_len=50):
    mask = [True]  # a2+ ions
    for ion in range(1, 3):  # b/y ions
        for chr in range(1, 4):
            for seq_idx in range(1, max_seq_len):
                mask.append(chr <= charge and seq_idx < seq_len)
    return mask

class MS2Dataset(Dataset):
    def __init__(self, df, matrix_data, token_to_id, max_length_input=40):
        self.df = df.reset_index(drop=True)
        self.matrix_data = matrix_data
        self.max_length_input = max_length_input
        self.token_to_id = token_to_id
        self.output_length = self.matrix_data.shape[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.loc[idx, 'peptide']
        charge = self.df.loc[idx, 'charge']
        input_ids = encode_input(seq, charge, self.token_to_id, self.max_length_input)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        ion_mask = get_ion_mask(len(seq), charge, self.max_length_input)
        ion_mask = torch.tensor(ion_mask, dtype=torch.bool)

        row = self.matrix_data[idx]
        probability = torch.tensor(row, dtype=torch.float)

        combined_input = torch.cat([input_ids, torch.full((self.output_length,), self.token_to_id['[PAD]'], dtype=torch.long)])
        output_pos_tensor = (combined_input == self.token_to_id["[OUTPUT]"]).nonzero(as_tuple=True)[0]
        output_pos = output_pos_tensor.item() if output_pos_tensor.numel() > 0 else -1

        return combined_input, probability, output_pos, ion_mask