
# Define special tokens
special_tokens = ["[PAD]", "[CLS]", "[CHARGE]", "[OUTPUT]"]

def build_token_vocab(sequences):
    """
    Build a vocabulary from peptide sequences and return both the token list and mapping.

    Args:
        sequences (List[str]): List of peptide sequences.

    Returns:
        all_tokens (List[str]): Full token list including special tokens.
        token_to_id (Dict[str, int]): Mapping from token to index.
    """
    amino_acids = sorted(set("".join(sequences)))
    all_tokens = special_tokens + amino_acids
    token_to_id = {t: i for i, t in enumerate(all_tokens)}
    return all_tokens, token_to_id


def encode_peptide(seq, token_to_id):
    """
    Encode a peptide sequence into token IDs using the provided vocabulary.

    Args:
        seq (str): Amino acid sequence.
        token_to_id (Dict[str, int]): Vocabulary mapping.

    Returns:
        List[int]: Encoded token IDs.
    """
    return [token_to_id[aa] for aa in seq if aa in token_to_id]


def encode_input(seq, charge, token_to_id, max_length_input):
    """
    Build the full input token ID sequence: [CLS] + peptide + [CHARGE]*c + [OUTPUT] + [PAD]...

    Args:
        seq (str): Peptide sequence.
        charge (int): Charge state.
        token_to_id (Dict[str, int]): Vocabulary.
        max_length_input (int): Max sequence length (input only).

    Returns:
        List[int]: Final input sequence of length max_length_input
    """
    CLS_ID = token_to_id["[CLS]"]
    CHARGE_ID = token_to_id["[CHARGE]"]
    OUTPUT_ID = token_to_id["[OUTPUT]"]
    PAD_ID = token_to_id["[PAD]"]

    input_ids = [CLS_ID] + encode_peptide(seq, token_to_id) + [CHARGE_ID] * charge + [OUTPUT_ID]
    input_ids = input_ids[:max_length_input]  # Truncate if too long
    input_ids += [PAD_ID] * (max_length_input - len(input_ids))  # Pad if too short
    return input_ids