# evaluate.py

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def run_inference(model, dataset, device, batch_size, output_length, token_to_id, output_path):
    """
    Run inference on the entire dataset and save predicted probabilities.

    Args:
        model: Trained model (should be in eval mode)
        dataset: MS2Dataset to run predictions on
        device: torch.device (cuda or cpu)
        batch_size: DataLoader batch size
        output_length: Number of output positions to extract
        token_to_id: Token vocabulary (used to find [OUTPUT] position)
        output_path: Where to save final predictions
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    all_preds = []

    with torch.no_grad():
        for input_ids, _, output_pos, _ in tqdm(loader, desc="Predicting"):
            input_ids = input_ids.to(device)
            output_pos = output_pos.to(device)

            logits = model(input_ids)             # [B, L, 1]
            probs = torch.sigmoid(logits).squeeze(-1)  # [B, L]

            B, L = probs.shape
            positions = torch.arange(L, device=device).unsqueeze(0)   # [1, L]
            start = output_pos.unsqueeze(1) + 1                       # [B, 1]
            mask = (positions >= start) & (positions < start + output_length)  # [B, L]

            batch_preds = torch.stack([
                probs[i, mask[i]]
                for i in range(B)
            ], dim=0)  # [B, output_length]

            all_preds.append(batch_preds.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    np.save(output_path, all_preds)
    print(f"Saved predictions to {output_path}")