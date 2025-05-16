# train/trainer.py

import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, scheduler, device, output_length, criterion=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_length = output_length
        self.criterion = criterion or nn.L1Loss(reduction='none')

        self.train_losses = []
        self.test_losses = []
        self.mse_list = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids, targets, out_pos, ion_mask = [x.to(self.device) for x in batch]
            preds = self.model(input_ids)

            B, L, _ = preds.size()
            mask = torch.zeros(B, L, dtype=torch.bool, device=self.device)
            for i in range(B):
                start = out_pos[i].item() + 1
                mask[i, start:start + self.output_length] = True

            preds_output = torch.sigmoid(preds[mask].view(B, self.output_length, 1))
            ion_mask = ion_mask.float().view(B, self.output_length, 1)
            loss_token = self.criterion(preds_output, targets.view(B, self.output_length, 1))
            masked_loss = loss_token * ion_mask
            loss_per_sample = masked_loss.sum(dim=(1, 2)) / (ion_mask.sum(dim=(1, 2)) + 1e-8)
            loss = loss_per_sample.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_mae = 0.0
        total_mse = 0.0

        with torch.no_grad():
            for input_ids, targets, out_pos, ion_mask in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                out_pos = out_pos.to(self.device)
                ion_mask = ion_mask.to(self.device).float().unsqueeze(-1)

                logits = self.model(input_ids)
                probs = torch.sigmoid(logits)
                B, L, _ = probs.shape

                positions = torch.arange(L, device=self.device).unsqueeze(0)
                start = out_pos.unsqueeze(1)
                valid_pos = (positions >= start + 1) & (positions < start + 1 + self.output_length)
                valid_pos = valid_pos.unsqueeze(-1)

                preds = probs[valid_pos].view(B, self.output_length, 1)
                abs_err = (preds - targets.view(B, self.output_length, 1)).abs() * ion_mask
                sq_err = (preds - targets.view(B, self.output_length, 1)).pow(2) * ion_mask

                sum_abs = abs_err.sum(dim=(1, 2))
                sum_sq = sq_err.sum(dim=(1, 2))
                cnt = ion_mask.sum(dim=(1, 2)).clamp(min=1e-8)

                mae = (sum_abs / cnt).mean().item()
                mse = (sum_sq / cnt).mean().item()

                total_mae += mae
                total_mse += mse

        avg_mae = total_mae / len(self.test_loader)
        avg_mse = total_mse / len(self.test_loader)

        self.test_losses.append(avg_mae)
        self.mse_list.append(avg_mse)
        self.scheduler.step(avg_mae)
        return avg_mae, avg_mse

    def run(self, epochs: int, save_every: int = 5, save_prefix: str = "checkpoint"):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch()
            test_loss, mse = self.evaluate()

            print(f"Train MAE: {train_loss:.4f}, Test MAE: {test_loss:.4f}, MSE: {mse:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")

            if (epoch + 1) % save_every == 0:
                torch.save(self.model.state_dict(), f"{save_prefix}_epoch{epoch + 1:02d}.pth")