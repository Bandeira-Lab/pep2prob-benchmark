import argparse
import torch
import pandas as pd
import numpy as np


# from model_prob import DecoderOnlyTransformer
from models.transformer.model_prob.transformer import DecoderOnlyTransformer
from models.transformer.dataloader.loader import MS2Dataset

from models.transformer.dataloader.tokenizer import build_token_vocab, encode_input
from models.transformer.train.trainer import Trainer
from models.transformer.evaluate import run_inference
from models.transformer.visualize import plot_losses

def parse_args():
    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument("--precursor_info_path", type=str, required=True,
                        help="Path to precursor_info.tsv")
    parser.add_argument("--split_path", type=str, required=True,
                        help="Path to train_test_split_set.npy")

    # Training options
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--save_prefix", type=str, default="model_run0")
    parser.add_argument("--max_length_input", type=int, default=40)

    # Optional flags
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--pred_output", type=str, default="predictions.npy")
    parser.add_argument("--plot", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    precursor_df = pd.read_csv(args.precursor_info_path, sep=',')
    cols = precursor_df.columns[5:]
    matrix = precursor_df[cols].to_numpy()
    split = np.load(args.split_path, allow_pickle=True).item()

    # Tokenizer
    all_tokens, token_to_id = build_token_vocab(precursor_df["peptide"].tolist())

    # Split data
    train_df = precursor_df.iloc[split["train"]].reset_index(drop=True)
    test_df  = precursor_df.iloc[split["test"]].reset_index(drop=True)
    train_matrix = matrix[split["train"]]
    test_matrix  = matrix[split["test"]]

    # Datasets
    train_dataset = MS2Dataset(train_df, train_matrix, token_to_id, max_length_input=args.max_length_input)
    test_dataset  = MS2Dataset(test_df,  test_matrix,  token_to_id, max_length_input=args.max_length_input)
    full_dataset  = MS2Dataset(precursor_df, matrix, token_to_id, max_length_input=args.max_length_input)
    output_length = train_dataset.output_length

    # Model
    model = DecoderOnlyTransformer(vocab_size=len(all_tokens), d_model=180, max_len=args.max_length_input + output_length)
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model = model.to(device)

    # Evaluation-only mode
    if args.eval_only:
        run_inference(model, full_dataset, device, args.batch_size, output_length, token_to_id, args.pred_output)
        return

    # Training setup
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    # Train
    trainer = Trainer(model, train_loader, test_loader, optimizer, scheduler, device, output_length)
    trainer.run(epochs=args.epochs, save_prefix=args.save_prefix)

    # Save logs
    np.save(f"{args.save_prefix}_train_losses.npy", np.array(trainer.train_losses))
    np.save(f"{args.save_prefix}_test_losses.npy",  np.array(trainer.test_losses))
    np.save(f"{args.save_prefix}_mse_list.npy",     np.array(trainer.mse_list))

    # Plot
    if args.plot:
        plot_losses(
            train_loss_path=f"{args.save_prefix}_train_losses.npy",
            test_loss_path=f"{args.save_prefix}_test_losses.npy",
            mse_path=f"{args.save_prefix}_mse_list.npy",
            save_dir="figures"
        )

    # Final inference
    run_inference(model, full_dataset, device, args.batch_size, output_length, token_to_id, args.pred_output)

if __name__ == "__main__":
    main()