
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_losses(train_loss_path, test_loss_path, mse_path, save_dir="figures"):
    """
    Load loss values from .npy files and save L1 + MSE plots.

    Args:
        train_loss_path (str): Path to .npy file for train MAE
        test_loss_path (str): Path to .npy file for test MAE
        mse_path (str): Path to .npy file for test MSE
        save_dir (str): Folder to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    train_losses = np.load(train_loss_path)
    test_losses  = np.load(test_loss_path)
    mse_losses   = np.load(mse_path)

    # Plot MAE (L1 loss)
    fig1, ax1 = plt.subplots()
    ax1.plot(train_losses, label="Train MAE")
    ax1.plot(test_losses, label="Test MAE")
    ax1.set_title("L1 Loss (Mean Absolute Error)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid()
    fig1.savefig(f"{save_dir}/l1_loss.png")
    print(f"Saved L1 loss plot to {save_dir}/l1_loss.png")

    # Plot MSE
    fig2, ax2 = plt.subplots()
    ax2.plot(mse_losses, label="Test MSE")
    ax2.set_title("Mean Squared Error")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid()
    fig2.savefig(f"{save_dir}/mse_loss.png")
    print(f"Saved MSE loss plot to {save_dir}/mse_loss.png")