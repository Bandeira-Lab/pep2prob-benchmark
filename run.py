"""
Main script for training various models on the pep2prob benchmark.
Supports multiple model types: Global, BoF, LR, ResNN, Transformer
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import numpy as np
import random

from pep2prob.dataset import Pep2ProbDataSet


def parse_arguments():
    """Parse command line arguments for model training."""
    parser = argparse.ArgumentParser(
        description="Train models on the pep2prob benchmark dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["Global", "BoF", "LR", "ResNN", "Transformer"],
        help="Model type to train"
    )
    
    # Data split index
    parser.add_argument(
        "--data_split",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Data split index (1-5)"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay for optimizer"
    )
    
    # Input sequence length constraints
    parser.add_argument(
        "--min_length_input",
        type=int,
        default=7,
        help="Minimum input sequence length"
    )
    
    parser.add_argument(
        "--max_length_input",
        type=int,
        default=40,
        help="Maximum input sequence length"
    )
    
    parser.add_argument(
        "--min_num_psm_for_statistics",
        type=int,
        default=10,
        help="Minimum number of PSMs required for statistics calculation"
    )
    
    # Additional optional parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save model outputs and checkpoints"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training (auto will detect best available)"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate the parsed arguments."""
    # Validate sequence length constraints
    if args.min_length_input >= args.max_length_input:
        raise ValueError(
            f"min_length_input ({args.min_length_input}) must be less than "
            f"max_length_input ({args.max_length_input})"
        )
    
    if args.min_length_input < 1:
        raise ValueError("min_length_input must be at least 1")
    
    # Validate training parameters
    if args.epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    
    if args.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    return True


def setup_device(device_arg):
    """Setup and return the appropriate device for training."""
    import torch
    
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    return device


def main():
    """Main function to run the training pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    try:
        validate_arguments(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print configuration
    print("=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Data split: {args.data_split}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Min input length: {args.min_length_input}")
    print(f"Max input length: {args.max_length_input}")
    print(f"Min PSMs for statistics: {args.min_num_psm_for_statistics}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 50)
    
    # Setup device
    device = setup_device(args.device)
    
    # Set random seed for reproducibility  
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Setup paths
    this_dir = Path(__file__).parent.resolve()
    data_dir = this_dir / "data"

    # Load data
    print("Loading data...")
    dataset = Pep2ProbDataSet(data_dir, split_idx=args.data_split, min_length_input=args.min_length_input, max_length_input=args.max_length_input, min_num_psm_for_statistics=args.min_num_psm_for_statistics, skip_download_if_exists=True)

    # TODO: Import and initialize the selected model
    print(f"\nInitializing {args.model} model...")
    
    if args.model == "Transformer":
        # Import Transformer model
        print("Loading Transformer model...")
        # TODO: from models.Transformer_baseline.main import train_transformer
        # train_transformer(args)
    elif args.model == "Global":
        print("Loading Global model...")
        # TODO: Implement Global model training
    elif args.model == "BoF":
        print("Loading Bag of Features model...")
        # TODO: Implement BoF model training
    elif args.model == "LR":
        print("Loading Linear Regression model...")
        # TODO: Implement LR model training
    elif args.model == "ResNN":
        print("Loading Residual Neural Network model...")
        # TODO: Implement ResNN model training
    
    print(f"\nTraining {args.model} model with data split {args.data_split}...")
    print("Training pipeline not yet implemented. This is a template structure.")
    
    # TODO: Load data based on data_split
    # TODO: Initialize model
    # TODO: Run training loop
    # TODO: Save results to output_dir
    
    print("Training completed!")


if __name__ == "__main__":
    main()