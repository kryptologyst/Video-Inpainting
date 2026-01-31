#!/usr/bin/env python3
"""Main training script for video inpainting."""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.train.trainer import VideoInpaintingTrainer
from src.utils.config import load_config, create_default_config
from src.utils.device import set_seed, ensure_dir


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train video inpainting model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["unet", "edvr"],
        default="unet",
        help="Model type to train"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for training"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file {args.config} not found, using default configuration")
        config = create_default_config()
    
    # Override config with command line arguments
    config.set("model.type", args.model)
    config.set("training.num_epochs", args.epochs)
    config.set("training.batch_size", args.batch_size)
    config.set("training.learning_rate", args.lr)
    config.set("paths.output_dir", args.output_dir)
    
    # Set up directories
    output_dir = Path(args.output_dir)
    ensure_dir(str(output_dir))
    ensure_dir(str(output_dir / "checkpoints"))
    ensure_dir(str(output_dir / "logs"))
    
    # Set seed
    set_seed(config.get("system.seed", 42))
    
    # Print configuration
    print("Configuration:")
    print(f"  Model: {config.get('model.type')}")
    print(f"  Epochs: {config.get('training.num_epochs')}")
    print(f"  Batch size: {config.get('training.batch_size')}")
    print(f"  Learning rate: {config.get('training.learning_rate')}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Use synthetic data: {args.synthetic}")
    
    # Create trainer
    trainer = VideoInpaintingTrainer(config.to_dict())
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        else:
            print(f"Checkpoint {args.resume} not found, starting from scratch")
    
    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == "__main__":
    main()
