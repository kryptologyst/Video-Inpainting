#!/usr/bin/env python3
"""Evaluation script for video inpainting models."""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.video_inpainting import UNetVideoInpainting, EDVRVideoInpainting
from src.data.dataset import create_data_loaders
from src.eval.metrics import MetricsCalculator
from src.utils.device import get_device, load_checkpoint
from src.utils.config import load_config, create_default_config


def evaluate_model(
    model_path: str,
    config_path: str = None,
    output_dir: str = "evaluation_results"
):
    """Evaluate a trained model.
    
    Args:
        model_path: Path to model checkpoint.
        config_path: Path to configuration file.
        output_dir: Directory to save evaluation results.
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = create_default_config()
    
    device = get_device()
    
    # Create model
    model_type = config.get("model.type", "unet")
    if model_type == "unet":
        model = UNetVideoInpainting(
            in_channels=config.get("model.in_channels", 3),
            out_channels=config.get("model.out_channels", 3),
            base_channels=config.get("model.base_channels", 64),
            num_frames=config.get("model.max_frames", 8)
        )
    elif model_type == "edvr":
        model = EDVRVideoInpainting(
            in_channels=config.get("model.in_channels", 3),
            out_channels=config.get("model.out_channels", 3),
            num_frames=config.get("model.max_frames", 8),
            base_channels=config.get("model.base_channels", 64)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, None, model_path, device)
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Model trained for {checkpoint['epoch']} epochs")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config.to_dict())
    
    # Initialize metrics calculator
    metrics_calculator = MetricsCalculator(device)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            # Move data to device
            masked_frames = batch['masked_frames'].to(device)
            target_frames = batch['frames'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            pred_frames = model(masked_frames, mask)
            
            # Calculate metrics
            metrics = metrics_calculator.calculate_all_metrics(
                pred_frames, target_frames, mask
            )
            all_metrics.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in avg_metrics.items():
        print(f"{key.upper()}: {value:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    
    with open(results_file, "w") as f:
        f.write("Video Inpainting Model Evaluation\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_type.upper()}\n")
        f.write(f"Checkpoint: {model_path}\n")
        f.write(f"Epochs trained: {checkpoint['epoch']}\n")
        f.write(f"Validation samples: {len(val_loader.dataset)}\n")
        f.write("\nMetrics:\n")
        for key, value in avg_metrics.items():
            f.write(f"{key.upper()}: {value:.4f}\n")
    
    print(f"\nResults saved to {results_file}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate video inpainting model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model checkpoint {args.model} not found!")
        return
    
    evaluate_model(args.model, args.config, args.output_dir)


if __name__ == "__main__":
    main()
