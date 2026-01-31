"""Quick start script for video inpainting."""

#!/usr/bin/env python3
"""
Quick start script for video inpainting project.
This script demonstrates the basic usage of the video inpainting models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.video_inpainting import UNetVideoInpainting
from src.data.dataset import SyntheticVideoDataset
from src.utils.device import get_device, set_seed
from src.utils.video import tensor_to_frames
from src.eval.metrics import MetricsCalculator


def main():
    """Run a quick demonstration of video inpainting."""
    print("Video Inpainting Quick Start")
    print("=" * 40)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing UNet model...")
    model = UNetVideoInpainting(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        num_frames=8
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create synthetic data
    print("Generating synthetic video data...")
    dataset = SyntheticVideoDataset(
        num_samples=1,
        max_frames=8,
        target_size=(128, 128),
        mask_ratio=0.15,
        mask_type='random'
    )
    
    sample = dataset[0]
    original_frames = sample['frames']
    masked_frames = sample['masked_frames']
    mask = sample['mask']
    
    print(f"Original frames shape: {original_frames.shape}")
    print(f"Masked frames shape: {masked_frames.shape}")
    
    # Perform inpainting
    print("Performing video inpainting...")
    model.eval()
    
    with torch.no_grad():
        masked_frames_tensor = masked_frames.unsqueeze(0).to(device)
        mask_tensor = mask.unsqueeze(0).to(device)
        
        inpainted_frames = model(masked_frames_tensor, mask_tensor)
        inpainted_frames = inpainted_frames.squeeze(0)
    
    # Convert back to numpy
    inpainted_np = tensor_to_frames(inpainted_frames.cpu())
    original_np = tensor_to_frames(original_frames)
    masked_np = tensor_to_frames(masked_frames)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics_calc = MetricsCalculator(device)
    
    metrics = metrics_calc.calculate_all_metrics(
        inpainted_frames.unsqueeze(0),
        original_frames.unsqueeze(0).to(device),
        mask.unsqueeze(0).to(device)
    )
    
    print("\nResults:")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"LPIPS: {metrics['lpips']:.4f}")
    
    if 'temporal_consistency' in metrics:
        print(f"Temporal Consistency: {metrics['temporal_consistency']:.4f}")
    
    # Save visualization
    print("\nSaving visualization...")
    num_frames_to_show = 4
    
    fig, axes = plt.subplots(3, num_frames_to_show, figsize=(15, 9))
    fig.suptitle('Video Inpainting Results', fontsize=16)
    
    for i in range(num_frames_to_show):
        # Original frame
        axes[0, i].imshow(original_np[i])
        axes[0, i].set_title(f'Original Frame {i+1}')
        axes[0, i].axis('off')
        
        # Masked frame
        axes[1, i].imshow(masked_np[i])
        axes[1, i].set_title(f'Masked Frame {i+1}')
        axes[1, i].axis('off')
        
        # Inpainted frame
        axes[2, i].imshow(inpainted_np[i])
        axes[2, i].set_title(f'Inpainted Frame {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Create assets directory if it doesn't exist
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    plt.savefig(assets_dir / "quick_start_results.png", dpi=150, bbox_inches='tight')
    print(f"Results saved to {assets_dir / 'quick_start_results.png'}")
    
    print("\nQuick start completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python scripts/train.py' to train a model")
    print("2. Run 'streamlit run demo/app.py' to launch the interactive demo")
    print("3. Check out the notebooks/ directory for more examples")


if __name__ == "__main__":
    main()
