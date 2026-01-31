"""Unit tests for video inpainting project."""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.video_inpainting import (
    UNetVideoInpainting, EDVRVideoInpainting, CharbonnierLoss, PerceptualLoss
)
from src.data.dataset import VideoInpaintingDataset, SyntheticVideoDataset
from src.eval.metrics import MetricsCalculator
from src.utils.device import get_device, set_seed
from src.utils.video import (
    frames_to_tensor, tensor_to_frames, create_random_mask, apply_mask
)


class TestModels:
    """Test model implementations."""
    
    def test_unet_forward(self):
        """Test UNet forward pass."""
        model = UNetVideoInpainting()
        device = get_device()
        model.to(device)
        
        # Create dummy input
        batch_size = 2
        num_frames = 4
        channels = 3
        height, width = 64, 64
        
        x = torch.randn(batch_size, num_frames, channels, height, width).to(device)
        mask = torch.randint(0, 2, (batch_size, num_frames, 1, height, width)).float().to(device)
        
        # Forward pass
        output = model(x, mask)
        
        # Check output shape
        assert output.shape == (batch_size, num_frames, channels, height, width)
        assert not torch.isnan(output).any()
    
    def test_edvr_forward(self):
        """Test EDVR forward pass."""
        model = EDVRVideoInpainting()
        device = get_device()
        model.to(device)
        
        # Create dummy input
        batch_size = 2
        num_frames = 4
        channels = 3
        height, width = 64, 64
        
        x = torch.randn(batch_size, num_frames, channels, height, width).to(device)
        mask = torch.randint(0, 2, (batch_size, num_frames, 1, height, width)).float().to(device)
        
        # Forward pass
        output = model(x, mask)
        
        # Check output shape
        assert output.shape == (batch_size, 1, channels, height, width)
        assert not torch.isnan(output).any()
    
    def test_charbonnier_loss(self):
        """Test Charbonnier loss."""
        loss_fn = CharbonnierLoss()
        
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestDataPipeline:
    """Test data loading and processing."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        dataset = SyntheticVideoDataset(
            num_samples=10,
            max_frames=4,
            target_size=(64, 64),
            mask_ratio=0.1
        )
        
        assert len(dataset) == 10
        
        # Test getting a sample
        sample = dataset[0]
        
        assert 'frames' in sample
        assert 'masked_frames' in sample
        assert 'mask' in sample
        
        assert sample['frames'].shape[0] == 4  # num_frames
        assert sample['frames'].shape[1] == 3  # channels
        assert sample['frames'].shape[2] == 64  # height
        assert sample['frames'].shape[3] == 64  # width
    
    def test_video_utils(self):
        """Test video utility functions."""
        # Test frame conversion
        frames = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        tensor = frames_to_tensor(frames)
        
        assert tensor.shape == (4, 3, 64, 64)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        
        # Test conversion back
        frames_back = tensor_to_frames(tensor)
        assert frames_back.shape == (4, 64, 64, 3)
        assert frames_back.dtype == np.uint8
        
        # Test mask creation
        mask = create_random_mask(64, 64, 0.1, 'random')
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert mask.sum() > 0  # Some pixels should be masked


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_metrics_calculator(self):
        """Test metrics calculation."""
        device = get_device()
        calculator = MetricsCalculator(device)
        
        # Create dummy data
        pred = torch.randn(2, 3, 64, 64).to(device)
        target = torch.randn(2, 3, 64, 64).to(device)
        
        # Calculate metrics
        metrics = calculator.calculate_all_metrics(pred, target)
        
        assert 'psnr' in metrics
        assert 'ssim' in metrics
        assert 'lpips' in metrics
        
        assert metrics['psnr'] >= 0
        assert 0 <= metrics['ssim'] <= 1
        assert metrics['lpips'] >= 0
    
    def test_psnr_calculation(self):
        """Test PSNR calculation."""
        device = get_device()
        calculator = MetricsCalculator(device)
        
        # Test with identical images (should give high PSNR)
        img = torch.randn(1, 3, 64, 64).to(device)
        psnr = calculator.calculate_psnr(img, img)
        
        assert psnr > 30  # Should be very high for identical images
        
        # Test with different images
        img2 = torch.randn(1, 3, 64, 64).to(device)
        psnr2 = calculator.calculate_psnr(img, img2)
        
        assert psnr2 < psnr  # Should be lower for different images


class TestUtils:
    """Test utility functions."""
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        rand1 = torch.randn(1)
        
        set_seed(42)
        rand2 = torch.randn(1)
        
        # Should be the same with same seed
        assert torch.allclose(rand1, rand2)


if __name__ == "__main__":
    pytest.main([__file__])
