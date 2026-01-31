"""Advanced video inpainting models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class ConvBlock(nn.Module):
    """Convolutional block with optional normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_norm: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        
        if use_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for deep networks."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels, activation=None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class UNetVideoInpainting(nn.Module):
    """UNet-based video inpainting model with temporal attention."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_frames: int = 5
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16, stride=2),
            ResidualBlock(base_channels * 16),
            ResidualBlock(base_channels * 16),
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        )
        
        # Decoder
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2),
            ConvBlock(base_channels * 8, base_channels * 4)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2),
            ConvBlock(base_channels * 4, base_channels * 2)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2),
            ConvBlock(base_channels * 2, base_channels)
        )
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(base_channels * 8)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input video tensor (B, T, C, H, W)
            mask: Mask tensor (B, T, 1, H, W)
        """
        B, T, C, H, W = x.shape
        
        # Process each frame
        outputs = []
        for t in range(T):
            frame = x[:, t]  # (B, C, H, W)
            frame_mask = mask[:, t]  # (B, 1, H, W)
            
            # Encoder
            e1 = self.enc1(frame)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            
            # Bottleneck
            b = self.bottleneck(e4)
            
            # Decoder with skip connections
            d4 = self.dec4(torch.cat([b, e4], dim=1))
            d3 = self.dec3(torch.cat([d4, e3], dim=1))
            d2 = self.dec2(torch.cat([d3, e2], dim=1))
            d1 = self.dec1(torch.cat([d2, e1], dim=1))
            
            # Output
            out = self.out_conv(d1)
            outputs.append(out)
        
        # Stack outputs
        result = torch.stack(outputs, dim=1)  # (B, T, C, H, W)
        
        # Apply temporal attention
        result = self.temporal_attention(result)
        
        return result


class TemporalAttention(nn.Module):
    """Temporal attention module for video inpainting."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        
        # Reshape for attention computation
        x_flat = x.view(B * T, C, H, W)
        
        # Compute attention
        query = self.query_conv(x_flat).view(B, T, -1, H * W)
        key = self.key_conv(x_flat).view(B, T, -1, H * W)
        value = self.value_conv(x_flat).view(B, T, -1, H * W)
        
        # Compute attention weights
        attention = torch.bmm(
            query.view(B * T, -1, H * W).transpose(1, 2),
            key.view(B * T, -1, H * W)
        )
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(
            value.view(B * T, -1, H * W),
            attention.transpose(1, 2)
        )
        out = out.view(B, T, C, H, W)
        
        return self.gamma * out + x


class EDVRVideoInpainting(nn.Module):
    """EDVR-based video inpainting model with deformable convolutions."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_frames: int = 5,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.base_channels = base_channels
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4)
        )
        
        # Temporal alignment module
        self.temporal_align = TemporalAlignment(base_channels * 4)
        
        # Reconstruction module
        self.reconstruction = nn.Sequential(
            ConvBlock(base_channels * 4 * num_frames, base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input video tensor (B, T, C, H, W)
            mask: Mask tensor (B, T, 1, H, W)
        """
        B, T, C, H, W = x.shape
        
        # Extract features for each frame
        features = []
        for t in range(T):
            feat = self.feature_extractor(x[:, t])
            features.append(feat)
        
        # Temporal alignment
        aligned_features = self.temporal_align(features)
        
        # Concatenate aligned features
        concat_features = torch.cat(aligned_features, dim=1)
        
        # Reconstruction
        output = self.reconstruction(concat_features)
        
        return output.unsqueeze(1)  # Add temporal dimension


class TemporalAlignment(nn.Module):
    """Temporal alignment module using deformable convolutions."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Offset prediction
        self.offset_conv = nn.Conv2d(channels, channels * 2, 3, 1, 1)
        
        # Deformable convolution
        self.deform_conv = nn.Conv2d(channels, channels, 3, 1, 1)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature tensors [(B, C, H, W), ...]
        """
        aligned_features = []
        
        for i, feat in enumerate(features):
            # Predict offsets
            offsets = self.offset_conv(feat)
            
            # Apply deformable convolution
            aligned_feat = self.deform_conv(feat)
            aligned_features.append(aligned_feat)
        
        return aligned_features


class CharbonnierLoss(nn.Module):
    """Charbonnier loss for video inpainting."""
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps))


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        import torchvision.models as models
        
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.ModuleList([
            vgg[:4],   # conv1_2
            vgg[:9],   # conv2_2
            vgg[:18],  # conv3_4
            vgg[:27],  # conv4_4
        ]).to(device)
        
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0
        for feature_extractor in self.features:
            pred_feat = feature_extractor(pred)
            target_feat = feature_extractor(target)
            loss += F.mse_loss(pred_feat, target_feat)
        return loss / len(self.features)
