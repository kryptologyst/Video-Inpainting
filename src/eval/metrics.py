"""Evaluation metrics for video inpainting."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from skimage.metrics import structural_similarity as ssim
import cv2


class MetricsCalculator:
    """Calculator for video inpainting evaluation metrics."""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # Initialize LPIPS model if available
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_available = True
        except ImportError:
            self.lpips_model = None
            self.lpips_available = False
    
    def calculate_psnr(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            pred: Predicted frames (B, T, C, H, W) or (B, C, H, W).
            target: Target frames (B, T, C, H, W) or (B, C, H, W).
            mask: Optional mask tensor (B, T, 1, H, W) or (B, 1, H, W).
            
        Returns:
            float: PSNR value in dB.
        """
        if pred.dim() == 5:  # Video
            pred = pred.view(-1, *pred.shape[2:])
            target = target.view(-1, *target.shape[2:])
            if mask is not None:
                mask = mask.view(-1, *mask.shape[2:])
        
        if mask is not None:
            # Only compute PSNR on unmasked regions
            pred_masked = pred * (1 - mask)
            target_masked = target * (1 - mask)
            mse = F.mse_loss(pred_masked, target_masked)
        else:
            mse = F.mse_loss(pred, target)
        
        if mse == 0:
            return float('inf')
        
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    
    def calculate_ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """Calculate Structural Similarity Index (SSIM).
        
        Args:
            pred: Predicted frames (B, T, C, H, W) or (B, C, H, W).
            target: Target frames (B, T, C, H, W) or (B, C, H, W).
            mask: Optional mask tensor (B, T, 1, H, W) or (B, 1, H, W).
            
        Returns:
            float: SSIM value.
        """
        if pred.dim() == 5:  # Video
            pred = pred.view(-1, *pred.shape[2:])
            target = target.view(-1, *target.shape[2:])
            if mask is not None:
                mask = mask.view(-1, *mask.shape[2:])
        
        # Convert to numpy for SSIM calculation
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        if mask is not None:
            mask_np = mask.detach().cpu().numpy()
            pred_np = pred_np * (1 - mask_np)
            target_np = target_np * (1 - mask_np)
        
        # Calculate SSIM for each image
        ssim_values = []
        for i in range(pred_np.shape[0]):
            if pred_np[i].shape[0] == 3:  # RGB
                ssim_val = ssim(
                    pred_np[i].transpose(1, 2, 0),
                    target_np[i].transpose(1, 2, 0),
                    multichannel=True,
                    channel_axis=2
                )
            else:  # Grayscale
                ssim_val = ssim(
                    pred_np[i, 0],
                    target_np[i, 0]
                )
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    def calculate_lpips(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """Calculate Learned Perceptual Image Patch Similarity (LPIPS).
        
        Args:
            pred: Predicted frames (B, T, C, H, W) or (B, C, H, W).
            target: Target frames (B, T, C, H, W) or (B, C, H, W).
            mask: Optional mask tensor (B, T, 1, H, W) or (B, 1, H, W).
            
        Returns:
            float: LPIPS value.
        """
        if not self.lpips_available:
            return 0.0
        
        if pred.dim() == 5:  # Video
            pred = pred.view(-1, *pred.shape[2:])
            target = target.view(-1, *target.shape[2:])
            if mask is not None:
                mask = mask.view(-1, *mask.shape[2:])
        
        if mask is not None:
            # Only compute LPIPS on unmasked regions
            pred_masked = pred * (1 - mask)
            target_masked = target * (1 - mask)
        else:
            pred_masked = pred
            target_masked = target
        
        # Normalize to [-1, 1] for LPIPS
        pred_norm = pred_masked * 2.0 - 1.0
        target_norm = target_masked * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_val = self.lpips_model(pred_norm, target_norm)
        
        return lpips_val.mean().item()
    
    def calculate_temporal_consistency(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate temporal consistency metrics.
        
        Args:
            pred: Predicted video (B, T, C, H, W).
            target: Target video (B, T, C, H, W).
            
        Returns:
            Dict containing temporal consistency metrics.
        """
        B, T, C, H, W = pred.shape
        
        # Calculate optical flow between consecutive frames
        pred_flows = []
        target_flows = []
        
        for b in range(B):
            for t in range(T - 1):
                # Convert to numpy for optical flow
                pred_frame1 = pred[b, t].detach().cpu().numpy().transpose(1, 2, 0)
                pred_frame2 = pred[b, t + 1].detach().cpu().numpy().transpose(1, 2, 0)
                target_frame1 = target[b, t].detach().cpu().numpy().transpose(1, 2, 0)
                target_frame2 = target[b, t + 1].detach().cpu().numpy().transpose(1, 2, 0)
                
                # Convert to grayscale
                pred_gray1 = cv2.cvtColor(pred_frame1, cv2.COLOR_RGB2GRAY)
                pred_gray2 = cv2.cvtColor(pred_frame2, cv2.COLOR_RGB2GRAY)
                target_gray1 = cv2.cvtColor(target_frame1, cv2.COLOR_RGB2GRAY)
                target_gray2 = cv2.cvtColor(target_frame2, cv2.COLOR_RGB2GRAY)
                
                # Calculate optical flow
                pred_flow = cv2.calcOpticalFlowPyrLK(
                    pred_gray1, pred_gray2, None, None
                )[0]
                target_flow = cv2.calcOpticalFlowPyrLK(
                    target_gray1, target_gray2, None, None
                )[0]
                
                if pred_flow is not None and target_flow is not None:
                    pred_flows.append(pred_flow)
                    target_flows.append(target_flow)
        
        if not pred_flows:
            return {'temporal_consistency': 0.0, 'flow_magnitude': 0.0}
        
        # Calculate flow magnitude difference
        flow_diffs = []
        for pred_flow, target_flow in zip(pred_flows, target_flows):
            pred_mag = np.sqrt(pred_flow[:, :, 0]**2 + pred_flow[:, :, 1]**2)
            target_mag = np.sqrt(target_flow[:, :, 0]**2 + target_flow[:, :, 1]**2)
            flow_diff = np.mean(np.abs(pred_mag - target_mag))
            flow_diffs.append(flow_diff)
        
        temporal_consistency = 1.0 / (1.0 + np.mean(flow_diffs))
        flow_magnitude = np.mean(flow_diffs)
        
        return {
            'temporal_consistency': temporal_consistency,
            'flow_magnitude': flow_magnitude
        }
    
    def calculate_all_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics.
        
        Args:
            pred: Predicted frames (B, T, C, H, W) or (B, C, H, W).
            target: Target frames (B, T, C, H, W) or (B, C, H, W).
            mask: Optional mask tensor (B, T, 1, H, W) or (B, 1, H, W).
            
        Returns:
            Dict containing all metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics['psnr'] = self.calculate_psnr(pred, target, mask)
        metrics['ssim'] = self.calculate_ssim(pred, target, mask)
        metrics['lpips'] = self.calculate_lpips(pred, target, mask)
        
        # Temporal consistency (only for video)
        if pred.dim() == 5:
            temporal_metrics = self.calculate_temporal_consistency(pred, target)
            metrics.update(temporal_metrics)
        
        return metrics


class MetricsTracker:
    """Track metrics across batches."""
    
    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update current metrics.
        
        Args:
            metrics: Dictionary of metric values.
        """
        self.current_metrics.update(metrics)
    
    def reset(self):
        """Reset current metrics."""
        self.current_metrics = {}
    
    def get_average(self) -> Dict[str, float]:
        """Get average metrics across all updates.
        
        Returns:
            Dict containing average metric values.
        """
        if not self.current_metrics:
            return {}
        
        return {k: v for k, v in self.current_metrics.items()}
    
    def log_metrics(self, prefix: str = "") -> str:
        """Format metrics for logging.
        
        Args:
            prefix: Prefix for metric names.
            
        Returns:
            Formatted string of metrics.
        """
        if not self.current_metrics:
            return ""
        
        metrics_str = []
        for key, value in self.current_metrics.items():
            if isinstance(value, float):
                metrics_str.append(f"{prefix}{key}: {value:.4f}")
            else:
                metrics_str.append(f"{prefix}{key}: {value}")
        
        return " | ".join(metrics_str)
