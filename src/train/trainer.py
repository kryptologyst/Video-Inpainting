"""Training script for video inpainting models."""

import os
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import numpy as np

from ..models.video_inpainting import (
    UNetVideoInpainting, EDVRVideoInpainting, CharbonnierLoss, PerceptualLoss
)
from ..data.dataset import create_data_loaders
from ..eval.metrics import MetricsCalculator, MetricsTracker
from ..utils.device import get_device, set_seed, save_checkpoint, load_checkpoint


class VideoInpaintingTrainer:
    """Trainer for video inpainting models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.device = get_device()
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss functions
        self.loss_fn = self._create_loss_function()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(self.device)
        
        # Initialize data loaders
        self.train_loader, self.val_loader = create_data_loaders(config)
        
        # Initialize logging
        self.writer = SummaryWriter(config.get('log_dir', 'logs'))
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project=config.get('project_name', 'video-inpainting'),
                config=config
            )
        
        # Training state
        self.current_epoch = 0
        self.best_psnr = 0.0
        self.global_step = 0
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model_type = self.config.get('model_type', 'unet')
        
        if model_type == 'unet':
            return UNetVideoInpainting(
                in_channels=self.config.get('in_channels', 3),
                out_channels=self.config.get('out_channels', 3),
                base_channels=self.config.get('base_channels', 64),
                num_frames=self.config.get('max_frames', 8)
            )
        elif model_type == 'edvr':
            return EDVRVideoInpainting(
                in_channels=self.config.get('in_channels', 3),
                out_channels=self.config.get('out_channels', 3),
                num_frames=self.config.get('max_frames', 8),
                base_channels=self.config.get('base_channels', 64)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-4)
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', None)
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 100)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        else:
            return None
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function."""
        loss_type = self.config.get('loss_type', 'charbonnier')
        
        if loss_type == 'charbonnier':
            return CharbonnierLoss()
        elif loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'perceptual':
            return PerceptualLoss(self.device)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics_tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            masked_frames = batch['masked_frames'].to(self.device)
            target_frames = batch['frames'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_frames = self.model(masked_frames, mask)
            
            # Calculate loss
            loss = self.loss_fn(pred_frames, target_frames)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', None) is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = self.metrics_calculator.calculate_all_metrics(
                    pred_frames, target_frames, mask
                )
                metrics['loss'] = loss.item()
                metrics_tracker.update(metrics)
            
            # Logging
            self.global_step += 1
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/PSNR', metrics['psnr'], self.global_step)
                self.writer.add_scalar('Train/SSIM', metrics['ssim'], self.global_step)
                
                if self.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/psnr': metrics['psnr'],
                        'train/ssim': metrics['ssim'],
                        'epoch': self.current_epoch,
                        'step': self.global_step
                    })
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'psnr': f"{metrics['psnr']:.2f}",
                'ssim': f"{metrics['ssim']:.4f}"
            })
        
        return metrics_tracker.get_average()
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                masked_frames = batch['masked_frames'].to(self.device)
                target_frames = batch['frames'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Forward pass
                pred_frames = self.model(masked_frames, mask)
                
                # Calculate loss
                loss = self.loss_fn(pred_frames, target_frames)
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_all_metrics(
                    pred_frames, target_frames, mask
                )
                metrics['loss'] = loss.item()
                metrics_tracker.update(metrics)
        
        return metrics_tracker.get_average()
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config.get('num_epochs', 100)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            print(f"Epoch {epoch}:")
            print(f"  Train: {train_metrics}")
            print(f"  Val: {val_metrics}")
            
            # Log to tensorboard
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                    'epoch': epoch
                })
            
            # Save checkpoint
            if val_metrics['psnr'] > self.best_psnr:
                self.best_psnr = val_metrics['psnr']
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['loss'],
                    os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'best_model.pth'),
                    metrics=val_metrics
                )
            
            # Save latest checkpoint
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_metrics['loss'],
                os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'latest_model.pth'),
                metrics=val_metrics
            )
        
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = load_checkpoint(
            self.model,
            self.optimizer,
            checkpoint_path,
            self.device
        )
        self.current_epoch = checkpoint['epoch']
        self.best_psnr = checkpoint.get('metrics', {}).get('psnr', 0.0)
        return checkpoint
