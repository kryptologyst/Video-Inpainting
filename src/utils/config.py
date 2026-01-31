"""Configuration management for video inpainting."""

import os
from typing import Dict, Any, Optional
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra


class Config:
    """Configuration class for video inpainting."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file.
        """
        if config_path is not None:
            self.cfg = OmegaConf.load(config_path)
        else:
            self.cfg = OmegaConf.create()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return OmegaConf.select(self.cfg, key, default=default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        OmegaConf.set(self.cfg, key, value)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary.
        
        Args:
            config_dict: Dictionary of configuration values.
        """
        self.cfg.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return OmegaConf.to_container(self.cfg, resolve=True)
    
    def save(self, path: str) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        OmegaConf.save(self.cfg, path)
    
    def merge(self, other_config: 'Config') -> None:
        """Merge with another configuration.
        
        Args:
            other_config: Another Config instance.
        """
        self.cfg = OmegaConf.merge(self.cfg, other_config.cfg)


def load_config(config_path: str) -> Config:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Config instance.
    """
    return Config(config_path)


def create_default_config() -> Config:
    """Create default configuration.
    
    Returns:
        Config instance with default values.
    """
    default_config = {
        'model': {
            'type': 'unet',
            'in_channels': 3,
            'out_channels': 3,
            'base_channels': 64,
            'max_frames': 8
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 4,
            'learning_rate': 1e-4,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'loss_type': 'charbonnier'
        },
        'data': {
            'max_frames': 8,
            'target_size': [256, 256],
            'mask_ratio': 0.1,
            'mask_type': 'random',
            'num_workers': 4,
            'num_train_samples': 1000,
            'num_val_samples': 200
        },
        'logging': {
            'log_dir': 'logs',
            'log_interval': 10,
            'use_wandb': False,
            'project_name': 'video-inpainting'
        },
        'paths': {
            'checkpoint_dir': 'checkpoints',
            'data_dir': 'data',
            'output_dir': 'outputs'
        },
        'system': {
            'seed': 42,
            'device': 'auto',
            'mixed_precision': False
        }
    }
    
    config = Config()
    config.update(default_config)
    return config


def setup_hydra_config() -> None:
    """Setup Hydra configuration."""
    cs = ConfigStore.instance()
    
    # Register default configuration
    default_config = create_default_config()
    cs.store(name="config", node=default_config.cfg)
    
    # Register model configurations
    cs.store(group="model", name="unet", node={
        'type': 'unet',
        'in_channels': 3,
        'out_channels': 3,
        'base_channels': 64,
        'max_frames': 8
    })
    
    cs.store(group="model", name="edvr", node={
        'type': 'edvr',
        'in_channels': 3,
        'out_channels': 3,
        'num_frames': 8,
        'base_channels': 64
    })
    
    # Register training configurations
    cs.store(group="training", name="default", node={
        'num_epochs': 100,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'loss_type': 'charbonnier'
    })
    
    cs.store(group="training", name="fast", node={
        'num_epochs': 20,
        'batch_size': 8,
        'learning_rate': 2e-4,
        'optimizer': 'adamw',
        'scheduler': 'step',
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'loss_type': 'mse'
    })


def get_config_from_hydra() -> Config:
    """Get configuration from Hydra.
    
    Returns:
        Config instance.
    """
    if GlobalHydra.instance().is_initialized():
        cfg = hydra.utils.instantiate(hydra.core.hydra_config.HydraConfig.get().config)
        return Config().update(cfg)
    else:
        return create_default_config()
