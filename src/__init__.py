"""Video inpainting package."""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models.video_inpainting import (
    UNetVideoInpainting,
    EDVRVideoInpainting,
    CharbonnierLoss,
    PerceptualLoss
)

from .data.dataset import (
    VideoInpaintingDataset,
    SyntheticVideoDataset,
    create_data_loaders
)

from .eval.metrics import (
    MetricsCalculator,
    MetricsTracker
)

from .utils.device import (
    get_device,
    set_seed,
    count_parameters,
    save_checkpoint,
    load_checkpoint
)

from .utils.video import (
    load_video_frames,
    frames_to_tensor,
    tensor_to_frames,
    create_random_mask,
    apply_mask,
    compute_optical_flow,
    warp_frame
)

from .utils.config import (
    Config,
    load_config,
    create_default_config
)

__all__ = [
    # Models
    "UNetVideoInpainting",
    "EDVRVideoInpainting", 
    "CharbonnierLoss",
    "PerceptualLoss",
    
    # Data
    "VideoInpaintingDataset",
    "SyntheticVideoDataset",
    "create_data_loaders",
    
    # Evaluation
    "MetricsCalculator",
    "MetricsTracker",
    
    # Utils
    "get_device",
    "set_seed",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "load_video_frames",
    "frames_to_tensor",
    "tensor_to_frames",
    "create_random_mask",
    "apply_mask",
    "compute_optical_flow",
    "warp_frame",
    "Config",
    "load_config",
    "create_default_config",
]
