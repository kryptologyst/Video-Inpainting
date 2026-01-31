"""Video processing utilities."""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
import decord
from decord import VideoReader


def load_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Load video frames using decord for efficient video reading.
    
    Args:
        video_path: Path to video file.
        max_frames: Maximum number of frames to load.
        target_size: Target (width, height) for resizing frames.
        
    Returns:
        np.ndarray: Video frames of shape (T, H, W, C).
    """
    vr = VideoReader(video_path)
    
    if max_frames is None:
        max_frames = len(vr)
    
    frame_indices = list(range(min(max_frames, len(vr))))
    frames = vr.get_batch(frame_indices).asnumpy()
    
    if target_size is not None:
        frames = np.array([
            cv2.resize(frame, target_size) for frame in frames
        ])
    
    return frames


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convert video frames to PyTorch tensor.
    
    Args:
        frames: Video frames of shape (T, H, W, C).
        
    Returns:
        torch.Tensor: Tensor of shape (T, C, H, W).
    """
    # Convert from (T, H, W, C) to (T, C, H, W)
    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    return frames_tensor / 255.0


def tensor_to_frames(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor back to video frames.
    
    Args:
        tensor: Tensor of shape (T, C, H, W) or (C, H, W).
        
    Returns:
        np.ndarray: Video frames of shape (T, H, W, C) or (H, W, C).
    """
    if tensor.dim() == 4:  # (T, C, H, W)
        frames = tensor.permute(0, 2, 3, 1).cpu().numpy()
    else:  # (C, H, W)
        frames = tensor.permute(1, 2, 0).cpu().numpy()
    
    return np.clip(frames * 255.0, 0, 255).astype(np.uint8)


def create_random_mask(
    height: int,
    width: int,
    mask_ratio: float = 0.1,
    mask_type: str = 'random'
) -> np.ndarray:
    """Create random mask for inpainting.
    
    Args:
        height: Frame height.
        width: Frame width.
        mask_ratio: Ratio of pixels to mask.
        mask_type: Type of mask ('random', 'center', 'block').
        
    Returns:
        np.ndarray: Binary mask (0=keep, 1=mask).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if mask_type == 'random':
        num_pixels = int(height * width * mask_ratio)
        flat_mask = np.zeros(height * width)
        flat_mask[:num_pixels] = 1
        np.random.shuffle(flat_mask)
        mask = flat_mask.reshape(height, width)
    
    elif mask_type == 'center':
        center_h, center_w = height // 2, width // 2
        mask_h = int(height * np.sqrt(mask_ratio))
        mask_w = int(width * np.sqrt(mask_ratio))
        mask[
            center_h - mask_h//2:center_h + mask_h//2,
            center_w - mask_w//2:center_w + mask_w//2
        ] = 1
    
    elif mask_type == 'block':
        block_h = int(height * np.sqrt(mask_ratio))
        block_w = int(width * np.sqrt(mask_ratio))
        start_h = np.random.randint(0, height - block_h + 1)
        start_w = np.random.randint(0, width - block_w + 1)
        mask[start_h:start_h + block_h, start_w:start_w + block_w] = 1
    
    return mask


def apply_mask(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to video frames.
    
    Args:
        frames: Video frames of shape (T, H, W, C).
        mask: Binary mask of shape (H, W).
        
    Returns:
        np.ndarray: Masked frames.
    """
    masked_frames = frames.copy()
    masked_frames[:, mask == 1] = 0  # Set masked pixels to black
    return masked_frames


def compute_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    method: str = 'farneback'
) -> np.ndarray:
    """Compute optical flow between two frames.
    
    Args:
        frame1: First frame (H, W, C).
        frame2: Second frame (H, W, C).
        method: Optical flow method ('farneback', 'lucas_kanade').
        
    Returns:
        np.ndarray: Optical flow field.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    if method == 'farneback':
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
    else:  # farneback
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
    
    return flow


def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp frame using optical flow.
    
    Args:
        frame: Input frame (H, W, C).
        flow: Optical flow field.
        
    Returns:
        np.ndarray: Warped frame.
    """
    h, w = frame.shape[:2]
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    
    for y in range(h):
        for x in range(w):
            flow_map[y, x] = [x + flow[y, x, 0], y + flow[y, x, 1]]
    
    warped = cv2.remap(
        frame, flow_map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    
    return warped
