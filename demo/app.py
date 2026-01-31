"""Streamlit demo application for video inpainting."""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from typing import Optional, Tuple
import time

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.video_inpainting import UNetVideoInpainting, EDVRVideoInpainting
from src.utils.device import get_device
from src.utils.video import (
    load_video_frames, frames_to_tensor, tensor_to_frames, 
    create_random_mask, apply_mask
)
from src.eval.metrics import MetricsCalculator


class VideoInpaintingDemo:
    """Demo application for video inpainting."""
    
    def __init__(self):
        """Initialize the demo."""
        self.device = get_device()
        self.model = None
        self.metrics_calculator = MetricsCalculator(self.device)
        
        # Initialize session state
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'current_video' not in st.session_state:
            st.session_state.current_video = None
        if 'inpainted_video' not in st.session_state:
            st.session_state.inpainted_video = None
    
    def load_model(self, model_type: str = 'unet') -> bool:
        """Load the inpainting model.
        
        Args:
            model_type: Type of model to load ('unet' or 'edvr').
            
        Returns:
            bool: True if model loaded successfully.
        """
        try:
            if model_type == 'unet':
                self.model = UNetVideoInpainting(
                    in_channels=3,
                    out_channels=3,
                    base_channels=64,
                    num_frames=8
                )
            elif model_type == 'edvr':
                self.model = EDVRVideoInpainting(
                    in_channels=3,
                    out_channels=3,
                    num_frames=8,
                    base_channels=64
                )
            else:
                st.error(f"Unknown model type: {model_type}")
                return False
            
            self.model.to(self.device)
            self.model.eval()
            st.session_state.model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def process_video(
        self,
        video_frames: np.ndarray,
        mask_ratio: float = 0.1,
        mask_type: str = 'random'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process video for inpainting.
        
        Args:
            video_frames: Input video frames (T, H, W, C).
            mask_ratio: Ratio of pixels to mask.
            mask_type: Type of mask ('random', 'center', 'block').
            
        Returns:
            Tuple of (original_frames, masked_frames, inpainted_frames).
        """
        if not st.session_state.model_loaded:
            st.error("Model not loaded. Please load a model first.")
            return None, None, None
        
        try:
            # Convert frames to tensor
            frames_tensor = frames_to_tensor(video_frames)  # (T, C, H, W)
            
            # Create mask
            T, C, H, W = frames_tensor.shape
            mask = create_random_mask(H, W, mask_ratio, mask_type)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = mask_tensor.expand(T, 1, H, W).to(self.device)
            
            # Apply mask
            masked_frames = frames_tensor.clone()
            masked_frames[mask_tensor.cpu() == 1] = 0
            
            # Move to device
            masked_frames = masked_frames.to(self.device)
            
            # Perform inpainting
            with torch.no_grad():
                inpainted_frames = self.model(masked_frames.unsqueeze(0), mask_tensor.unsqueeze(0))
                inpainted_frames = inpainted_frames.squeeze(0)  # Remove batch dimension
            
            # Convert back to numpy
            original_frames = tensor_to_frames(frames_tensor)
            masked_frames_np = tensor_to_frames(masked_frames.cpu())
            inpainted_frames_np = tensor_to_frames(inpainted_frames.cpu())
            
            return original_frames, masked_frames_np, inpainted_frames_np
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return None, None, None
    
    def calculate_metrics(
        self,
        original: np.ndarray,
        inpainted: np.ndarray
    ) -> dict:
        """Calculate evaluation metrics.
        
        Args:
            original: Original frames (T, H, W, C).
            inpainted: Inpainted frames (T, H, W, C).
            
        Returns:
            Dictionary of metrics.
        """
        try:
            # Convert to tensors
            original_tensor = torch.from_numpy(original).permute(0, 3, 1, 2).float() / 255.0
            inpainted_tensor = torch.from_numpy(inpainted).permute(0, 3, 1, 2).float() / 255.0
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                inpainted_tensor, original_tensor
            )
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def run(self):
        """Run the demo application."""
        st.set_page_config(
            page_title="Video Inpainting Demo",
            page_icon="🎬",
            layout="wide"
        )
        
        st.title("Video Inpainting Demo")
        st.markdown("Upload a video and watch AI fill in missing parts!")
        
        # Sidebar for model selection
        with st.sidebar:
            st.header("Model Configuration")
            
            model_type = st.selectbox(
                "Model Type",
                ["unet", "edvr"],
                help="Choose the inpainting model architecture"
            )
            
            if st.button("Load Model"):
                with st.spinner("Loading model..."):
                    success = self.load_model(model_type)
                    if success:
                        st.success(f"{model_type.upper()} model loaded successfully!")
                    else:
                        st.error("Failed to load model")
            
            st.header("Inpainting Parameters")
            
            mask_ratio = st.slider(
                "Mask Ratio",
                min_value=0.05,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="Percentage of pixels to mask"
            )
            
            mask_type = st.selectbox(
                "Mask Type",
                ["random", "center", "block"],
                help="Type of mask to apply"
            )
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Input Video")
            
            # Video upload
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for inpainting"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    # Load video frames
                    video_frames = load_video_frames(
                        tmp_path,
                        max_frames=8,
                        target_size=(256, 256)
                    )
                    
                    st.session_state.current_video = video_frames
                    
                    # Display video info
                    st.info(f"Loaded video with {len(video_frames)} frames")
                    
                    # Show first frame
                    if len(video_frames) > 0:
                        st.image(video_frames[0], caption="First frame", use_column_width=True)
                    
                    # Process button
                    if st.button("Process Video", disabled=not st.session_state.model_loaded):
                        with st.spinner("Processing video..."):
                            original, masked, inpainted = self.process_video(
                                video_frames, mask_ratio, mask_type
                            )
                            
                            if original is not None:
                                st.session_state.inpainted_video = {
                                    'original': original,
                                    'masked': masked,
                                    'inpainted': inpainted
                                }
                                
                                st.success("Video processed successfully!")
                
                except Exception as e:
                    st.error(f"Error loading video: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        
        with col2:
            st.header("Results")
            
            if st.session_state.inpainted_video is not None:
                results = st.session_state.inpainted_video
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Original", "Masked", "Inpainted", "Comparison"])
                
                with tab1:
                    st.subheader("Original Video")
                    self.display_video_frames(results['original'])
                
                with tab2:
                    st.subheader("Masked Video")
                    self.display_video_frames(results['masked'])
                
                with tab3:
                    st.subheader("Inpainted Video")
                    self.display_video_frames(results['inpainted'])
                
                with tab4:
                    st.subheader("Side-by-Side Comparison")
                    self.display_comparison(results['original'], results['inpainted'])
                
                # Calculate and display metrics
                st.subheader("Evaluation Metrics")
                metrics = self.calculate_metrics(results['original'], results['inpainted'])
                
                if metrics:
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    
                    with col_metrics1:
                        st.metric("PSNR", f"{metrics.get('psnr', 0):.2f} dB")
                    
                    with col_metrics2:
                        st.metric("SSIM", f"{metrics.get('ssim', 0):.4f}")
                    
                    with col_metrics3:
                        st.metric("LPIPS", f"{metrics.get('lpips', 0):.4f}")
                    
                    if 'temporal_consistency' in metrics:
                        st.metric("Temporal Consistency", f"{metrics['temporal_consistency']:.4f}")
            else:
                st.info("Process a video to see results here")
    
    def display_video_frames(self, frames: np.ndarray):
        """Display video frames as a sequence.
        
        Args:
            frames: Video frames (T, H, W, C).
        """
        if len(frames) == 0:
            st.warning("No frames to display")
            return
        
        # Create columns for frames
        cols = st.columns(min(len(frames), 4))
        
        for i, frame in enumerate(frames):
            with cols[i % 4]:
                st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
    
    def display_comparison(self, original: np.ndarray, inpainted: np.ndarray):
        """Display side-by-side comparison.
        
        Args:
            original: Original frames (T, H, W, C).
            inpainted: Inpainted frames (T, H, W, C).
        """
        if len(original) == 0 or len(inpainted) == 0:
            st.warning("No frames to compare")
            return
        
        # Show first few frames side by side
        num_frames = min(len(original), len(inpainted), 4)
        
        for i in range(num_frames):
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(original[i], caption=f"Original Frame {i+1}", use_column_width=True)
            
            with col2:
                st.image(inpainted[i], caption=f"Inpainted Frame {i+1}", use_column_width=True)


def main():
    """Main function to run the demo."""
    demo = VideoInpaintingDemo()
    demo.run()


if __name__ == "__main__":
    main()
