"""
Video processing utilities for the Emotion Detection App.
"""
import os
import cv2
import numpy as np
import tempfile
from typing import Tuple, List, Dict, Any, Generator
import time
from tqdm import tqdm

from utils.config import (
    DEFAULT_FRAME_SAMPLE_RATE,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    ALLOWED_EXTENSIONS,
    MAX_VIDEO_SIZE_MB
)

def validate_video_file(video_file) -> Tuple[bool, str]:
    """
    Validate uploaded video file format and size.
    
    Args:
        video_file: Streamlit uploaded video file
    
    Returns:
        (is_valid, message): Tuple indicating validity and error message if invalid
    """
    if video_file is None:
        return False, "No file uploaded"
    
    # Check file extension
    file_extension = video_file.name.split(".")[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file format. Please upload: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size
    file_size_mb = video_file.size / (1024 * 1024)
    if file_size_mb > MAX_VIDEO_SIZE_MB:
        return False, f"File too large. Maximum allowed: {MAX_VIDEO_SIZE_MB} MB"
    
    return True, "Valid video file"

def save_uploaded_file(video_file) -> str:
    """
    Save uploaded video file to temp directory.
    
    Args:
        video_file: Streamlit uploaded video file
        
    Returns:
        Temporary file path
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{video_file.name.split('.')[-1]}")
    temp_file.write(video_file.read())
    temp_file_path = temp_file.name
    temp_file.close()
    
    return temp_file_path

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get video metadata.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video info
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
        "duration_formatted": format_time(duration)
    }

def format_time(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"

def process_video_frames(
    video_path: str,
    frame_processor_func,
    frame_sample_rate: int = DEFAULT_FRAME_SAMPLE_RATE,
    progress_bar=None,
    preview_mode: bool = False,
    max_frames: int = None
) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """
    Process video frames with a given processor function.
    
    Args:
        video_path: Path to video file
        frame_processor_func: Function to process each frame
        frame_sample_rate: Process every Nth frame
        progress_bar: Streamlit progress bar to update
        preview_mode: If True, only process a few frames for preview
        max_frames: Maximum number of frames to process
    
    Yields:
        (processed_frame, metadata): Processed frame and metadata
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames and preview_mode:
        total_frames = min(total_frames, max_frames)
    
    frame_count = 0
    processed_count = 0
    processing_times = []
    skipped_frames = 0
    
    # Adjust frame_sample_rate based on preview mode
    if preview_mode:
        # If in preview mode, process fewer frames
        frame_sample_rate = max(5, frame_sample_rate)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret or (max_frames and frame_count >= max_frames):
            break
            
        # Process every Nth frame
        if frame_count % frame_sample_rate == 0:
            start_time = time.time()
            
            # Process frame
            try:
                processed_frame, metadata = frame_processor_func(frame)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Update metadata
                metadata.update({
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps if fps > 0 else 0,
                    "processing_time": processing_time,
                    "current_fps": 1.0 / processing_time if processing_time > 0 else 0
                })
                
                processed_count += 1
                
                # Yield the processed frame
                yield processed_frame, metadata
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                skipped_frames += 1
                yield frame, {"error": str(e), "frame_number": frame_count}
        else:
            skipped_frames += 1
        
        # Update progress bar if provided
        if progress_bar is not None and total_frames > 0:
            progress_bar.progress((frame_count + 1) / total_frames)
            
        frame_count += 1
    
    cap.release()

def resize_frame(frame: np.ndarray, width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        width: Target width
        height: Target height
        
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    # Calculate aspect ratio
    aspect = w / h
    
    # Calculate new dimensions
    if width / height > aspect:
        new_width = int(height * aspect)
        new_height = height
    else:
        new_width = width
        new_height = int(width / aspect)
    
    # Resize frame
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA) 