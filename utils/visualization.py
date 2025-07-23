"""
Visualization utilities for the Emotion Detection App.
"""

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any

from utils.config import EMOTIONS, EMOTION_COLORS

def draw_face_bbox(
    frame: np.ndarray,
    bbox: List[float],
    emotion: str = None,
    confidence: float = None,
    color=None
) -> np.ndarray:
    """
    Draw bounding box around detected face with emotion label.
    
    Args:
        frame: Input frame
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        emotion: Detected emotion
        confidence: Detection confidence score
        color: Box color (BGR format)
        
    Returns:
        Frame with bounding box
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Use emotion color or default color
    if emotion and emotion in EMOTION_COLORS:
        color = EMOTION_COLORS[emotion]
        # Convert RGB to BGR for OpenCV
        color = (color[2], color[1], color[0])  
    elif color is None:
        color = (0, 255, 0)  # Default: Green
        
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    label_parts = []
    if emotion:
        label_parts.append(emotion.capitalize())
    if confidence is not None:
        label_parts.append(f"{confidence:.2f}")
    
    label = ": ".join(label_parts)
    
    # Draw label background
    if label:
        font_scale = 0.6
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get size of the text box
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame, 
            (x1, y1 - 20), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            frame, 
            label, 
            (x1, y1 - 5), 
            font, 
            font_scale, 
            (255, 255, 255), 
            font_thickness
        )
        
    return frame

def draw_emotion_bars(
    frame: np.ndarray,
    emotion_scores: Dict[str, float],
    position: Tuple[int, int] = (20, 50),
    width: int = 150,
    height: int = 15,
    spacing: int = 20
) -> np.ndarray:
    """
    Draw horizontal bars representing emotion confidence scores.
    
    Args:
        frame: Input frame
        emotion_scores: Dictionary mapping emotions to scores
        position: Top-left position of the bars
        width: Width of the bars
        height: Height of each bar
        spacing: Vertical spacing between bars
        
    Returns:
        Frame with emotion bars
    """
    x, y = position
    
    # Sort emotions by score (descending)
    sorted_emotions = sorted(
        emotion_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Draw background
    bar_panel_height = len(sorted_emotions) * (height + spacing) - spacing + 10
    cv2.rectangle(
        frame,
        (x - 5, y - 25),
        (x + width + 5, y + bar_panel_height),
        (0, 0, 0),
        -1
    )
    
    # Draw title
    cv2.putText(
        frame,
        "Emotions:",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    # Draw bars for each emotion
    for i, (emotion, score) in enumerate(sorted_emotions):
        bar_y = y + i * (height + spacing)
        
        # Bar background
        cv2.rectangle(
            frame,
            (x, bar_y),
            (x + width, bar_y + height),
            (100, 100, 100),
            -1
        )
        
        # Get color for this emotion
        if emotion in EMOTION_COLORS:
            color = EMOTION_COLORS[emotion]
            # Convert RGB to BGR for OpenCV
            color = (color[2], color[1], color[0])
        else:
            color = (200, 200, 200)  # Default gray
            
        # Bar fill based on score
        bar_width = int(width * score)
        cv2.rectangle(
            frame,
            (x, bar_y),
            (x + bar_width, bar_y + height),
            color,
            -1
        )
        
        # Emotion label
        cv2.putText(
            frame,
            f"{emotion.capitalize()}: {score:.2f}",
            (x + width + 10, bar_y + height - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
        
    return frame

def create_emotion_timeline(results_df: pd.DataFrame) -> px.line:
    """
    Create interactive timeline of emotions over time.
    
    Args:
        results_df: DataFrame with emotion results
        
    Returns:
        Plotly figure object
    """
    # Ensure DataFrame has required columns
    required_cols = ['timestamp'] + EMOTIONS
    if not all(col in results_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Melt DataFrame to long format for plotting
    plot_df = results_df.melt(
        id_vars=['timestamp'],
        value_vars=EMOTIONS,
        var_name='emotion',
        value_name='confidence'
    )
    
    # Create line plot
    fig = px.line(
        plot_df,
        x='timestamp',
        y='confidence',
        color='emotion',
        title='Emotion Timeline',
        labels={
            'timestamp': 'Time (seconds)',
            'confidence': 'Confidence Score',
            'emotion': 'Emotion'
        },
        color_discrete_map={
            emotion: f"rgb{EMOTION_COLORS[emotion]}"
            for emotion in EMOTIONS if emotion in EMOTION_COLORS
        }
    )
    
    # Add layout settings
    fig.update_layout(
        xaxis_title='Time (seconds)',
        yaxis_title='Confidence Score',
        legend_title='Emotion',
        hovermode='x unified',
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_emotion_distribution(results_df: pd.DataFrame) -> px.bar:
    """
    Create bar chart showing overall emotion distribution.
    
    Args:
        results_df: DataFrame with emotion results
        
    Returns:
        Plotly figure object
    """
    # Calculate average for each emotion
    emotion_means = results_df[EMOTIONS].mean().reset_index()
    emotion_means.columns = ['emotion', 'average_confidence']
    
    # Sort by average confidence (descending)
    emotion_means = emotion_means.sort_values(
        by='average_confidence',
        ascending=False
    )
    
    # Create bar chart
    fig = px.bar(
        emotion_means,
        x='emotion',
        y='average_confidence',
        title='Overall Emotion Distribution',
        labels={
            'emotion': 'Emotion',
            'average_confidence': 'Average Confidence'
        },
        color='emotion',
        color_discrete_map={
            emotion: f"rgb{EMOTION_COLORS[emotion]}"
            for emotion in EMOTIONS if emotion in EMOTION_COLORS
        }
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Emotion',
        yaxis_title='Average Confidence',
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def add_fps_counter(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (20, 30)
) -> np.ndarray:
    """
    Add FPS counter to the frame.
    
    Args:
        frame: Input frame
        fps: Current FPS value
        position: Position of the counter
        
    Returns:
        Frame with FPS counter
    """
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),  # Green
        2
    )
    
    return frame 