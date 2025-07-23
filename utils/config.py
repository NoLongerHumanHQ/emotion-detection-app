"""
Configuration file for the Emotion Detection Application
"""

# Model paths
FACE_DETECTION_MODEL_PATH = "models/weights/yolov8n-face.pt"
EMOTION_MODEL_PATH = "models/weights/emotion_model.h5"

# Default settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_FRAME_SAMPLE_RATE = 1  # Process every frame
MAX_FRAME_SAMPLE_RATE = 10  # Process every 10th frame

# Emotions
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
EMOTION_COLORS = {
    "angry": (255, 0, 0),       # Red
    "disgust": (0, 128, 0),     # Green
    "fear": (128, 0, 128),      # Purple
    "happy": (255, 255, 0),     # Yellow
    "neutral": (192, 192, 192), # Silver
    "sad": (0, 0, 255),         # Blue
    "surprised": (255, 165, 0)  # Orange
}

# Video processing
ALLOWED_EXTENSIONS = ["mp4", "avi", "mov"]
MAX_VIDEO_SIZE_MB = 100
VIDEO_WIDTH = 640  # Default width for display
VIDEO_HEIGHT = 480  # Default height for display

# Face detection
FACE_PADDING = 30  # Pixels to add around face for better emotion detection

# App settings
APP_TITLE = "Emotion Detection App"
APP_DESCRIPTION = "Detect emotions in videos and webcam streams using YOLO v8 and deep learning."
APP_THEME_PRIMARY_COLOR = "#FF4B4B"  # Streamlit red 