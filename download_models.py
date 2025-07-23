"""
Script to download the required models for the Emotion Detection App.
"""

import os
import sys
import requests
import zipfile
import io
from tqdm import tqdm
from pathlib import Path
import urllib.request

def download_with_progress(url, save_path):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        save_path: Path to save the file to
    """
    print(f"Downloading {url} to {save_path}")
    
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Open the URL
    with urllib.request.urlopen(url) as response:
        file_size = int(response.headers.get('Content-Length', 0))
        
        # Check if file already exists and has the same size
        if os.path.exists(save_path) and os.path.getsize(save_path) == file_size:
            print(f"File already exists and is complete: {save_path}")
            return
        
        # Set up the progress bar
        print(f"Downloading {file_size / (1024 * 1024):.1f} MB")
        
        # Use tqdm to create a progress bar
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(save_path)) as pbar:
            # Download the file
            with open(save_path, 'wb') as f:
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Download complete: {save_path}")

def download_ultralytics_model(model_name, save_dir="models/weights"):
    """
    Download a YOLOv8 model from Ultralytics.
    
    Args:
        model_name: Name of the model to download (e.g., yolov8n.pt)
        save_dir: Directory to save the model to
    """
    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
    save_path = os.path.join(save_dir, model_name)
    
    download_with_progress(url, save_path)

def download_yolov8_face():
    """Download the YOLOv8 face detection model."""
    # YOLOv8n-face model from ultralytics
    print("Downloading YOLOv8 face detection model...")
    
    # URL for YOLOv8n-face model
    url = "https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
    save_path = "models/weights/yolov8n-face.pt"
    
    download_with_progress(url, save_path)

def download_emotion_model():
    """Download the emotion classification model."""
    # Emotion model
    print("Downloading emotion classification model...")
    
    # For a pre-trained emotion model, we can use models from Hugging Face
    # This is a placeholder URL - you would need to replace with an actual model URL
    url = "https://github.com/priya-dwivedi/face_and_emotion_detection/blob/master/emotion_detector_models/model_v6_23.hdf5?raw=true"
    save_path = "models/weights/emotion_model.h5"
    
    download_with_progress(url, save_path)
    
    # Note: In a real-world scenario, you might want to train your own emotion model
    # or find a more suitable pre-trained model.
    print("""
Note: This is a basic pre-trained emotion model. For best results, 
you may want to train your own model on the FER-2013 dataset.
""")

def main():
    """Main function to download all required models."""
    print("Downloading models for Emotion Detection App...")
    
    # Create models/weights directory if it doesn't exist
    os.makedirs("models/weights", exist_ok=True)
    
    try:
        # Download YOLOv8 face detection model
        download_yolov8_face()
        
        # Alternatively, download the standard YOLOv8n model
        # The standard model can detect people but is not specialized for faces
        # However, it can still work reasonably well
        print("Downloading standard YOLOv8n model as a backup...")
        download_ultralytics_model("yolov8n.pt")
        
        # Download emotion model
        # download_emotion_model()
        print("Note: Emotion model needs to be trained or downloaded separately.")
        print("For now, the app will use a simple model created at runtime.")
        
        print("\nAll models downloaded successfully!")
        print("You can now run the Emotion Detection App with: streamlit run app.py")
        
    except Exception as e:
        print(f"Error downloading models: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 