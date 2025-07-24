# Emotion Detection App

A real-time emotion detection application using YOLOv8 for face detection and a CNN model for emotion classification. Built with Streamlit for an easy-to-use web interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Local Setup and Running](#local-setup-and-running)
- [Usage](#usage)
- [Configuration](#configuration)
- [Models](#models)
- [Deployment](#deployment)
- [Performance Optimization](#performance-optimization)
- [Limitations](#limitations)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This application provides comprehensive emotion detection capabilities through multiple input methods. It combines state-of-the-art face detection using YOLOv8 with a custom Convolutional Neural Network for emotion classification, offering real-time analysis through an intuitive web interface.

## Features

### Core Functionality
- **Face Detection**: Utilizes YOLOv8 for accurate face detection in images and videos
- **Emotion Classification**: Classifies emotions into 7 categories:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprised

### Input Options
- **Video Upload**: Process videos in MP4, AVI, and MOV formats
- **Real-time Webcam**: Live emotion detection using your webcam
- **Image Upload**: Analyze static images (JPG, PNG, JPEG formats)

### Analysis and Visualization
- **Emotion Timeline**: Track emotion changes over time in video analysis
- **Emotion Distribution Charts**: Visual breakdown of detected emotions
- **Frame-by-Frame Analysis**: Detailed examination of individual video frames
- **Export Capabilities**: Download analysis results in CSV or JSON format

### Performance Features
- **Adjustable Frame Sampling**: Process every Nth frame for optimized performance
- **Configurable Confidence Thresholds**: Fine-tune detection sensitivity
- **Real-time Processing**: Optimized for live webcam feeds

## Technology Stack

- **Frontend Framework**: Streamlit
- **Face Detection**: YOLOv8 (Ultralytics)
- **Emotion Classification**: Custom CNN based on FER-2013 architecture
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Plotly
- **Deep Learning**: PyTorch/TensorFlow (model dependent)

## Project Structure

```
emotion-detection-app/
├── app.py                    # Main Streamlit application
├── download_models.py        # Script to download required models
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── LICENSE                   # License file
├── models/
│   ├── __init__.py
│   ├── face_detector.py      # YOLOv8 face detection wrapper
│   ├── emotion_classifier.py # Emotion classification model
│   └── weights/              # Pre-trained model weights directory
├── utils/
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── video_processor.py    # Video processing utilities
│   └── visualization.py     # Visualization utilities
└── assets/                   # Static assets (images, demos)
```

## Prerequisites

Before installing the application, ensure you have:

- **Python 3.8 or higher**
- **pip package manager**
- **Git** (for cloning the repository)
- **Webcam** (optional, for real-time detection features)
- **Sufficient RAM** (minimum 4GB recommended for video processing)
- **GPU support** (optional, for enhanced performance)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/NoLongerHumanHQ/emotion-detection-app.git
cd emotion-detection-app
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv emotion_detection_env

# Activate virtual environment
# On Windows:
emotion_detection_env\Scripts\activate
# On macOS/Linux:
source emotion_detection_env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Required Models

```bash
python download_models.py
```

This script will download:
- YOLOv8n-face model for face detection
- Pre-trained emotion classification model weights
- Any additional required model files

## Local Setup and Running

### Quick Start

1. **Navigate to project directory**:
   ```bash
   cd emotion-detection-app
   ```

2. **Activate virtual environment** (if created):
   ```bash
   # Windows
   emotion_detection_env\Scripts\activate
   # macOS/Linux
   source emotion_detection_env/bin/activate
   ```

3. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the application**:
   - Open your web browser
   - Navigate to `http://localhost:8501` (default Streamlit port)
   - The application interface will load automatically

### Alternative Running Methods

#### Custom Port
```bash
streamlit run app.py --server.port 8502
```

#### Network Access
```bash
streamlit run app.py --server.address 0.0.0.0
```

#### Debug Mode
```bash
streamlit run app.py --logger.level debug
```

### Troubleshooting Installation

If you encounter issues:

1. **Update pip**:
   ```bash
   pip install --upgrade pip
   ```

2. **Install specific package versions**:
   ```bash
   pip install streamlit==1.28.0
   pip install ultralytics==8.0.0
   ```

3. **Clear pip cache**:
   ```bash
   pip cache purge
   ```

4. **Check Python version**:
   ```bash
   python --version
   ```

## Usage

### 1. Video Analysis

1. Select "Upload Video" from the input options
2. Upload a video file (supported formats: MP4, AVI, MOV)
3. Configure processing settings in the sidebar:
   - Detection confidence threshold
   - Frame sampling rate
4. Click "Process Video" to begin analysis
5. View results including:
   - Emotion timeline graph
   - Emotion distribution pie chart
   - Frame-by-frame breakdown
6. Export results using the download buttons

### 2. Real-time Webcam Detection

1. Select "Use Webcam" from the input options
2. Grant camera permissions when prompted
3. Click "START" to begin the webcam feed
4. Real-time emotion detection will overlay on the video feed
5. Adjust settings in real-time using the sidebar controls
6. Click "STOP" to end the session

### 3. Image Analysis

1. Select "Upload Image" from the input options
2. Upload an image file (supported formats: JPG, JPEG, PNG)
3. Click "Detect Emotions" to process the image
4. View results showing:
   - Detected faces with bounding boxes
   - Emotion labels and confidence scores
   - Overall emotion distribution

## Configuration

### Sidebar Settings

- **Detection Confidence Threshold**: Adjust minimum confidence score for face detection (0.1 - 1.0)
- **Process Every Nth Frame**: Control frame sampling rate for video processing (1-10)
- **Model Selection**: Choose between available face detection models
- **Output Format**: Select export format (CSV, JSON)

### Advanced Configuration

Edit `utils/config.py` for advanced settings:

```python
# Model configurations
FACE_DETECTION_MODEL = "yolov8n-face.pt"
EMOTION_MODEL_PATH = "models/weights/emotion_model.pth"

# Processing parameters
DEFAULT_CONFIDENCE = 0.5
MAX_VIDEO_SIZE_MB = 100
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]

# Performance settings
USE_GPU = True
BATCH_SIZE = 32
```

## Models

### Face Detection Model
- **Architecture**: YOLOv8n-face
- **Fallback**: YOLOv8n (general object detection)
- **Input Size**: 640x640 pixels
- **Performance**: ~50 FPS on GPU, ~10 FPS on CPU

### Emotion Classification Model
- **Architecture**: Custom CNN based on FER-2013
- **Input Size**: 48x48 pixels (grayscale)
- **Classes**: 7 emotion categories
- **Accuracy**: Varies based on training data and conditions

## Deployment

### Streamlit Community Cloud

1. Fork the repository to your GitHub account
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy with one click
5. Access your deployed app via the provided URL

### Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Select Streamlit as the SDK
3. Connect your GitHub repository
4. Configure environment variables if needed
5. Deploy and share your application

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt
RUN python download_models.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## Performance Optimization

### For Better Speed
- Increase the "Process every Nth frame" value for videos
- Use GPU acceleration if available
- Reduce video resolution before processing
- Increase confidence threshold to reduce false positives

### For Better Accuracy
- Use lower confidence thresholds
- Process every frame (set N=1)
- Ensure good lighting conditions
- Use high-quality input images/videos

### Resource Management
- Monitor RAM usage for large video files
- Close other applications during processing
- Use the lightweight YOLOv8n-face model for resource-constrained environments

## Limitations

### Known Issues
- **Webcam functionality**: May not work in deployed Streamlit apps due to browser security restrictions
- **Large video files**: Processing time increases significantly with file size
- **Model accuracy**: Default emotion model may require additional training for optimal performance
- **Lighting conditions**: Performance degrades in poor lighting
- **Multiple faces**: Processing speed decreases with more faces in frame

### System Requirements
- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB or higher for video processing
- **Storage**: At least 2GB free space for models and temporary files
- **Internet connection**: Required for initial model downloads

## Datasets

The emotion classification model can be trained on various datasets:

### Recommended Datasets
- **FER-2013**: Facial Expression Recognition 2013 dataset
- **CK+**: Extended Cohn-Kanade dataset  
- **JAFFE**: Japanese Female Facial Expression database
- **AffectNet**: Large-scale facial expression database

### Custom Dataset Training
To train on custom data:

1. Prepare your dataset in the required format
2. Modify `models/emotion_classifier.py`
3. Run the training script:
   ```bash
   python train_emotion_model.py --dataset path/to/dataset
   ```

## Contributing

We welcome contributions to improve the emotion detection app:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Submit a pull request**

### Development Setup
```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Include type hints where appropriate

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** for YOLOv8 implementation
- **[Streamlit](https://streamlit.io/)** for the web framework
- **[FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)** dataset for emotion classification training
- **OpenCV** community for computer vision tools
- **PyTorch/TensorFlow** teams for deep learning frameworks

---

For additional support or questions, please open an issue on the GitHub repository or contact the maintainers.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Streamlit](https://streamlit.io/) for the web framework
- [FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) dataset for emotion classification 
