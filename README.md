# Emotion Detection App

A real-time emotion detection application using YOLOv8 for face detection and a CNN model for emotion classification. Built with Streamlit for an easy-to-use web interface.

![Emotion Detection App](https://i.imgur.com/LUWdio5.png)

## Features

- **Face Detection**: Detects faces in images and videos using YOLOv8
- **Emotion Classification**: Classifies emotions into 7 categories: angry, disgust, fear, happy, neutral, sad, surprised
- **Multiple Input Options**:
  - Upload videos (MP4, AVI, MOV)
  - Use your webcam for real-time detection
  - Upload images (JPG, PNG)
- **Analysis & Visualization**:
  - Emotion timeline for videos
  - Emotion distribution charts
  - Frame-by-frame analysis
- **Export Options**: Download results as CSV or JSON
- **Performance Optimization**: Adjustable frame sampling rate and confidence thresholds

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/emotion-detection-app.git
cd emotion-detection-app
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Download the required models:
```
python download_models.py
```

### Running the App

1. Start the Streamlit app:
```
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage

### Video Analysis

1. Select "Upload Video" from the input options
2. Upload a video file (MP4, AVI, or MOV format)
3. Click "Process Video" to start analysis
4. View the results, including emotion timeline and distribution

### Webcam Mode

1. Select "Use Webcam" from the input options
2. Click "START" to begin the webcam feed
3. Real-time emotion detection will be displayed on your webcam feed

### Image Analysis

1. Select "Upload Image" from the input options
2. Upload an image file (JPG, JPEG, or PNG format)
3. Click "Detect Emotions" to process the image
4. View the detected faces and emotion distribution

## Configuration

Use the sidebar to adjust settings:

- **Detection Confidence Threshold**: Adjust the minimum confidence score for face detection
- **Process every Nth frame**: Increase this value to speed up processing for longer videos

## Model Information

- **Face Detection**: YOLOv8n-face (or fallback to YOLOv8n)
- **Emotion Classification**: Custom CNN model based on FER-2013 architecture

## Free Deployment Options

### Streamlit Community Cloud

1. Fork this repository to your GitHub account
2. Log in to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Deploy your app by connecting to your GitHub repository

### Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Select Streamlit as the SDK
3. Connect your GitHub repository

## Datasets

The emotion classification model can be trained on the following datasets:

- [FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- [AffectNet](http://mohammadmahoor.com/affectnet/)

## Project Structure

```
project/
├── app.py                  # Main Streamlit application
├── download_models.py      # Script to download required models
├── requirements.txt        # Python dependencies
├── models/
│   ├── __init__.py
│   ├── face_detector.py    # YOLOv8 face detection wrapper
│   ├── emotion_classifier.py # Emotion classification model
│   └── weights/            # Pre-trained model weights
├── utils/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── video_processor.py  # Video processing utilities
│   └── visualization.py    # Visualization utilities
└── README.md
```

## Performance Optimization

- Use the frame sampling rate slider to process every Nth frame for faster analysis
- The app uses the lightweight YOLOv8n-face model for optimal performance
- For resource-constrained environments, increase the confidence threshold

## Limitations

- Webcam functionality may not work in deployed Streamlit apps due to browser security restrictions
- Large video files may take a significant amount of time to process
- The default emotion model may require training for optimal accuracy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Streamlit](https://streamlit.io/) for the web framework
- [FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) dataset for emotion classification 