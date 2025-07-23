"""
Emotion Detection App using YOLOv8 for face detection and deep learning for emotion recognition.
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
import tempfile
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Import utility modules
from utils.config import (
    APP_TITLE, 
    APP_DESCRIPTION,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAME_SAMPLE_RATE,
    MAX_FRAME_SAMPLE_RATE,
    EMOTIONS,
    VIDEO_WIDTH,
    VIDEO_HEIGHT
)
from utils.video_processor import (
    validate_video_file, 
    save_uploaded_file, 
    get_video_info, 
    process_video_frames, 
    resize_frame
)
from utils.visualization import (
    draw_face_bbox, 
    draw_emotion_bars, 
    create_emotion_timeline,
    create_emotion_distribution,
    add_fps_counter
)

# Import models
from models.face_detector import FaceDetector
from models.emotion_classifier import EmotionClassifier

# Initialize models
@st.cache_resource
def load_models():
    """Load face detection and emotion classification models."""
    # Check if models directory exists, if not create it
    if not os.path.exists("models/weights"):
        os.makedirs("models/weights", exist_ok=True)
        
    # Check if face detection model exists, otherwise use standard YOLOv8n model
    face_model_path = "models/weights/yolov8n-face.pt"
    if not os.path.exists(face_model_path):
        # Fallback to standard YOLOv8n model
        face_model_path = "models/weights/yolov8n.pt"
        
        if not os.path.exists(face_model_path):
            st.warning(
                "Face detection model not found. Please run `python download_models.py` first, "
                "or the app will attempt to download it automatically (which may take some time)."
            )
            # Try downloading the standard YOLOv8n model
            try:
                from ultralytics import YOLO
                YOLO("yolov8n.pt")
            except Exception as e:
                st.error(f"Failed to download model: {str(e)}")

    try:
        # Load face detector
        face_detector = FaceDetector(
            model_path=face_model_path,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
        )
        
        # Load emotion classifier
        emotion_classifier = EmotionClassifier()
        
        return face_detector, emotion_classifier
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def process_frame(frame, face_detector, emotion_classifier, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Process a single frame for face detection and emotion recognition.
    
    Args:
        frame: Input frame
        face_detector: FaceDetector instance
        emotion_classifier: EmotionClassifier instance
        confidence_threshold: Confidence threshold for detection
        
    Returns:
        processed_frame: Frame with annotations
        metadata: Detection metadata
    """
    # Resize frame to improve performance
    resized_frame = resize_frame(frame, VIDEO_WIDTH, VIDEO_HEIGHT)
    
    # Detect faces
    faces = face_detector.detect_faces(resized_frame)
    
    # Extract face regions
    faces_with_regions = face_detector.extract_face_regions(resized_frame, faces)
    
    # Process each face for emotion detection
    all_emotion_scores = []
    metadata = {
        "faces_detected": len(faces),
        "emotions": {}
    }
    
    for face in faces_with_regions:
        # Extract bbox
        bbox = face["bbox"]
        conf = face["confidence"]
        
        # Skip if confidence is below threshold
        if conf < confidence_threshold:
            continue
            
        # Get the face image (already preprocessed by extract_face_regions)
        face_img_gray = face["face_img_gray"]
        
        # Predict emotion
        emotion_scores = emotion_classifier.predict_emotion(face_img_gray)
        all_emotion_scores.append(emotion_scores)
        
        # Get dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        emotion_name = dominant_emotion[0]
        emotion_conf = dominant_emotion[1]
        
        # Draw face bounding box with emotion label
        draw_face_bbox(
            resized_frame, 
            bbox, 
            emotion=emotion_name,
            confidence=emotion_conf
        )
        
        # Update metadata
        for emotion, score in emotion_scores.items():
            if emotion not in metadata["emotions"]:
                metadata["emotions"][emotion] = []
            metadata["emotions"][emotion].append(score)
            
    # Draw emotion bars (using average of all detected faces)
    if all_emotion_scores:
        # Calculate average emotion scores across all faces
        avg_emotion_scores = {}
        for emotion in EMOTIONS:
            emotion_values = [scores.get(emotion, 0.0) for scores in all_emotion_scores]
            avg_emotion_scores[emotion] = sum(emotion_values) / len(emotion_values) if emotion_values else 0.0
            
        # Draw emotion bars on frame
        draw_emotion_bars(resized_frame, avg_emotion_scores)
        
        # Update metadata with averages
        metadata["avg_emotions"] = avg_emotion_scores
        
    return resized_frame, metadata

class VideoProcessor(VideoTransformerBase):
    """Video transformer for webcam processing."""
    
    def __init__(self, face_detector, emotion_classifier, confidence_threshold):
        self.face_detector = face_detector
        self.emotion_classifier = emotion_classifier
        self.confidence_threshold = confidence_threshold
        self.last_time = time.time()
        self.fps = 0
        
    def transform(self, frame):
        """Process video frame from webcam."""
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
        self.last_time = current_time
        
        # Process frame for face detection and emotion recognition
        img = frame.to_ndarray(format="bgr24")
        processed_frame, _ = process_frame(
            img, 
            self.face_detector,
            self.emotion_classifier,
            self.confidence_threshold
        )
        
        # Add FPS counter
        add_fps_counter(processed_frame, self.fps)
        
        return processed_frame

def main():
    """Main function for the Streamlit app."""
    # Set page config
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Display header
    st.title("Emotion Detection App")
    st.markdown(APP_DESCRIPTION)
    
    # Load models (with caching)
    face_detector, emotion_classifier = load_models()
    
    if face_detector is None or emotion_classifier is None:
        st.error("Failed to load models. Please check the logs and try again.")
        return
    
    # Create sidebar for settings
    st.sidebar.title("Settings")
    
    # Confidence threshold slider
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence score to consider a detection valid."
    )
    
    # Frame sample rate slider (for video processing)
    frame_sample_rate = st.sidebar.slider(
        "Process every Nth frame",
        min_value=1,
        max_value=MAX_FRAME_SAMPLE_RATE,
        value=DEFAULT_FRAME_SAMPLE_RATE,
        step=1,
        help="Higher values mean faster but less smooth processing."
    )
    
    # Display stats
    with st.sidebar.expander("Model Information", expanded=False):
        st.write("Face Detection Model: YOLOv8")
        st.write("Emotion Classification: CNN")
        st.write(f"Emotions: {', '.join(EMOTIONS)}")
    
    # Main app content
    input_option = st.radio(
        "Select Input Source:",
        ["Upload Video", "Use Webcam", "Upload Image"]
    )
    
    # Results storage
    results_df = None
    
    # Handle different input options
    if input_option == "Upload Video":
        # Video upload
        video_file = st.file_uploader(
            "Upload video file", 
            type=["mp4", "avi", "mov"]
        )
        
        if video_file is not None:
            # Validate video file
            is_valid, message = validate_video_file(video_file)
            
            if not is_valid:
                st.error(message)
            else:
                # Save uploaded file to temp directory
                temp_path = save_uploaded_file(video_file)
                
                # Display video info
                video_info = get_video_info(temp_path)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Duration: {video_info['duration_formatted']}")
                with col2:
                    st.write(f"FPS: {video_info['fps']:.2f}")
                with col3:
                    st.write(f"Frames: {video_info['frame_count']}")
                
                # Process button
                if st.button("Process Video"):
                    # Setup progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create result columns
                    video_col, metrics_col = st.columns([3, 1])
                    
                    # Display the first frame initially
                    video_placeholder = video_col.empty()
                    metrics_placeholder = metrics_col.empty()
                    
                    # Initialize results list
                    results = []
                    
                    # Process video frames
                    frame_processor = lambda frame: process_frame(
                        frame,
                        face_detector,
                        emotion_classifier,
                        confidence_threshold
                    )
                    
                    # Start timing
                    start_time = time.time()
                    
                    # Process the video
                    for i, (processed_frame, metadata) in enumerate(
                        process_video_frames(
                            temp_path,
                            frame_processor,
                            frame_sample_rate=frame_sample_rate,
                            progress_bar=progress_bar
                        )
                    ):
                        # Update status
                        elapsed = time.time() - start_time
                        status_text.text(f"Processing: {elapsed:.2f} seconds elapsed")
                        
                        # Display the processed frame
                        video_placeholder.image(processed_frame, channels="BGR")
                        
                        # Display metrics
                        metrics_info = f"""
                        **Frame:** {metadata.get('frame_number', i)}
                        **Faces Detected:** {metadata.get('faces_detected', 0)}
                        **Processing FPS:** {1.0 / metadata.get('processing_time', 1.0):.2f}
                        """
                        metrics_placeholder.markdown(metrics_info)
                        
                        # Collect results for analysis
                        result_entry = {
                            "frame": metadata.get('frame_number', i),
                            "timestamp": metadata.get('timestamp', 0),
                            "faces_detected": metadata.get('faces_detected', 0),
                            "processing_time": metadata.get('processing_time', 0)
                        }
                        
                        # Add emotion scores
                        if "avg_emotions" in metadata:
                            for emotion, score in metadata["avg_emotions"].items():
                                result_entry[emotion] = score
                                
                        results.append(result_entry)
                    
                    # Clean up
                    status_text.text(f"Processing complete! Total time: {time.time() - start_time:.2f} seconds")
                    
                    # Convert results to DataFrame for analysis
                    results_df = pd.DataFrame(results)
                    
                    # Display results if available
                    if not results_df.empty:
                        st.subheader("Analysis Results")
                        
                        # Show emotion timeline
                        try:
                            if all(emotion in results_df.columns for emotion in EMOTIONS):
                                st.plotly_chart(
                                    create_emotion_timeline(results_df),
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.warning(f"Could not generate emotion timeline: {str(e)}")
                        
                        # Show emotion distribution
                        try:
                            if all(emotion in results_df.columns for emotion in EMOTIONS):
                                st.plotly_chart(
                                    create_emotion_distribution(results_df),
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.warning(f"Could not generate emotion distribution: {str(e)}")
                        
                        # Show data table
                        with st.expander("View Data Table", expanded=False):
                            st.dataframe(results_df)
                        
                        # Export options
                        st.subheader("Export Results")
                        export_format = st.selectbox(
                            "Export Format",
                            ["CSV", "JSON"]
                        )
                        
                        if st.button("Export Data"):
                            if export_format == "CSV":
                                csv_data = results_df.to_csv(index=False)
                                st.download_button(
                                    "Download CSV",
                                    csv_data,
                                    "emotion_analysis.csv",
                                    "text/csv"
                                )
                            else:
                                json_data = results_df.to_json(orient="records")
                                st.download_button(
                                    "Download JSON",
                                    json_data,
                                    "emotion_analysis.json",
                                    "application/json"
                                )
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
    
    elif input_option == "Use Webcam":
        st.write("Note: This may not work in deployed Streamlit apps due to webcam access restrictions.")
        
        # Webcam settings
        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Create processor instance
        processor = VideoProcessor(
            face_detector,
            emotion_classifier,
            confidence_threshold
        )
        
        # Start WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            video_transformer_factory=lambda: processor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            video_html_attrs={"style": {"width": "100%", "height": "auto"}},
        )
        
        # Information text
        if webrtc_ctx.state.playing:
            st.write("Webcam is active! Detecting faces and emotions...")
        else:
            st.write("Click 'START' to begin webcam emotion detection.")
    
    elif input_option == "Upload Image":
        # Image upload
        image_file = st.file_uploader(
            "Upload image file",
            type=["jpg", "jpeg", "png"]
        )
        
        if image_file is not None:
            # Read and display the original image
            image = Image.open(image_file)
            image_np = np.array(image)
            
            # If image is in RGB format, convert to BGR for OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            
            # Process button
            if st.button("Detect Emotions"):
                # Process the image
                processed_image, metadata = process_frame(
                    image_np,
                    face_detector,
                    emotion_classifier,
                    confidence_threshold
                )
                
                # Convert BGR back to RGB for display
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                # Display results
                st.subheader("Processed Image")
                st.image(processed_image_rgb, use_column_width=True)
                
                # Display metadata
                st.subheader("Detection Results")
                st.write(f"Faces detected: {metadata.get('faces_detected', 0)}")
                
                # Show emotion distribution if faces detected
                if metadata.get("avg_emotions"):
                    emotion_data = {
                        "emotion": list(metadata["avg_emotions"].keys()),
                        "confidence": list(metadata["avg_emotions"].values())
                    }
                    emotion_df = pd.DataFrame(emotion_data)
                    
                    # Sort by confidence (descending)
                    emotion_df = emotion_df.sort_values(
                        by="confidence",
                        ascending=False
                    ).reset_index(drop=True)
                    
                    # Display as bar chart
                    st.bar_chart(
                        emotion_df.set_index("emotion")["confidence"],
                        use_container_width=True
                    )

    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        """
        This app uses YOLOv8 for face detection and a CNN model for emotion classification.
        """
    )
    
    # Credits/Links section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Credits")
    st.sidebar.markdown(
        """
        - [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
        - [FER-2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
        """
    )

if __name__ == "__main__":
    main() 