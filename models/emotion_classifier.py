"""
Emotion classification module for the Emotion Detection App.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from typing import Dict, List, Any

from utils.config import EMOTION_MODEL_PATH, EMOTIONS

class EmotionClassifier:
    """Emotion classifier model wrapper class."""
    
    def __init__(self, model_path: str = EMOTION_MODEL_PATH):
        """
        Initialize the emotion classifier.
        
        Args:
            model_path: Path to the emotion model
        """
        self._model_loaded = False
        
        # Check if model exists
        if os.path.exists(model_path):
            try:
                # Load pre-trained model
                self.model = load_model(model_path)
                self._model_loaded = True
                print(f"Emotion classifier loaded from: {model_path}")
            except Exception as e:
                print(f"Failed to load emotion model: {str(e)}")
                self._create_default_model()
        else:
            # If model not found, create a simple one (will need training)
            print(f"Emotion model not found at: {model_path}. Creating a default model.")
            self._create_default_model()
            
            # Ensure directory exists for saving
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    def _create_default_model(self):
        """Create a default emotion classification CNN model."""
        try:
            # Create simple CNN model architecture
            model = Sequential([
                # First convolutional block
                Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
                Conv2D(32, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                # Second convolutional block
                Conv2D(64, kernel_size=(3, 3), activation='relu'),
                Conv2D(64, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                # Flatten and dense layers
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(len(EMOTIONS), activation='softmax')
            ])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            self._model_loaded = True
            print("Default emotion model created. Note: This model needs to be trained.")
            
        except Exception as e:
            print(f"Failed to create default model: {str(e)}")
            self._model_loaded = False
    
    @property
    def is_ready(self) -> bool:
        """Check if the model is ready to use."""
        return self._model_loaded
    
    def preprocess_image(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for the emotion classifier.
        
        Args:
            face_img: Input face image (grayscale)
            
        Returns:
            Preprocessed face image
        """
        # Ensure image is grayscale
        if len(face_img.shape) == 3 and face_img.shape[2] > 1:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
        # Resize to 48x48 if needed
        if face_img.shape[0] != 48 or face_img.shape[1] != 48:
            face_img = cv2.resize(face_img, (48, 48))
        
        # Normalize pixel values
        face_img = face_img.astype('float32') / 255.0
        
        # Reshape to match model input
        face_img = face_img.reshape((1, 48, 48, 1))
        
        return face_img
    
    def predict_emotion(self, face_img: np.ndarray) -> Dict[str, float]:
        """
        Predict emotions for a given face image.
        
        Args:
            face_img: Input face image (grayscale)
            
        Returns:
            Dictionary mapping emotions to confidence scores
        """
        if not self._model_loaded:
            print("Model not loaded. Cannot predict emotion.")
            return {emotion: 0.0 for emotion in EMOTIONS}
        
        try:
            # Preprocess image
            processed_img = self.preprocess_image(face_img)
            
            # Get predictions
            preds = self.model.predict(processed_img, verbose=0)[0]
            
            # Create dictionary of emotion scores
            emotion_scores = {emotion: float(score) for emotion, score in zip(EMOTIONS, preds)}
            
            return emotion_scores
            
        except Exception as e:
            print(f"Error predicting emotion: {str(e)}")
            return {emotion: 0.0 for emotion in EMOTIONS}
    
    def predict_batch(self, face_images: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Predict emotions for a batch of face images.
        
        Args:
            face_images: List of face images
            
        Returns:
            List of dictionaries mapping emotions to confidence scores
        """
        if not self._model_loaded:
            print("Model not loaded. Cannot predict emotions.")
            return [{emotion: 0.0 for emotion in EMOTIONS} for _ in face_images]
        
        try:
            # Preprocess all images
            processed_imgs = np.array([
                self.preprocess_image(face_img)[0] for face_img in face_images
            ])
            
            # Get predictions for batch
            batch_preds = self.model.predict(processed_imgs, verbose=0)
            
            # Create list of emotion score dictionaries
            emotion_scores_list = [
                {emotion: float(score) for emotion, score in zip(EMOTIONS, preds)}
                for preds in batch_preds
            ]
            
            return emotion_scores_list
            
        except Exception as e:
            print(f"Error predicting batch emotions: {str(e)}")
            return [{emotion: 0.0 for emotion in EMOTIONS} for _ in face_images]
    
    def save_model(self, model_path: str = EMOTION_MODEL_PATH):
        """
        Save the model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if not self._model_loaded:
            print("No model to save.")
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            self.model.save(model_path)
            print(f"Model saved to: {model_path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def train(
        self, 
        train_data, 
        validation_data=None, 
        epochs: int = 50, 
        batch_size: int = 32
    ):
        """
        Train the emotion classifier model.
        
        Args:
            train_data: Training data (x_train, y_train)
            validation_data: Validation data (x_val, y_val)
            epochs: Number of training epochs
            batch_size: Training batch size
        
        Returns:
            Training history
        """
        if not self._model_loaded:
            print("Model not initialized properly. Cannot train.")
            return None
        
        try:
            x_train, y_train = train_data
            
            # Train the model
            history = self.model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                verbose=1
            )
            
            return history
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None

# Import here to avoid circular imports
import cv2 