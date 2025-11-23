"""
Prediction module for Rock-Paper-Scissors image classification.
Handles single image predictions and batch predictions.
"""

import numpy as np
from PIL import Image
import cv2
from typing import Dict, Tuple, List
import os


class Predictor:
    """Handles predictions using the trained RPS model."""
    
    def __init__(self, model, preprocessor, class_names: List[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model: Trained Keras model
            preprocessor: ImagePreprocessor instance
            class_names: List of class names
        """
        self.model = model
        self.preprocessor = preprocessor
        self.class_names = class_names or ['rock', 'paper', 'scissors']
    
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Predict the class of a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Preprocess image
        img = self.preprocessor.load_and_preprocess_image(image_path)
        img_batch = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = self.class_names[predicted_class_idx]
        
        # Get probabilities for all classes
        class_probabilities = {
            self.class_names[i]: float(predictions[0][i])
            for i in range(len(self.class_names))
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'all_predictions': predictions[0].tolist()
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_array(self, image_array: np.ndarray) -> Dict:
        """
        Predict from a numpy array (already preprocessed).
        
        Args:
            image_array: Preprocessed image array
            
        Returns:
            Dictionary with prediction results
        """
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = self.class_names[predicted_class_idx]
        
        # Get probabilities for all classes
        class_probabilities = {
            self.class_names[i]: float(predictions[0][i])
            for i in range(len(self.class_names))
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'all_predictions': predictions[0].tolist()
        }

