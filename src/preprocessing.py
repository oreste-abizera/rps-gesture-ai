"""
Data preprocessing module for Rock-Paper-Scissors image classification.
Handles image loading, augmentation, and preparation for model training.
"""

import os
import numpy as np
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List
import pandas as pd


class ImagePreprocessor:
    """Handles preprocessing of image data for the RPS classification model."""
    
    def __init__(self, img_size: Tuple[int, int] = (150, 150), normalize: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            img_size: Target image size (width, height)
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.img_size = img_size
        self.normalize = normalize
        self.label_encoder = LabelEncoder()
        self.classes = None
        
    def load_images_from_directory(self, directory: str, label: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from a directory.
        
        Args:
            directory: Path to directory containing images
            label: Optional label for all images in directory
            
        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        images = []
        labels = []
        
        if not os.path.exists(directory):
            raise ValueError(f"Directory {directory} does not exist")
        
        # If label is provided, use it for all images
        if label:
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(directory, filename)
                    try:
                        img = self.load_and_preprocess_image(img_path)
                        images.append(img)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
        else:
            # Load from subdirectories (each subdirectory is a class)
            for class_name in os.listdir(directory):
                class_path = os.path.join(directory, class_name)
                if os.path.isdir(class_path):
                    for filename in os.listdir(class_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, filename)
                            try:
                                img = self.load_and_preprocess_image(img_path)
                                images.append(img)
                                labels.append(class_name)
                            except Exception as e:
                                print(f"Error loading {img_path}: {e}")
                                continue
        
        return np.array(images), np.array(labels)
    
    def load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image from {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.img_size)
        
        # Normalize pixel values to [0, 1]
        if self.normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
    
    def encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Encode string labels to integers.
        
        Args:
            labels: Array of string labels
            
        Returns:
            Encoded labels as integers
        """
        if self.classes is None:
            self.classes = np.unique(labels)
            self.label_encoder.fit(self.classes)
        
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        Decode integer labels back to strings.
        
        Args:
            encoded_labels: Array of encoded integer labels
            
        Returns:
            Decoded string labels
        """
        if self.label_encoder.classes_ is None:
            raise ValueError("Label encoder not fitted. Call encode_labels first.")
        
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        if self.classes is None:
            return []
        return list(self.classes)
    
    def prepare_training_data(self, train_dir: str, test_dir: str = None) -> Tuple:
        """
        Prepare training and optionally test data.
        
        Args:
            train_dir: Directory containing training images
            test_dir: Optional directory containing test images
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test) or (X_train, y_train, None, None)
        """
        # Load training data
        print("Loading training data...")
        X_train, y_train = self.load_images_from_directory(train_dir)
        y_train_encoded = self.encode_labels(y_train)
        
        # Load test data if provided
        if test_dir and os.path.exists(test_dir):
            print("Loading test data...")
            X_test, y_test = self.load_images_from_directory(test_dir)
            y_test_encoded = self.encode_labels(y_test)
            return X_train, y_train_encoded, X_test, y_test_encoded
        else:
            return X_train, y_train_encoded, None, None
    
    def augment_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to an image.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Augmented image
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        
        # Random rotation
        angle = np.random.uniform(-15, 15)
        center = (self.img_size[0] // 2, self.img_size[1] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, self.img_size)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 1)
        
        return img

