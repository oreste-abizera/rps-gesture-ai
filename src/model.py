"""
Model creation and training module for Rock-Paper-Scissors classification.
Uses a CNN architecture with transfer learning capabilities.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from typing import Tuple, Dict
import joblib


class RPSModel:
    """Rock-Paper-Scissors classification model using CNN."""
    
    def __init__(self, img_size: Tuple[int, int] = (150, 150), num_classes: int = 3):
        """
        Initialize the model.
        
        Args:
            img_size: Input image size (width, height)
            num_classes: Number of classes (rock, paper, scissors)
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = ['rock', 'paper', 'scissors']
        
    def build_model(self, use_transfer_learning: bool = True) -> keras.Model:
        """
        Build the CNN model architecture.
        
        Args:
            use_transfer_learning: Whether to use MobileNetV2 as base
            
        Returns:
            Compiled Keras model
        """
        if use_transfer_learning:
            # Use MobileNetV2 as base model (transfer learning)
            base_model = MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False  # Freeze base model initially
            
            # Build model
            inputs = keras.Input(shape=(*self.img_size, 3))
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            self.model = keras.Model(inputs, outputs)
        else:
            # Custom CNN architecture
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.2,
        use_early_stopping: bool = True,
        use_reduce_lr: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data to use for validation
            use_early_stopping: Whether to use early stopping
            use_reduce_lr: Whether to use learning rate reduction
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Setup callbacks
        callback_list = []
        
        if use_early_stopping:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss' if (X_val is not None or validation_split > 0) else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stopping)
        
        if use_reduce_lr:
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss' if (X_val is not None or validation_split > 0) else 'loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001,
                verbose=1
            )
            callback_list.append(reduce_lr)
        
        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_loss' if (X_val is not None or validation_split > 0) else 'loss',
            save_best_only=True,
            verbose=1
        )
        callback_list.append(checkpoint)
        
        # Train model
        if X_val is not None and y_val is not None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callback_list,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callback_list,
                verbose=1
            )
        
        return self.history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Loss
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)[0]
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'loss': float(test_loss),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Input images
            
        Returns:
            Tuple of (predicted_classes, prediction_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load a trained model first.")
        
        predictions = self.model.predict(X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        return predicted_classes, predictions
    
    def save_model(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Save class names
        class_names_path = filepath.replace('.h5', '_classes.json')
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f)
    
    def load_model(self, filepath: str):
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        
        # Load class names if available
        class_names_path = filepath.replace('.h5', '_classes.json')
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        
        print(f"Model loaded from {filepath}")

