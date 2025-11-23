"""
Locust load testing script for Rock-Paper-Scissors ML Pipeline API.
Simulates multiple users making requests to the prediction endpoint.
"""

from locust import HttpUser, task, between
import random
import os
import base64


class MLPipelineUser(HttpUser):
    """Simulates a user interacting with the ML Pipeline API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Health check
        self.client.get("/api/health")
    
    @task(3)
    def predict_image(self):
        """Simulate image prediction request."""
        # Use a sample image from the test set
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_images_dir = os.path.join(base_dir, "data", "raw", "test")
        
        # Randomly select a class
        class_name = random.choice(["rock", "paper", "scissors"])
        class_dir = os.path.join(test_images_dir, class_name)
        
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                image_path = os.path.join(class_dir, random.choice(images))
                
                with open(image_path, 'rb') as f:
                    files = {'image': (os.path.basename(image_path), f, 'image/png')}
                    self.client.post("/api/predict", files=files, name="/api/predict")
    
    @task(1)
    def get_health(self):
        """Check API health."""
        self.client.get("/api/health", name="/api/health")
    
    @task(1)
    def get_uptime(self):
        """Get uptime statistics."""
        self.client.get("/api/uptime", name="/api/uptime")
    
    @task(1)
    def get_metrics(self):
        """Get model metrics."""
        self.client.get("/api/metrics", name="/api/metrics")
    
    @task(1)
    def get_stats(self):
        """Get dataset statistics."""
        self.client.get("/api/stats", name="/api/stats")


class HighLoadUser(HttpUser):
    """Simulates high-load scenario with rapid requests."""
    
    wait_time = between(0.1, 0.5)  # Very short wait time
    
    @task(10)
    def rapid_predictions(self):
        """Make rapid prediction requests."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_images_dir = os.path.join(base_dir, "data", "raw", "test")
        class_name = random.choice(["rock", "paper", "scissors"])
        class_dir = os.path.join(test_images_dir, class_name)
        
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                image_path = os.path.join(class_dir, random.choice(images))
                
                with open(image_path, 'rb') as f:
                    files = {'image': (os.path.basename(image_path), f, 'image/png')}
                    self.client.post("/api/predict", files=files, name="/api/predict (high load)")

