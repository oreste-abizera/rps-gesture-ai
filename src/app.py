"""
Flask API for Rock-Paper-Scissors classification model.
Provides endpoints for prediction, data upload, retraining, and monitoring.
"""

import os
import json
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import threading
import time
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import ImagePreprocessor
from model import RPSModel
from prediction import Predictor

# Initialize Flask app
app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'data/uploaded'
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)

# Database setup
Base = declarative_base()
db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'training_data.db')
engine = create_engine(f'sqlite:///{db_path}', echo=False)
Session = sessionmaker(bind=engine)


class UploadedImage(Base):
    """Database model for uploaded images."""
    __tablename__ = 'uploaded_images'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(500), nullable=False)
    label = Column(String(50))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    used_for_training = Column(Integer, default=0)  # 0 = not used, 1 = used


class ModelMetrics(Base):
    """Database model for storing model metrics."""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True)
    model_version = Column(String(100))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    loss = Column(Float)
    trained_at = Column(DateTime, default=datetime.utcnow)


class ModelUptime(Base):
    """Database model for tracking model uptime."""
    __tablename__ = 'model_uptime'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20))  # 'up' or 'down'
    response_time = Column(Float)  # in milliseconds


# Create tables
Base.metadata.create_all(engine)

# Global variables
model = None
preprocessor = None
predictor = None
model_loaded = False
training_in_progress = False
start_time = datetime.utcnow()


def load_model():
    """Load the trained model."""
    global model, preprocessor, predictor, model_loaded
    
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'rps_model.h5')
        if not os.path.exists(model_path):
            print("Model not found. Please train the model first.")
            return False
        
        # Initialize preprocessor
        preprocessor = ImagePreprocessor(img_size=(150, 150))
        
        # Load model
        rps_model = RPSModel(img_size=(150, 150), num_classes=3)
        rps_model.load_model(model_path)
        model = rps_model.model
        
        # Initialize predictor
        predictor = Predictor(model, preprocessor, rps_model.class_names)
        
        model_loaded = True
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def retrain_model():
    """Retrain the model with uploaded data."""
    global training_in_progress, model, preprocessor, predictor, model_loaded
    
    if training_in_progress:
        return {"status": "error", "message": "Training already in progress"}
    
    training_in_progress = True
    
    try:
        session = Session()
        
        # Get all uploaded images that haven't been used for training
        uploaded_images = session.query(UploadedImage).filter(
            UploadedImage.used_for_training == 0
        ).all()
        
        if len(uploaded_images) == 0:
            training_in_progress = False
            return {"status": "error", "message": "No new data to train on"}
        
        # Load existing training data
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_dir = os.path.join(base_dir, 'data', 'raw', 'train')
        test_dir = os.path.join(base_dir, 'data', 'raw', 'test')
        
        preprocessor = ImagePreprocessor(img_size=(150, 150))
        X_train, y_train, X_test, y_test = preprocessor.prepare_training_data(
            train_dir,
            test_dir
        )
        
        # Load uploaded images
        uploaded_X = []
        uploaded_y = []
        
        for img_record in uploaded_images:
            if os.path.exists(img_record.filepath):
                try:
                    img = preprocessor.load_and_preprocess_image(img_record.filepath)
                    uploaded_X.append(img)
                    uploaded_y.append(img_record.label)
                except Exception as e:
                    print(f"Error loading uploaded image {img_record.filepath}: {e}")
        
        if len(uploaded_X) > 0:
            uploaded_X = np.array(uploaded_X)
            uploaded_y_encoded = preprocessor.encode_labels(np.array(uploaded_y))
            
            # Combine with existing training data
            X_train = np.concatenate([X_train, uploaded_X], axis=0)
            y_train = np.concatenate([y_train, uploaded_y_encoded], axis=0)
        
        # Create and train model
        rps_model = RPSModel(img_size=(150, 150), num_classes=3)
        rps_model.build_model(use_transfer_learning=True)
        
        # Train model
        history = rps_model.train(
            X_train, y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=15,
            batch_size=32,
            use_early_stopping=True,
            use_reduce_lr=True
        )
        
        # Evaluate model
        if X_test is not None and y_test is not None:
            metrics = rps_model.evaluate(X_test, y_test)
            
            # Save metrics to database
            metric_record = ModelMetrics(
                model_version=f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                loss=metrics['loss']
            )
            session.add(metric_record)
        
        # Save model
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'rps_model.h5')
        rps_model.save_model(model_path)
        
        # Mark uploaded images as used
        for img_record in uploaded_images:
            img_record.used_for_training = 1
        session.commit()
        session.close()
        
        # Reload model
        load_model()
        
        training_in_progress = False
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "metrics": metrics if X_test is not None else None
        }
        
    except Exception as e:
        training_in_progress = False
        return {"status": "error", "message": str(e)}


# Load model on startup
load_model()


@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    uptime_seconds = (datetime.utcnow() - start_time).total_seconds()
    return jsonify({
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "uptime_seconds": uptime_seconds,
        "training_in_progress": training_in_progress
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict endpoint for single image."""
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500
    
    start_time_req = time.time()
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save temporary file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(base_dir, 'data', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Make prediction
        result = predictor.predict_single_image(temp_path)
        
        # Calculate response time
        response_time = (time.time() - start_time_req) * 1000  # in milliseconds
        
        # Log uptime
        session = Session()
        uptime_record = ModelUptime(
            status='up',
            response_time=response_time
        )
        session.add(uptime_record)
        session.commit()
        session.close()
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify(result)
        
    except Exception as e:
        response_time = (time.time() - start_time_req) * 1000
        session = Session()
        uptime_record = ModelUptime(status='down', response_time=response_time)
        session.add(uptime_record)
        session.commit()
        session.close()
        
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload():
    """Upload endpoint for training data."""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        label = request.form.get('label', '')
        
        if not label:
            return jsonify({"error": "Label is required"}), 400
        
        if label not in ['rock', 'paper', 'scissors']:
            return jsonify({"error": "Label must be one of: rock, paper, scissors"}), 400
        
        uploaded_files = []
        session = Session()
        
        for file in files:
            if file.filename == '':
                continue
            
            filename = secure_filename(file.filename)
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            # Create label directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            upload_dir = os.path.join(base_dir, app.config['UPLOAD_FOLDER'])
            label_dir = os.path.join(upload_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            
            # Save file
            filepath = os.path.join(label_dir, filename)
            file.save(filepath)
            
            # Save to database
            img_record = UploadedImage(
                filename=filename,
                filepath=filepath,
                label=label
            )
            session.add(img_record)
            uploaded_files.append(filename)
        
        session.commit()
        session.close()
        
        return jsonify({
            "status": "success",
            "message": f"Uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining."""
    if training_in_progress:
        return jsonify({"error": "Training already in progress"}), 400
    
    # Run retraining in background thread
    thread = threading.Thread(target=retrain_model)
    thread.start()
    
    return jsonify({
        "status": "started",
        "message": "Retraining started in background"
    })


@app.route('/api/retrain/status', methods=['GET'])
def retrain_status():
    """Get retraining status."""
    return jsonify({
        "training_in_progress": training_in_progress
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model metrics."""
    session = Session()
    metrics = session.query(ModelMetrics).order_by(ModelMetrics.trained_at.desc()).limit(10).all()
    session.close()
    
    metrics_list = [{
        "id": m.id,
        "model_version": m.model_version,
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1_score": m.f1_score,
        "loss": m.loss,
        "trained_at": m.trained_at.isoformat()
    } for m in metrics]
    
    return jsonify(metrics_list)


@app.route('/api/uptime', methods=['GET'])
def get_uptime():
    """Get model uptime statistics."""
    session = Session()
    
    # Get recent uptime records
    recent_records = session.query(ModelUptime).order_by(
        ModelUptime.timestamp.desc()
    ).limit(100).all()
    
    session.close()
    
    if not recent_records:
        return jsonify({
            "uptime_percentage": 0,
            "average_response_time": 0,
            "total_requests": 0,
            "recent_records": []
        })
    
    # Calculate uptime percentage
    up_count = sum(1 for r in recent_records if r.status == 'up')
    uptime_percentage = (up_count / len(recent_records)) * 100
    
    # Calculate average response time
    response_times = [r.response_time for r in recent_records if r.response_time]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    records = [{
        "timestamp": r.timestamp.isoformat(),
        "status": r.status,
        "response_time": r.response_time
    } for r in recent_records]
    
    return jsonify({
        "uptime_percentage": uptime_percentage,
        "average_response_time": avg_response_time,
        "total_requests": len(recent_records),
        "recent_records": records
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics."""
    session = Session()
    
    # Count uploaded images by label
    rock_count = session.query(UploadedImage).filter(UploadedImage.label == 'rock').count()
    paper_count = session.query(UploadedImage).filter(UploadedImage.label == 'paper').count()
    scissors_count = session.query(UploadedImage).filter(UploadedImage.label == 'scissors').count()
    total_uploaded = session.query(UploadedImage).count()
    unused_count = session.query(UploadedImage).filter(UploadedImage.used_for_training == 0).count()
    
    session.close()
    
    # Count raw data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_rock_dir = os.path.join(base_dir, 'data', 'raw', 'train', 'rock')
    train_paper_dir = os.path.join(base_dir, 'data', 'raw', 'train', 'paper')
    train_scissors_dir = os.path.join(base_dir, 'data', 'raw', 'train', 'scissors')
    
    train_rock = len(os.listdir(train_rock_dir)) if os.path.exists(train_rock_dir) else 0
    train_paper = len(os.listdir(train_paper_dir)) if os.path.exists(train_paper_dir) else 0
    train_scissors = len(os.listdir(train_scissors_dir)) if os.path.exists(train_scissors_dir) else 0
    
    return jsonify({
        "raw_data": {
            "train": {
                "rock": train_rock,
                "paper": train_paper,
                "scissors": train_scissors,
                "total": train_rock + train_paper + train_scissors
            }
        },
        "uploaded_data": {
            "rock": rock_count,
            "paper": paper_count,
            "scissors": scissors_count,
            "total": total_uploaded,
            "unused": unused_count
        }
    })


if __name__ == '__main__':
    # Create necessary directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base_dir, 'data', 'temp'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'data', 'uploaded'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    
    # Use PORT environment variable if set (for Render, Heroku, etc.)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

