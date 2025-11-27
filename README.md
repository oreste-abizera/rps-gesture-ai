# Rock-Paper-Scissors Gesture Classification ML Pipeline

A complete end-to-end Machine Learning pipeline for classifying Rock-Paper-Scissors hand gestures from images. This project demonstrates the full ML lifecycle including data preprocessing, model training, evaluation, deployment, monitoring, and retraining capabilities.

## 🎯 Project Overview

This project implements a production-ready ML pipeline for image classification using:
- **Dataset**: Rock-Paper-Scissors hand gesture images from Kaggle
- **Model Architecture**: CNN with Transfer Learning (MobileNetV2)
- **Framework**: TensorFlow/Keras
- **API**: Flask REST API
- **Frontend**: Modern web UI with real-time visualizations
- **Deployment**: Docker containerization
- **Load Testing**: Locust for performance testing

## 🎥 Demo Video

Youtube Link: [https://youtu.be/OCWhXyKQHqE](https://youtu.be/OCWhXyKQHqE)

## Demo Urls

Web UI Interface: [https://rps-ml-pipeline-production.up.railway.app](https://rps-ml-pipeline-production.up.railway.app)

API DOCS: [https://rps-ml-pipeline-production.up.railway.app/docs](https://rps-ml-pipeline-production.up.railway.app/docs)


## 📋 Features

### Core Functionality
- **Image Classification**: Predict Rock, Paper, or Scissors from uploaded images
- **Model Training**: Train models with optimization techniques (Early Stopping, Learning Rate Reduction, Transfer Learning)
- **Model Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1 Score, Loss)
- **Data Upload**: Bulk upload images for retraining
- **Model Retraining**: Trigger retraining with newly uploaded data
- **Monitoring**: Real-time model uptime, response time, and performance metrics
- **Visualizations**: Interactive charts for data distribution, model performance, and feature analysis

### Technical Features
- Transfer Learning with MobileNetV2
- Data augmentation capabilities
- SQLite database for tracking uploaded data and metrics
- RESTful API with multiple endpoints
- Docker containerization for easy deployment
- Load testing with Locust
- Scalable architecture with multiple container support

## 📁 Project Structure

```
rps-gesture-pipeline/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Multi-container setup
├── locustfile.py           # Load testing script
│
├── notebook/
│   └── rps_gesture_pipeline.ipynb  # Jupyter notebook with full pipeline
│
├── src/
│   ├── preprocessing.py     # Data preprocessing module
│   ├── model.py            # Model architecture and training
│   ├── prediction.py       # Prediction module
│   ├── app.py              # Flask API application
│   └── train.py            # Standalone training script
│
├── templates/
│   └── index.html          # Web UI template
│
├── static/
│   ├── css/
│   │   └── style.css       # Styling
│   └── js/
│       └── app.js          # Frontend JavaScript
│
├── data/
│   ├── raw/                # Original dataset
│   │   ├── train/          # Training images
│   │   └── test/           # Test images
│   ├── temp/               # Temporary files
│   └── uploaded/           # User-uploaded images for retraining
│
└── models/                 # Saved models and database
    ├── rps_model.h5        # Trained model (generated)
    └── training_data.db    # SQLite database (generated)
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- Git

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/oreste-abizera/rps-gesture-ai
   cd rps-gesture-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the dataset**
   - Ensure your dataset is in `data/raw/train/` and `data/raw/test/` directories
   - Each directory should have subdirectories: `rock/`, `paper/`, `scissors/`

5. **Train the initial model**
   
   Option A: Using the Jupyter notebook
   ```bash
   jupyter notebook notebook/rps_gesture_pipeline.ipynb
   ```
   Run all cells to train and evaluate the model.
   
   Option B: Using the training script
   ```bash
   python src/train.py --epochs 20 --batch-size 32
   ```

6. **Run the Flask application**
   ```bash
   python src/app.py
   ```

7. **Access the web UI**
   - Open your browser and navigate to: `http://localhost:5000`

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   - Main service: `http://localhost:5001`
   - Additional instances: `http://localhost:5002`, `http://localhost:5003`

3. **Run with a single container**
   ```bash
   docker build -t rps-ml-pipeline .
   docker run -p 5001:5000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models rps-ml-pipeline
   ```

## 📊 Model Evaluation

The model is evaluated using multiple metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Correctness of positive predictions
- **Recall**: Ability to find all positive instances
- **F1 Score**: Harmonic mean of precision and recall
- **Loss**: Training/validation loss

All metrics are displayed in the Jupyter notebook and available via the API.

## 🔌 API Endpoints

### Prediction
- `POST /api/predict` - Predict class for a single image
  - Body: `multipart/form-data` with `image` file

### Data Management
- `POST /api/upload` - Upload images for retraining
  - Body: `multipart/form-data` with `files[]` and `label`
- `GET /api/stats` - Get dataset statistics

### Model Management
- `POST /api/retrain` - Trigger model retraining
- `GET /api/retrain/status` - Get retraining status

### Monitoring
- `GET /api/health` - Health check endpoint
- `GET /api/uptime` - Model uptime statistics
- `GET /api/metrics` - Model performance metrics

## 🧪 Load Testing with Locust

Test the API performance under load:

1. **Start the Flask application**
   ```bash
   python src/app.py
   ```

2. **Run Locust**
   ```bash
   locust -f locustfile.py --host=http://localhost:5001
   ```

3. **Access Locust UI**
   - Open `http://localhost:8089` in your browser
   - Set number of users and spawn rate
   - Start the test and monitor results

4. **Test with multiple containers**
   - Start multiple containers: `docker-compose up --scale web=3`
   - Use a load balancer or test each container individually
   - Compare response times and throughput

## 📈 Visualizations

The web UI provides several visualizations:

1. **Dataset Distribution**: Bar chart showing class distribution in training data
2. **Model Performance Metrics**: Accuracy, Precision, Recall, F1 Score
3. **Feature Analysis - Image Size Distribution**: Distribution of image sizes
4. **Feature Analysis - Class Balance**: Pie chart of class distribution
5. **Feature Analysis - Training Progress**: Model accuracy over training iterations
6. **Response Time Over Time**: Line chart of API response times
7. **Uptime History**: Model availability over time

## 🎓 Model Architecture

The model uses **Transfer Learning** with MobileNetV2 as the base:

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 150x150x3 (RGB images)
- **Output**: 3 classes (Rock, Paper, Scissors)
- **Optimization Techniques**:
  - Early Stopping (prevents overfitting)
  - Learning Rate Reduction (adaptive learning)
  - Dropout layers (regularization)
  - Global Average Pooling

## 🔄 Retraining Process

1. **Upload Data**: Use the "Upload Data" tab to upload new images with labels
2. **Trigger Retraining**: Click "Start Retraining" in the "Retrain Model" tab
3. **Background Processing**: Retraining runs in a background thread
4. **Model Update**: New model is automatically loaded after training completes
5. **Metrics Tracking**: All training metrics are saved to the database

## 📝 Dataset Information

- **Source**: [Kaggle - Rock-Paper-Scissors Dataset](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset)
- **Training Images**: ~840 images per class (2,520 total)
- **Test Images**: ~124 images per class (372 total)
- **Classes**: Rock, Paper, Scissors
- **Format**: PNG images

## 🐛 Troubleshooting

### Model not found error
- Ensure you've trained the model first using the notebook or training script
- Check that `models/rps_model.h5` exists

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using Python 3.9+

### Docker build issues
- Ensure Docker has enough memory allocated (recommended: 4GB+)
- Check that all files are in the correct directories

### Database errors
- The database is created automatically on first run
- Ensure write permissions in the `models/` directory

## 📄 License

This project is created for educational purposes as part of the Machine Learning Pipeline assignment.

## 👤 Author

Created as part of the African Leadership University Machine Learning Pipeline assignment.

## 🙏 Acknowledgments

- Dataset: [Kaggle - Rock-Paper-Scissors Dataset](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset)
- TensorFlow/Keras for deep learning framework
- Flask for API development
- Plotly for interactive visualizations

---

**Note**: This is a complete ML pipeline implementation demonstrating best practices in MLOps, including model versioning, monitoring, and retraining capabilities.

