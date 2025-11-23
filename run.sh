#!/bin/bash

# Script to run the ML Pipeline application

echo "Rock-Paper-Scissors ML Pipeline"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if model exists
if [ ! -f "models/rps_model.h5" ]; then
    echo "Model not found. Training model..."
    python src/train.py --epochs 20 --batch-size 32
fi

# Run the application
echo "Starting Flask application..."
python src/app.py

