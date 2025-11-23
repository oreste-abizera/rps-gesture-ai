FROM python:3.9-slim

# Use a specific Debian version for compatibility
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
# Note: For headless OpenCV, we don't need GL libraries
# Split into multiple RUN commands for better caching and error handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/temp data/uploaded models static/css static/js templates

# Expose port (Render uses PORT env var, default to 5000)
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=src/app.py
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Run the application
# Use PORT environment variable if set (Render provides this)
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 --threads 2 src.app:app

