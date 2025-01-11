# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY models/anomaly_detector.weights.h5 /app/models/
COPY src/test_anomaly_detector.py /app/src/
COPY src/train_model.py /app/src/

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/anomaly_detector.weights.h5

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Command to run the application
CMD ["python", "src/test_anomaly_detector.py"]
