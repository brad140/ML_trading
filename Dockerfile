# Multi-stage Dockerfile for ML Trading System
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY README.md .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose ports
EXPOSE 5000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["python", "-m", "src.app"]

# Production stage
FROM base as production

# Install additional production dependencies
RUN pip install gunicorn

# Copy production configuration
COPY docker/production.conf .

# Run with gunicorn
CMD ["gunicorn", "--config", "production.conf", "src.app:app"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install jupyter ipython

# Copy development configuration
COPY docker/development.conf .

# Run in development mode
CMD ["python", "-m", "src.app", "--debug"]

# ML Model stage
FROM base as ml-model

# Copy model files
COPY models/ ./models/

# Create model serving script
COPY docker/model_server.py .

# Install model serving dependencies
RUN pip install fastapi uvicorn

# Expose model serving port
EXPOSE 8000

# Run model server
CMD ["python", "model_server.py"]

# Kafka stage
FROM base as kafka-producer

# Install Kafka dependencies
RUN pip install kafka-python

# Copy Kafka producer script
COPY docker/kafka_producer.py .

# Run Kafka producer
CMD ["python", "kafka_producer.py"]

# Monitoring stage
FROM base as monitoring

# Install monitoring dependencies
RUN pip install prometheus-client grafana-api

# Copy monitoring scripts
COPY docker/monitoring/ ./monitoring/

# Expose monitoring port
EXPOSE 9090

# Run monitoring
CMD ["python", "monitoring/metrics_collector.py"]
