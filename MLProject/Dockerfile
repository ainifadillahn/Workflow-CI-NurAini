# Base image dengan MLflow
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies sistem
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow dan dependencies
RUN pip install --no-cache-dir \
    mlflow==2.9.0 \
    scikit-learn \
    pandas \
    numpy \
    gunicorn

# Copy semua file dari MLProject
COPY . /app/

# Set environment variables untuk MLflow
ENV MLFLOW_TRACKING_URI=/app/mlruns

# Expose port untuk model serving
EXPOSE 8080

# Command untuk serve model dengan MLflow
CMD ["mlflow", "models", "serve", \
     "-m", "mlruns/0/latest/artifacts/model", \
     "-h", "0.0.0.0", \
     "-p", "8080", \
     "--no-conda"]