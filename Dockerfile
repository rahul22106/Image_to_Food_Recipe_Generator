FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install torch first (CPU version)
RUN pip install --no-cache-dir \
    torch==2.0.0+cpu \
    torchvision==0.15.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install numpy and pandas with compatible versions
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3

# Install other dependencies
RUN pip install --no-cache-dir \
    pillow==10.0.0 \
    fastapi==0.104.0 \
    uvicorn==0.24.0 \
    requests==2.31.0 \
    sentence-transformers \
    python-multipart \
    opencv-python-headless

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

CMD ["python", "app.py"]