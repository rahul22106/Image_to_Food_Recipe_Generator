FROM python:3.9-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install torch first (CPU version for smaller image)
RUN pip install --no-cache-dir \
    torch==2.0.0+cpu \
    torchvision==0.15.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install compatible numpy and pandas versions
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3

# Install other core dependencies
RUN pip install --no-cache-dir \
    pillow==10.0.0 \
    fastapi==0.104.0 \
    uvicorn==0.24.0 \
    requests==2.31.0 \
    sentence-transformers \
    python-multipart \
    opencv-python-headless \
    pydantic \
    python-box \
    pyyaml \
    boto3 \
    tqdm

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

CMD ["python", "app.py"]