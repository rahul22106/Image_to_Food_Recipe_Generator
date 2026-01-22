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

# **CHANGE HERE: Install everything in ONE STEP with constraint**
RUN pip install --no-cache-dir \
    --constraint <(echo "numpy==1.24.3") \
    torch==2.0.0+cpu \
    torchvision==0.15.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    --constraint <(echo "numpy==1.24.3") \
    -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD ["python", "app.py"]