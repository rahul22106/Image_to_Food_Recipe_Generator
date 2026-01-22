FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies with pip
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of your application
COPY . .

# Your application command
CMD ["python", "app.py"]