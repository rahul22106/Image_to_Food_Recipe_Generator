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

# 1. Create constraint file
RUN echo "numpy==1.24.3" > constraints.txt

# 2. Install torch with numpy constraint
RUN pip install --no-cache-dir \
    -c constraints.txt \
    torch==2.0.0+cpu \
    torchvision==0.15.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Install requirements with numpy constraint
RUN pip install --no-cache-dir \
    -c constraints.txt \
    -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD ["python", "app.py"]