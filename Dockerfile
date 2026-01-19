FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (torch may need some)
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
    torch==2.8.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install other core dependencies
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    pandas==2.0.0 \
    pillow==10.0.0 \
    fastapi==0.104.0 \
    uvicorn==0.24.0 \
    requests==2.31.0

# Install any additional requirements from requirements.txt
RUN if [ -f requirements.txt ] && [ -s requirements.txt ]; then \
    pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy application code
COPY . .

CMD ["python", "app.py"]