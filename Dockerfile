FROM python:3.10-slim  

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install PyTorch from their index
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install other packages from PyPI
RUN pip install --no-cache-dir \
    numpy==2.0.2 \
    pandas \
    scikit-learn

# Copy the rest of your application
COPY . .

CMD ["python", "app.py"]