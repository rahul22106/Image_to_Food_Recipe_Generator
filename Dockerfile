FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Create constraints file to LOCK numpy
RUN echo "numpy==1.24.3" > /tmp/constraints.txt

# Install ALL packages with constraints file
COPY requirements.txt .

# ONE COMMAND - Install everything with numpy constraint applied
RUN pip install --no-cache-dir \
    -c /tmp/constraints.txt \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn \
    torch==2.0.0+cpu \
    torchvision==0.15.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install transformers and sentence-transformers WITH constraint
RUN pip install --no-cache-dir \
    -c /tmp/constraints.txt \
    transformers \
    sentence-transformers

# Install remaining requirements WITH constraint
RUN pip install --no-cache-dir \
    -c /tmp/constraints.txt \
    -r requirements.txt

# Verify numpy version
RUN python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); assert numpy.__version__ == '1.24.3', f'Wrong numpy: {numpy.__version__}'"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]