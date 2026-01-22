FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and UV
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libopenblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.cargo/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install dependencies with UV (much faster!)
RUN uv pip install --system --no-cache \
    numpy \
    pandas \
    scikit-learn \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache \
    transformers \
    sentence-transformers

RUN uv pip install --system --no-cache -r requirements.txt

# Verify installations
RUN python -c "import numpy; import pandas; import torch; print(f'numpy: {numpy.__version__}, pandas: {pandas.__version__}, torch: {torch.__version__}')"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]