FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Verify uv is installed
RUN uv --version

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies with uv
RUN uv pip install --system --no-cache \
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