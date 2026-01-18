FROM python:3.9-alpine

WORKDIR /app

# Install minimal system dependencies for Python packages
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    curl

# Copy requirements
COPY requirements_light.txt requirements.txt

# Install Python dependencies in small batches
RUN pip install --no-cache-dir --upgrade pip

# Install core dependencies first
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    pandas==2.0.0 \
    pillow==10.0.0

# Install from requirements if any
RUN if [ -f requirements.txt ] && [ -s requirements.txt ]; then \
    pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy application code
COPY . .

# Remove build dependencies to save space
RUN apk del gcc musl-dev linux-headers

CMD ["python", "app.py"]