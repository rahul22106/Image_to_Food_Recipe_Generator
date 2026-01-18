FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install using pre-built manylinux wheels (no compilation)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.24.0 \
    pandas==2.0.0 \
    pillow==10.0.0 \
    fastapi==0.104.0 \
    uvicorn==0.24.0 \
    requests==2.31.0

# Install any additional requirements
RUN if [ -f requirements.txt ] && [ -s requirements.txt ]; then \
    pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy application code
COPY . .

CMD ["python", "app.py"]