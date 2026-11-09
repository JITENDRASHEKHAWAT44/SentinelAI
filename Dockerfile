# Sentinel AI - Production Dockerfile
FROM python:3.12-slim

# Prevent Python from writing .pyc files & enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for compiling psycopg and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application directories
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY data/ /app/data/

# Create uploads directory if not present
RUN mkdir -p /app/data/uploads

EXPOSE 8000

# Start FastAPI server (supports dynamic cloud PORT or defaults to 8000)
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
