# StockVision AI — Dockerfile
# Multi-stage build for production deployment

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Builder stage ─────────────────────────────────────────────────────────
FROM base AS builder

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Production stage ──────────────────────────────────────────────────────
FROM base AS production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/       ./src/
COPY api/       ./api/
COPY dashboard/ ./dashboard/
COPY sql/       ./sql/

# Create required directories
RUN mkdir -p data/raw data/processed data/external models logs reports

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000  
EXPOSE 8501  

# Default command: run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── To run Streamlit instead: ─────────────────────────────────────────────
# CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
