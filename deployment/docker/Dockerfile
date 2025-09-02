# Beverly Knits ERP System - Production Docker Image
# Multi-stage build for optimized production deployment

# Stage 1: Builder stage
FROM python:3.11-slim as builder

# Set build arguments and environment variables
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=3.0.0

LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="Beverly Knits ERP" \
      org.label-schema.description="Textile Manufacturing ERP System with AI-powered optimization" \
      org.label-schema.url="https://beverlyknits.com" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.version=$VERSION \
      org.label-schema.vendor="Beverly Knits" \
      org.label-schema.schema-version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    cmake \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements files
WORKDIR /app
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-ml.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# Stage 2: Runtime stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libxml2 \
    libxslt1.1 \
    libjpeg62-turbo \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r bki && useradd -r -g bki -m -d /home/bki bki

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=bki:bki . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data/processed /app/data/raw /app/cache && \
    chown -R bki:bki /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/New_York

# Expose the application port
EXPOSE 5006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5006/health || exit 1

# Switch to non-root user
USER bki

# Default command - run the main ERP application
CMD ["python", "beverly_comprehensive_erp.py"]