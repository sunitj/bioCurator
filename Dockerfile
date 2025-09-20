# Multi-stage build for BioCurator
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_ENV=development

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ "$BUILD_ENV" = "development" ]; then \
        pip install --no-cache-dir -r requirements-dev.txt; \
    fi

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_MODE=${APP_MODE:-development}

# Create non-root user
RUN useradd -m -u 1000 biocurator && \
    mkdir -p /app /logs /data && \
    chown -R biocurator:biocurator /app /logs /data

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=biocurator:biocurator . .

# Switch to non-root user
USER biocurator

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m src.health.check || exit 1

# Default command
CMD ["python", "-m", "src.main"]