# Multi-stage build for BioCurator
FROM python:3.11-slim AS builder

# Set build arguments
ARG BUILD_ENV=development

# Install build dependencies and UV
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to PATH (UV installs to /root/.local/bin)
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY README.md .

# Create virtual environment and install dependencies
RUN uv venv \
    && . .venv/bin/activate \
    && if [ "$BUILD_ENV" = "development" ]; then \
        uv pip install -e ".[dev]"; \
    else \
        uv pip install -e .; \
    fi

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_MODE=${APP_MODE:-development} \
    PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN useradd -m -u 1000 biocurator && \
    mkdir -p /app /logs /data && \
    chown -R biocurator:biocurator /app /logs /data

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=biocurator:biocurator /app/.venv /app/.venv

# Copy application code
COPY --chown=biocurator:biocurator . .

# Switch to non-root user
USER biocurator

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m src.health.check || exit 1

# Default command
CMD ["python", "-m", "src.main"]