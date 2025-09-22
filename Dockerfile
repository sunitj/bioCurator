# Multi-stage build for BioCurator following UV best practices
FROM python:3.12-slim-bookworm AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set build arguments
ARG BUILD_ENV=development

# Set UV environment variables for optimization
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY uv.lock pyproject.toml ./

# Install dependencies only (without the project itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$BUILD_ENV" = "development" ]; then \
        uv sync --frozen --no-install-project; \
    else \
        uv sync --frozen --no-install-project --no-dev; \
    fi

# Copy the rest of the application code
COPY . /app

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$BUILD_ENV" = "development" ]; then \
        uv sync --frozen; \
    else \
        uv sync --frozen --no-dev; \
    fi

# Production stage
FROM base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_MODE=${APP_MODE:-development} \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN useradd -m -u 1000 biocurator && \
    mkdir -p /app /logs /data && \
    chown -R biocurator:biocurator /app /logs /data

# Set working directory
WORKDIR /app

# Copy virtual environment and application from builder
COPY --from=builder --chown=biocurator:biocurator /app/.venv /app/.venv
COPY --from=builder --chown=biocurator:biocurator /app /app

# Switch to non-root user
USER biocurator

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m src.health.check || exit 1

# Default command
CMD ["python", "-m", "src.main"]