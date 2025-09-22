# BioCurator Makefile
.PHONY: help build test lint format clean setup health docker-up docker-down

# Default target
help:
	@echo "BioCurator Development Commands"
	@echo "================================"
	@echo "make setup       - Set up development environment"
	@echo "make build       - Build Docker containers"
	@echo "make test        - Run test suite"
	@echo "make lint        - Run code linters"
	@echo "make format      - Format code"
	@echo "make health      - Check system health"
	@echo "make docker-up   - Start Docker services"
	@echo "make docker-down - Stop Docker services"
	@echo "make clean       - Clean build artifacts"

# Setup development environment
setup:
	@echo "Setting up development environment with UV..."
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --python 3.11; \
		source .venv/bin/activate && uv pip install -e ".[dev]"; \
	else \
		echo "UV not found. Run ./scripts/setup_venv.sh instead"; \
		exit 1; \
	fi
	pre-commit install
	@echo "Environment setup complete!"

# Build Docker containers
build:
	@echo "Building Docker containers..."
	docker-compose build --no-cache

# Run tests with coverage
test:
	@echo "Running tests with coverage..."
	pytest tests/ \
		--cov=src \
		--cov-report=html \
		--cov-report=term \
		--cov-report=xml \
		--cov-fail-under=70

# Run linters
lint:
	@echo "Running linters..."
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports
	black --check src/ tests/
	isort --check-only src/ tests/

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

# Check system health
health:
	@echo "Checking system health..."
	@curl -s http://localhost:8080/health | python -m json.tool || echo "Health endpoint not available"
	@curl -s http://localhost:8080/ready | python -m json.tool || echo "Ready endpoint not available"

# Start Docker services
docker-up:
	@echo "Starting Docker services..."
	docker-compose -f docker-compose.yml -f docker-compose.$(APP_MODE).yml up -d

# Stop Docker services
docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	rm -rf build/ dist/ *.egg-info
	@echo "Clean complete!"

# Security scan
security-scan:
	@echo "Running security scan..."
	uv run pip-audit
	uv run bandit -r src/

# Generate SBOM
sbom:
	@echo "Generating Software Bill of Materials..."
	uv add --dev cyclonedx-bom
	uv run cyclonedx-py --format json --output-file sbom.json .

# Run development server
dev:
	@echo "Starting development server..."
	APP_MODE=development python -m src.main --reload

# Run production server
prod:
	@echo "Starting production server..."
	APP_MODE=production python -m src.main