#!/bin/bash
# Setup script for BioCurator with UV

set -e

echo "Setting up BioCurator development environment with UV..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "UV version: $(uv --version)"

# Create virtual environment
echo "Creating virtual environment..."
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e ".[dev]"

# Run checks
echo "Running basic checks..."
python -c "import src; print('✓ Package imports successfully')"
python -c "from src.config import settings; print('✓ Config loads successfully')"
python -c "from src.main import create_app; print('✓ App creates successfully')"

echo "✅ Setup complete! Activate the environment with:"
echo "  source .venv/bin/activate"