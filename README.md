# BioCurator

Memory-augmented multi-agent system for scientific literature curation and analysis.

## Overview

BioCurator demonstrates how AI agents can develop domain expertise through collaborative literature analysis, using a sophisticated multi-modal memory architecture and safety-first development approach.

## Quick Start

### Development Mode (Local Models - Zero Cost)

```bash
# Set up development environment with UV
./scripts/setup_venv.sh
source .venv/bin/activate
export APP_MODE=development

# Configure environment (optional - uses defaults if not set)
cp .env.example .env
# Edit .env to set JUPYTER_TOKEN and other configurations

# Run with local models (Ollama)
docker-compose -f docker-compose.yml -f docker-compose.development.yml up

# Access services:
# - BioCurator API: http://localhost:8080
# - Jupyter Lab: http://localhost:8888 (token: see JUPYTER_TOKEN in .env or default: biocurator-dev)
# - Ollama API: http://localhost:11434

# Check system health
make health
```

### Production Mode (Cloud Models)

```bash
# Set up production environment with UV
./scripts/setup_venv.sh
source .venv/bin/activate
export APP_MODE=production

# Run with cloud models
docker-compose -f docker-compose.yml -f docker-compose.production.yml up
```

## Architecture

<!-- Architecture Diagram Placeholder: docs/images/architecture.png -->
<!-- TODO: Add detailed system architecture diagram showing component interactions -->

```text
┌─────────────────────────────────────────────────────────┐
│                     Agent Orchestra                      │
├─────────────────────────────────────────────────────────┤
│  Research    Literature    Deep      Domain    Knowledge │
│  Director      Scout      Reader   Specialist   Weaver  │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│                   Safety Controls                        │
│  Circuit Breakers │ Rate Limiting │ Cost Tracking       │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│                  Memory Systems                          │
│   Neo4j   │   Qdrant   │  PostgreSQL  │  Redis │ SQLite │
└──────────────────────────────────────────────────────────┘
```

## Key Features

- **Multi-Agent Coordination**: Specialized agents for literature discovery, analysis, and synthesis
- **Multi-Modal Memory**: Knowledge graph, vector embeddings, episodic memory, and procedural patterns
- **Safety-First Design**: Circuit breakers, rate limiting, cost tracking, and anomaly detection
- **Development Mode**: Free local model operation with Ollama (DeepSeek-R1, Llama 3.1, Qwen 2.5)
- **Production Ready**: Cloud model integration with comprehensive monitoring and observability

## Development

### Requirements

- Python 3.11+
- [UV package manager](https://docs.astral.sh/uv/) (installed automatically by setup script)
- Docker and Docker Compose

### Setup

```bash
# Automated setup with UV
./scripts/setup_venv.sh
source .venv/bin/activate

# Manual setup alternative
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Common Commands

```bash
# Run tests
make test

# Run linting
make lint

# Format code
make format

# Build containers
make build

# View metrics
curl http://localhost:9090/metrics

# Check health
curl http://localhost:8080/health
```

## Documentation

- [Architecture Decision Records](docs/adr/)
- [API Documentation](docs/api/)
- [Development Guide](docs/development.md)
- [Safety Controls](docs/safety.md)

## Testing

The project maintains:

- >=70% overall test coverage
- >=85% coverage for safety-critical modules
- Comprehensive integration tests
- Performance benchmarks

## License

Apache 2.0 - See LICENSE file for details
