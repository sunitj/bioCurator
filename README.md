# BioCurator

Memory-augmented multi-agent system for scientific literature curation and analysis.

## Overview

BioCurator demonstrates how AI agents can develop domain expertise through collaborative literature analysis, using a sophisticated multi-modal memory architecture and safety-first development approach.

## Quick Start

### Development Mode (Local Models - Zero Cost)

```bash
# Set up development environment with UV
export UV_LINK_MODE=copy
./scripts/setup_venv.sh
source .venv/bin/activate
export APP_MODE=development

# Configure environment (optional - uses defaults if not set)
cp .env.example .env
# Edit .env to set JUPYTER_TOKEN and other configurations

# Run with local models (Ollama) - services start in dependency order
docker-compose -f docker-compose.yml -f docker-compose.development.yml up -d

# Wait for all services to be healthy (takes ~30-60 seconds)
docker-compose ps  # Check status

# Access services (replace localhost with your server IP if remote):
# - BioCurator API: http://localhost:8080/
# - Health Status: http://localhost:8080/health/
# - Neo4j Browser: http://localhost:7474/ (user: neo4j, password: dev_password)
# - Jupyter Lab: http://localhost:8888/ (token: biocurator-dev or JUPYTER_TOKEN)
# - Ollama API: http://localhost:11434/

# Verify system health
curl -s http://localhost:8080/health/ | python -m json.tool
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

## Troubleshooting

### Common Issues

1. **Services fail to start or restart continuously**
   - Check Docker logs: `docker logs <container-name>`
   - Neo4j memory settings require specific format in Docker Compose
   - Ensure all required ports are available: 8080, 7474, 7687, 6333, 5432, 6379, 8086

2. **Application can't connect to databases**
   - Verify environment variables are set in docker-compose files
   - Services must use container names (e.g., `redis`, `postgres`) not `localhost`
   - Check that all services are healthy: `docker-compose ps`

3. **Health endpoint shows "unhealthy" but system works**
   - This is expected if optional backends (like InfluxDB) aren't initialized
   - Check individual component status in the health response
   - Only required backends (Redis, PostgreSQL, Neo4j, Qdrant) need to be healthy

4. **Cannot access endpoints from browser (EC2/Remote)**
   - Ensure security groups allow inbound traffic on required ports
   - Use server's public IP instead of localhost
   - Consider SSH tunneling for secure development access

5. **Fresh start after issues**
   ```bash
   docker-compose down
   docker volume rm $(docker volume ls -q | grep biocurator)  # Removes all data
   docker-compose build --no-cache app
   docker-compose -f docker-compose.yml -f docker-compose.development.yml up -d
   ```

## License

Apache 2.0 - See LICENSE file for details
