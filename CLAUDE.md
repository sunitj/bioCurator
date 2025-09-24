# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioCurator is a memory-augmented multi-agent system for scientific literature curation and analysis, demonstrating how AI agents can develop domain expertise through collaborative literature analysis.

## Architecture

### Core Agent Ecosystem
- **Research Director**: Strategic coordination and workflow orchestration (✅ **PR #3 - IMPLEMENTED**)
  - Multi-agent workflow orchestration
  - Task allocation and load balancing
  - Quality control and result synthesis
  - Inter-agent communication coordination
- **Literature Scout**: Paper discovery and acquisition using GPT-4o (🚧 Future PR #6)
- **Deep Reader**: Content analysis and extraction using Claude Sonnet 4 (🚧 Future PR #7)
- **Domain Specialist**: Scientific validation using Claude Sonnet 4 + specialized models (🚧 Future PR #10)
- **Knowledge Weaver**: Synthesis and connection identification using GPT-4o (🚧 Future PR #11)
- **Memory Keeper**: Memory management and curation using Claude Sonnet 4 (🚧 Future PR #12)

### Memory Systems
- **Neo4j**: Conceptual knowledge graph for papers, authors, concepts, methods, findings
- **Qdrant**: Semantic memory store with dense embeddings
- **PostgreSQL**: Episodic memory for agent interaction histories
- **Redis**: Working memory for active analysis contexts
- **SQLite**: Procedural memory for workflow patterns

### Safety Architecture (✅ **PR #3 - FULLY INTEGRATED**)
- **Multi-Mode Operation**: Development (local models), Production (cloud models), Hybrid
- **Circuit Breakers**: Per-agent circuit breakers with configurable thresholds (closed/open/half-open states)
- **Rate Limiting**: Token bucket algorithm with burst capacity and refill rates
- **Cost Tracking**: Real-time budget monitoring and enforcement with pluggable price catalogs
- **Behavior Monitoring**: Anomaly detection for rogue agent behavior with statistical analysis
- **Safety Event Bus**: Comprehensive safety event logging and alerting system

## Development Setup

### Requirements
- Python 3.11+
- [UV package manager](https://docs.astral.sh/uv/) (auto-installed by setup script)
- Docker and Docker Compose

### Quick Start
```bash
# Automated setup with UV (recommended)
./scripts/setup_venv.sh
source .venv/bin/activate

# Manual setup alternative
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Local Development with Ollama
```bash
# Setup Ollama for development mode
scripts/setup_ollama.sh
scripts/download_models.py

# Development models (cost-free)
- ollama/deepseek-r1:32b (reasoning)
- ollama/llama3.1:8b (general)
- ollama/qwen2.5:7b (technical)
```

### Docker Environment
```bash
# Development environment
export APP_MODE=development
docker-compose -f docker-compose.yml -f docker-compose.development.yml up

# Production environment
export APP_MODE=production
docker-compose -f docker-compose.yml -f docker-compose.production.yml up

# Memory services (PR #2)
docker-compose -f docker-compose.memory.yml up

# Initialize memory system
python scripts/setup_memory.py
```

## Common Commands

### Build and Development
```bash
# Setup development environment with UV
./scripts/setup_venv.sh && source .venv/bin/activate

# Makefile targets (PR #1)
make build      # Build Docker containers
make lint       # Run code linters
make test       # Run test suite
make format     # Format code
make health     # Check system health
make clean      # Clean build artifacts
```

### Testing
```bash
# Run all tests (coverage target: >=70% initial, >=85% for safety module)
pytest tests/

# Run specific test categories
pytest tests/agents/
pytest tests/memory/
pytest tests/safety/

# Integration tests
pytest tests/integration/

# Performance benchmarking (PR #2.5)
python scripts/benchmark_models.py
```

### Paper Ingestion
```bash
# Ingest papers from sources
python scripts/ingest_papers.py --source pubmed --query "protein folding"
python scripts/ingest_papers.py --source arxiv --category "cs.AI"
```

### Agent Workflows
```bash
# Run basic analysis workflow
python examples/basic_workflow.py

# Run complete workflow
python examples/complete_workflow.py

# Safety demonstration
python examples/safety_demo.py
```

### System Health and Monitoring
```bash
# Check system health
python scripts/health_check.py

# Health/readiness endpoints (PR #1)
curl http://localhost:8080/health
curl http://localhost:8080/ready

# Monitor agent performance
python scripts/monitor_agents.py

# Inspect safety state (PR #1.5)
python -m src.safety.cli --list-breakers
python -m src.safety.cli --export-state > safety_state.json

# Prometheus metrics endpoint (PR #1)
curl http://localhost:9090/metrics
```

### Memory Management
```bash
# Initialize memory systems (PR #2)
python scripts/setup_memory.py

# Reset all memory data (DESTRUCTIVE)
python scripts/setup_memory.py --reset

# Health check all memory backends
python -c "
import asyncio
from src.config.loader import load_config
from src.memory.manager import DefaultMemoryManager
async def check():
    config = load_config()
    async with DefaultMemoryManager(config.database) as manager:
        health = await manager.health_check_all()
        for name, status in health.items():
            print(f'{name}: {status.message}')
asyncio.run(check())
"

# Memory backend endpoints
curl http://localhost:7474  # Neo4j browser
curl http://localhost:6333/collections  # Qdrant collections
curl http://localhost:8086  # InfluxDB UI

# Generate embeddings for papers (future)
python scripts/generate_embeddings.py

# Benchmark memory performance (future)
python scripts/benchmark_memory.py
```

### Agent System Management (✅ **PR #3 - NEW**)
```bash
# Run agent workflow examples
python examples/basic_workflow.py      # Basic multi-agent coordination demo
python examples/safety_demo.py        # Safety controls demonstration

# Agent configuration files
cagents.yaml                          # Base agent configuration
cagents.development.yaml              # Development mode overrides (local models)
cagents.production.yaml               # Production mode overrides (cloud models)

# Check agent health through API
curl http://localhost:8080/health | jq '.components[] | select(.name | startswith("agent"))'

# Agent registry and coordination
python -c "
import asyncio
from src.config.loader import load_config
from src.agents.registry import AgentRegistry
from src.memory.manager import DefaultMemoryManager

async def check_agents():
    config = load_config()
    async with DefaultMemoryManager(config.database) as memory_manager:
        registry = AgentRegistry(config, memory_manager)
        await registry.load_configuration(['cagents.yaml', 'cagents.development.yaml'])
        # Registry operations...
asyncio.run(check_agents())
"

# Task queue operations (when implemented with full database)
# python -c "from src.coordination.task_queue import TaskQueue; ..."

# Agent performance monitoring
# Access through /health endpoint or agent monitor API
```

## Development Workflow

### Phase 1: Foundation (Current)
Focus on basic infrastructure, memory setup, and simple agent coordination with safety controls.

### Phase 2: Intelligence
Implement advanced agent behaviors, memory-informed decisions, and temporal analysis.

### Phase 3: Innovation  
Develop emergent specialization, collaborative problem solving, and predictive intelligence.

## Configuration

### Application Modes (APP_MODE environment variable)
- `development`: Local models only, zero cost, strict safety controls
- `production`: Cloud models, configurable budgets, balanced controls  
- `hybrid`: Local-first with cloud escalation, adaptive controls

### Agent Configuration (cagents.yaml)
- Development mode: Uses local Ollama models for cost-free experimentation
- Production mode: Uses cloud foundation models for maximum capability
- Hybrid mode: Local-first with cloud escalation for complex tasks

### Safety Controls (configs/)
- development.yaml: Strict safety controls, zero cost budget, hard guard against cloud models
- production.yaml: Balanced controls, configurable budget
- hybrid.yaml: Adaptive controls based on task complexity

### Critical Configuration Requirements
- Central config loader with schema validation (PR #1)
- Failure on missing critical configuration keys
- Environment + YAML configuration support
- Configurable safety thresholds and circuit breaker states

## Key Implementation Areas

### Literature Processing
- PubMed, arXiv, bioRxiv integration
- PDF parsing and metadata extraction
- Scientific embeddings (SciBERT, BioBERT)

### Agent Coordination
- Docker cagents orchestration
- Inter-agent communication protocols
- Task allocation and load balancing
- Conflict resolution mechanisms

### Safety Systems (PR #1.5)
- **Circuit Breakers**: Configurable states (closed/open/half-open) with thresholds
  - Error rate percentage, rolling window duration, minimum volume
  - Half-open probing strategy with max_probes and probe_interval
- **Rate Limiting**: Token bucket algorithm with burst and refill parameters
  - Per-agent and global quotas
  - Configurable burst capacity and refill rates
- **Cost Tracking**: Pluggable price catalogs and budget enforcement
  - Mock local pricing for development
  - Cloud reference pricing for production
  - Per-agent and per-session aggregation
- **Behavior Monitoring**: Anomaly detection with rule-based detectors
  - Rapid repeat requests detection
  - Escalating latency monitoring
  - Statistical anomaly detection (z-score)
- **Safety Audit Log**: Structured JSON logging of safety events
  - Circuit trips, rate limit blocks, budget violations, anomalies
  - Log rotation by size or time
- **Safety Metrics**: Prometheus-compatible metrics
  - circuit_breaker_trips_total, rate_limit_blocks_total
  - anomaly_events_total, cost_budget_warnings_total

## Performance Optimization

### Local Model Optimization (PR #2.5)
- **Model Quantization**: Pre/post metrics tracking
  - Accuracy delta, latency improvement, memory reduction percentages
  - CI regression gates (>10% latency or >3% accuracy degradation fails)
- **Quality Bridge**: Local vs cloud model comparison
  - Semantic similarity scoring (>=0.85 threshold for summarization)
  - Escalation policy with safety event emission
  - Cost pre-check before any cloud fallback
- **Capability Profiles**: Per-model performance characteristics
  - Latency averages, context windows, token limits
  - Estimated costs, strengths, and weaknesses arrays
- **Intelligent Caching**: LRU + TTL eviction strategies
  - cache_hits_total, cache_misses_total, cache_hit_ratio metrics
  - Configurable cache sizes and expiration policies
- **Benchmarking Framework**: JSON task schema
  - Task types aligned to agent roles (search, extraction, synthesis, reasoning)
  - Automated regression detection and reporting

### System Optimization
- Parallel processing workflows
- Memory usage optimization
- Resource pool management
- Predictive scaling

## Research Focus Areas

1. **Multi-Modal Memory**: How to unify knowledge across different memory types
2. **Agent Specialization**: How agents develop expertise through experience
3. **Emergent Coordination**: Self-organizing workflows for complex tasks
4. **Foundation Model Orchestra**: Coordinating multiple AI models effectively

## Architecture Decision Records (ADRs)

The project uses ADRs to document significant architectural decisions:

- **ADR-0001**: Logging and configuration decisions (PR #1)
- **ADR-0002**: Project structure (PR #1)
- **ADR-0003**: Circuit breaker and safety architecture (PR #1.5)
- **ADR-0004**: Local model optimization and quality bridge (PR #2.5)

ADRs are stored in `docs/adr/` and follow the standard ADR template.

## Quality Requirements

### Testing Coverage
- Initial target: >=70% overall coverage (PR #1)
- Safety module: >=85% coverage (PR #1.5)
- Core components: >90% coverage target

### CI/CD Pipeline
- Stages: lint → test → security scan → build
- Security scanning with pip-audit or safety
- Coverage report artifacts
- SBOM generation (CycloneDX format)

### Logging and Monitoring
- Structured JSON logging with correlation IDs
- Request ID injection for tracing
- Centralized error handling patterns
- Prometheus metrics exposition

## Important Notes

- Always use development mode (local models) for initial testing to avoid costs
- Development mode enforces hard guard against cloud model invocation
- Implement safety controls before any agent coordination work
- Test circuit breakers and rate limiting before production deployment
- Monitor cost tracking actively when using cloud models
- Use memory systems efficiently to avoid performance degradation at scale
- All PRs should be <=500 LOC for effective review
- Follow ADR framework for architectural decisions