# ADR-0006: Agent Coordination Architecture

## Status

Accepted

## Context

BioCurator requires a sophisticated multi-agent coordination system to orchestrate literature analysis workflows. The system must support multiple specialized agents working collaboratively while maintaining safety controls, performance monitoring, and graceful error handling.

Key requirements include:
- **Safety-first design**: Circuit breakers, rate limiting, cost tracking
- **Multi-mode operation**: Development (local models), production (cloud models), hybrid
- **Async coordination**: Non-blocking inter-agent communication
- **Persistent task management**: Reliable task queuing with failure recovery
- **Performance monitoring**: Real-time metrics and behavior analysis
- **Graceful degradation**: System continues operating when individual agents fail

## Decision

We will implement a layered agent coordination architecture with the following components:

### 1. Agent Infrastructure Layer

**Base Agent Class (`src/agents/base.py`)**
- Safety-integrated agent base with circuit breakers, rate limiting, cost tracking
- Memory system access through unified manager interface
- Async task execution with semaphore-based concurrency control
- Health monitoring and status reporting
- Graceful lifecycle management (start/stop)

**Agent Registry (`src/agents/registry.py`)**
- Centralized agent discovery and lifecycle management
- Configuration loading from YAML files with mode-specific overrides
- Health checking across all registered agents
- Dynamic agent creation using importlib with error handling

**Agent Configuration (`src/agents/config.py`)**
- Pydantic-based configuration schemas with validation
- Multi-mode support (development/production/hybrid)
- Per-agent safety controls and memory permissions
- Extensible configuration merging system

### 2. Communication & Coordination Layer

**Message Protocol (`src/coordination/protocols.py`)**
- Async message passing with correlation IDs for request/response patterns
- Priority-based message queuing with timeout handling
- Broadcast capabilities for system-wide notifications
- Error propagation and recovery mechanisms

**Task Queue (`src/coordination/task_queue.py`)**
- PostgreSQL-backed persistent task storage
- Dependency management with completion checking
- Retry logic with exponential backoff
- Performance metrics collection (execution time, queue wait)

**Safety Coordinator (`src/coordination/safety_coordinator.py`)**
- Per-agent safety control enforcement
- Load balancing with capability-based task allocation
- System-wide safety monitoring and alerting
- Agent performance tracking and anomaly detection

### 3. Monitoring & Safety Layer

**Agent Monitor (`src/monitoring/agent_monitor.py`)**
- Real-time performance metrics collection
- Behavior anomaly detection with configurable thresholds
- Historical data retention with automatic cleanup
- System-wide health reporting

**Safety Integration**
- Individual circuit breakers per agent ID
- Per-agent rate limiting with configurable quotas
- Cost tracking aggregation by agent and time window
- Safety event emission for monitoring and alerting

### 4. Configuration Architecture

**Multi-mode Configuration System**
```yaml
# Base configuration (cagents.yaml)
agents:
  research_director:
    class: "src.agents.coordinator.ResearchDirectorAgent"
    role: "Strategic coordination"
    safety:
      max_requests_per_minute: 30

# Development mode override (cagents.development.yaml)
models:
  research_director:
    primary: "deepseek-r1:32b"  # Local model
safety:
  enforce_zero_cost: true
  cloud_model_guard: true

# Production mode override (cagents.production.yaml)
models:
  research_director:
    primary: "claude-sonnet-4"  # Cloud model
safety:
  enforce_zero_cost: false
  cost_monitoring_enabled: true
```

### 5. Research Director Implementation

**Orchestration Capabilities**
- Workflow orchestration with task decomposition
- Quality control and result synthesis
- Agent capability discovery and task allocation
- Inter-agent message handling and coordination

## Implementation Details

### Agent Safety Integration Pattern
```python
class BaseAgent(ABC):
    def __init__(self, agent_id, config, system_config, memory_manager, event_bus):
        # Individual safety controls per agent
        self.circuit_breaker = CircuitBreaker(f"agent_{agent_id}", config.circuit_breaker)
        self.rate_limiter = RateLimiter(f"agent_{agent_id}", config.rate_limiter)
        self.cost_tracker = CostTracker(config.max_cost_budget_per_hour)

    async def execute_task(self, task):
        # Safety checks before execution
        await self._check_safety_controls(task)

        # Execute with circuit breaker protection
        async with self.circuit_breaker:
            return await self._execute_task_impl(task)
```

### Configuration Merging Strategy
1. Load base configuration (`cagents.yaml`)
2. Identify mode-specific configuration based on `APP_MODE`
3. Deep merge mode configuration over base configuration
4. Validate final configuration with Pydantic schemas
5. Apply development mode safety guards if applicable

### Task Queue Design
- **Task Status Flow**: pending → assigned → running → completed/failed/cancelled
- **Retry Logic**: Failed tasks scheduled for retry with exponential backoff
- **Dependency Management**: Tasks wait for prerequisite tasks to complete
- **Performance Tracking**: Execution time and queue wait time metrics

### Health Monitoring Integration
- Agent status included in `/health` endpoint responses
- Individual agent circuit breaker states exposed
- Task queue statistics and agent performance metrics
- System-wide health aggregation with unhealthy agent detection

## Consequences

### Positive
- **Safety-first design**: Multiple layers of protection against failures
- **Scalable architecture**: Async design supports high concurrency
- **Multi-mode flexibility**: Single codebase supports development and production
- **Comprehensive monitoring**: Full visibility into agent behavior and performance
- **Graceful degradation**: System continues operating with partial failures
- **Extensible design**: Easy to add new agents and capabilities

### Negative
- **Complexity**: Multi-layered architecture requires careful coordination
- **Resource usage**: Monitoring and safety controls add overhead
- **Configuration management**: Multi-mode configuration can be complex
- **Database dependency**: Task queue requires PostgreSQL for persistence

### Risk Mitigations
- **Extensive testing**: Unit, integration, and safety system tests
- **Comprehensive examples**: Working demonstrations of all features
- **Development mode**: Zero-cost local development with strict safety guards
- **Health monitoring**: Early detection of issues through monitoring
- **Graceful shutdown**: Proper cleanup of resources and running tasks

## Implementation Status

✅ **Phase 1 Complete (PR #3)**
- Base agent infrastructure with safety integration
- Agent registry with lifecycle management
- Research Director agent with basic orchestration
- Configuration system with multi-mode support
- Task queue with PostgreSQL persistence
- Safety coordinator with monitoring integration
- Health endpoints with agent status reporting
- Working examples and comprehensive tests

🚧 **Future Phases**
- **PR #6**: Literature Scout agent implementation
- **PR #7**: Deep Reader agent with content analysis
- **PR #10**: Domain Specialist agent for validation
- **PR #11**: Knowledge Weaver agent for synthesis
- **PR #12**: Memory Keeper agent for knowledge curation

## Validation

The architecture has been validated through:
1. **Working Examples**: `examples/basic_workflow.py` demonstrates end-to-end functionality
2. **Safety Demo**: `examples/safety_demo.py` validates all safety controls
3. **Comprehensive Tests**: >90% test coverage for agent system components
4. **Health Integration**: Agent status properly exposed through health endpoints
5. **Development Mode**: Zero-cost operation with local models validated

The implementation successfully meets all acceptance criteria for PR #3 and provides a solid foundation for future agent implementations.