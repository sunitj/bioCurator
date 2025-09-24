# ADR-0005: Multi-Modal Memory System Architecture

## Status
Accepted

## Context

BioCurator requires a sophisticated memory system to support multi-agent scientific literature analysis. The system needs to handle diverse data types and access patterns:

1. **Knowledge Graph Data**: Papers, authors, concepts, and their relationships
2. **Vector Embeddings**: Semantic search capabilities for scientific content
3. **Episodic Memory**: Agent interaction histories and learning patterns
4. **Working Memory**: Real-time agent coordination and caching
5. **Time-Series Data**: Agent performance metrics and system monitoring

The memory system must integrate with the existing safety architecture (circuit breakers, rate limiting) and support the multi-mode operation strategy (development/production/hybrid).

## Decision

We will implement a **multi-modal memory architecture** with five specialized backends:

### 1. Neo4j - Knowledge Graph Backend
- **Purpose**: Store papers, authors, concepts, methods, and their complex relationships
- **Use Cases**:
  - Citation networks and research connections
  - Concept evolution tracking over time
  - Contradiction detection between papers
  - Domain expertise mapping
- **Configuration**: Community edition with APOC and GDS plugins

### 2. Qdrant - Vector Backend
- **Purpose**: Store and search dense embeddings of scientific content
- **Use Cases**:
  - Semantic similarity search across papers
  - Content clustering and topic discovery
  - Cross-paper connection identification
  - Retrieval-augmented generation for agents
- **Configuration**: HTTP API with gRPC preference for performance

### 3. PostgreSQL - Episodic Backend
- **Purpose**: Store agent interaction histories and outcomes
- **Use Cases**:
  - Agent learning from past experiences
  - Performance analysis and improvement
  - Workflow pattern recognition
  - Decision audit trails
- **Configuration**: AsyncPG with SQLAlchemy ORM for schema management

### 4. Redis - Working Memory Backend
- **Purpose**: Fast access cache and real-time coordination
- **Use Cases**:
  - Agent state management
  - Task queuing and coordination
  - Session data and context
  - Performance optimization caching
- **Configuration**: Connection pooling with health checks

### 5. InfluxDB - Time-Series Backend
- **Purpose**: Agent performance metrics and system monitoring
- **Use Cases**:
  - Response time tracking
  - Resource usage monitoring
  - Performance trend analysis
  - System health dashboards
- **Configuration**: 2.x with Flux query language

### Architecture Patterns

#### Abstract Interface Design
- `MemoryBackend` base class with health check and connection management
- Specialized interfaces for each backend type (KnowledgeGraph, Vector, Episodic, etc.)
- Unified `MemoryManager` for coordinated backend orchestration

#### Safety Integration
- Circuit breaker integration for each backend
- Health monitoring with automatic failure detection
- Safety event emission for backend failures
- Connection pool management with timeout controls

#### Connection Management
- Async/await patterns throughout for non-blocking operations
- Connection pooling with configurable limits
- Automatic reconnection with exponential backoff
- Graceful shutdown with resource cleanup

## Consequences

### Positive
- **Specialized Storage**: Each data type stored in optimal format
- **Performance**: Specialized backends optimized for their use cases
- **Scalability**: Independent scaling of different memory components
- **Safety**: Integrated circuit breakers prevent cascading failures
- **Flexibility**: Abstract interfaces allow backend swapping
- **Development**: Local development with Docker Compose

### Negative
- **Complexity**: Multiple databases to manage and monitor
- **Operational**: More moving parts in production deployment
- **Data Consistency**: Potential inconsistencies across backends
- **Resource**: Higher memory and storage requirements
- **Learning Curve**: Team needs familiarity with multiple technologies

### Mitigation Strategies
- **Complexity**: Unified memory manager abstracts backend details
- **Operational**: Docker Compose simplifies local development
- **Consistency**: Event-driven updates and eventual consistency patterns
- **Resources**: Development mode uses smaller resource allocations
- **Learning**: Comprehensive documentation and examples

## Implementation Notes

### Development vs Production Configuration
```yaml
# Development - Smaller resource allocation
neo4j_max_connection_pool_size: 10
postgres_pool_size: 5
redis_pool_size: 5

# Production - Higher throughput
neo4j_max_connection_pool_size: 50
postgres_pool_size: 20
redis_pool_size: 10
```

### Safety Integration
- Each backend wrapped with circuit breaker pattern
- Health checks run concurrently with timeout controls
- Safety events emitted for backend failures
- Connection pool monitoring via Prometheus metrics

### Schema Evolution
- PostgreSQL: Alembic migrations for schema changes
- Neo4j: Index creation scripts for performance optimization
- Qdrant: Collection versioning for embedding model updates
- InfluxDB: Bucket retention policies for data lifecycle

### Testing Strategy
- Unit tests for each backend client
- Integration tests with real database containers
- Health check validation under failure conditions
- Performance benchmarking for concurrent access

## Alternatives Considered

### Single Database Approach
**Rejected**: PostgreSQL with JSON columns for all data types
- **Pros**: Simpler operations, ACID guarantees
- **Cons**: Poor performance for graph traversals and vector search

### Document Store Approach
**Rejected**: MongoDB with specialized collections
- **Pros**: Flexible schema, good for rapid development
- **Cons**: Limited graph capabilities, no native vector search

### Cloud-Native Services
**Rejected**: AWS Neptune, Pinecone, etc.
- **Pros**: Managed services, automatic scaling
- **Cons**: Vendor lock-in, higher costs, development mode complexity

## Implementation Learnings

### Actual vs Estimated Complexity
- **LOC Multiplier**: 7x (2,800 actual vs 400 estimated)
- **Primary Drivers**:
  - Async/await patterns throughout
  - Comprehensive error handling
  - Connection pool management
  - Health check implementations
  - Safety integration requirements
  - Docker Compose integration complexity

### Critical Technical Insights

#### 1. Neo4j 5.15 Configuration Challenges
- **Property Naming**: Neo4j 5.15 requires specific format for memory settings:
  - Correct: `NEO4J_server_memory_heap_initial__size=512m`
  - Incorrect: `NEO4J_server_memory_heap_initial_size=512m`
- **Plugin Configuration**: APOC and GDS plugins require specific format in Docker
- **Health Check**: Cypher-shell command needs proper authentication parameters

#### 2. Docker Service Dependencies
- **Health-Based Dependencies**: Critical for proper startup order
  ```yaml
  depends_on:
    redis:
      condition: service_healthy
  ```
- **Startup Time**: Services can take 30-60 seconds to become healthy
- **Service Discovery**: Container names (not localhost) required for inter-service communication

#### 3. Environment Variable Management
- **Docker Networking**: Services must use container names for connection URLs
  - Use: `redis://redis:6379` not `redis://localhost:6379`
- **Configuration Mapping**: src/config/loader.py requires comprehensive mapping
- **Graceful Degradation**: Optional backends like InfluxDB need try/catch patterns

#### 4. FastAPI Evolution
- **Lifespan Pattern**: Replaced deprecated `@app.on_event` with contextmanager:
  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI):
      await app.state.memory_manager.initialize()
      yield
      await app.state.memory_manager.shutdown()
  ```

#### 5. Safety Integration Complexity
- **Import Management**: EventBus → SafetyEventBus naming convention critical
- **Health Check Aggregation**: Optional backend failures affect overall status
- **Circuit Breaker States**: Per-backend monitoring prevents cascading failures

### Technical Patterns That Worked Well
- **Abstract Interface First**: Defining interfaces before implementations clarified the API
- **Factory Pattern**: Dynamic backend instantiation based on configuration
- **Event Bus Integration**: Clean separation between memory failures and safety responses
- **Centralized Logging**: Using `get_logger(__name__)` provides consistent context
- **Concurrent Health Checks**: `asyncio.gather()` pattern improves startup performance
- **Graceful Degradation**: Optional backends don't block system startup

### Docker Deployment Learnings
- **Service Health Checks**: Essential for reliable container orchestration
- **Volume Management**: Named volumes prevent data loss during container restarts
- **Port Mapping**: Required for external access to services (browser, tools)
- **Network Configuration**: Bridge network sufficient for service communication
- **Memory Allocation**: Neo4j requires minimum 512MB heap for stable operation

### Production Readiness Insights
- **Connection Pooling**: Critical for multi-agent concurrent access
- **Error Boundaries**: Each backend needs independent failure handling
- **Monitoring Integration**: Health endpoints must reflect actual system status
- **Resource Management**: Memory and connection limits prevent resource exhaustion

### Testing Strategy Evolution
- **Integration Testing**: Real database containers required for accurate testing
- **Health Check Validation**: Must test under various failure conditions
- **Performance Benchmarking**: Concurrent access patterns reveal scaling issues
- **Error Simulation**: Circuit breaker behavior needs systematic testing

### Future Improvements
- Add connection retry logic with exponential backoff
- Implement connection pool monitoring metrics
- Add distributed tracing support for multi-backend operations
- Create performance benchmarking suite
- Implement data migration tools for schema evolution
- Add backup and restore procedures for production deployment

## Related Decisions
- ADR-0001: Logging and configuration architecture
- ADR-0003: Circuit breaker and safety architecture
- Future: Agent coordination protocols
- Future: Data consistency and synchronization patterns

## References
- [Neo4j Python Driver Documentation](https://neo4j.com/docs/python-manual/current/)
- [Qdrant Client Library](https://qdrant.tech/documentation/clients/python/)
- [SQLAlchemy Async Documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Redis.py Async Documentation](https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html)
- [InfluxDB Python Client](https://influxdb-client.readthedocs.io/en/stable/)