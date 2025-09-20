# ADR-0002: Project Structure and Organization

**Status:** Accepted  
**Date:** 2024-01-15  
**Deciders:** Development Team

## Context

BioCurator is a complex multi-agent system that requires clear organization for:
- Multi-modal memory systems (Neo4j, Qdrant, PostgreSQL, Redis, SQLite)
- Agent coordination and orchestration
- Safety controls and monitoring
- Model management and optimization
- Testing and quality assurance
- Documentation and ADRs

The project structure must support both development and production deployments while maintaining clear separation of concerns.

## Decision

We will organize the project with the following structure:

```
bioCurator/
├── src/                           # Application source code
│   ├── agents/                    # Agent implementations
│   ├── config/                    # Configuration management
│   ├── coordination/              # Agent coordination
│   ├── health/                    # Health monitoring
│   ├── logging/                   # Structured logging
│   ├── memory/                    # Memory systems
│   ├── metrics/                   # Prometheus metrics
│   ├── models/                    # AI model interfaces
│   ├── safety/                    # Safety controls
│   └── web/                       # Web interface
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── performance/               # Performance tests
├── configs/                       # Configuration files
├── docs/                          # Documentation
│   └── adr/                       # Architecture Decision Records
├── scripts/                       # Utility scripts
├── benchmarks/                    # Performance benchmarks
└── docker-compose*.yml           # Container orchestration
```

### Key Principles
- **Domain-driven organization**: Modules organized by business domain
- **Layered architecture**: Clear separation between infrastructure and application layers
- **Test organization**: Tests mirror source structure with additional categorization
- **Configuration externalization**: All environment-specific config external to source
- **Documentation co-location**: ADRs and docs alongside code

## Rationale

### Source Code Organization
- **Domain modules**: Each major system component gets its own module
- **Flat hierarchy**: Avoid deep nesting that makes imports complex
- **Clear dependencies**: Module dependencies flow in one direction
- **Shared utilities**: Common code in well-defined shared modules

### Test Structure
- **Mirror source**: Easy to find tests for any source file
- **Test categories**: Unit, integration, and performance tests separated
- **Fixtures**: Shared test fixtures in conftest.py files
- **Markers**: Pytest markers for selective test execution

### Configuration Strategy
- **External configs**: Environment-specific settings outside source tree
- **Version control**: Base configurations versioned, secrets excluded
- **Override pattern**: Environment variables override file settings
- **Schema validation**: All configuration validated at startup

### Alternatives Considered
1. **Feature-based structure**: Rejected due to cross-cutting concerns
2. **Monolithic modules**: Rejected due to complexity and maintainability
3. **Deep hierarchies**: Rejected due to import complexity
4. **Embedded configuration**: Rejected due to deployment inflexibility

## Consequences

### Positive
- Clear separation of concerns and responsibilities
- Easy to locate code and tests for any feature
- Supports both local development and containerized deployment
- Scalable structure that can grow with the project
- Clear patterns for adding new components

### Negative
- Initial setup complexity with multiple directories
- Potential for circular imports if module boundaries aren't respected
- Requires discipline to maintain organization over time
- More files and directories to navigate initially

### Implementation Guidelines
- Use absolute imports to avoid relative import issues
- Keep `__init__.py` files minimal with clear exports
- Document module purposes and dependencies
- Use dependency injection to manage inter-module dependencies
- Regular refactoring to maintain clean boundaries

## References

- [Python Application Layout](https://docs.python-guide.org/writing/structure/)
- [Domain-Driven Design](https://martinfowler.com/tags/domain%20driven%20design.html)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [12-Factor App](https://12factor.net/)