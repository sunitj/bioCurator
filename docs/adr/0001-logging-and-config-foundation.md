# ADR 0001: Logging and Configuration Foundation

## Status

Accepted

## Context

Early establishment of structured logging and centralized configuration is critical for multi-agent observability, reproducibility, and safety feature evolution. The PRD emphasizes traceability (agent behavior analysis, safety auditing) and multi-mode operations (development, production, hybrid) requiring environment-aware configurable behavior. Without a unified approach, later layers (safety events, model benchmarking, memory interactions) would fragment instrumentation and increase debugging cost.

## Decision

Implement a centralized configuration loader supporting layered precedence (environment variables > mode-specific YAML > defaults). Implement structured JSON logging with correlation ID propagation and standardized log fields. Provide pluggable sinks (stdout initially; expansion later). Expose a metrics bootstrap to ensure early instrumentation adoption.

## Details

- Config Loader: `src/config/loader.py` leveraging pydantic (or dataclasses + validators) to enforce schema.
- Mode Selection: `APP_MODE` env var required; accepted values: `development`, `production`, `hybrid`.
- Logging: JSON formatter with keys: timestamp, level, message, module, function, correlation_id, agent_id(optional), event_type(optional), duration_ms(optional).
- Correlation IDs: Generated per inbound request / CLI invocation; contextvars used for propagation.
- Metrics Stub: Provide `observability/metrics.py` with helper to register counters/gauges; Prometheus exposition endpoint or WSGI middleware placeholder.
- Error Handling: Standardized logging for unhandled exceptions via hook.

## Alternatives Considered

- Defer structured logging until agents introduced (rejected: retrofitting is expensive).
- Use plain `.env` loading without schema validation (rejected: silent config drift risk).
- Adopt full OpenTelemetry stack now (postponed: premature complexity, add later after baseline stability).

## Consequences

Positives:

- Reduces future cost of integrating safety, benchmarking, memory instrumentation.
- Enables early correlation of actions across subsystems.
- Enforces discipline in config evolution.

Negatives:

- Slight upfront implementation overhead.
- Requires developers to follow structured logging pattern from day one.

## Validation

- CI ensures config schema validation passes.
- Example log snapshot added to README.
- Health command demonstrates correlation ID generation.

## Follow-ups

- Potential ADR for tracing expansion (OpenTelemetry) once multi-agent workflows stabilize.
- Add log redaction strategy for sensitive fields.
