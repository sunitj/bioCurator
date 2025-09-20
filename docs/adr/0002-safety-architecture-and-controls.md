# ADR 0002: Safety Architecture and Controls

## Status

Accepted

## Context

The PRD mandates a robust safety layer: circuit breakers, rate limiting, cost governance, anomaly detection, and multi-mode guard rails. Early clarity on semantics prevents ad hoc policies and strengthens later research credibility (emergent coordination vs. safety interventions). A formalized event model and state machine definition enable observability, replay, and audit functions.

## Decision

Define a unified Safety Controller architecture implementing: circuit breaker state machine, hierarchical rate limiting, cost tracker with model pricing abstraction, anomaly monitor with baseline phase, and a safety event bus. All safety-relevant actions create structured events consumed by logging + metrics + (future) UI dashboards. Production vs. development mode differences enforced at runtime (e.g., cloud model call prohibition in development mode).

## Components

- Circuit Breaker: Per external dependency (model endpoint) and optionally per agent-task tuple. States: CLOSED -> OPEN (on threshold breach) -> HALF_OPEN (probing) -> CLOSED (success) or OPEN (failure).
- Rate Limiter: Token bucket with namespace levels (global, agent, model). Burst size + refill rate defined per mode.
- Cost Tracker: Aggregates cost per (agent, session, model) with pricing provider abstraction; triggers warnings and hard cutoffs.
- Behavior Monitor: Rule-based detectors initially (rapid identical requests, time-skew, oscillating breaker states) with future statistical expansion.
- Event Bus: In-memory publish-subscribe interface emitting SafetyEvent objects.
- Audit Log: Append-only JSON lines file (rotate by size) capturing canonical representation of SafetyEvent.

## Event Schema (SafetyEvent)

Fields: id (uuid), timestamp, event_type, agent_id(optional), model_id(optional), breaker_id(optional), cost_snapshot(optional), reason, metadata(dict).

Event Types: CIRCUIT_TRIPPED, CIRCUIT_RESET, RATE_LIMIT_BLOCK, COST_WARNING, COST_BUDGET_EXCEEDED, ANOMALY_DETECTED, MODE_VIOLATION, ESCALATION_DENIED.

## Threshold Strategy

- Circuit Breaker Trigger: error_rate >= configured threshold (default 50%) with minimum N calls in rolling window (default 20) OR consecutive_failures >= max_consecutive_failures (default 5).
- Half-Open Probe: limited probe_count (default 3) spaced by probe_interval (default 5s).
- Rate Limits: expressed as tokens_per_minute; defaults differ by mode (development lower, production higher, hybrid adaptive planned later).
- Cost Budget: Hard limit per session (dev: 0 cloud dollars) and soft warning at 80% of user-configured budget.

## Alternatives Considered

- Use external service (e.g., Envoy / service mesh) for circuit management (rejected: complexity not justified initially, fine-grained agent semantics needed).
- Statistical anomaly detection first (rejected: needs baseline, risk of false positives early).
- Global-only rate limiting (rejected: lacks fairness between agents; specialization requires isolation).

## Consequences

Positives:

- Deterministic and inspectable safety behavior supports research claims.
- Modular event-driven design simplifies future UI and analytics integration.
- Clear demarcation between development and production reduces accidental spend.

Negatives:

- Additional implementation and test surface early.
- Must maintain event schema backward compatibility.

## Validation

- Unit tests for state transitions, threshold triggers, anomaly detection cases, cost enforcement.
- Synthetic load test to confirm breaker and rate limiter interplay.
- Audit log inspection test verifying JSON schema conformity.

## Follow-ups

- Introduce statistical anomaly module (EWMA latency, z-score pattern detection).
- Persist safety events to PostgreSQL for historical analytics.
- Add real-time WebSocket feed for safety dashboard.
