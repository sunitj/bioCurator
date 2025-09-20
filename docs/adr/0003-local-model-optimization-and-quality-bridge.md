# ADR 0003: Local Model Optimization and Quality Bridge

## Status

Accepted

## Context

The project requires development-mode reliance on local models while ensuring production-grade analytical reliability. A systematic quality bridge is needed to compare local (Ollama-hosted) models against cloud baseline models (Claude, GPT-4, etc.) across tasks relevant to emerging agents: search relevance, summarization, extraction, synthesis/reasoning. Performance optimization (quantization, caching) must not degrade semantic fidelity beyond acceptable thresholds. Without codified benchmarks, escalation policies become subjective and cost controls weaken.

## Decision

Establish a benchmark-driven evaluation framework producing quantitative quality and performance metrics, stored as version-controlled artifacts. Define capability profiles per model supporting dynamic selection and fallback logic. Implement an escalation policy triggered by benchmark deltas or live performance deficits, always routed through safety + cost governance. Optimize local model efficiency (quantization, caching) with mandatory pre/post quality diff reporting. Introduce regression gating in CI to prevent silent degradation.

## Benchmark Taxonomy

Task Categories: SEARCH, SUMMARIZATION, EXTRACTION, SYNTHESIS. (Future: TEMPORAL_ANALYSIS, COMPARATIVE_ANALYSIS.) Each benchmark case references evaluation method (semantic similarity, ROUGE/BLEU, structured extraction F1, reasoning rubric). Inputs stored in lightweight JSON schema.

## Capability Profiles

For each model record: name, version/hash, modality, latency_ms_avg, context_window, cost_per_1k_tokens (estimated for local = 0 / energy placeholder), max_output_tokens, strengths[], weaknesses[], last_benchmark_id, quality_scores{task_type: score}.

## Escalation Policy

Local model is primary in development and hybrid modes. Escalate to cloud if: (a) live task confidence score < threshold and benchmark delta persists across N=3 consecutive tasks; or (b) task category explicitly marked as high_risk requiring cloud validation; or (c) anomaly detector flags repeated failure pattern. All escalations emit SAFETY event (ESCALATION_REQUEST -> ESCALATION_APPROVED/DENIED) including projected marginal cost.

## Optimization Techniques

- Quantization (int8 / mixed precision) with recorded latency & memory impacts.
- Response caching keyed by normalized prompt + task metadata; LRU + TTL eviction.
- Optional prompt compression / token optimization strategies (future).

## Quality Bridge Metrics

- Semantic similarity (embedding cosine) for summarization/synthesis.
- Exact + partial match F1 for extraction tasks.
- Relevance@K / MRR for search tasks.
- Rubric-based scored reasoning (LLM-as-judge, double-evaluated with variance cap) for synthesis.

## Regression Gate

CI fails if: performance_latency_regression_pct > 10 OR quality_drop_pct > 3 (any critical task) unless override label applied (benchmark-override) with justification.

## Alternatives Considered

- Manual spot checks (rejected: unscalable & subjective).
- Rely solely on cloud models (rejected: cost & dev iteration friction).
- Single blended score obscuring task nuances (rejected: loses optimization signal).

## Consequences

Positives:

- Objective measurement supports credible performance claims.
- Enables adaptive model selection aligned with cost governance.
- Early detection of quality regressions from optimization.

Negatives:

- Benchmark maintenance overhead.
- Requires disciplined artifact versioning.

## Validation

- Benchmark run produces JSON + Markdown summary committed or attached as artifact.
- Quantization diff includes accuracy_delta, latency_improvement_pct, memory_reduction_pct.
- Simulated escalation test triggers safety event path.

## Follow-ups

- Add retrieval-aware benchmarks post memory integration.
- Introduce active learning loop collecting real task outcomes for continual model calibration.
- Integrate visualization dashboard for benchmark trend analysis.
