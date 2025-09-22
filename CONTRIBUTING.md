# Contributing to BioCurator

Welcome! This guide outlines the workflow, quality bars, and architectural governance required for contributions.

## Core Principles

- Small, reviewable PRs (target <= 500 LOC changed)
- Each PR independently testable & revert-safe
- Architectural consistency enforced via ADRs
- Observability and safety are first-class, not afterthoughts

## Branch Strategy

- `main`: Production-ready, tagged releases only
- `develop`: Integrated, always green, next release candidate
- `feature/<slug>`: Single feature / change set
- `docs/<slug>`: Documentation or content updates only

## Acceptance Criteria & DoD

Early PRs (see roadmap) have refined acceptance criteria blocks inside `/.ideas/biocurator_pr_roadmap.md`:

- PR #1: Logging, config, health, metrics, ADR bootstrap, security scan
- PR #1.5: Circuit breaker semantics, safety event bus, audit log, cost & anomaly controls
- PR #2.5: Benchmarks, capability profiles, escalation policy, regression gating

A PR is DONE when:

1. All listed MUST criteria satisfied (SHOULD where feasible)
2. Tests pass (no skipped critical tests) & coverage does not regress baseline
3. Linting passes (code + markdown where applicable)
4. ADRs updated/added if architectural decision changed or introduced
5. Documentation updated for new public APIs / CLI scripts

## ADR Process

- ADRs stored in `docs/adr/` with sequential numbering
- Status values: Proposed -> Accepted -> (optionally) Superseded
- New ADR required if:
  - Introducing a new subsystem boundary
  - Changing public integration contracts
  - Modifying safety or optimization policies
- Update `docs/adr/README.md` index in same PR

## Configuration & Modes

- `APP_MODE` must be explicitly set (development|production|hybrid)
- Config values validated via loader (`src/config/loader.py`)
- Environment overrides: prefix `BIOCURATOR_` to override YAML

## Observability

- Use structured logging (JSON). Include: `timestamp`, `level`, `message`, `correlation_id`, `component`, optional `agent_id`
- Metrics: Register through `observability.metrics` registry only
- Add new counters/gauges sparingly; document purpose if non-obvious

## Safety Integration

If code triggers external model calls or orchestrates agents:

- Emit safety or cost-related events (placeholder until full event bus merged)
- Respect development mode prohibition of paid/cloud models

## Benchmarks & Model Changes

- For model-impacting PRs (performance, quality, caching): attach or generate benchmark artifact
- Update capability profile if characteristics change
- Add label `model-profile-change` if modifying `configs/model_profiles.yaml`

## Testing Standards

- Unit tests for new logic paths
- Avoid broad integration tests unless needed for regression coverage
- Deterministic tests (seed randomness; avoid external network unless mocked)
- Provide factory/helpers for repeated test constructions

## Development Setup

### Environment Setup with UV

```bash
# Automated setup
./scripts/setup_venv.sh
source .venv/bin/activate

# Manual setup
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Tooling Commands

```bash
make setup         # set up development environment (legacy)
make lint          # run linters / style checks
make test          # run test suite with coverage
make format        # format code with black/ruff
make health        # invoke health_check script
make clean         # clean build artifacts
```

## Commit & PR Hygiene

- Conventional-ish commit subjects (feat:, fix:, chore:, docs:, refactor:, test:, perf:)
- PR description references roadmap section and ADR numbers touched (e.g. "Implements PR #1 criteria; relates ADR 0001")
- Include checklist reflecting refined acceptance criteria items

## Code Review Guidelines

Reviewer looks for:

- Alignment with ADRs & roadmap acceptance criteria
- Clear separation of concerns & minimal surface area
- Adequate tests & failure path coverage
- Observability hooks present (logs/metrics) where meaningful
- No silent exception swallowing

## Adding New Dependencies

- Update `pyproject.toml` with new dependencies
- Use appropriate section: `dependencies`, `dev`, `ml`, or `docs`
- Justify in PR description (why existing libs insufficient)
- Pin version ranges appropriately (>=X.Y.Z)
- Security scan passes
- Run `uv pip compile` if needed for lock files

## Performance & Benchmarks

- Provide before/after numbers for performance-impacting changes
- Include environment description (machine, model variant)

## Security & Secrets

- Never commit secrets (use `.env.example` to document variables)
- External service keys must be loaded from environment only

## Getting Help

- Open a draft PR early for directional feedback
- Reference ADR if unsure; propose new one if gap exists

Thank you for contributing to BioCurator. This discipline enables credible technical storytelling and sustainable evolution.
