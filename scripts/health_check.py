#!/usr/bin/env python3
import json
import sys
from typing import Dict, Any

from config import load_config, ConfigError  # type: ignore
from observability import get_registry, counter  # type: ignore

# Register basic metric increments to show instrumentation test
startup_counter = counter("health_check_startups", "Number of health check invocations")
startup_counter.inc()


def main() -> int:
    status: Dict[str, Any] = {"ok": True, "components": {}, "errors": []}
    try:
        cfg = load_config()
        status["components"]["config"] = {
            "mode": cfg.mode,
            "logging_level": cfg.logging_level,
        }
    except ConfigError as e:
        status["ok"] = False
        status["errors"].append(str(e))

    # Metrics snapshot
    status["components"]["metrics"] = get_registry().export()

    print(json.dumps(status, indent=2))
    return 0 if status["ok"] else 1


if __name__ == "main__":  # pragma: no cover (typo guard)
    # Guard: fallback to proper invocation
    sys.exit(main())

if __name__ == "__main__":
    sys.exit(main())
