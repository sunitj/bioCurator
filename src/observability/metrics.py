import threading
import time
from typing import Any, Callable, Dict, Optional


class Metric:
    def __init__(self, name: str, metric_type: str, description: str = ""):
        self.name = name
        self.type = metric_type
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        if self.type != "counter":
            raise TypeError("inc only valid for counter")
        with self._lock:
            self._value += amount

    def set(self, value: float) -> None:
        if self.type != "gauge":
            raise TypeError("set only valid for gauge")
        with self._lock:
            self._value = value

    def get(self) -> float:
        with self._lock:
            return self._value

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "value": self.get(),
            "description": self.description,
        }


class MetricsRegistry:
    def __init__(self) -> None:
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "") -> Metric:
        return self._register(name, "counter", description)

    def gauge(self, name: str, description: str = "") -> Metric:
        return self._register(name, "gauge", description)

    def _register(self, name: str, metric_type: str, description: str) -> Metric:
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            metric = Metric(name, metric_type, description)
            self._metrics[name] = metric
            return metric

    def export(self) -> Dict[str, Any]:
        return {name: m.snapshot() for name, m in self._metrics.items()}

    def to_prometheus(self) -> str:
        lines = []
        for m in self._metrics.values():
            metric_name = m.name.replace("-", "_")
            if m.description:
                lines.append(f"# HELP {metric_name} {m.description}")
            lines.append(f"# TYPE {metric_name} {m.type}")
            lines.append(f"{metric_name} {m.get()}")
        return "\n".join(lines)


_global_registry: Optional[MetricsRegistry] = None


def get_registry() -> MetricsRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = MetricsRegistry()
    return _global_registry


def counter(name: str, description: str = "") -> Metric:
    return get_registry().counter(name, description)


def gauge(name: str, description: str = "") -> Metric:
    return get_registry().gauge(name, description)


def timed(metric_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    c = counter(metric_name + "_calls", description="Total calls")
    g = gauge(metric_name + "_latency_ms", description="Last call latency ms")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000.0
                c.inc()
                g.set(duration_ms)

        return wrapper

    return decorator


if __name__ == "__main__":  # basic smoke test
    c = counter("example_counter", "Example counter")
    g = gauge("example_gauge", "Example gauge")
    c.inc()
    g.set(42)
    print(get_registry().export())
