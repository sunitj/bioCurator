"""Behavior monitoring and anomaly detection for safety controls."""

import time
import statistics
from abc import ABC, abstractmethod
from collections import deque
from threading import Lock
from typing import Dict, Any, List, Optional, Deque

from pydantic import BaseModel

from ..logging import get_logger
from .event_bus import SafetyEventType, emit_safety_event

logger = get_logger(__name__)


class BehaviorMetric(BaseModel):
    """Individual behavior metric."""

    timestamp: float
    agent_id: Optional[str]
    metric_type: str
    value: float
    metadata: Dict[str, Any] = {}


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""

    @abstractmethod
    def check_anomaly(self, metrics: List[BehaviorMetric]) -> Optional[Dict[str, Any]]:
        """Check for anomalies in metrics. Return anomaly details if found."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get detector name."""
        pass


class RapidRepeatRequestsDetector(AnomalyDetector):
    """Detect rapid repeat requests pattern."""

    def __init__(self, max_requests: int = 10, time_window: float = 60.0):
        """Initialize detector."""
        self.max_requests = max_requests
        self.time_window = time_window

    def check_anomaly(self, metrics: List[BehaviorMetric]) -> Optional[Dict[str, Any]]:
        """Check for rapid repeat requests."""
        if len(metrics) < self.max_requests:
            return None

        # Get recent request metrics
        now = time.time()
        recent_requests = [
            m for m in metrics
            if m.metric_type == "request" and now - m.timestamp <= self.time_window
        ]

        if len(recent_requests) >= self.max_requests:
            return {
                "detector": self.get_name(),
                "anomaly_type": "rapid_repeat_requests",
                "request_count": len(recent_requests),
                "time_window": self.time_window,
                "threshold": self.max_requests,
                "severity": "WARNING",
            }

        return None

    def get_name(self) -> str:
        """Get detector name."""
        return "rapid_repeat_requests"


class EscalatingLatencyDetector(AnomalyDetector):
    """Detect escalating latency pattern."""

    def __init__(self, threshold_multiplier: float = 3.0, min_samples: int = 5):
        """Initialize detector."""
        self.threshold_multiplier = threshold_multiplier
        self.min_samples = min_samples

    def check_anomaly(self, metrics: List[BehaviorMetric]) -> Optional[Dict[str, Any]]:
        """Check for escalating latency."""
        latency_metrics = [m for m in metrics if m.metric_type == "latency"]

        if len(latency_metrics) < self.min_samples:
            return None

        # Get recent latencies
        recent_latencies = sorted(latency_metrics, key=lambda x: x.timestamp)[-self.min_samples:]
        latency_values = [m.value for m in recent_latencies]

        if len(latency_values) < self.min_samples:
            return None

        # Check if latest latency is significantly higher than average
        avg_latency = statistics.mean(latency_values[:-1])  # Exclude latest
        latest_latency = latency_values[-1]

        if latest_latency > avg_latency * self.threshold_multiplier:
            return {
                "detector": self.get_name(),
                "anomaly_type": "escalating_latency",
                "latest_latency": latest_latency,
                "average_latency": avg_latency,
                "threshold_multiplier": self.threshold_multiplier,
                "severity": "WARNING",
            }

        return None

    def get_name(self) -> str:
        """Get detector name."""
        return "escalating_latency"


class StatisticalAnomalyDetector(AnomalyDetector):
    """Detect statistical anomalies using z-score."""

    def __init__(self, z_threshold: float = 3.0, min_samples: int = 10):
        """Initialize detector."""
        self.z_threshold = z_threshold
        self.min_samples = min_samples

    def check_anomaly(self, metrics: List[BehaviorMetric]) -> Optional[Dict[str, Any]]:
        """Check for statistical anomalies."""
        latency_metrics = [m for m in metrics if m.metric_type == "latency"]

        if len(latency_metrics) < self.min_samples:
            return None

        latency_values = [m.value for m in latency_metrics]

        try:
            mean_latency = statistics.mean(latency_values)
            stdev_latency = statistics.stdev(latency_values)

            if stdev_latency == 0:
                return None

            # Check latest value
            latest_latency = latency_values[-1]
            z_score = (latest_latency - mean_latency) / stdev_latency

            if abs(z_score) > self.z_threshold:
                return {
                    "detector": self.get_name(),
                    "anomaly_type": "statistical_anomaly",
                    "z_score": z_score,
                    "threshold": self.z_threshold,
                    "latest_value": latest_latency,
                    "mean_value": mean_latency,
                    "std_deviation": stdev_latency,
                    "severity": "INFO",
                }

        except statistics.StatisticsError:
            # Not enough variance in data
            pass

        return None

    def get_name(self) -> str:
        """Get detector name."""
        return "statistical_anomaly"


class BehaviorMonitor:
    """Behavior monitoring with anomaly detection."""

    def __init__(self, baseline_mode: bool = True, max_metrics: int = 1000):
        """Initialize behavior monitor."""
        self.baseline_mode = baseline_mode
        self.max_metrics = max_metrics

        # Metrics storage
        self._metrics: Deque[BehaviorMetric] = deque(maxlen=max_metrics)
        self._agent_metrics: Dict[str, Deque[BehaviorMetric]] = {}

        # Anomaly detectors
        self._detectors: List[AnomalyDetector] = [
            RapidRepeatRequestsDetector(),
            EscalatingLatencyDetector(),
            StatisticalAnomalyDetector(),
        ]

        # Stats
        self._anomaly_count = 0

        # Thread safety
        self._lock = Lock()

    def record_metric(
        self,
        metric_type: str,
        value: float,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a behavior metric."""
        metric = BehaviorMetric(
            timestamp=time.time(),
            agent_id=agent_id,
            metric_type=metric_type,
            value=value,
            metadata=metadata or {}
        )

        with self._lock:
            # Add to global metrics
            self._metrics.append(metric)

            # Add to agent-specific metrics
            if agent_id:
                if agent_id not in self._agent_metrics:
                    self._agent_metrics[agent_id] = deque(maxlen=self.max_metrics)
                self._agent_metrics[agent_id].append(metric)

            # Check for anomalies (unless in baseline mode)
            if not self.baseline_mode:
                self._check_anomalies([metric])

    def _check_anomalies(self, new_metrics: List[BehaviorMetric]) -> None:
        """Check for anomalies in recent metrics."""
        # Check global anomalies
        recent_metrics = list(self._metrics)[-100:]  # Last 100 metrics
        self._run_anomaly_detection(recent_metrics, None)

        # Check per-agent anomalies
        for agent_id, metrics in self._agent_metrics.items():
            agent_recent = list(metrics)[-50:]  # Last 50 per agent
            self._run_anomaly_detection(agent_recent, agent_id)

    def _run_anomaly_detection(
        self, metrics: List[BehaviorMetric], agent_id: Optional[str]
    ) -> None:
        """Run anomaly detection on metrics."""
        for detector in self._detectors:
            try:
                anomaly = detector.check_anomaly(metrics)
                if anomaly:
                    self._handle_anomaly(anomaly, agent_id)
            except Exception as e:
                logger.error(
                    "Error in anomaly detector",
                    detector=detector.get_name(),
                    error=str(e)
                )

    def _handle_anomaly(self, anomaly: Dict[str, Any], agent_id: Optional[str]) -> None:
        """Handle detected anomaly."""
        with self._lock:
            self._anomaly_count += 1

        emit_safety_event(
            event_type=SafetyEventType.ANOMALY_DETECTED,
            component="behavior_monitor",
            message=f"Anomaly detected: {anomaly['anomaly_type']}",
            agent_id=agent_id,
            metadata=anomaly,
            severity=anomaly.get("severity", "INFO")
        )

        logger.warning(
            "Behavior anomaly detected",
            anomaly_type=anomaly["anomaly_type"],
            detector=anomaly["detector"],
            agent_id=agent_id,
            details=anomaly
        )

    def add_detector(self, detector: AnomalyDetector) -> None:
        """Add custom anomaly detector."""
        with self._lock:
            self._detectors.append(detector)

    def remove_detector(self, detector_name: str) -> bool:
        """Remove anomaly detector by name."""
        with self._lock:
            for i, detector in enumerate(self._detectors):
                if detector.get_name() == detector_name:
                    del self._detectors[i]
                    return True
            return False

    def set_baseline_mode(self, enabled: bool) -> None:
        """Enable or disable baseline mode."""
        with self._lock:
            self.baseline_mode = enabled
            logger.info(
                "Behavior monitor baseline mode changed",
                baseline_mode=enabled
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get behavior monitor statistics."""
        with self._lock:
            detector_names = [d.get_name() for d in self._detectors]

            agent_metric_counts = {
                agent_id: len(metrics)
                for agent_id, metrics in self._agent_metrics.items()
            }

            return {
                "baseline_mode": self.baseline_mode,
                "total_metrics": len(self._metrics),
                "agent_metrics": agent_metric_counts,
                "anomaly_count": self._anomaly_count,
                "active_detectors": detector_names,
                "max_metrics": self.max_metrics,
            }

    def get_recent_metrics(
        self, count: int = 100, agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent metrics."""
        with self._lock:
            if agent_id and agent_id in self._agent_metrics:
                metrics = list(self._agent_metrics[agent_id])[-count:]
            else:
                metrics = list(self._metrics)[-count:]

            return [
                {
                    "timestamp": m.timestamp,
                    "agent_id": m.agent_id,
                    "metric_type": m.metric_type,
                    "value": m.value,
                    "metadata": m.metadata,
                }
                for m in metrics
            ]