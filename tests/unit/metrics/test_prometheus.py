"""Tests for Prometheus metrics."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.metrics.prometheus import (
    REQUEST_COUNT,
    REQUEST_DURATION,
    record_health_check_metrics,
    record_request_metrics,
)


class TestPrometheusMetrics:
    """Test Prometheus metrics functionality."""

    def test_metrics_endpoint_enabled(self, test_client: TestClient):
        """Test metrics endpoint when Prometheus is enabled."""
        response = test_client.get("/metrics/")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"

        # Check for expected metric types in response
        content = response.text
        assert "biocurator_requests_total" in content
        assert "biocurator_request_duration_seconds" in content

    @patch("src.config.settings.monitoring.prometheus_enabled", False)
    def test_metrics_endpoint_disabled(self, test_client: TestClient):
        """Test metrics endpoint when Prometheus is disabled."""
        response = test_client.get("/metrics/")

        assert response.status_code == 404
        assert response.text == "Metrics disabled"

    def test_metrics_health_endpoint(self, test_client: TestClient):
        """Test metrics health endpoint."""
        response = test_client.get("/metrics/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "prometheus_enabled" in data
        assert "registry_collectors" in data
        assert data["prometheus_enabled"] is True

    def test_record_request_metrics(self):
        """Test recording request metrics."""
        # Clear any existing metrics
        REQUEST_COUNT._value.clear()
        REQUEST_DURATION._sum.clear()

        # Record some metrics
        record_request_metrics("GET", "/health", 200, 0.150)
        record_request_metrics("POST", "/api", 201, 0.250)
        record_request_metrics("GET", "/health", 500, 0.050)

        # Check that metrics were recorded
        # Note: This is a basic test - in practice you'd check the actual values
        assert len(REQUEST_COUNT._value) > 0
        assert len(REQUEST_DURATION._sum) > 0

    def test_record_health_check_metrics(self):
        """Test recording health check metrics."""
        from src.metrics.prometheus import HEALTH_CHECK_COUNT, HEALTH_CHECK_DURATION

        # Clear any existing metrics
        HEALTH_CHECK_COUNT._value.clear()
        HEALTH_CHECK_DURATION._sum.clear()

        # Record some metrics
        record_health_check_metrics("redis", "healthy", 0.050)
        record_health_check_metrics("config", "healthy", 0.001)
        record_health_check_metrics("redis", "unhealthy", 5.000)

        # Check that metrics were recorded
        assert len(HEALTH_CHECK_COUNT._value) > 0
        assert len(HEALTH_CHECK_DURATION._sum) > 0

    def test_metrics_registry_isolation(self):
        """Test that custom registry is isolated from default."""
        from prometheus_client import REGISTRY

        from src.metrics.prometheus import metrics_registry

        # Our custom registry should be different from the default
        assert metrics_registry is not REGISTRY

        # Custom registry should have our metrics
        collector_names = set()
        for collector in metrics_registry._collector_to_names.keys():
            collector_names.update(metrics_registry._collector_to_names[collector])

        expected_metrics = [
            "biocurator_requests_total",
            "biocurator_request_duration_seconds",
            "biocurator_health_checks_total",
            "biocurator_circuit_breaker_trips_total",
        ]

        for metric in expected_metrics:
            assert metric in collector_names

    def test_metrics_labels(self):
        """Test that metrics have correct labels."""
        # Clear metrics
        REQUEST_COUNT._value.clear()

        # Record metric with labels
        record_request_metrics("GET", "/health", 200, 0.100)

        # Check that labels were applied correctly
        samples = list(REQUEST_COUNT.collect())[0].samples
        assert len(samples) > 0

        # Find our sample
        sample = next((s for s in samples if s.labels["method"] == "GET"), None)
        assert sample is not None
        assert sample.labels["endpoint"] == "/health"
        assert sample.labels["status"] == "200"
