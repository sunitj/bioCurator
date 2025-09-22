"""Tests for health endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_endpoint_success(self, test_client: TestClient):
        """Test successful health check."""
        response = test_client.get("/health/")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "components" in data
        assert len(data["components"]) >= 2  # config and logging

        # Check component structure
        for component in data["components"]:
            assert "name" in component
            assert "status" in component
            assert "message" in component
            assert component["status"] in ["healthy", "unhealthy", "unknown"]

    def test_readiness_endpoint_success(self, test_client: TestClient):
        """Test successful readiness check."""
        response = test_client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ready"
        assert "message" in data

    def test_liveness_endpoint_success(self, test_client: TestClient):
        """Test successful liveness check."""
        response = test_client.get("/health/live")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "alive"
        assert "message" in data

    def test_health_endpoint_components(self, test_client: TestClient):
        """Test health endpoint returns expected components."""
        response = test_client.get("/health/")

        assert response.status_code == 200
        data = response.json()

        component_names = [comp["name"] for comp in data["components"]]
        assert "config" in component_names
        assert "logging" in component_names

    def test_health_endpoint_metadata(self, test_client: TestClient):
        """Test health endpoint includes metadata."""
        response = test_client.get("/health/")

        assert response.status_code == 200
        data = response.json()

        # Find config component
        config_component = next(
            (comp for comp in data["components"] if comp["name"] == "config"), None
        )

        assert config_component is not None
        assert "metadata" in config_component
        assert "app_mode" in config_component["metadata"]
        assert "log_level" in config_component["metadata"]
        assert config_component["metadata"]["app_mode"] == "development"
