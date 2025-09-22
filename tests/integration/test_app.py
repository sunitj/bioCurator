"""Integration tests for the main application."""

import pytest
from fastapi.testclient import TestClient


class TestApplicationIntegration:
    """Test full application integration."""

    def test_app_startup(self, test_client: TestClient):
        """Test that the application starts up correctly."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "BioCurator"
        assert data["version"] == "0.1.0"
        assert data["mode"] == "development"
        assert data["status"] == "running"

    def test_health_check_integration(self, test_client: TestClient):
        """Test health check integration with all components."""
        response = test_client.get("/health/")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert len(data["components"]) >= 2

        # All components should be healthy in test environment
        for component in data["components"]:
            assert component["status"] == "healthy"

    def test_metrics_integration(self, test_client: TestClient):
        """Test metrics integration."""
        # First, make some requests to generate metrics
        test_client.get("/")
        test_client.get("/health/")
        test_client.get("/health/ready")

        # Then check metrics
        response = test_client.get("/metrics/")

        assert response.status_code == 200
        content = response.text

        # Should contain request metrics
        assert "biocurator_requests_total" in content
        assert "biocurator_request_duration_seconds" in content

    def test_cors_integration(self, test_client: TestClient):
        """Test CORS middleware integration."""
        response = test_client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_error_handling_integration(self, test_client: TestClient):
        """Test error handling integration."""
        # Request non-existent endpoint
        response = test_client.get("/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
