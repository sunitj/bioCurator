"""InfluxDB client for time-series operations."""

import time
from typing import Any

from influxdb_client import Point
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

from ..logging import get_logger
from .interfaces import HealthStatus, TimeSeriesBackend

logger = get_logger(__name__)


class InfluxClient(TimeSeriesBackend):
    """InfluxDB client for time-series operations with safety integration."""

    def __init__(self, config: dict[str, Any]):
        """Initialize InfluxDB client.

        Args:
            config: Database configuration containing InfluxDB settings
        """
        super().__init__(config)
        self.url = config.get("influxdb_url", "http://localhost:8086")
        self.token = config.get("influxdb_token", "dev_token_12345")
        self.org = config.get("influxdb_org", "biocurator")
        self.bucket = config.get("influxdb_bucket", "agent_metrics")
        self.timeout = config.get("influxdb_timeout", 30)

        self._client: InfluxDBClientAsync | None = None
        self._write_api = None
        self._query_api = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to InfluxDB."""
        if self._connected:
            logger.warning("InfluxDB client already connected")
            return

        try:
            logger.info("Connecting to InfluxDB", url=self.url)

            self._client = InfluxDBClientAsync(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=self.timeout * 1000  # Convert to milliseconds
            )

            # Get write and query APIs
            self._write_api = self._client.write_api()
            self._query_api = self._client.query_api()

            # Test connection by getting health
            health = await self._client.health()
            if health.status == "pass":
                self._connected = True
                logger.info("InfluxDB connection established successfully")
            else:
                raise RuntimeError(f"InfluxDB health check failed: {health.message}")

        except InfluxDBError as e:
            logger.error("InfluxDB connection failed", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error connecting to InfluxDB", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close connection to InfluxDB."""
        if self._client and self._connected:
            try:
                await self._client.close()
                logger.info("InfluxDB connection closed")
            except Exception as e:
                logger.error("Error closing InfluxDB connection", error=str(e))
            finally:
                self._connected = False

    async def health_check(self) -> HealthStatus:
        """Check the health status of InfluxDB."""
        if not self._connected or not self._client:
            return HealthStatus(
                is_healthy=False,
                message="Not connected to InfluxDB"
            )

        try:
            start_time = time.time()

            health = await self._client.health()
            if health.status == "pass":
                latency_ms = (time.time() - start_time) * 1000

                return HealthStatus(
                    is_healthy=True,
                    message="InfluxDB is healthy",
                    latency_ms=latency_ms
                )
            else:
                return HealthStatus(
                    is_healthy=False,
                    message=f"InfluxDB health check failed: {health.message}"
                )

        except InfluxDBError as e:
            logger.warning("InfluxDB health check failed", error=str(e))
            return HealthStatus(
                is_healthy=False,
                message=f"InfluxDB unavailable: {str(e)}"
            )
        except Exception as e:
            logger.error("InfluxDB health check error", error=str(e))
            return HealthStatus(
                is_healthy=False,
                message=f"Health check error: {str(e)}"
            )

    async def ping(self) -> float:
        """Ping InfluxDB and return latency in milliseconds."""
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to InfluxDB")

        start_time = time.time()

        try:
            await self._client.ping()
            return (time.time() - start_time) * 1000
        except Exception as e:
            logger.error("InfluxDB ping failed", error=str(e))
            raise

    async def write_point(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: dict[str, Any],
        timestamp: int | None = None
    ) -> bool:
        """Write a single data point."""
        if not self._connected or not self._write_api:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            # Create point
            point = Point(measurement)

            # Add tags
            for key, value in tags.items():
                point = point.tag(key, value)

            # Add fields
            for key, value in fields.items():
                point = point.field(key, value)

            # Set timestamp if provided
            if timestamp:
                point = point.time(timestamp)

            # Write point
            await self._write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=point
            )

            logger.debug(
                "Wrote InfluxDB point",
                measurement=measurement,
                tags=tags,
                fields=list(fields.keys())
            )
            return True

        except InfluxDBError as e:
            logger.error("InfluxDB error writing point", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error writing point", error=str(e))
            raise

    async def write_points(self, points: list[dict[str, Any]]) -> bool:
        """Write multiple data points."""
        if not self._connected or not self._write_api:
            raise RuntimeError("Not connected to InfluxDB")

        if not points:
            return True

        try:
            # Convert point dictionaries to Point objects
            point_objects = []
            for point_data in points:
                measurement = point_data.get("measurement")
                if not measurement:
                    raise ValueError("Point must have 'measurement' field")

                point = Point(measurement)

                # Add tags
                tags = point_data.get("tags", {})
                for key, value in tags.items():
                    point = point.tag(key, value)

                # Add fields
                fields = point_data.get("fields", {})
                for key, value in fields.items():
                    point = point.field(key, value)

                # Set timestamp if provided
                timestamp = point_data.get("timestamp")
                if timestamp:
                    point = point.time(timestamp)

                point_objects.append(point)

            # Write all points
            await self._write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=point_objects
            )

            logger.debug("Wrote InfluxDB points", count=len(points))
            return True

        except InfluxDBError as e:
            logger.error("InfluxDB error writing points", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error writing points", error=str(e))
            raise

    async def query(
        self,
        query: str,
        start_time: str | None = None,
        stop_time: str | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Flux query."""
        if not self._connected or not self._query_api:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            # Add time range to query if provided
            if start_time or stop_time:
                time_range = "|> range("
                if start_time:
                    time_range += f"start: {start_time}"
                if stop_time:
                    if start_time:
                        time_range += f", stop: {stop_time}"
                    else:
                        time_range += f"stop: {stop_time}"
                time_range += ")"

                # Insert time range after from() clause
                if "|> from(" in query and time_range not in query:
                    query = query.replace("|> from(", "|> from(").replace(")", f") {time_range}", 1)

            # Execute query
            result = await self._query_api.query(query, org=self.org)

            # Convert to list of dictionaries
            records = []
            for table in result:
                for record in table.records:
                    record_dict = {
                        "time": record.get_time(),
                        "measurement": record.get_measurement(),
                        "field": record.get_field(),
                        "value": record.get_value()
                    }
                    # Add tags
                    for key, value in record.values.items():
                        if key.startswith("_") or key in ["result", "table"]:
                            continue
                        record_dict[key] = value

                    records.append(record_dict)

            logger.debug("Executed InfluxDB query", records_count=len(records))
            return records

        except InfluxDBError as e:
            logger.error("InfluxDB error executing query", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error executing query", error=str(e))
            raise

    async def get_agent_metrics(
        self,
        agent_id: str,
        metric_name: str,
        hours_back: int = 24
    ) -> list[dict[str, Any]]:
        """Get metrics for a specific agent."""
        if not self._connected or not self._query_api:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            # Build Flux query
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: -{hours_back}h)
                |> filter(fn: (r) => r._measurement == "{metric_name}")
                |> filter(fn: (r) => r.agent_id == "{agent_id}")
                |> sort(columns: ["_time"])
            '''

            result = await self._query_api.query(query, org=self.org)

            # Convert to list of dictionaries
            metrics = []
            for table in result:
                for record in table.records:
                    metric_dict = {
                        "time": record.get_time(),
                        "agent_id": agent_id,
                        "metric": metric_name,
                        "value": record.get_value(),
                        "field": record.get_field()
                    }
                    # Add additional tags
                    for key, value in record.values.items():
                        if key.startswith("_") or key in ["result", "table", "agent_id"]:
                            continue
                        metric_dict[key] = value

                    metrics.append(metric_dict)

            logger.debug(
                "Retrieved agent metrics",
                agent_id=agent_id,
                metric=metric_name,
                hours_back=hours_back,
                count=len(metrics)
            )
            return metrics

        except InfluxDBError as e:
            logger.error("InfluxDB error getting agent metrics", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error getting agent metrics", error=str(e))
            raise
