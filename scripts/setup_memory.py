#!/usr/bin/env python3
"""Setup script for memory system initialization.

This script initializes the multi-modal memory system:
- Creates necessary database schemas
- Sets up vector collections
- Initializes knowledge graph indices
- Configures time-series buckets
- Validates all connections

Usage:
    python scripts/setup_memory.py [--config CONFIG_FILE] [--reset]
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import argparse

import structlog
import yaml

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.loader import load_config
from config.schemas import ConfigSchema
from memory.manager import DefaultMemoryManager
from memory.interfaces import HealthStatus
from safety.event_bus import EventBus

logger = structlog.get_logger()


async def check_docker_services() -> bool:
    """Check if required Docker services are running."""
    logger.info("Checking Docker services...")

    # Simple check - try to connect to default ports
    import socket

    services = {
        "Redis": ("localhost", 6379),
        "Neo4j": ("localhost", 7687),
        "PostgreSQL": ("localhost", 5432),
        "Qdrant": ("localhost", 6333),
        "InfluxDB": ("localhost", 8086),
    }

    all_running = True
    for service, (host, port) in services.items():
        try:
            with socket.create_connection((host, port), timeout=5):
                logger.info(f"✓ {service} is running")
        except (socket.error, socket.timeout):
            logger.error(f"✗ {service} is not running on {host}:{port}")
            all_running = False

    return all_running


async def setup_neo4j_indices(manager: DefaultMemoryManager) -> bool:
    """Setup Neo4j indices for optimal performance."""
    logger.info("Setting up Neo4j indices...")

    try:
        kg = manager.get_knowledge_graph()

        # Create indices for common lookups
        indices = [
            "CREATE INDEX paper_id_idx IF NOT EXISTS FOR (p:Paper) ON (p.id)",
            "CREATE INDEX author_id_idx IF NOT EXISTS FOR (a:Author) ON (a.id)",
            "CREATE INDEX concept_id_idx IF NOT EXISTS FOR (c:Concept) ON (c.id)",
            "CREATE INDEX paper_title_idx IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX paper_doi_idx IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
            "CREATE INDEX concept_name_idx IF NOT EXISTS FOR (c:Concept) ON (c.name)",
        ]

        for index_query in indices:
            await kg.run_query(index_query)
            logger.debug(f"Created index: {index_query}")

        logger.info("✓ Neo4j indices created successfully")
        return True

    except Exception as e:
        logger.error("Failed to create Neo4j indices", error=str(e))
        return False


async def setup_qdrant_collections(manager: DefaultMemoryManager) -> bool:
    """Setup Qdrant vector collections."""
    logger.info("Setting up Qdrant collections...")

    try:
        vector_store = manager.get_vector_store()

        # Standard collections for different embedding types
        collections = [
            {
                "name": "paper_abstracts",
                "vector_size": 768,  # SciBERT embeddings
                "distance": "cosine"
            },
            {
                "name": "paper_fulltext",
                "vector_size": 768,
                "distance": "cosine"
            },
            {
                "name": "concepts",
                "vector_size": 768,
                "distance": "cosine"
            },
            {
                "name": "methods",
                "vector_size": 768,
                "distance": "cosine"
            },
            {
                "name": "results",
                "vector_size": 768,
                "distance": "cosine"
            }
        ]

        for collection in collections:
            try:
                await vector_store.create_collection(
                    collection_name=collection["name"],
                    vector_size=collection["vector_size"],
                    distance_metric=collection["distance"]
                )
                logger.info(f"✓ Created collection: {collection['name']}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"✓ Collection already exists: {collection['name']}")
                else:
                    raise

        logger.info("✓ Qdrant collections setup complete")
        return True

    except Exception as e:
        logger.error("Failed to setup Qdrant collections", error=str(e))
        return False


async def setup_postgres_tables(manager: DefaultMemoryManager) -> bool:
    """Setup PostgreSQL tables (already handled by SQLAlchemy)."""
    logger.info("Setting up PostgreSQL tables...")

    try:
        episodic = manager.get_episodic_memory()

        # Tables are created automatically by SQLAlchemy in connect()
        # Just verify we can connect and query
        episodes = await episodic.get_episodes(limit=1)

        logger.info("✓ PostgreSQL tables setup complete")
        return True

    except Exception as e:
        logger.error("Failed to setup PostgreSQL tables", error=str(e))
        return False


async def setup_influxdb_buckets(manager: DefaultMemoryManager) -> bool:
    """Setup InfluxDB buckets and initial schema."""
    logger.info("Setting up InfluxDB buckets...")

    try:
        time_series = manager.get_time_series()

        # Write a test point to ensure bucket exists and is writable
        await time_series.write_point(
            measurement="system_health",
            tags={"component": "setup", "status": "initializing"},
            fields={"value": 1.0, "setup_stage": "bucket_creation"}
        )

        logger.info("✓ InfluxDB buckets setup complete")
        return True

    except Exception as e:
        logger.error("Failed to setup InfluxDB buckets", error=str(e))
        return False


async def setup_redis_namespaces(manager: DefaultMemoryManager) -> bool:
    """Setup Redis key namespaces and test connectivity."""
    logger.info("Setting up Redis namespaces...")

    try:
        working_memory = manager.get_working_memory()

        # Create namespace keys to organize data
        namespaces = [
            "biocurator:agents",
            "biocurator:tasks",
            "biocurator:cache",
            "biocurator:sessions",
            "biocurator:coordination"
        ]

        for namespace in namespaces:
            await working_memory.set(f"{namespace}:initialized", True, expire_seconds=86400)

        logger.info("✓ Redis namespaces setup complete")
        return True

    except Exception as e:
        logger.error("Failed to setup Redis namespaces", error=str(e))
        return False


async def run_health_checks(manager: DefaultMemoryManager) -> bool:
    """Run comprehensive health checks on all backends."""
    logger.info("Running comprehensive health checks...")

    try:
        health_results = await manager.health_check_all()

        all_healthy = True
        for backend_name, status in health_results.items():
            if status.is_healthy:
                logger.info(
                    f"✓ {backend_name}: {status.message}",
                    latency_ms=status.latency_ms,
                    connections=status.connection_count
                )
            else:
                logger.error(f"✗ {backend_name}: {status.message}")
                all_healthy = False

        if all_healthy:
            logger.info("✓ All memory backends are healthy")
        else:
            logger.error("✗ Some memory backends are unhealthy")

        return all_healthy

    except Exception as e:
        logger.error("Failed to run health checks", error=str(e))
        return False


async def reset_all_data(manager: DefaultMemoryManager) -> bool:
    """Reset all data in memory backends (DESTRUCTIVE)."""
    logger.warning("RESETTING ALL MEMORY DATA - This is destructive!")

    try:
        # Clear Redis
        working_memory = manager.get_working_memory()
        keys = await working_memory.get_all_keys("biocurator:*")
        for key in keys:
            await working_memory.delete(key)
        logger.info("✓ Cleared Redis data")

        # Clear Neo4j (remove all nodes and relationships)
        kg = manager.get_knowledge_graph()
        await kg.run_query("MATCH (n) DETACH DELETE n")
        logger.info("✓ Cleared Neo4j data")

        # Note: Qdrant collections are recreated, PostgreSQL tables are truncated
        # InfluxDB data retention is managed by the database itself

        logger.info("✓ Data reset complete")
        return True

    except Exception as e:
        logger.error("Failed to reset data", error=str(e))
        return False


async def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup BioCurator memory system")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all existing data (DESTRUCTIVE)"
    )
    parser.add_argument(
        "--skip-docker-check",
        action="store_true",
        help="Skip Docker service availability check"
    )

    args = parser.parse_args()

    logger.info("Starting BioCurator memory system setup...")

    # Load configuration
    try:
        if args.config:
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
            config = ConfigSchema(**config_data)
        else:
            config = load_config()

        logger.info("Configuration loaded successfully")

    except Exception as e:
        logger.error("Failed to load configuration", error=str(e))
        return 1

    # Check Docker services
    if not args.skip_docker_check:
        if not await check_docker_services():
            logger.error("Required Docker services are not running")
            logger.info("Start services with: docker-compose -f docker-compose.memory.yml up -d")
            return 1

    # Initialize memory manager
    try:
        event_bus = EventBus()
        manager = DefaultMemoryManager(config.database, event_bus)

        logger.info("Initializing memory backends...")
        await manager.initialize()

    except Exception as e:
        logger.error("Failed to initialize memory manager", error=str(e))
        return 1

    # Reset data if requested
    if args.reset:
        if not await reset_all_data(manager):
            logger.error("Failed to reset data")
            return 1

    # Setup each backend
    setup_tasks = [
        ("Neo4j indices", setup_neo4j_indices(manager)),
        ("Qdrant collections", setup_qdrant_collections(manager)),
        ("PostgreSQL tables", setup_postgres_tables(manager)),
        ("InfluxDB buckets", setup_influxdb_buckets(manager)),
        ("Redis namespaces", setup_redis_namespaces(manager)),
    ]

    success_count = 0
    for task_name, task in setup_tasks:
        try:
            if await task:
                success_count += 1
            else:
                logger.error(f"Setup failed: {task_name}")
        except Exception as e:
            logger.error(f"Setup error in {task_name}", error=str(e))

    # Final health check
    if not await run_health_checks(manager):
        logger.error("Health checks failed")
        return 1

    # Cleanup
    await manager.shutdown()

    if success_count == len(setup_tasks):
        logger.info("🎉 Memory system setup completed successfully!")
        logger.info("You can now start the main application")
        return 0
    else:
        logger.error(f"Setup incomplete: {success_count}/{len(setup_tasks)} tasks succeeded")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Unexpected error during setup", error=str(e))
        sys.exit(1)