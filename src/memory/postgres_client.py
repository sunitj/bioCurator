"""PostgreSQL client for episodic memory operations."""

import time
from typing import Any

from sqlalchemy import JSON, Column, DateTime, String, select
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool

from ..logging import get_logger
from .interfaces import EpisodicBackend, HealthStatus

logger = get_logger(__name__)

Base = declarative_base()


class AgentEpisode(Base):
    """SQLAlchemy model for agent episodes."""

    __tablename__ = "agent_episodes"

    id = Column(String, primary_key=True)
    agent_id = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False, index=True)
    context = Column(JSON, nullable=False)
    outcome = Column(JSON, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=True)


class PostgresClient(EpisodicBackend):
    """PostgreSQL client for episodic memory with safety integration."""

    def __init__(self, config: dict[str, Any]):
        """Initialize PostgreSQL client.

        Args:
            config: Database configuration containing PostgreSQL settings
        """
        super().__init__(config)
        self.url = config.get("postgres_url", "postgresql://user:password@localhost:5432/biocurator")

        # Connection pool settings
        self.pool_size = config.get("postgres_pool_size", 20)
        self.max_overflow = config.get("postgres_max_overflow", 30)
        self.pool_timeout = config.get("postgres_pool_timeout", 30)
        self.pool_recycle = config.get("postgres_pool_recycle", 3600)

        self._engine = None
        self._session_factory = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to PostgreSQL."""
        if self._connected:
            logger.warning("PostgreSQL client already connected")
            return

        try:
            logger.info("Connecting to PostgreSQL")

            # Convert psycopg2 URL to asyncpg URL
            async_url = self.url.replace("postgresql://", "postgresql+asyncpg://")

            # Create async engine with connection pool
            self._engine = create_async_engine(
                async_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                echo=False,  # Set to True for SQL debugging
            )

            # Create async session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Test connection and create tables
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._connected = True
            logger.info("PostgreSQL connection established successfully")

        except OperationalError as e:
            logger.error("PostgreSQL connection failed", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error connecting to PostgreSQL", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close connection to PostgreSQL."""
        if self._engine and self._connected:
            try:
                await self._engine.dispose()
                logger.info("PostgreSQL connection closed")
            except Exception as e:
                logger.error("Error closing PostgreSQL connection", error=str(e))
            finally:
                self._connected = False

    async def health_check(self) -> HealthStatus:
        """Check the health status of PostgreSQL."""
        if not self._connected or not self._engine:
            return HealthStatus(
                is_healthy=False,
                message="Not connected to PostgreSQL"
            )

        try:
            start_time = time.time()

            async with self._session_factory() as session:
                # Simple health check query
                result = await session.execute(select(1))
                row = result.scalar()

                if row == 1:
                    latency_ms = (time.time() - start_time) * 1000

                    # Get connection pool info
                    pool = self._engine.pool
                    connection_count = pool.checkedout() if hasattr(pool, 'checkedout') else None

                    return HealthStatus(
                        is_healthy=True,
                        message="PostgreSQL is healthy",
                        latency_ms=latency_ms,
                        connection_count=connection_count
                    )
                else:
                    return HealthStatus(
                        is_healthy=False,
                        message="PostgreSQL health check query failed"
                    )

        except OperationalError as e:
            logger.warning("PostgreSQL health check failed", error=str(e))
            return HealthStatus(
                is_healthy=False,
                message=f"PostgreSQL unavailable: {str(e)}"
            )
        except Exception as e:
            logger.error("PostgreSQL health check error", error=str(e))
            return HealthStatus(
                is_healthy=False,
                message=f"Health check error: {str(e)}"
            )

    async def ping(self) -> float:
        """Ping PostgreSQL and return latency in milliseconds."""
        if not self._connected or not self._session_factory:
            raise RuntimeError("Not connected to PostgreSQL")

        start_time = time.time()

        try:
            async with self._session_factory() as session:
                await session.execute(select(1))
            return (time.time() - start_time) * 1000
        except Exception as e:
            logger.error("PostgreSQL ping failed", error=str(e))
            raise

    async def store_episode(
        self,
        episode_id: str,
        agent_id: str,
        action: str,
        context: dict[str, Any],
        outcome: dict[str, Any],
        timestamp: float | None = None
    ) -> bool:
        """Store an agent episode."""
        if not self._connected or not self._session_factory:
            raise RuntimeError("Not connected to PostgreSQL")

        episode_timestamp = timestamp or time.time()

        try:
            async with self._session_factory() as session:
                episode = AgentEpisode(
                    id=episode_id,
                    agent_id=agent_id,
                    action=action,
                    context=context,
                    outcome=outcome,
                    timestamp=episode_timestamp,
                    created_at=time.time()
                )

                session.add(episode)
                await session.commit()

                logger.debug(
                    "Stored agent episode",
                    episode_id=episode_id,
                    agent_id=agent_id,
                    action=action
                )
                return True

        except SQLAlchemyError as e:
            logger.error("PostgreSQL error storing episode", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error storing episode", error=str(e))
            raise

    async def get_episodes(
        self,
        agent_id: str | None = None,
        action_type: str | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict[str, Any]]:
        """Retrieve episodes matching criteria."""
        if not self._connected or not self._session_factory:
            raise RuntimeError("Not connected to PostgreSQL")

        try:
            async with self._session_factory() as session:
                query = select(AgentEpisode)

                # Add filters
                if agent_id:
                    query = query.where(AgentEpisode.agent_id == agent_id)
                if action_type:
                    query = query.where(AgentEpisode.action == action_type)

                # Add ordering and pagination
                query = query.order_by(AgentEpisode.timestamp.desc())
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                episodes = result.scalars().all()

                # Convert to dict format
                episode_dicts = []
                for episode in episodes:
                    episode_dict = {
                        "id": episode.id,
                        "agent_id": episode.agent_id,
                        "action": episode.action,
                        "context": episode.context,
                        "outcome": episode.outcome,
                        "timestamp": episode.timestamp,
                        "created_at": episode.created_at,
                        "updated_at": episode.updated_at
                    }
                    episode_dicts.append(episode_dict)

                logger.debug("Retrieved episodes", count=len(episode_dicts))
                return episode_dicts

        except SQLAlchemyError as e:
            logger.error("PostgreSQL error getting episodes", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error getting episodes", error=str(e))
            raise

    async def get_agent_history(
        self,
        agent_id: str,
        hours_back: int = 24,
        action_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Get recent history for an agent."""
        if not self._connected or not self._session_factory:
            raise RuntimeError("Not connected to PostgreSQL")

        try:
            async with self._session_factory() as session:
                # Calculate time threshold
                time_threshold = time.time() - (hours_back * 3600)

                query = select(AgentEpisode).where(
                    AgentEpisode.agent_id == agent_id,
                    AgentEpisode.timestamp >= time_threshold
                )

                # Filter by action types if specified
                if action_types:
                    query = query.where(AgentEpisode.action.in_(action_types))

                query = query.order_by(AgentEpisode.timestamp.desc())

                result = await session.execute(query)
                episodes = result.scalars().all()

                # Convert to dict format
                history = []
                for episode in episodes:
                    episode_dict = {
                        "id": episode.id,
                        "agent_id": episode.agent_id,
                        "action": episode.action,
                        "context": episode.context,
                        "outcome": episode.outcome,
                        "timestamp": episode.timestamp,
                        "created_at": episode.created_at,
                        "updated_at": episode.updated_at
                    }
                    history.append(episode_dict)

                logger.debug(
                    "Retrieved agent history",
                    agent_id=agent_id,
                    hours_back=hours_back,
                    count=len(history)
                )
                return history

        except SQLAlchemyError as e:
            logger.error("PostgreSQL error getting agent history", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error getting agent history", error=str(e))
            raise

    async def update_episode_outcome(
        self,
        episode_id: str,
        outcome: dict[str, Any]
    ) -> bool:
        """Update the outcome of an existing episode."""
        if not self._connected or not self._session_factory:
            raise RuntimeError("Not connected to PostgreSQL")

        try:
            async with self._session_factory() as session:
                # Find the episode
                query = select(AgentEpisode).where(AgentEpisode.id == episode_id)
                result = await session.execute(query)
                episode = result.scalar_one_or_none()

                if not episode:
                    logger.warning("Episode not found for update", episode_id=episode_id)
                    return False

                # Update the outcome
                episode.outcome = outcome
                episode.updated_at = time.time()

                await session.commit()

                logger.debug("Updated episode outcome", episode_id=episode_id)
                return True

        except SQLAlchemyError as e:
            logger.error("PostgreSQL error updating episode", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error updating episode", error=str(e))
            raise
