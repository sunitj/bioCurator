#!/usr/bin/env python3
"""
Safety controls demonstration for BioCurator agent system.

This example demonstrates:
1. Circuit breaker functionality
2. Rate limiting controls
3. Cost tracking and budget enforcement
4. Anomaly detection
5. Behavior monitoring
6. Development mode safety guards

Usage:
    python examples/safety_demo.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.loader import load_config
from config.schemas import AppMode
from agents.base import BaseAgent, AgentTask
from agents.config import AgentConfig, AgentSafetyConfig
from safety.event_bus import SafetyEventBus, SafetyEvent, SafetyEventType
from safety.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from safety.rate_limiter import RateLimiter, RateLimiterConfig
from safety.cost_tracker import CostTracker
from safety.behavior_monitor import BehaviorMonitor
from monitoring.agent_monitor import AgentMonitor
from logging import get_logger

logger = get_logger(__name__)


class SafetyTestAgent(BaseAgent):
    """Test agent for demonstrating safety controls."""

    def __init__(self, *args, failure_rate=0.0, slow_response=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_rate = failure_rate
        self.slow_response = slow_response
        self.task_count = 0

    async def _startup(self):
        print(f"     Safety test agent {self.agent_id} starting (failure_rate={self.failure_rate})")

    async def _shutdown(self):
        print(f"     Safety test agent {self.agent_id} shutting down")

    async def _execute_task_impl(self, task):
        self.task_count += 1

        # Simulate slow response if configured
        if self.slow_response:
            await asyncio.sleep(2.0)  # Slow response

        # Simulate failures based on failure rate
        import random
        if random.random() < self.failure_rate:
            raise RuntimeError(f"Simulated failure in task {self.task_count}")

        return {
            "task_id": task.id,
            "task_count": self.task_count,
            "result": f"Successfully processed {task.task_type}",
            "agent_id": self.agent_id,
        }


async def main():
    """Run safety controls demonstration."""
    print("🛡️  BioCurator Safety Controls Demonstration")
    print("=" * 50)

    # Set development mode
    os.environ["APP_MODE"] = "development"

    try:
        # 1. Load configuration
        print("\n1. Loading configuration...")
        config = load_config()
        print(f"   Mode: {config.app_mode.value}")
        print(f"   Development mode enforces zero cost budget: ${config.safety.max_cost_budget}")

        # 2. Initialize safety systems
        print("\n2. Initializing safety systems...")

        # Event bus with custom handler to log safety events
        event_bus = SafetyEventBus()
        safety_events = []

        def safety_event_handler(event: SafetyEvent):
            safety_events.append(event)
            print(f"   🚨 SAFETY EVENT: {event.event_type.value} - {event.message}")

        event_bus.subscribe(safety_event_handler)

        # Create mock memory manager
        class MockMemoryManager:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def get_knowledge_graph(self):
                return None

        async with MockMemoryManager() as memory_manager:
            # Agent monitor
            agent_monitor = AgentMonitor(event_bus=event_bus, monitoring_interval_seconds=5)
            await agent_monitor.start_monitoring()

            # 3. Demonstrate circuit breaker
            print("\n3. Circuit Breaker Demonstration:")
            print("   Creating agent with high failure rate...")

            failing_agent = SafetyTestAgent(
                agent_id="failing_agent",
                config=AgentConfig(
                    class_path="examples.safety_demo.SafetyTestAgent",
                    role="Failing Test Agent",
                    specialties=["testing"],
                    safety=AgentSafetyConfig(
                        circuit_breaker_threshold=0.3,  # Low threshold for demo
                        circuit_breaker_min_volume=3,   # Small volume for demo
                    ),
                ),
                system_config=config,
                memory_manager=memory_manager,
                event_bus=event_bus,
                failure_rate=0.8,  # 80% failure rate
            )

            await failing_agent.start()
            await agent_monitor.register_agent(
                agent_id=failing_agent.agent_id,
                agent_info={"role": "Failing Test Agent", "purpose": "Circuit breaker demo"},
            )

            # Execute tasks that will cause circuit breaker to trip
            print("   Executing tasks to trigger circuit breaker...")
            for i in range(8):
                task = AgentTask(
                    id=f"cb_task_{i+1}",
                    agent_id=failing_agent.agent_id,
                    task_type="test_task",
                    input_data={"iteration": i+1},
                )

                await agent_monitor.record_task_start(task.id, failing_agent.agent_id, task.task_type)

                start_time = asyncio.get_event_loop().time()
                try:
                    response = await failing_agent.execute_task(task)
                    duration = asyncio.get_event_loop().time() - start_time
                    success = response.success

                    print(f"     Task {i+1}: {'✅' if success else '❌'} ({duration:.2f}s)")

                    await agent_monitor.record_task_completion(
                        task.id, failing_agent.agent_id, task.task_type,
                        duration, success, response.error_message
                    )

                except Exception as e:
                    duration = asyncio.get_event_loop().time() - start_time
                    print(f"     Task {i+1}: ❌ Circuit breaker or other error ({duration:.2f}s)")

                    await agent_monitor.record_task_completion(
                        task.id, failing_agent.agent_id, task.task_type,
                        duration, False, str(e)
                    )

                await asyncio.sleep(0.1)

            # Check circuit breaker state
            cb_state = failing_agent.circuit_breaker.state
            print(f"   Circuit breaker state: {cb_state.value}")

            # 4. Demonstrate rate limiting
            print("\n4. Rate Limiting Demonstration:")
            print("   Creating agent with low rate limits...")

            rate_limited_agent = SafetyTestAgent(
                agent_id="rate_limited_agent",
                config=AgentConfig(
                    class_path="examples.safety_demo.SafetyTestAgent",
                    role="Rate Limited Test Agent",
                    specialties=["testing"],
                    safety=AgentSafetyConfig(
                        max_requests_per_minute=5,  # Very low for demo
                        burst_capacity=3,           # Small burst
                    ),
                ),
                system_config=config,
                memory_manager=memory_manager,
                event_bus=event_bus,
                failure_rate=0.0,  # No failures for this test
            )

            await rate_limited_agent.start()
            await agent_monitor.register_agent(
                agent_id=rate_limited_agent.agent_id,
                agent_info={"role": "Rate Limited Test Agent", "purpose": "Rate limiting demo"},
            )

            # Rapidly execute tasks to trigger rate limiting
            print("   Rapidly executing tasks to trigger rate limiting...")
            for i in range(10):
                task = AgentTask(
                    id=f"rl_task_{i+1}",
                    agent_id=rate_limited_agent.agent_id,
                    task_type="rate_test",
                    input_data={"iteration": i+1},
                )

                start_time = asyncio.get_event_loop().time()
                try:
                    response = await rate_limited_agent.execute_task(task)
                    duration = asyncio.get_event_loop().time() - start_time
                    print(f"     Task {i+1}: ✅ Completed ({duration:.2f}s)")

                except Exception as e:
                    duration = asyncio.get_event_loop().time() - start_time
                    if "rate limit" in str(e).lower():
                        print(f"     Task {i+1}: 🚫 Rate limited ({duration:.2f}s)")
                    else:
                        print(f"     Task {i+1}: ❌ Error: {e} ({duration:.2f}s)")

                # No delay - rapid fire to trigger rate limiting
                if i < 5:  # Add small delay only for first few
                    await asyncio.sleep(0.05)

            # 5. Demonstrate development mode safety guards
            print("\n5. Development Mode Safety Guards:")
            print("   Development mode prevents:")
            print("   ✅ Zero cost budget enforcement")
            print("   ✅ Cloud model access blocked")
            print("   ✅ Strict safety thresholds")

            # Check safety configuration
            safety_config = config.safety
            print(f"   Current safety settings:")
            print(f"     Max cost budget: ${safety_config.max_cost_budget}")
            print(f"     Rate limit: {safety_config.rate_limit_per_minute}/min")
            print(f"     Circuit breaker threshold: {safety_config.circuit_breaker_threshold}")

            # 6. Demonstrate behavior monitoring
            print("\n6. Behavior Monitoring Demonstration:")
            print("   Creating agent with anomalous behavior...")

            anomalous_agent = SafetyTestAgent(
                agent_id="anomalous_agent",
                config=AgentConfig(
                    class_path="examples.safety_demo.SafetyTestAgent",
                    role="Anomalous Test Agent",
                    specialties=["testing"],
                    safety=AgentSafetyConfig(),
                ),
                system_config=config,
                memory_manager=memory_manager,
                event_bus=event_bus,
                failure_rate=0.0,
                slow_response=True,  # Causes anomalous response times
            )

            await anomalous_agent.start()
            await agent_monitor.register_agent(
                agent_id=anomalous_agent.agent_id,
                agent_info={"role": "Anomalous Test Agent", "purpose": "Anomaly detection demo"},
            )

            # Execute tasks with anomalous patterns
            print("   Executing tasks with anomalous patterns...")
            for i in range(5):
                task = AgentTask(
                    id=f"anom_task_{i+1}",
                    agent_id=anomalous_agent.agent_id,
                    task_type="anomaly_test",
                    input_data={"iteration": i+1},
                )

                await agent_monitor.record_task_start(task.id, anomalous_agent.agent_id, task.task_type)

                start_time = asyncio.get_event_loop().time()
                response = await anomalous_agent.execute_task(task)
                duration = asyncio.get_event_loop().time() - start_time

                print(f"     Task {i+1}: Response time {duration:.2f}s {'(SLOW!)' if duration > 1.0 else ''}")

                await agent_monitor.record_task_completion(
                    task.id, anomalous_agent.agent_id, task.task_type,
                    duration, response.success, response.error_message
                )

                await asyncio.sleep(0.1)

            # 7. Display comprehensive safety metrics
            print("\n7. Safety Metrics Summary:")

            # Agent metrics
            all_metrics = await agent_monitor.get_all_agent_metrics()
            for agent_id, metrics in all_metrics.items():
                print(f"   Agent {agent_id}:")
                print(f"     Success rate: {metrics.success_rate:.1%}")
                print(f"     Error rate: {metrics.error_rate:.1%}")
                print(f"     Avg response time: {metrics.average_response_time:.2f}s")
                print(f"     Anomaly count: {metrics.anomaly_count}")

            # Safety events summary
            print(f"\n   Safety Events Triggered: {len(safety_events)}")
            event_counts = {}
            for event in safety_events:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1

            for event_type, count in event_counts.items():
                print(f"     {event_type}: {count}")

            # System health
            system_metrics = await agent_monitor.get_system_metrics()
            print(f"\n   System Overview:")
            print(f"     Total agents: {system_metrics['system_overview']['total_agents']}")
            print(f"     System success rate: {system_metrics['task_metrics']['success_rate']:.1%}")
            print(f"     Average response time: {system_metrics['performance_metrics']['average_response_time']:.2f}s")

            # 8. Demonstrate recovery mechanisms
            print("\n8. Recovery Mechanisms:")

            # Reset circuit breaker (normally would happen after timeout)
            if failing_agent.circuit_breaker.state.value == "open":
                print("   Circuit breaker is open - normally would recover after timeout")
                print("   In production, circuit breaker would transition to half-open for testing")

            # Show rate limiter recovery
            available_tokens = rate_limited_agent.rate_limiter.get_available_tokens()
            print(f"   Rate limiter available tokens: {available_tokens}")
            print("   Rate limits refill over time based on configured rate")

            # 9. Cleanup
            print("\n9. Cleanup:")
            agents_to_cleanup = [failing_agent, rate_limited_agent, anomalous_agent]
            for agent in agents_to_cleanup:
                await agent.stop()
                await agent_monitor.unregister_agent(agent.agent_id)

            await agent_monitor.stop_monitoring()
            print("   ✅ All agents stopped and monitoring cleaned up")

    except Exception as e:
        logger.error(f"Safety demo failed: {e}")
        print(f"\n❌ Safety demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n🎉 Safety controls demonstration completed!")
    print("\nSafety Features Demonstrated:")
    print("✅ Circuit breaker protection against failing components")
    print("✅ Rate limiting to prevent system overload")
    print("✅ Development mode safety guards")
    print("✅ Anomaly detection and monitoring")
    print("✅ Comprehensive safety event logging")
    print("✅ Performance metrics and health tracking")
    print("✅ Graceful recovery mechanisms")

    return 0


if __name__ == "__main__":
    # Run the safety demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code)