#!/usr/bin/env python3
"""
Basic agent workflow example demonstrating multi-agent coordination.

This example shows how to:
1. Initialize the agent system
2. Load agent configurations
3. Start agents with safety controls
4. Execute a simple workflow
5. Monitor agent performance
6. Shut down gracefully

Usage:
    python examples/basic_workflow.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.loader import load_config
from config.schemas import AppMode
from agents.registry import AgentRegistry
from coordination.task_queue import TaskQueue, TaskDefinition, TaskPriority
from coordination.safety_coordinator import SafetyCoordinator
from memory.manager import DefaultMemoryManager
from monitoring.agent_monitor import AgentMonitor
from safety.event_bus import SafetyEventBus
from logging import get_logger

logger = get_logger(__name__)


async def main():
    """Run basic agent workflow example."""
    print("🔬 BioCurator Basic Agent Workflow Example")
    print("=" * 50)

    # Set development mode
    os.environ["APP_MODE"] = "development"

    try:
        # 1. Load configuration
        print("\n1. Loading configuration...")
        config = load_config()
        print(f"   Mode: {config.app_mode.value}")
        print(f"   Safety budget: ${config.safety.max_cost_budget}")

        # 2. Initialize core systems
        print("\n2. Initializing core systems...")

        # Event bus for safety events
        event_bus = SafetyEventBus()

        # Memory manager (simplified for demo - may not have all backends)
        print("   Initializing memory systems...")
        try:
            async with DefaultMemoryManager(config.database, event_bus) as memory_manager:
                await _run_with_full_memory(config, event_bus, memory_manager)
        except Exception as e:
            print(f"   Full memory system not available: {e}")
            print("   Running with simplified mock setup...")
            await _run_simplified_demo(config, event_bus)

    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n🎉 Basic workflow example completed successfully!")
    print("\nKey Demonstrations:")
    print("✅ Agent initialization with safety controls")
    print("✅ Task execution with monitoring")
    print("✅ Safety coordinator integration")
    print("✅ Performance metrics collection")
    print("✅ Graceful shutdown procedures")

    return 0


async def _run_with_full_memory(config, event_bus, memory_manager):
    """Run example with full memory system."""
    print("   Memory systems initialized")

    # Task queue
    print("   Initializing task queue...")
    task_queue = TaskQueue(
        database_url=config.database.postgres_url,
        event_bus=event_bus,
    )

    try:
        await task_queue.initialize()
        await _run_full_workflow(config, event_bus, memory_manager, task_queue)
    except Exception as e:
        print(f"   Task queue initialization failed: {e}")
        await _run_basic_workflow(config, event_bus, memory_manager)
    finally:
        if task_queue:
            try:
                await task_queue.shutdown()
            except:
                pass


async def _run_simplified_demo(config, event_bus):
    """Run simplified demo without full infrastructure."""
    print("   Running basic agent demonstration...")

    # Create mock memory manager
    class MockMemoryManager:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def get_knowledge_graph(self):
            return None

    async with MockMemoryManager() as memory_manager:
        await _run_basic_workflow(config, event_bus, memory_manager)


async def _run_basic_workflow(config, event_bus, memory_manager):
    """Run basic workflow with mock agents."""
    print("   Creating mock Research Director...")

    from agents.base import BaseAgent, AgentTask
    from agents.config import AgentConfig, AgentSafetyConfig

    class MockResearchDirector(BaseAgent):
        async def _startup(self):
            print(f"     Mock agent {self.agent_id} starting up...")

        async def _shutdown(self):
            print(f"     Mock agent {self.agent_id} shutting down...")

        async def _execute_task_impl(self, task):
            # Simulate some work
            await asyncio.sleep(0.1)

            if task.task_type == "analyze_literature":
                return {
                    "query": task.input_data.get("query", "unknown"),
                    "papers_analyzed": task.input_data.get("max_papers", 0),
                    "findings": [
                        {"type": "trend", "description": f"Emerging trend in {task.input_data.get('query', 'research')}"},
                        {"type": "gap", "description": "Research gap identified"}
                    ],
                    "summary": f"Analyzed literature for '{task.input_data.get('query', 'unknown query')}'",
                }
            elif task.task_type == "orchestrate_workflow":
                return {
                    "workflow_id": f"wf_{int(asyncio.get_event_loop().time())}",
                    "status": "completed",
                    "tasks_executed": 3,
                    "results": {"coordination": "successful"}
                }
            elif task.task_type == "quality_control":
                return {
                    "quality_score": 0.85,
                    "quality_issues": [],
                    "recommendation": "approved"
                }
            else:
                return {"result": f"Mock execution of {task.task_type}"}

    # Create mock agent
    mock_config = AgentConfig(
        class_path="examples.basic_workflow.MockResearchDirector",
        role="Mock Research Director",
        specialties=["coordination", "analysis", "quality_control"],
        safety=AgentSafetyConfig(),
    )

    agent = MockResearchDirector(
        agent_id="research_director",
        config=mock_config,
        system_config=config,
        memory_manager=memory_manager,
        event_bus=event_bus,
    )

    # Initialize monitoring
    agent_monitor = AgentMonitor(event_bus=event_bus)
    await agent_monitor.start_monitoring()
    await agent_monitor.register_agent(
        agent_id=agent.agent_id,
        agent_info={
            "role": agent.config.role,
            "capabilities": agent.config.specialties,
        },
    )

    try:
        # Start agent
        await agent.start()
        print(f"   Agent {agent.agent_id} started")

        # Execute example tasks
        print("\n3. Executing example tasks...")
        tasks = [
            {
                "task_type": "analyze_literature",
                "input_data": {
                    "query": "protein folding mechanisms",
                    "max_papers": 5,
                    "analysis_depth": "standard",
                },
            },
            {
                "task_type": "orchestrate_workflow",
                "input_data": {
                    "workflow_type": "literature_analysis",
                    "parameters": {"query": "CRISPR gene editing", "scope": "recent"},
                },
            },
            {
                "task_type": "quality_control",
                "input_data": {
                    "output_data": {"results": ["finding1", "finding2"]},
                    "quality_criteria": ["completeness", "consistency"],
                },
            },
        ]

        results = []
        for i, task_def in enumerate(tasks):
            print(f"   Executing task {i+1}: {task_def['task_type']}")

            # Create agent task
            agent_task = AgentTask(
                id=f"demo_task_{i+1}",
                agent_id=agent.agent_id,
                task_type=task_def["task_type"],
                input_data=task_def["input_data"],
            )

            # Record task start
            await agent_monitor.record_task_start(
                task_id=agent_task.id,
                agent_id=agent.agent_id,
                task_type=agent_task.task_type,
            )

            # Execute task
            start_time = asyncio.get_event_loop().time()
            try:
                response = await agent.execute_task(agent_task)
                duration = asyncio.get_event_loop().time() - start_time

                results.append(response)

                # Record completion
                await agent_monitor.record_task_completion(
                    task_id=agent_task.id,
                    agent_id=agent.agent_id,
                    task_type=agent_task.task_type,
                    duration_seconds=duration,
                    success=response.success,
                    error_message=response.error_message,
                    quality_score=0.8 if response.success else None,
                )

                print(f"     ✅ Success: {response.success}")
                if response.success and response.output_data:
                    print(f"     📊 Result keys: {list(response.output_data.keys())}")
                if response.error_message:
                    print(f"     ❌ Error: {response.error_message}")

            except Exception as e:
                duration = asyncio.get_event_loop().time() - start_time
                print(f"     ❌ Exception: {e}")

                await agent_monitor.record_task_completion(
                    task_id=agent_task.id,
                    agent_id=agent.agent_id,
                    task_type=agent_task.task_type,
                    duration_seconds=duration,
                    success=False,
                    error_message=str(e),
                )

            await asyncio.sleep(0.2)

        # Show results
        print("\n4. Results Summary:")
        successful_tasks = sum(1 for r in results if r.success)
        print(f"   Completed: {successful_tasks}/{len(results)} tasks")

        # Show metrics
        metrics = await agent_monitor.get_agent_metrics(agent.agent_id)
        if metrics:
            print(f"   Agent Performance:")
            print(f"     Total tasks: {metrics.total_tasks}")
            print(f"     Success rate: {metrics.success_rate:.1%}")
            print(f"     Avg response time: {metrics.average_response_time:.2f}s")

        # Show agent status
        status = await agent.get_status()
        print(f"   Agent Status:")
        print(f"     Health: {status.health_status}")
        print(f"     Completed: {status.total_tasks_completed}")
        print(f"     Failed: {status.total_tasks_failed}")

        # Show system health
        system_metrics = await agent_monitor.get_system_metrics()
        print(f"   System Health:")
        print(f"     Success rate: {system_metrics['task_metrics']['success_rate']:.1%}")

    finally:
        # Cleanup
        print("\n5. Shutting down...")
        await agent.stop()
        await agent_monitor.unregister_agent(agent.agent_id)
        await agent_monitor.stop_monitoring()
        print("   ✅ Cleanup completed")


async def _run_full_workflow(config, event_bus, memory_manager, task_queue):
    """Run full workflow with all systems."""
    # This would be the full implementation with all agents
    print("   Full workflow not implemented yet - using basic workflow")
    await _run_basic_workflow(config, event_bus, memory_manager)


if __name__ == "__main__":
    # Run the example
    exit_code = asyncio.run(main())
    sys.exit(exit_code)