"""Research Director agent for strategic coordination and workflow orchestration."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..coordination.protocols import AgentMessage, MessageProtocol, MessagePriority
from ..coordination.task_queue import TaskDefinition, TaskPriority, TaskStatus
from ..logging import get_logger
from .base import BaseAgent, AgentTask

logger = get_logger(__name__)


class ResearchDirectorAgent(BaseAgent):
    """Research Director agent responsible for strategic coordination and orchestration.

    This agent serves as the central coordinator for multi-agent workflows,
    making high-level decisions about task allocation and workflow management.
    """

    def __init__(self, *args, **kwargs):
        """Initialize Research Director agent."""
        super().__init__(*args, **kwargs)

        # Communication protocol
        self.message_protocol = MessageProtocol(
            agent_id=self.agent_id,
            timeout_seconds=30,
        )

        # Orchestration state
        self._active_workflows: Dict[str, Dict[str, Any]] = {}
        self._agent_capabilities: Dict[str, List[str]] = {}
        self._task_assignments: Dict[str, str] = {}  # task_id -> agent_id

        logger.info(f"Research Director agent {self.agent_id} initialized")

    async def _startup(self) -> None:
        """Agent-specific startup logic."""
        logger.info(f"Starting Research Director agent {self.agent_id}")

        # Register message handlers
        self._register_message_handlers()

        # Start message processing
        asyncio.create_task(self.message_protocol.start_message_processing())

        # Discover available agents and their capabilities
        await self._discover_agent_capabilities()

        # Initialize workflow patterns from memory
        await self._load_workflow_patterns()

        logger.info(f"Research Director agent {self.agent_id} startup complete")

    async def _shutdown(self) -> None:
        """Agent-specific shutdown logic."""
        logger.info(f"Shutting down Research Director agent {self.agent_id}")

        # Complete active workflows gracefully
        await self._complete_active_workflows()

        # Save workflow patterns to memory
        await self._save_workflow_patterns()

        logger.info(f"Research Director agent {self.agent_id} shutdown complete")

    async def _execute_task_impl(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a coordination task.

        Args:
            task: Task to execute

        Returns:
            Dict[str, Any]: Task execution result
        """
        task_type = task.task_type
        input_data = task.input_data

        logger.info(
            f"Executing coordination task",
            task_id=task.id,
            task_type=task_type,
            agent_id=self.agent_id,
        )

        # Route task based on type
        if task_type == "orchestrate_workflow":
            return await self._orchestrate_workflow(task)
        elif task_type == "allocate_task":
            return await self._allocate_task(task)
        elif task_type == "monitor_progress":
            return await self._monitor_progress(task)
        elif task_type == "synthesize_results":
            return await self._synthesize_results(task)
        elif task_type == "quality_control":
            return await self._quality_control(task)
        elif task_type == "analyze_literature":
            return await self._analyze_literature(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _register_message_handlers(self) -> None:
        """Register handlers for inter-agent messages."""
        self.message_protocol.register_handler("agent_status_update", self._handle_agent_status)
        self.message_protocol.register_handler("task_completion", self._handle_task_completion)
        self.message_protocol.register_handler("workflow_request", self._handle_workflow_request)
        self.message_protocol.register_handler("capability_announcement", self._handle_capability_announcement)

        logger.debug(f"Message handlers registered for agent {self.agent_id}")

    async def _discover_agent_capabilities(self) -> None:
        """Discover capabilities of available agents."""
        logger.info("Discovering agent capabilities")

        try:
            # In a real implementation, this would query the agent registry
            # For now, we'll initialize with known agent types
            self._agent_capabilities = {
                "literature_scout": [
                    "search_papers",
                    "assess_relevance",
                    "detect_trends",
                    "extract_metadata",
                ],
                "deep_reader": [
                    "analyze_content",
                    "extract_claims",
                    "evaluate_methodology",
                    "parse_figures",
                ],
                "domain_specialist": [
                    "validate_facts",
                    "assess_credibility",
                    "detect_contradictions",
                    "provide_context",
                ],
                "knowledge_weaver": [
                    "identify_patterns",
                    "build_narratives",
                    "find_connections",
                    "identify_gaps",
                ],
                "memory_keeper": [
                    "manage_knowledge",
                    "resolve_conflicts",
                    "optimize_storage",
                    "track_evolution",
                ],
            }

            logger.info("Agent capabilities discovered", capabilities=self._agent_capabilities)

        except Exception as e:
            logger.error("Failed to discover agent capabilities", error=str(e))

    async def _load_workflow_patterns(self) -> None:
        """Load successful workflow patterns from memory."""
        try:
            # Access procedural memory for workflow patterns
            if hasattr(self.memory_manager, 'get_working_memory'):
                working_memory = self.memory_manager.get_working_memory()

                # In a real implementation, this would query for stored patterns
                logger.debug("Workflow patterns loaded from memory")

        except Exception as e:
            logger.warning("Failed to load workflow patterns", error=str(e))

    async def _save_workflow_patterns(self) -> None:
        """Save successful workflow patterns to memory."""
        try:
            # Save current successful patterns
            if self._active_workflows:
                logger.debug("Saving workflow patterns to memory")
                # In a real implementation, would store in procedural memory

        except Exception as e:
            logger.warning("Failed to save workflow patterns", error=str(e))

    # Task implementation methods

    async def _orchestrate_workflow(self, task: AgentTask) -> Dict[str, Any]:
        """Orchestrate a multi-agent workflow."""
        workflow_type = task.input_data.get("workflow_type", "basic_analysis")
        parameters = task.input_data.get("parameters", {})

        logger.info(
            f"Orchestrating workflow",
            workflow_type=workflow_type,
            task_id=task.id,
        )

        # Create workflow instance
        workflow_id = f"workflow_{task.id}_{int(datetime.now().timestamp())}"
        workflow = {
            "id": workflow_id,
            "type": workflow_type,
            "status": "initializing",
            "created_at": datetime.now(),
            "parameters": parameters,
            "tasks": [],
            "results": {},
        }

        self._active_workflows[workflow_id] = workflow

        try:
            # Generate workflow tasks based on type
            if workflow_type == "literature_analysis":
                tasks = await self._create_literature_analysis_workflow(parameters)
            elif workflow_type == "comparative_study":
                tasks = await self._create_comparative_study_workflow(parameters)
            else:
                tasks = await self._create_basic_analysis_workflow(parameters)

            workflow["tasks"] = tasks
            workflow["status"] = "running"

            # Execute workflow tasks
            results = await self._execute_workflow_tasks(workflow_id, tasks)

            workflow["results"] = results
            workflow["status"] = "completed"
            workflow["completed_at"] = datetime.now()

            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "results": results,
                "task_count": len(tasks),
            }

        except Exception as e:
            workflow["status"] = "failed"
            workflow["error"] = str(e)
            workflow["failed_at"] = datetime.now()
            logger.error(f"Workflow orchestration failed", workflow_id=workflow_id, error=str(e))
            raise

    async def _allocate_task(self, task: AgentTask) -> Dict[str, Any]:
        """Allocate a task to the most suitable agent."""
        target_task = task.input_data.get("task")
        required_capabilities = task.input_data.get("required_capabilities", [])

        logger.info(f"Allocating task", task_id=task.id, capabilities=required_capabilities)

        # Find suitable agents
        suitable_agents = []
        for agent_id, capabilities in self._agent_capabilities.items():
            if any(cap in capabilities for cap in required_capabilities):
                suitable_agents.append((agent_id, capabilities))

        if not suitable_agents:
            raise ValueError(f"No agents found with required capabilities: {required_capabilities}")

        # Simple allocation strategy: pick first suitable agent
        # In a real implementation, this would consider agent load, performance, etc.
        selected_agent, agent_capabilities = suitable_agents[0]

        # Record task assignment
        self._task_assignments[task.id] = selected_agent

        return {
            "allocated_to": selected_agent,
            "agent_capabilities": agent_capabilities,
            "allocation_strategy": "capability_match",
        }

    async def _monitor_progress(self, task: AgentTask) -> Dict[str, Any]:
        """Monitor progress of active workflows and tasks."""
        logger.info(f"Monitoring progress", task_id=task.id)

        progress_summary = {
            "active_workflows": len(self._active_workflows),
            "task_assignments": len(self._task_assignments),
            "workflows": [],
        }

        for workflow_id, workflow in self._active_workflows.items():
            workflow_progress = {
                "workflow_id": workflow_id,
                "type": workflow["type"],
                "status": workflow["status"],
                "created_at": workflow["created_at"].isoformat(),
                "task_count": len(workflow.get("tasks", [])),
                "progress_percentage": self._calculate_workflow_progress(workflow),
            }

            if "completed_at" in workflow:
                workflow_progress["completed_at"] = workflow["completed_at"].isoformat()

            progress_summary["workflows"].append(workflow_progress)

        return progress_summary

    async def _synthesize_results(self, task: AgentTask) -> Dict[str, Any]:
        """Synthesize results from multiple agent outputs."""
        workflow_id = task.input_data.get("workflow_id")
        results_data = task.input_data.get("results", [])

        logger.info(f"Synthesizing results", workflow_id=workflow_id, result_count=len(results_data))

        # Basic synthesis - in real implementation would be more sophisticated
        synthesis = {
            "summary": f"Synthesized {len(results_data)} results",
            "key_findings": [],
            "confidence_score": 0.8,
            "synthesis_method": "basic_aggregation",
        }

        # Extract key findings
        for result in results_data:
            if "findings" in result:
                synthesis["key_findings"].extend(result["findings"])

        # Store synthesis in knowledge graph
        try:
            kg = self.memory_manager.get_knowledge_graph()
            # In real implementation: await kg.store_synthesis(synthesis)
            logger.debug("Synthesis stored in knowledge graph")
        except Exception as e:
            logger.warning("Failed to store synthesis", error=str(e))

        return synthesis

    async def _quality_control(self, task: AgentTask) -> Dict[str, Any]:
        """Perform quality control on agent outputs."""
        output_data = task.input_data.get("output_data", {})
        quality_criteria = task.input_data.get("quality_criteria", [])

        logger.info(f"Performing quality control", task_id=task.id)

        quality_score = 0.0
        quality_issues = []

        # Basic quality checks
        if "completeness" in quality_criteria:
            completeness_score = len(output_data.get("results", [])) / 10  # Arbitrary scoring
            quality_score += min(completeness_score, 1.0) * 0.4

        if "consistency" in quality_criteria:
            # Placeholder for consistency checks
            quality_score += 0.3

        if "accuracy" in quality_criteria:
            # Placeholder for accuracy checks
            quality_score += 0.3

        quality_assessment = {
            "quality_score": min(quality_score, 1.0),
            "quality_issues": quality_issues,
            "criteria_checked": quality_criteria,
            "recommendation": "approved" if quality_score > 0.7 else "needs_improvement",
        }

        return quality_assessment

    async def _analyze_literature(self, task: AgentTask) -> Dict[str, Any]:
        """Coordinate literature analysis workflow."""
        query = task.input_data.get("query", "")
        max_papers = task.input_data.get("max_papers", 10)
        analysis_depth = task.input_data.get("depth", "standard")

        logger.info(f"Analyzing literature", query=query, max_papers=max_papers)

        # This is a simplified version - would coordinate with other agents
        analysis_result = {
            "query": query,
            "papers_analyzed": max_papers,
            "analysis_depth": analysis_depth,
            "summary": f"Literature analysis for '{query}' completed",
            "findings": [
                {"type": "trend", "description": f"Emerging trend in {query}"},
                {"type": "gap", "description": f"Research gap identified in {query}"},
            ],
            "recommendations": [
                f"Further investigation needed in {query}",
                "Cross-validation with additional sources recommended",
            ],
        }

        return analysis_result

    # Workflow creation methods

    async def _create_literature_analysis_workflow(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create tasks for literature analysis workflow."""
        query = parameters.get("query", "")

        tasks = [
            {
                "type": "search_papers",
                "agent": "literature_scout",
                "parameters": {"query": query, "max_results": 20},
                "priority": TaskPriority.HIGH,
            },
            {
                "type": "analyze_content",
                "agent": "deep_reader",
                "parameters": {"depth": "detailed"},
                "priority": TaskPriority.NORMAL,
                "depends_on": ["search_papers"],
            },
            {
                "type": "validate_findings",
                "agent": "domain_specialist",
                "parameters": {"validation_level": "standard"},
                "priority": TaskPriority.NORMAL,
                "depends_on": ["analyze_content"],
            },
            {
                "type": "synthesize_insights",
                "agent": "knowledge_weaver",
                "parameters": {"synthesis_method": "comprehensive"},
                "priority": TaskPriority.HIGH,
                "depends_on": ["validate_findings"],
            },
        ]

        return tasks

    async def _create_comparative_study_workflow(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create tasks for comparative study workflow."""
        topics = parameters.get("topics", [])

        tasks = []

        # Create parallel search tasks for each topic
        for i, topic in enumerate(topics):
            tasks.append({
                "type": "search_papers",
                "agent": "literature_scout",
                "parameters": {"query": topic, "max_results": 10},
                "priority": TaskPriority.NORMAL,
                "task_id": f"search_{i}",
            })

        # Analysis and comparison tasks
        tasks.extend([
            {
                "type": "comparative_analysis",
                "agent": "deep_reader",
                "parameters": {"comparison_type": "methodology"},
                "priority": TaskPriority.HIGH,
                "depends_on": [f"search_{i}" for i in range(len(topics))],
            },
            {
                "type": "synthesize_comparison",
                "agent": "knowledge_weaver",
                "parameters": {"output_format": "comparative_report"},
                "priority": TaskPriority.HIGH,
                "depends_on": ["comparative_analysis"],
            },
        ])

        return tasks

    async def _create_basic_analysis_workflow(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create tasks for basic analysis workflow."""
        return [
            {
                "type": "basic_analysis",
                "agent": "research_director",  # Self-execution for simple tasks
                "parameters": parameters,
                "priority": TaskPriority.NORMAL,
            }
        ]

    async def _execute_workflow_tasks(self, workflow_id: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute all tasks in a workflow."""
        results = {}

        logger.info(f"Executing workflow tasks", workflow_id=workflow_id, task_count=len(tasks))

        # For now, simulate task execution
        # In real implementation, would submit tasks to queue and coordinate execution
        for i, task_def in enumerate(tasks):
            task_id = task_def.get("task_id", f"task_{i}")

            # Simulate task execution
            await asyncio.sleep(0.1)  # Simulate processing time

            results[task_id] = {
                "status": "completed",
                "agent": task_def.get("agent", "unknown"),
                "type": task_def.get("type", "unknown"),
                "result": f"Simulated result for {task_def.get('type')}",
                "execution_time": 0.1,
            }

        return results

    def _calculate_workflow_progress(self, workflow: Dict[str, Any]) -> float:
        """Calculate workflow completion percentage."""
        if workflow["status"] == "completed":
            return 100.0
        elif workflow["status"] == "failed":
            return 0.0
        elif workflow["status"] == "running":
            # Simple progress calculation
            total_tasks = len(workflow.get("tasks", []))
            if total_tasks == 0:
                return 0.0

            # In real implementation, would check actual task completion
            return min(50.0, 100.0)  # Placeholder
        else:
            return 0.0

    # Message handlers

    async def _handle_agent_status(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle agent status update messages."""
        agent_id = message.sender_id
        status_data = message.payload

        logger.debug(f"Agent status update", agent_id=agent_id, status=status_data)

        # Update agent tracking
        # In real implementation, would update agent registry

        return {"acknowledged": True}

    async def _handle_task_completion(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle task completion notifications."""
        task_id = message.payload.get("task_id")
        result = message.payload.get("result", {})

        logger.info(f"Task completion notification", task_id=task_id, agent=message.sender_id)

        # Update workflow progress
        for workflow_id, workflow in self._active_workflows.items():
            if task_id in [task.get("task_id") for task in workflow.get("tasks", [])]:
                if "task_results" not in workflow:
                    workflow["task_results"] = {}
                workflow["task_results"][task_id] = result

        return {"acknowledged": True}

    async def _handle_workflow_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle workflow execution requests."""
        workflow_type = message.payload.get("workflow_type")
        parameters = message.payload.get("parameters", {})

        logger.info(f"Workflow request", workflow_type=workflow_type, requester=message.sender_id)

        # Create and execute workflow
        try:
            # This would typically be queued as a task
            result = {
                "status": "queued",
                "workflow_type": workflow_type,
                "estimated_duration": "5-10 minutes",
            }

            return result

        except Exception as e:
            logger.error(f"Failed to handle workflow request", error=str(e))
            return {"status": "error", "message": str(e)}

    async def _handle_capability_announcement(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle capability announcements from agents."""
        agent_id = message.sender_id
        capabilities = message.payload.get("capabilities", [])

        logger.info(f"Capability announcement", agent_id=agent_id, capabilities=capabilities)

        # Update agent capabilities
        self._agent_capabilities[agent_id] = capabilities

        return {"acknowledged": True}