#!/usr/bin/env python3
"""
Agent selector with multi-factor scoring.

Intelligently selects optimal agents for tasks using:
- Specialization matching (40%)
- Performance history (30%)
- Workload balancing (20%)
- Cost optimization (10%)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .performance_tracker import AgentMetrics, PerformanceTracker
from .task_graph import Task

logger = logging.getLogger(__name__)

# Scoring weights (must sum to 1.0)
SPECIALIZATION_WEIGHT = 0.4
PERFORMANCE_WEIGHT = 0.3
WORKLOAD_WEIGHT = 0.2
COST_WEIGHT = 0.1


@dataclass
class AgentAssignment:
    """Result of agent selection with confidence score."""

    agent_id: str
    confidence: float  # 0.0-1.0
    score_breakdown: Dict[str, float]  # Individual factor scores
    alternatives: List[str] = field(default_factory=list)  # Top 2-3 alternatives
    reasoning: str = ""  # Why this agent was selected


class AgentSelector:
    """
    Select optimal agent for each task using multi-factor analysis.

    Uses weighted scoring across four dimensions:
    - Specialization match (40%): Language/framework/domain alignment
    - Performance history (30%): Success rate and execution speed
    - Workload balance (20%): Current agent availability
    - Cost optimization (10%): Resource efficiency

    Example:
        >>> tracker = PerformanceTracker()
        >>> selector = AgentSelector(tracker)
        >>> task = Task(
        ...     name="Build REST API",
        ...     agent_id="",
        ...     metadata={"type": "backend", "language": "python", "framework": "fastapi"}
        ... )
        >>> assignment = selector.select_agent(task)
        >>> print(f"Selected: {assignment.agent_id} ({assignment.confidence:.1%})")
    """

    def __init__(
        self, tracker: PerformanceTracker, agents_dir: str = ".ai-workspace/agents"
    ):
        """
        Initialize agent selector.

        Args:
            tracker: PerformanceTracker for historical performance
            agents_dir: Directory containing agent definitions
        """
        self.tracker = tracker
        self.agents_dir = agents_dir
        self.agent_capabilities = self._load_agent_capabilities()
        logger.info(
            f"AgentSelector initialized with {len(self.agent_capabilities)} agents"
        )

    def _load_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Load agent metadata from agents directory.

        Scans .ai-workspace/agents/ for agent files and extracts:
        - Specializations (languages, frameworks, domains)
        - Categories (00-orchestration, 01-development, etc.)
        - Descriptions and capabilities

        Returns:
            Dict mapping agent_id to capabilities dict
        """
        capabilities = {}
        agents_path = Path(self.agents_dir)

        if not agents_path.exists():
            logger.warning(f"Agents directory not found: {agents_path}")
            return capabilities

        # Scan all subdirectories for .md files
        for agent_file in agents_path.rglob("*.md"):
            try:
                agent_data = self._parse_agent_file(agent_file)
                if agent_data:
                    agent_id = agent_data["agent_id"]
                    capabilities[agent_id] = agent_data
                    logger.debug(f"Loaded agent: {agent_id}")
            except Exception as e:
                logger.error(f"Error parsing {agent_file}: {e}")

        return capabilities

    def _parse_agent_file(self, agent_file: Path) -> Optional[Dict[str, Any]]:
        """
        Parse agent markdown file to extract metadata.

        Expected format:
        ---
        name: agent-name
        description: Agent description
        tools: tool1, tool2, tool3
        ---

        Args:
            agent_file: Path to agent markdown file

        Returns:
            Dict with agent metadata, or None if parsing fails
        """
        try:
            content = agent_file.read_text(encoding="utf-8")

            # Extract YAML frontmatter
            frontmatter_match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if not frontmatter_match:
                return None

            frontmatter = frontmatter_match.group(1)

            # Extract fields
            name_match = re.search(r"name:\s*(.+)", frontmatter)
            desc_match = re.search(r"description:\s*(.+)", frontmatter)
            tools_match = re.search(r"tools:\s*(.+)", frontmatter)

            if not name_match:
                return None

            agent_id = name_match.group(1).strip()
            description = desc_match.group(1).strip() if desc_match else ""
            tools = (
                [t.strip() for t in tools_match.group(1).split(",")]
                if tools_match
                else []
            )

            # Extract category from path
            category = "unknown"
            for part in agent_file.parts:
                if re.match(r"\d{2}-", part):
                    category = part
                    break

            # Extract specializations from description and content
            specializations = self._extract_specializations(description, content)

            return {
                "agent_id": agent_id,
                "description": description,
                "tools": tools,
                "category": category,
                "specializations": specializations,
                "file_path": str(agent_file),
            }

        except Exception as e:
            logger.error(f"Error parsing agent file {agent_file}: {e}")
            return None

    def _extract_specializations(
        self, description: str, content: str
    ) -> Dict[str, List[str]]:
        """
        Extract specializations from agent description and content.

        Args:
            description: Agent description
            content: Full agent content

        Returns:
            Dict with languages, frameworks, and domains
        """
        text = (description + " " + content).lower()

        # Known languages
        languages = []
        for lang in [
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "c++",
            "csharp",
            "ruby",
            "php",
        ]:
            if lang in text:
                languages.append(lang)

        # Known frameworks
        frameworks = []
        framework_patterns = [
            "fastapi",
            "django",
            "flask",
            "express",
            "react",
            "vue",
            "angular",
            "nextjs",
            "next.js",
            "spring",
            "spring boot",
            "laravel",
            "rails",
            "actix",
            "gin",
            "echo",
            "nest",
            "nuxt",
            "svelte",
            "kubernetes",
            "docker",
            "terraform",
            "ansible",
        ]
        for framework in framework_patterns:
            if framework in text:
                frameworks.append(framework)

        # Known domains
        domains = []
        domain_patterns = [
            "backend",
            "frontend",
            "fullstack",
            "mobile",
            "devops",
            "database",
            "api",
            "microservices",
            "testing",
            "security",
            "performance",
            "cloud",
            "data",
            "ml",
            "ai",
            "blockchain",
            "iot",
            "fintech",
        ]
        for domain in domain_patterns:
            if domain in text:
                domains.append(domain)

        return {
            "languages": languages,
            "frameworks": frameworks,
            "domains": domains,
        }

    def select_agent(self, task: Task) -> AgentAssignment:
        """
        Select optimal agent using multi-factor scoring.

        Scoring formula:
        - Specialization match: 40%
        - Performance history: 30%
        - Workload balance: 20%
        - Cost optimization: 10%

        Args:
            task: Task to assign

        Returns:
            AgentAssignment with selected agent and confidence
        """
        candidates = self._get_candidate_agents(task)

        if not candidates:
            logger.warning(f"No candidate agents found for task: {task.name}")
            # Return default assignment with low confidence
            return AgentAssignment(
                agent_id="orchestrator",
                confidence=0.0,
                score_breakdown={},
                alternatives=[],
                reasoning="No suitable agents found; defaulting to orchestrator",
            )

        # Calculate scores for all candidates
        scores = {}
        breakdowns = {}

        for agent_id in candidates:
            specialization = self._specialization_score(agent_id, task)
            performance = self._performance_score(agent_id, task)
            workload = self._workload_score(agent_id)
            cost = self._cost_score(agent_id, task)

            total_score = (
                specialization * SPECIALIZATION_WEIGHT
                + performance * PERFORMANCE_WEIGHT
                + workload * WORKLOAD_WEIGHT
                + cost * COST_WEIGHT
            )

            scores[agent_id] = total_score
            breakdowns[agent_id] = {
                "specialization": specialization,
                "performance": performance,
                "workload": workload,
                "cost": cost,
            }

        # Select best agent
        best_agent = max(scores, key=scores.get)
        confidence = scores[best_agent]

        # Get top 3 alternatives
        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = [agent for agent, _ in sorted_agents[1:4]]

        # Build reasoning
        breakdown = breakdowns[best_agent]
        reasoning = self._build_reasoning(best_agent, breakdown, task)

        logger.info(
            f"Selected {best_agent} for task '{task.name}' "
            f"with confidence {confidence:.1%}"
        )

        return AgentAssignment(
            agent_id=best_agent,
            confidence=confidence,
            score_breakdown=breakdown,
            alternatives=alternatives,
            reasoning=reasoning,
        )

    def _get_candidate_agents(self, task: Task) -> List[str]:
        """
        Get candidate agents for task.

        Filters agents based on:
        - Task type matches agent category
        - Agent has required capabilities
        - Agent is not explicitly excluded

        Returns:
            List of candidate agent IDs
        """
        candidates = []
        task_metadata = task.metadata or {}

        # Extract task requirements
        task_type = task_metadata.get("type", "general")
        task_language = task_metadata.get("language", "").lower()
        task_framework = task_metadata.get("framework", "").lower()
        task_domain = task_metadata.get("domain", "").lower()

        for agent_id, capabilities in self.agent_capabilities.items():
            # Check if agent matches any task requirements
            specs = capabilities["specializations"]

            # Language match
            if task_language and task_language in specs["languages"]:
                candidates.append(agent_id)
                continue

            # Framework match
            if task_framework and task_framework in specs["frameworks"]:
                candidates.append(agent_id)
                continue

            # Domain match
            if task_domain and task_domain in specs["domains"]:
                candidates.append(agent_id)
                continue

            # Task type in description (fallback)
            if task_type in capabilities["description"].lower():
                candidates.append(agent_id)
                continue

        # If no specific matches, return all agents as candidates
        if not candidates:
            candidates = list(self.agent_capabilities.keys())

        logger.debug(f"Found {len(candidates)} candidate agents for task '{task.name}'")
        return candidates

    def _specialization_score(self, agent_id: str, task: Task) -> float:
        """
        Calculate specialization match score (0.0-1.0).

        Checks:
        - Language match (Python task → python-pro: 1.0)
        - Framework match (Django task → django-developer: 1.0)
        - Domain match (API design → api-designer: 1.0)
        - Partial matches get proportional scores

        Args:
            agent_id: Agent to score
            task: Task to match against

        Returns:
            Specialization score 0.0-1.0
        """
        if agent_id not in self.agent_capabilities:
            return 0.0

        capabilities = self.agent_capabilities[agent_id]
        specs = capabilities["specializations"]
        task_metadata = task.metadata or {}

        # Extract task requirements
        task_language = task_metadata.get("language", "").lower()
        task_framework = task_metadata.get("framework", "").lower()
        task_domain = task_metadata.get("domain", "").lower()

        score = 0.0
        matches = 0
        total_checks = 0

        # Check language match
        if task_language:
            total_checks += 1
            if task_language in specs["languages"]:
                score += 1.0
                matches += 1

        # Check framework match
        if task_framework:
            total_checks += 1
            if task_framework in specs["frameworks"]:
                score += 1.0
                matches += 1

        # Check domain match
        if task_domain:
            total_checks += 1
            if task_domain in specs["domains"]:
                score += 1.0
                matches += 1

        # Check if agent name matches task type
        if agent_id.replace("-", " ") in task.name.lower():
            score += 0.5

        # Check if task name appears in agent description
        task_words = set(task.name.lower().split())
        desc_words = set(capabilities["description"].lower().split())
        common_words = task_words & desc_words
        if common_words:
            score += 0.3 * (len(common_words) / len(task_words))

        # Normalize score
        if total_checks > 0:
            normalized_score = score / (total_checks + 1)  # +1 for name/desc bonus
        else:
            normalized_score = min(score, 1.0)

        return min(normalized_score, 1.0)

    def _performance_score(self, agent_id: str, task: Task) -> float:
        """
        Calculate performance score from history (0.0-1.0).

        Uses PerformanceTracker metrics:
        - Success rate for similar tasks
        - Average execution time (normalized)
        - Output quality scores

        Formula: success_rate * (1 - normalized_time) * quality

        Args:
            agent_id: Agent to score
            task: Task type for filtering history

        Returns:
            Performance score 0.0-1.0
        """
        task_type = (
            task.metadata.get("type", "general") if task.metadata else "general"
        )

        # Get metrics for similar tasks
        metrics: AgentMetrics = self.tracker.get_agent_metrics(
            agent_id=agent_id, task_type=task_type, days=30
        )

        if metrics.total_executions == 0:
            # No history: return neutral score (slight penalty for unknown)
            return 0.6

        # Calculate time efficiency (normalize to 5 minutes)
        max_duration = 300  # 5 minutes in seconds
        time_factor = min(metrics.avg_duration_seconds / max_duration, 1.0)

        # Calculate quality factor (use avg quality or default to 0.7)
        quality_factor = (
            metrics.avg_quality_score if metrics.avg_quality_score > 0 else 0.7
        )

        # Combine factors
        performance = metrics.success_rate * (1 - time_factor * 0.3) * quality_factor

        return min(performance, 1.0)

    def _workload_score(self, agent_id: str) -> float:
        """
        Calculate workload balance score (0.0-1.0).

        Checks agent's current load:
        - Number of active tasks
        - Recent task count (last hour)
        - Resource utilization

        Score = 1.0 - (current_load / max_load)

        Args:
            agent_id: Agent to score

        Returns:
            Workload score 0.0-1.0 (1.0 = fully available)
        """
        # Check recent activity (last hour) as proxy for workload
        try:
            cursor = self.tracker.db.cursor()
            one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()

            query = """
                SELECT COUNT(*) FROM agent_executions
                WHERE agent_id = ? AND start_time >= ?
            """
            cursor.execute(query, (agent_id, one_hour_ago))
            recent_tasks = cursor.fetchone()[0]

            # Normalize: 0 tasks = 1.0, 10+ tasks = 0.5
            max_load = 10
            load_factor = min(recent_tasks / max_load, 0.5)
            score = 1.0 - load_factor

            return score

        except Exception as e:
            logger.error(f"Error calculating workload for {agent_id}: {e}")
            return 1.0  # Default to available

    def _cost_score(self, agent_id: str, task: Task) -> float:
        """
        Calculate cost efficiency score (0.0-1.0).

        Based on:
        - Average execution time for similar tasks
        - Resource usage patterns
        - Prefer faster agents when quality equivalent

        Args:
            agent_id: Agent to score
            task: Task for cost estimation

        Returns:
            Cost score 0.0-1.0 (1.0 = most efficient)
        """
        task_type = (
            task.metadata.get("type", "general") if task.metadata else "general"
        )

        # Get metrics for similar tasks
        metrics: AgentMetrics = self.tracker.get_agent_metrics(
            agent_id=agent_id, task_type=task_type, days=30
        )

        if metrics.total_executions == 0:
            # No history: return neutral cost score
            return 0.7

        # Calculate time efficiency (faster = better)
        # Normalize to 5 minutes: 0s = 1.0, 5min = 0.5, 10min+ = 0.0
        max_duration = 300  # 5 minutes
        time_efficiency = max(0.0, 1.0 - (metrics.avg_duration_seconds / max_duration))

        # Calculate resource efficiency (lower CPU/memory = better)
        # Normalize CPU: 0% = 1.0, 50% = 0.5, 100% = 0.0
        cpu_efficiency = max(0.0, 1.0 - (metrics.avg_cpu_percent / 100))

        # Normalize memory: 0MB = 1.0, 500MB = 0.5, 1GB+ = 0.0
        memory_efficiency = max(0.0, 1.0 - (metrics.avg_memory_mb / 1024))

        # Combine factors (time is most important for cost)
        cost_score = (
            time_efficiency * 0.6 + cpu_efficiency * 0.2 + memory_efficiency * 0.2
        )

        return min(cost_score, 1.0)

    def _build_reasoning(
        self, agent_id: str, breakdown: Dict[str, float], task: Task
    ) -> str:
        """
        Build human-readable reasoning for agent selection.

        Args:
            agent_id: Selected agent
            breakdown: Score breakdown
            task: Task being assigned

        Returns:
            Reasoning string
        """
        reasons = []

        # Find strongest factor
        max_factor = max(breakdown.items(), key=lambda x: x[1])
        factor_name, factor_score = max_factor

        if factor_score > 0.8:
            reasons.append(f"Excellent {factor_name} match ({factor_score:.1%})")
        elif factor_score > 0.6:
            reasons.append(f"Good {factor_name} match ({factor_score:.1%})")

        # Add specific reasons
        if breakdown["specialization"] > 0.7:
            reasons.append("Strong domain expertise")

        if breakdown["performance"] > 0.7:
            reasons.append("Proven track record")

        if breakdown["workload"] > 0.8:
            reasons.append("High availability")

        if breakdown["cost"] > 0.7:
            reasons.append("Cost-efficient")

        if not reasons:
            reasons.append("Best available option")

        return f"Selected {agent_id}: {', '.join(reasons)}"

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent capabilities dict, or None if not found
        """
        return self.agent_capabilities.get(agent_id)

    def list_agents_by_category(self, category: str) -> List[str]:
        """
        List all agents in a specific category.

        Args:
            category: Category name (e.g., "01-development")

        Returns:
            List of agent IDs in the category
        """
        return [
            agent_id
            for agent_id, caps in self.agent_capabilities.items()
            if caps["category"] == category
        ]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AgentSelector(agents={len(self.agent_capabilities)}, "
            f"dir={self.agents_dir})"
        )
