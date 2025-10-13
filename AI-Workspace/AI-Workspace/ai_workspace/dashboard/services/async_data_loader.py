"""Async data loader for AI Workspace Dashboard.

Non-blocking I/O operations for loading workspace data with parallel processing
to eliminate blocking and improve concurrent request handling.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ReuseCheck:
    """Represents a reuse check record."""

    task_name: str
    search_completed: bool
    reuse_analyzed: bool
    reuse_percentage: float
    timestamp: str
    files_checked: List[str]
    approved_to_create: bool


@dataclass
class PhaseGate:
    """Represents a phase gate record."""

    phase: str
    task_name: str
    timestamp: str
    artifacts: Dict[str, Any]
    exit_criteria_met: List[bool]
    approved_by: str
    approval_timestamp: Optional[str]


class AsyncDataLoader:
    """Async data loader with non-blocking I/O.

    Provides parallel loading capabilities to eliminate blocking operations
    and improve dashboard performance under concurrent load.
    """

    def __init__(
        self, workspace_root: Optional[Path] = None, max_workers: int = 4
    ) -> None:
        """Initialize async data loader.

        Args:
            workspace_root: Root directory of workspace (defaults to current directory)
            max_workers: Maximum number of thread pool workers for CPU-bound operations
        """
        self.root = workspace_root or Path.cwd()
        self.workspace = self.root / ".ai-workspace"
        self.agent_workspace = self.root / ".agent-workspace"
        self.cache_dir = self.workspace / "cache"
        self.handoffs_dir = self.agent_workspace / "handoffs"
        self.decisions_dir = self.agent_workspace / "decisions"

        # Thread pool for CPU-bound parsing operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def load_reuse_checks_async(self) -> List[ReuseCheck]:
        """Load reuse check history asynchronously.

        Returns:
            List of reuse check records
        """
        cache_file = self.cache_dir / "reuse_checks.json"

        if not cache_file.exists():
            return []

        try:
            # Async file read
            async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                content = await f.read()

            # Parse JSON in thread pool (CPU-bound)
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(self.executor, json.loads, content)

            # Convert to ReuseCheck objects
            checks = []
            for task_name, check_data in data.items():
                checks.append(ReuseCheck(**check_data))

            return sorted(checks, key=lambda x: x.timestamp, reverse=True)

        except Exception as e:
            logger.error(f"Error loading reuse checks: {e}", exc_info=True)
            return []

    async def load_phase_gates_async(
        self, task_name: Optional[str] = None
    ) -> List[PhaseGate]:
        """Load phase gate records asynchronously.

        Args:
            task_name: Optional task filter

        Returns:
            List of phase gate records
        """
        if not self.handoffs_dir.exists():
            return []

        try:
            pattern = f"gate-*-{task_name}.json" if task_name else "gate-*.json"
            gate_files = list(self.handoffs_dir.glob(pattern))

            # Load all gate files in parallel
            tasks = [self._load_gate_file(gate_file) for gate_file in gate_files]
            gates = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out errors and None values
            valid_gates = [g for g in gates if isinstance(g, PhaseGate)]

            return sorted(valid_gates, key=lambda x: x.timestamp, reverse=True)

        except Exception as e:
            logger.error(f"Error loading phase gates: {e}", exc_info=True)
            return []

    async def _load_gate_file(self, gate_file: Path) -> Optional[PhaseGate]:
        """Load a single gate file asynchronously.

        Args:
            gate_file: Path to gate file

        Returns:
            PhaseGate object or None on error
        """
        try:
            async with aiofiles.open(gate_file, "r", encoding="utf-8") as f:
                content = await f.read()

            # Parse JSON in thread pool
            loop = asyncio.get_event_loop()
            gate_data = await loop.run_in_executor(self.executor, json.loads, content)

            return PhaseGate(
                phase=gate_data.get("phase", "unknown"),
                task_name=gate_data.get("task_name", "default"),
                timestamp=gate_data.get("timestamp", ""),
                artifacts=gate_data.get("artifacts", {}),
                exit_criteria_met=gate_data.get("exit_criteria_met", []),
                approved_by=gate_data.get("approved_by", ""),
                approval_timestamp=gate_data.get("approval_timestamp"),
            )

        except Exception as e:
            logger.error(f"Error loading gate file {gate_file}: {e}")
            return None

    async def load_stack_config_async(self) -> Optional[Dict[str, Any]]:
        """Load detected stack configuration asynchronously.

        Returns:
            Stack configuration dictionary or None
        """
        config_file = self.root / ".ai-workspace-config.yml"

        if not config_file.exists():
            return None

        try:
            async with aiofiles.open(config_file, "r", encoding="utf-8") as f:
                content = await f.read()

            # Parse YAML in thread pool
            loop = asyncio.get_event_loop()
            config = await loop.run_in_executor(
                self.executor, yaml.safe_load, content
            )

            return config

        except Exception as e:
            logger.error(f"Error loading stack config: {e}", exc_info=True)
            return None

    async def load_adrs_async(self) -> List[Dict[str, str]]:
        """Load architectural decision records asynchronously.

        Returns:
            List of ADR metadata
        """
        if not self.decisions_dir.exists():
            return []

        try:
            adr_files = list(self.decisions_dir.glob("adr-*.md"))

            # Load all ADR files in parallel
            tasks = [self._load_adr_file(adr_file) for adr_file in adr_files]
            adrs = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out errors and None values
            valid_adrs = [a for a in adrs if isinstance(a, dict)]

            return sorted(valid_adrs, key=lambda x: x.get("date", ""), reverse=True)

        except Exception as e:
            logger.error(f"Error loading ADRs: {e}", exc_info=True)
            return []

    async def _load_adr_file(self, adr_file: Path) -> Optional[Dict[str, str]]:
        """Load a single ADR file asynchronously.

        Args:
            adr_file: Path to ADR file

        Returns:
            ADR metadata dictionary or None on error
        """
        try:
            async with aiofiles.open(adr_file, "r", encoding="utf-8") as f:
                # Read first 20 lines for metadata
                lines = []
                async for line in f:
                    lines.append(line)
                    if len(lines) >= 20:
                        break

            # Parse metadata
            title = ""
            status = ""
            date = ""

            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                elif "Status:" in line:
                    status = line.split("Status:")[1].strip()
                elif "Date:" in line:
                    date = line.split("Date:")[1].strip()

            return {
                "file": adr_file.name,
                "path": str(adr_file),
                "title": title or adr_file.stem,
                "status": status or "Unknown",
                "date": date or "",
            }

        except Exception as e:
            logger.error(f"Error loading ADR file {adr_file}: {e}")
            return None

    async def get_project_info_async(self) -> Dict[str, Any]:
        """Get basic project information asynchronously.

        Returns:
            Dictionary with project metadata
        """
        config = await self.load_stack_config_async()

        if not config:
            return {
                "name": self.root.name,
                "type": "unknown",
                "primary_language": "unknown",
                "detected_stack": {},
            }

        project_data = config.get("project", {})

        return {
            "name": self.root.name,
            "type": project_data.get("project_type", "unknown"),
            "primary_language": project_data.get("primary_language", "unknown"),
            "detected_stack": project_data.get("detected_stack", {}),
            "detection_date": project_data.get("detection_date", ""),
        }

    async def get_workspace_stats_async(self) -> Dict[str, int]:
        """Get workspace statistics asynchronously.

        Returns:
            Dictionary with counts of various workspace items
        """
        # Run all counting operations in parallel
        reuse_checks_task = self.load_reuse_checks_async()
        phase_gates_task = self.load_phase_gates_async()
        adrs_task = self.load_adrs_async()

        # Count agents and scripts in thread pool (directory operations)
        loop = asyncio.get_event_loop()
        agents_task = loop.run_in_executor(self.executor, self._count_agents)
        scripts_task = loop.run_in_executor(self.executor, self._count_scripts)

        # Wait for all tasks
        reuse_checks, phase_gates, adrs, agents_count, scripts_count = await asyncio.gather(
            reuse_checks_task,
            phase_gates_task,
            adrs_task,
            agents_task,
            scripts_task,
            return_exceptions=True,
        )

        return {
            "total_reuse_checks": len(reuse_checks) if isinstance(reuse_checks, list) else 0,
            "total_phase_gates": len(phase_gates) if isinstance(phase_gates, list) else 0,
            "total_adrs": len(adrs) if isinstance(adrs, list) else 0,
            "agents_available": agents_count if isinstance(agents_count, int) else 0,
            "scripts_available": scripts_count if isinstance(scripts_count, int) else 0,
        }

    def _count_agents(self) -> int:
        """Count agent files (runs in thread pool)."""
        agents_dir = self.workspace / "agents"
        if agents_dir.exists():
            return len(list(agents_dir.rglob("*.md")))
        return 0

    def _count_scripts(self) -> int:
        """Count script files (runs in thread pool)."""
        scripts_dir = self.workspace / "scripts"
        if scripts_dir.exists():
            return len(list(scripts_dir.glob("*.py")))
        return 0

    async def load_agents_async(self, agent_files: List[Path]) -> List[Dict[str, Any]]:
        """Load agent files in parallel without blocking.

        Args:
            agent_files: List of agent file paths

        Returns:
            List of parsed agent dictionaries
        """
        tasks = [self._load_agent_file(f) for f in agent_files]
        agents = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        return [a for a in agents if isinstance(a, dict)]

    async def _load_agent_file(self, path: Path) -> Dict[str, Any]:
        """Load single agent file asynchronously.

        Args:
            path: Path to agent file

        Returns:
            Agent metadata dictionary
        """
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()

            # Parse agent metadata in thread pool
            loop = asyncio.get_event_loop()
            agent = await loop.run_in_executor(
                self.executor, self._parse_agent_metadata, content, path
            )

            return agent

        except Exception as e:
            logger.error(f"Error loading agent file {path}: {e}")
            return {}

    def _parse_agent_metadata(self, content: str, path: Path) -> Dict[str, Any]:
        """Parse agent metadata (runs in thread pool).

        Args:
            content: File content
            path: File path

        Returns:
            Agent metadata dictionary
        """
        import re

        metadata = {
            "name": "",
            "description": "",
            "tools": "",
            "model": "",
            "file_path": str(path),
            "file_name": path.stem,
            "category": path.parent.name,
            "content": content[:500],
        }

        # Extract YAML frontmatter
        yaml_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)

            # Simple YAML parsing
            for line in yaml_content.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key in ["name", "description", "tools", "model"]:
                        metadata[key] = value

        return metadata

    async def close(self) -> None:
        """Clean up resources."""
        self.executor.shutdown(wait=False)
