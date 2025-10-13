"""Read-only data loader for AI Workspace dashboard.

This module provides services to load data from workspace files
WITHOUT modifying any core system files.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


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


class WorkspaceDataLoader:
    """Read-only data loader for dashboard.

    NEVER modifies core workspace files. All operations are read-only.
    """

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        """Initialize data loader.

        Args:
            workspace_root: Root directory of workspace (defaults to current directory)
        """
        self.root = workspace_root or Path.cwd()
        self.workspace = self.root / ".ai-workspace"
        self.agent_workspace = self.root / ".agent-workspace"
        self.cache_dir = self.workspace / "cache"
        self.handoffs_dir = self.agent_workspace / "handoffs"
        self.decisions_dir = self.agent_workspace / "decisions"

    def load_reuse_checks(self) -> List[ReuseCheck]:
        """Load reuse check history from cache.

        Returns:
            List of reuse check records
        """
        cache_file = self.cache_dir / "reuse_checks.json"

        if not cache_file.exists():
            return []

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            checks = []
            for task_name, check_data in data.items():
                checks.append(ReuseCheck(**check_data))

            return sorted(checks, key=lambda x: x.timestamp, reverse=True)

        except Exception as e:
            print(f"Error loading reuse checks: {e}")
            return []

    def load_phase_gates(self, task_name: Optional[str] = None) -> List[PhaseGate]:
        """Load phase gate records.

        Args:
            task_name: Optional task filter

        Returns:
            List of phase gate records
        """
        if not self.handoffs_dir.exists():
            return []

        gates = []

        try:
            pattern = f"gate-*-{task_name}.json" if task_name else "gate-*.json"

            for gate_file in self.handoffs_dir.glob(pattern):
                with open(gate_file, "r", encoding="utf-8") as f:
                    gate_data = json.load(f)

                gates.append(
                    PhaseGate(
                        phase=gate_data.get("phase", "unknown"),
                        task_name=gate_data.get("task_name", "default"),
                        timestamp=gate_data.get("timestamp", ""),
                        artifacts=gate_data.get("artifacts", {}),
                        exit_criteria_met=gate_data.get("exit_criteria_met", []),
                        approved_by=gate_data.get("approved_by", ""),
                        approval_timestamp=gate_data.get("approval_timestamp"),
                    )
                )

            return sorted(gates, key=lambda x: x.timestamp, reverse=True)

        except Exception as e:
            print(f"Error loading phase gates: {e}")
            return []

    def load_stack_config(self) -> Optional[Dict[str, Any]]:
        """Load detected stack configuration.

        Returns:
            Stack configuration dictionary or None
        """
        config_file = self.root / ".ai-workspace-config.yml"

        if not config_file.exists():
            return None

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading stack config: {e}")
            return None

    def load_adrs(self) -> List[Dict[str, str]]:
        """Load architectural decision records.

        Returns:
            List of ADR metadata
        """
        if not self.decisions_dir.exists():
            return []

        adrs = []

        try:
            for adr_file in self.decisions_dir.glob("adr-*.md"):
                # Read first few lines for metadata
                with open(adr_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()[:20]  # First 20 lines for metadata

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

                adrs.append(
                    {
                        "file": adr_file.name,
                        "path": str(adr_file),
                        "title": title or adr_file.stem,
                        "status": status or "Unknown",
                        "date": date or "",
                    }
                )

            return sorted(adrs, key=lambda x: x["date"], reverse=True)

        except Exception as e:
            print(f"Error loading ADRs: {e}")
            return []

    def get_project_info(self) -> Dict[str, Any]:
        """Get basic project information.

        Returns:
            Dictionary with project metadata
        """
        config = self.load_stack_config()

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

    def get_workspace_stats(self) -> Dict[str, int]:
        """Get workspace statistics.

        Returns:
            Dictionary with counts of various workspace items
        """
        stats = {
            "total_reuse_checks": 0,
            "total_phase_gates": 0,
            "total_adrs": 0,
            "agents_available": 0,
            "scripts_available": 0,
        }

        # Count reuse checks
        stats["total_reuse_checks"] = len(self.load_reuse_checks())

        # Count phase gates
        stats["total_phase_gates"] = len(self.load_phase_gates())

        # Count ADRs
        stats["total_adrs"] = len(self.load_adrs())

        # Count agents
        agents_dir = self.workspace / "agents"
        if agents_dir.exists():
            stats["agents_available"] = len(list(agents_dir.rglob("*.md")))

        # Count scripts
        scripts_dir = self.workspace / "scripts"
        if scripts_dir.exists():
            stats["scripts_available"] = len(list(scripts_dir.glob("*.py")))

        return stats
