"""Typed data models for AI Workspace Dashboard.

This module provides strongly-typed dataclasses and TypedDict definitions
for all dashboard data structures with full validation and serialization support.

All models follow Python 3.13+ type hint standards with complete PEP 484 compliance.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import yaml


# =============================================================================
# TypedDict Definitions (JSON Schemas)
# =============================================================================


class ReuseAnalysisDict(TypedDict, total=False):
    """TypedDict for reuse analysis JSON structure.

    Represents the structure of data stored in .ai-workspace/cache/reuse_checks.json.

    Attributes:
        task_name: Name of the task being analyzed
        search_completed: Whether codebase search was performed
        reuse_analyzed: Whether reuse analysis was completed
        reuse_percentage: Percentage of reusable code found (0-100)
        timestamp: ISO 8601 timestamp of the analysis
        files_checked: List of file paths that were analyzed
        approved_to_create: Whether new code creation was approved
    """

    task_name: str
    search_completed: bool
    reuse_analyzed: bool
    reuse_percentage: float
    timestamp: str
    files_checked: List[str]
    approved_to_create: bool


class PhaseGateDict(TypedDict, total=False):
    """TypedDict for phase gate JSON structure.

    Represents the structure of gate-*.json files in .agent-workspace/handoffs/.

    Attributes:
        phase: Development phase (discovery, design, implementation, verification, integration)
        task_name: Name of the task this gate applies to
        timestamp: ISO 8601 timestamp of gate creation
        artifacts: Dictionary of required artifacts and their status
        exit_criteria_met: List of boolean values indicating criteria completion
        approved_by: Agent or user who approved the gate
        approval_timestamp: ISO 8601 timestamp of approval (optional)
    """

    phase: str
    task_name: str
    timestamp: str
    artifacts: Dict[str, Any]
    exit_criteria_met: List[bool]
    approved_by: str
    approval_timestamp: Optional[str]


class ProjectManifestDict(TypedDict, total=False):
    """TypedDict for project manifest JSON structure.

    Represents the structure of .agent-workspace/manifest.json.

    Attributes:
        project: Project name
        version: Semantic version string
        protocol: Protocol version (e.g., "v2")
        initiated: ISO 8601 timestamp of project initiation
        description: Project description
        location: Absolute path to project directory
        agents: List of agent configurations
        handoffs: List of handoff records
        current_phase: Current development phase
        phase_gates: Dictionary of phase gate configurations
        metadata: Additional project metadata
    """

    project: str
    version: str
    protocol: str
    initiated: str
    description: str
    location: str
    agents: List[Dict[str, Any]]
    handoffs: List[Dict[str, Any]]
    current_phase: str
    phase_gates: Dict[str, Any]
    metadata: Dict[str, Any]


class MetricsSummaryDict(TypedDict, total=False):
    """TypedDict for aggregated metrics JSON structure.

    Represents calculated metrics from various data sources.

    Attributes:
        total_reuse_checks: Total number of reuse checks performed
        avg_reuse_percentage: Average reuse percentage across all checks
        total_phase_gates: Total number of phase gates
        completed_gates: Number of completed gates
        gate_completion_percentage: Percentage of gates completed
        quality_score: Overall quality score (0-100)
        reuse_compliance: Reuse principle compliance percentage
        check_before_create_compliance: Check-before-create compliance percentage
        gate_compliance: Phase gate compliance percentage
        primary_language: Primary programming language detected
        agents_used: List of agent names that have worked on the project
    """

    total_reuse_checks: int
    avg_reuse_percentage: float
    total_phase_gates: int
    completed_gates: int
    gate_completion_percentage: float
    quality_score: float
    reuse_compliance: float
    check_before_create_compliance: float
    gate_compliance: float
    primary_language: str
    agents_used: List[str]


# =============================================================================
# Dataclass Definitions (Domain Models)
# =============================================================================


@dataclass
class ReuseAnalysis:
    """Domain model for reuse analysis data.

    Represents a single reuse check operation with validation and serialization.

    Attributes:
        file: Source file path that was analyzed
        reuse_percentage: Percentage of reusable code found (0-100)
        matches: List of matching files/patterns found
        timestamp: When the analysis was performed
        task_name: Name of the task being analyzed
        search_completed: Whether codebase search was performed
        reuse_analyzed: Whether reuse analysis was completed
        approved_to_create: Whether new code creation was approved
        files_checked: List of file paths that were analyzed
    """

    file: str
    reuse_percentage: float
    matches: List[str]
    timestamp: datetime
    task_name: str = ""
    search_completed: bool = False
    reuse_analyzed: bool = False
    approved_to_create: bool = False
    files_checked: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate data after initialization.

        Raises:
            ValueError: If reuse_percentage is not in valid range or other validation fails
        """
        if not 0 <= self.reuse_percentage <= 100:
            raise ValueError(
                f"reuse_percentage must be between 0 and 100, got {self.reuse_percentage}"
            )

        if not self.file:
            raise ValueError("file path cannot be empty")

        if not isinstance(self.matches, list):
            raise ValueError("matches must be a list")

        # Convert timestamp string to datetime if needed
        if isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            except ValueError as e:
                raise ValueError(f"Invalid timestamp format: {e}") from e

    @classmethod
    def from_json(cls, file_path: Path) -> List["ReuseAnalysis"]:
        """Load reuse analysis records from JSON cache file.

        Args:
            file_path: Path to the reuse_checks.json file

        Returns:
            List of ReuseAnalysis instances

        Raises:
            FileNotFoundError: If the cache file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If data validation fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Reuse cache file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data: Dict[str, ReuseAnalysisDict] = json.load(f)

        analyses: List[ReuseAnalysis] = []

        for task_name, check_data in data.items():
            analyses.append(
                cls(
                    file=check_data.get("task_name", task_name),
                    reuse_percentage=check_data["reuse_percentage"],
                    matches=check_data.get("files_checked", []),
                    timestamp=check_data["timestamp"],
                    task_name=task_name,
                    search_completed=check_data.get("search_completed", False),
                    reuse_analyzed=check_data.get("reuse_analyzed", False),
                    approved_to_create=check_data.get("approved_to_create", False),
                    files_checked=check_data.get("files_checked", []),
                )
            )

        return analyses

    def to_dict(self) -> ReuseAnalysisDict:
        """Serialize to dictionary format.

        Returns:
            Dictionary representation compatible with ReuseAnalysisDict
        """
        return ReuseAnalysisDict(
            task_name=self.task_name,
            search_completed=self.search_completed,
            reuse_analyzed=self.reuse_analyzed,
            reuse_percentage=self.reuse_percentage,
            timestamp=self.timestamp.isoformat(),
            files_checked=self.files_checked,
            approved_to_create=self.approved_to_create,
        )

    def validate(self) -> bool:
        """Validate data integrity.

        Returns:
            True if all validation checks pass

        Raises:
            ValueError: If validation fails
        """
        if not self.task_name:
            raise ValueError("task_name cannot be empty")

        if self.reuse_analyzed and not self.search_completed:
            raise ValueError("Cannot have reuse_analyzed=True without search_completed=True")

        if self.approved_to_create and self.reuse_percentage < 70:
            # Warning: This is allowed but should be documented via ADR
            pass

        return True


@dataclass
class PhaseGate:
    """Domain model for phase gate data.

    Represents a single phase gate checkpoint with validation.

    Attributes:
        name: Human-readable name of the gate
        status: Current status (pending, in-progress, completed, failed)
        checklist_items: List of checklist items for this gate
        completion_date: When the gate was completed (None if not completed)
        phase: Development phase this gate belongs to
        task_name: Name of the task this gate applies to
        timestamp: When the gate was created
        artifacts: Dictionary of required artifacts and their status
        exit_criteria_met: List of boolean values indicating criteria completion
        approved_by: Agent or user who approved the gate
        approval_timestamp: When the gate was approved
    """

    name: str
    status: str
    checklist_items: List[str]
    completion_date: Optional[datetime] = None
    phase: str = "unknown"
    task_name: str = "default"
    timestamp: datetime = field(default_factory=datetime.now)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    exit_criteria_met: List[bool] = field(default_factory=list)
    approved_by: str = ""
    approval_timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Validate data after initialization.

        Raises:
            ValueError: If validation fails
        """
        valid_statuses = {"pending", "in-progress", "completed", "failed", "blocked"}
        if self.status not in valid_statuses:
            raise ValueError(
                f"Invalid status '{self.status}'. Must be one of: {valid_statuses}"
            )

        valid_phases = {"discovery", "design", "implementation", "verification", "integration"}
        if self.phase not in valid_phases and self.phase != "unknown":
            raise ValueError(
                f"Invalid phase '{self.phase}'. Must be one of: {valid_phases}"
            )

        # Convert string timestamps to datetime
        if isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            except ValueError as e:
                raise ValueError(f"Invalid timestamp format: {e}") from e

        if isinstance(self.completion_date, str):
            try:
                self.completion_date = datetime.fromisoformat(
                    self.completion_date.replace('Z', '+00:00')
                )
            except ValueError as e:
                raise ValueError(f"Invalid completion_date format: {e}") from e

        if isinstance(self.approval_timestamp, str):
            try:
                self.approval_timestamp = datetime.fromisoformat(
                    self.approval_timestamp.replace('Z', '+00:00')
                )
            except ValueError as e:
                raise ValueError(f"Invalid approval_timestamp format: {e}") from e

    @classmethod
    def from_json(cls, file_path: Path) -> "PhaseGate":
        """Load phase gate from JSON file.

        Args:
            file_path: Path to the gate-*.json file

        Returns:
            PhaseGate instance

        Raises:
            FileNotFoundError: If the gate file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If data validation fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Gate file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data: PhaseGateDict = json.load(f)

        # Determine status based on exit criteria
        exit_criteria = data.get("exit_criteria_met", [])
        if all(exit_criteria):
            status = "completed"
        elif any(exit_criteria):
            status = "in-progress"
        else:
            status = "pending"

        # Extract checklist items from artifacts
        artifacts = data.get("artifacts", {})
        checklist_items = list(artifacts.keys())

        return cls(
            name=f"{data['phase']}-{data['task_name']}",
            status=status,
            checklist_items=checklist_items,
            completion_date=data.get("approval_timestamp"),
            phase=data["phase"],
            task_name=data["task_name"],
            timestamp=data["timestamp"],
            artifacts=artifacts,
            exit_criteria_met=exit_criteria,
            approved_by=data.get("approved_by", ""),
            approval_timestamp=data.get("approval_timestamp"),
        )

    @classmethod
    def from_manifest(cls, manifest_data: Dict[str, Any], phase_name: str) -> "PhaseGate":
        """Create PhaseGate from manifest.json phase_gates section.

        Args:
            manifest_data: Dictionary from manifest.json
            phase_name: Name of the phase (e.g., "discovery", "design")

        Returns:
            PhaseGate instance

        Raises:
            ValueError: If phase_name not found in manifest or data is invalid
        """
        phase_gates = manifest_data.get("phase_gates", {})

        if phase_name not in phase_gates:
            raise ValueError(f"Phase '{phase_name}' not found in manifest")

        phase_data = phase_gates[phase_name]
        status = phase_data.get("status", "pending")

        # Extract artifacts as checklist items
        artifacts = phase_data.get("required_artifacts", [])
        checklist_items = []
        artifact_dict = {}

        for artifact in artifacts:
            if isinstance(artifact, str):
                # Check if it's marked as completed
                completed = "✓" in artifact or "completed" in artifact.lower()
                artifact_name = artifact.replace("✓", "").replace("(", "").replace(")", "").strip()
                checklist_items.append(artifact_name)
                artifact_dict[artifact_name] = {"completed": completed}

        return cls(
            name=f"{phase_name}-gate",
            status=status,
            checklist_items=checklist_items,
            completion_date=None if status != "completed" else datetime.now(),
            phase=phase_name,
            task_name=manifest_data.get("project", "default"),
            timestamp=datetime.now(),
            artifacts=artifact_dict,
            exit_criteria_met=[
                artifact_dict[item].get("completed", False)
                for item in checklist_items
            ],
            approved_by="",
            approval_timestamp=None,
        )

    def to_dict(self) -> PhaseGateDict:
        """Serialize to dictionary format.

        Returns:
            Dictionary representation compatible with PhaseGateDict
        """
        return PhaseGateDict(
            phase=self.phase,
            task_name=self.task_name,
            timestamp=self.timestamp.isoformat(),
            artifacts=self.artifacts,
            exit_criteria_met=self.exit_criteria_met,
            approved_by=self.approved_by,
            approval_timestamp=(
                self.approval_timestamp.isoformat()
                if self.approval_timestamp else None
            ),
        )

    def validate(self) -> bool:
        """Validate data integrity.

        Returns:
            True if all validation checks pass

        Raises:
            ValueError: If validation fails
        """
        if not self.name:
            raise ValueError("Gate name cannot be empty")

        if self.status == "completed" and not all(self.exit_criteria_met):
            raise ValueError("Cannot have status=completed without all exit criteria met")

        if self.approval_timestamp and not self.approved_by:
            raise ValueError("Cannot have approval_timestamp without approved_by")

        if len(self.exit_criteria_met) != len(self.checklist_items):
            raise ValueError(
                "Length of exit_criteria_met must match checklist_items "
                f"({len(self.exit_criteria_met)} vs {len(self.checklist_items)})"
            )

        return True

    @property
    def is_completed(self) -> bool:
        """Check if the gate is completed.

        Returns:
            True if all exit criteria are met
        """
        return all(self.exit_criteria_met) if self.exit_criteria_met else False

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage.

        Returns:
            Percentage of completed criteria (0-100)
        """
        if not self.exit_criteria_met:
            return 0.0

        completed = sum(1 for met in self.exit_criteria_met if met)
        total = len(self.exit_criteria_met)

        return (completed / total * 100) if total > 0 else 0.0


@dataclass
class ProjectManifest:
    """Domain model for project manifest data.

    Represents the complete project manifest with all metadata.

    Attributes:
        project_name: Name of the project
        primary_language: Primary programming language
        agents_used: List of agent names that have worked on the project
        gates: List of PhaseGate instances
        version: Semantic version string
        protocol: Protocol version
        initiated: When the project was initiated
        description: Project description
        location: Absolute path to project directory
        current_phase: Current development phase
        metadata: Additional metadata dictionary
    """

    project_name: str
    primary_language: str
    agents_used: List[str]
    gates: List[PhaseGate]
    version: str = "1.0.0"
    protocol: str = "v2"
    initiated: datetime = field(default_factory=datetime.now)
    description: str = ""
    location: str = ""
    current_phase: str = "discovery"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data after initialization.

        Raises:
            ValueError: If validation fails
        """
        if not self.project_name:
            raise ValueError("project_name cannot be empty")

        valid_protocols = {"v1", "v2", "v3"}
        if self.protocol not in valid_protocols:
            raise ValueError(
                f"Invalid protocol '{self.protocol}'. Must be one of: {valid_protocols}"
            )

        # Convert string timestamp to datetime
        if isinstance(self.initiated, str):
            try:
                self.initiated = datetime.fromisoformat(self.initiated.replace('Z', '+00:00'))
            except ValueError as e:
                raise ValueError(f"Invalid initiated timestamp: {e}") from e

    @classmethod
    def from_yaml(cls, file_path: Path) -> "ProjectManifest":
        """Load project manifest from YAML config file.

        Args:
            file_path: Path to the .ai-workspace-config.yml file

        Returns:
            ProjectManifest instance

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the file contains invalid YAML
            ValueError: If data validation fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        project_data = data.get("project", {})

        return cls(
            project_name=file_path.parent.name,
            primary_language=project_data.get("primary_language", "unknown"),
            agents_used=[],  # Will be populated from manifest.json if available
            gates=[],  # Will be populated from phase gate files
            version=data.get("version", "1.0.0"),
            protocol="v2",
            initiated=project_data.get("detection_date", datetime.now().isoformat()),
            description=project_data.get("project_type", ""),
            location=str(file_path.parent),
            current_phase="discovery",
            metadata=data,
        )

    @classmethod
    def from_json(cls, file_path: Path) -> "ProjectManifest":
        """Load project manifest from JSON manifest file.

        Args:
            file_path: Path to the manifest.json file

        Returns:
            ProjectManifest instance

        Raises:
            FileNotFoundError: If the manifest file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If data validation fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data: ProjectManifestDict = json.load(f)

        # Extract agent names
        agents = data.get("agents", [])
        agents_used = [agent.get("name", "") for agent in agents if agent.get("name")]

        # Create gates from phase_gates section
        gates: List[PhaseGate] = []
        phase_gates_data = data.get("phase_gates", {})

        for phase_name in ["discovery", "design", "implementation", "verification", "integration"]:
            if phase_name in phase_gates_data:
                try:
                    gate = PhaseGate.from_manifest(data, phase_name)
                    gates.append(gate)
                except ValueError:
                    # Skip invalid gates
                    pass

        # Determine primary language from metadata
        metadata = data.get("metadata", {})
        primary_language = "unknown"

        return cls(
            project_name=data.get("project", "unknown"),
            primary_language=primary_language,
            agents_used=agents_used,
            gates=gates,
            version=data.get("version", "1.0.0"),
            protocol=data.get("protocol", "v2"),
            initiated=data.get("initiated", datetime.now().isoformat()),
            description=data.get("description", ""),
            location=data.get("location", ""),
            current_phase=data.get("current_phase", "discovery"),
            metadata=metadata,
        )

    def to_dict(self) -> ProjectManifestDict:
        """Serialize to dictionary format.

        Returns:
            Dictionary representation compatible with ProjectManifestDict
        """
        return ProjectManifestDict(
            project=self.project_name,
            version=self.version,
            protocol=self.protocol,
            initiated=self.initiated.isoformat(),
            description=self.description,
            location=self.location,
            agents=[{"name": agent} for agent in self.agents_used],
            handoffs=[],
            current_phase=self.current_phase,
            phase_gates={
                gate.phase: gate.to_dict()
                for gate in self.gates
            },
            metadata=self.metadata,
        )

    def validate(self) -> bool:
        """Validate data integrity.

        Returns:
            True if all validation checks pass

        Raises:
            ValueError: If validation fails
        """
        if not self.project_name:
            raise ValueError("project_name cannot be empty")

        # Validate all gates
        for gate in self.gates:
            gate.validate()

        return True

    @property
    def total_gates(self) -> int:
        """Get total number of gates.

        Returns:
            Count of gates
        """
        return len(self.gates)

    @property
    def completed_gates(self) -> int:
        """Get number of completed gates.

        Returns:
            Count of completed gates
        """
        return sum(1 for gate in self.gates if gate.is_completed)

    @property
    def completion_percentage(self) -> float:
        """Calculate overall project completion percentage.

        Returns:
            Percentage of completed gates (0-100)
        """
        if not self.gates:
            return 0.0

        return (self.completed_gates / self.total_gates * 100)


@dataclass
class MetricsSummary:
    """Domain model for aggregated metrics.

    Aggregates reuse data, gates, and quality metrics.

    Attributes:
        total_reuse_checks: Total number of reuse checks performed
        avg_reuse_percentage: Average reuse percentage across all checks
        total_phase_gates: Total number of phase gates
        completed_gates: Number of completed gates
        gate_completion_percentage: Percentage of gates completed
        quality_score: Overall quality score (0-100)
        reuse_compliance: Reuse principle compliance percentage
        check_before_create_compliance: Check-before-create compliance percentage
        gate_compliance: Phase gate compliance percentage
        primary_language: Primary programming language detected
        agents_used: List of agent names that have worked on the project
        reuse_distribution: Distribution of reuse percentages
        recent_activity: List of recent activities
    """

    total_reuse_checks: int
    avg_reuse_percentage: float
    total_phase_gates: int
    completed_gates: int
    gate_completion_percentage: float
    quality_score: float
    reuse_compliance: float
    check_before_create_compliance: float
    gate_compliance: float
    primary_language: str
    agents_used: List[str]
    reuse_distribution: Dict[str, int] = field(default_factory=dict)
    recent_activity: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate data after initialization.

        Raises:
            ValueError: If validation fails
        """
        # Validate percentages are in range
        for attr in [
            "avg_reuse_percentage",
            "gate_completion_percentage",
            "quality_score",
            "reuse_compliance",
            "check_before_create_compliance",
            "gate_compliance",
        ]:
            value = getattr(self, attr)
            if not 0 <= value <= 100:
                raise ValueError(f"{attr} must be between 0 and 100, got {value}")

        # Validate counts are non-negative
        for attr in ["total_reuse_checks", "total_phase_gates", "completed_gates"]:
            value = getattr(self, attr)
            if value < 0:
                raise ValueError(f"{attr} cannot be negative, got {value}")

    @classmethod
    def from_components(
        cls,
        reuse_analyses: List[ReuseAnalysis],
        gates: List[PhaseGate],
        primary_language: str = "unknown",
        agents_used: Optional[List[str]] = None,
    ) -> "MetricsSummary":
        """Create MetricsSummary from component data.

        Args:
            reuse_analyses: List of ReuseAnalysis instances
            gates: List of PhaseGate instances
            primary_language: Primary programming language
            agents_used: List of agent names

        Returns:
            MetricsSummary instance

        Raises:
            ValueError: If data validation fails
        """
        # Calculate reuse metrics
        total_reuse = len(reuse_analyses)
        avg_reuse = (
            sum(r.reuse_percentage for r in reuse_analyses) / total_reuse
            if total_reuse > 0
            else 0.0
        )

        # Calculate reuse distribution
        distribution = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
        for analysis in reuse_analyses:
            pct = analysis.reuse_percentage
            if pct < 20:
                distribution["0-20"] += 1
            elif pct < 40:
                distribution["20-40"] += 1
            elif pct < 60:
                distribution["40-60"] += 1
            elif pct < 80:
                distribution["60-80"] += 1
            else:
                distribution["80-100"] += 1

        # Calculate gate metrics
        total_gates = len(gates)
        completed = sum(1 for gate in gates if gate.is_completed)
        gate_completion = (completed / total_gates * 100) if total_gates > 0 else 0.0

        # Calculate compliance metrics
        reuse_compliance = (
            sum(1 for r in reuse_analyses if r.reuse_percentage >= 70) / total_reuse * 100
            if total_reuse > 0
            else 0.0
        )

        check_compliance = (
            sum(1 for r in reuse_analyses if r.search_completed) / total_reuse * 100
            if total_reuse > 0
            else 0.0
        )

        gate_compliance = gate_completion

        # Calculate overall quality score
        quality = (reuse_compliance + check_compliance + gate_compliance) / 3

        return cls(
            total_reuse_checks=total_reuse,
            avg_reuse_percentage=round(avg_reuse, 2),
            total_phase_gates=total_gates,
            completed_gates=completed,
            gate_completion_percentage=round(gate_completion, 2),
            quality_score=round(quality, 2),
            reuse_compliance=round(reuse_compliance, 2),
            check_before_create_compliance=round(check_compliance, 2),
            gate_compliance=round(gate_compliance, 2),
            primary_language=primary_language,
            agents_used=agents_used or [],
            reuse_distribution=distribution,
            recent_activity=[],
        )

    def to_dict(self) -> MetricsSummaryDict:
        """Serialize to dictionary format.

        Returns:
            Dictionary representation compatible with MetricsSummaryDict
        """
        return MetricsSummaryDict(
            total_reuse_checks=self.total_reuse_checks,
            avg_reuse_percentage=self.avg_reuse_percentage,
            total_phase_gates=self.total_phase_gates,
            completed_gates=self.completed_gates,
            gate_completion_percentage=self.gate_completion_percentage,
            quality_score=self.quality_score,
            reuse_compliance=self.reuse_compliance,
            check_before_create_compliance=self.check_before_create_compliance,
            gate_compliance=self.gate_compliance,
            primary_language=self.primary_language,
            agents_used=self.agents_used,
        )

    def validate(self) -> bool:
        """Validate data integrity.

        Returns:
            True if all validation checks pass

        Raises:
            ValueError: If validation fails
        """
        if self.completed_gates > self.total_phase_gates:
            raise ValueError(
                f"completed_gates ({self.completed_gates}) cannot exceed "
                f"total_phase_gates ({self.total_phase_gates})"
            )

        return True
