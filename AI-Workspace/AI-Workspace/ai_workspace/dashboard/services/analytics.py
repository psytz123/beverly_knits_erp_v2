"""Analytics and metrics calculation service.

Provides analysis and calculations on workspace data for dashboard visualization.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .data_loader import PhaseGate, ReuseCheck, WorkspaceDataLoader


class MetricsAnalyzer:
    """Calculate metrics and analytics from workspace data."""

    def __init__(self, data_loader: WorkspaceDataLoader) -> None:
        """Initialize analyzer.

        Args:
            data_loader: Data loader instance for accessing workspace data
        """
        self.loader = data_loader

    def calculate_reuse_summary(self) -> Dict[str, Any]:
        """Calculate reuse analysis summary.

        Returns:
            Dictionary with reuse metrics
        """
        checks = self.loader.load_reuse_checks()

        if not checks:
            return {
                "total_checks": 0,
                "avg_reuse_percentage": 0.0,
                "high_reuse_count": 0,
                "medium_reuse_count": 0,
                "low_reuse_count": 0,
                "recent_checks": [],
            }

        # Calculate statistics
        total = len(checks)
        avg_reuse = sum(c.reuse_percentage for c in checks) / total

        # Categorize by reuse level
        high_reuse = sum(1 for c in checks if c.reuse_percentage >= 70)
        medium_reuse = sum(
            1 for c in checks if 30 <= c.reuse_percentage < 70
        )
        low_reuse = sum(1 for c in checks if c.reuse_percentage < 30)

        return {
            "total_checks": total,
            "avg_reuse_percentage": round(avg_reuse, 2),
            "high_reuse_count": high_reuse,
            "medium_reuse_count": medium_reuse,
            "low_reuse_count": low_reuse,
            "recent_checks": [
                {
                    "task_name": c.task_name,
                    "reuse_percentage": c.reuse_percentage,
                    "timestamp": c.timestamp,
                    "approved": c.approved_to_create,
                }
                for c in checks[:10]
            ],
        }

    def calculate_reuse_distribution(self) -> Dict[str, int]:
        """Calculate distribution of reuse percentages.

        Returns:
            Dictionary with percentage buckets and counts
        """
        checks = self.loader.load_reuse_checks()

        if not checks:
            return {
                "0-20": 0,
                "20-40": 0,
                "40-60": 0,
                "60-80": 0,
                "80-100": 0,
            }

        distribution = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}

        for check in checks:
            pct = check.reuse_percentage
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

        return distribution

    def find_reuse_violations(self) -> List[Dict[str, Any]]:
        """Find reuse check violations.

        Returns:
            List of violations (blocked creations)
        """
        checks = self.loader.load_reuse_checks()

        violations = []

        for check in checks:
            if check.reuse_percentage < 70 and not check.approved_to_create:
                violations.append(
                    {
                        "task_name": check.task_name,
                        "reuse_percentage": check.reuse_percentage,
                        "timestamp": check.timestamp,
                        "reason": "Low reuse percentage - ADR required",
                    }
                )

        return violations

    def calculate_gate_progress(self) -> Dict[str, Any]:
        """Calculate phase gate progress.

        Returns:
            Dictionary with gate completion status
        """
        gates = self.loader.load_phase_gates()

        # Define gate order
        gate_order = ["discovery", "design", "implementation", "verification", "integration"]

        # Count completed gates by phase
        phase_counts: Dict[str, int] = {phase: 0 for phase in gate_order}

        for gate in gates:
            if gate.phase in phase_counts:
                # Check if all exit criteria met
                if all(gate.exit_criteria_met):
                    phase_counts[gate.phase] += 1

        # Calculate overall progress
        total_gates = len(gates)
        completed_gates = sum(
            1
            for gate in gates
            if gate.exit_criteria_met and all(gate.exit_criteria_met)
        )

        # Determine current phase
        current_phase = "discovery"
        for phase in gate_order:
            if phase_counts[phase] == 0:
                current_phase = phase
                break
        else:
            current_phase = "integration"

        return {
            "total_gates": total_gates,
            "completed_gates": completed_gates,
            "completion_percentage": (
                round((completed_gates / total_gates) * 100, 1) if total_gates > 0 else 0
            ),
            "current_phase": current_phase,
            "phase_counts": phase_counts,
            "phase_order": gate_order,
        }

    def calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate quality metrics.

        Returns:
            Dictionary with quality indicators
        """
        checks = self.loader.load_reuse_checks()
        gates = self.loader.load_phase_gates()

        # Principle 1: Less is More (Reuse)
        reuse_compliance = (
            sum(1 for c in checks if c.reuse_percentage >= 70) / len(checks) * 100
            if checks
            else 0
        )

        # Principle 3: Check Before Create
        check_compliance = (
            sum(1 for c in checks if c.search_completed) / len(checks) * 100
            if checks
            else 0
        )

        # Principle 4: Phase Gate Reviews
        gate_compliance = (
            sum(1 for g in gates if all(g.exit_criteria_met)) / len(gates) * 100
            if gates
            else 0
        )

        return {
            "reuse_compliance": round(reuse_compliance, 1),
            "check_before_create_compliance": round(check_compliance, 1),
            "gate_compliance": round(gate_compliance, 1),
            "overall_quality_score": round(
                (reuse_compliance + check_compliance + gate_compliance) / 3, 1
            ),
        }

    def get_language_stats(self) -> Dict[str, int]:
        """Get language usage statistics from stack config.

        Returns:
            Dictionary with language names and confidence scores
        """
        config = self.loader.load_stack_config()

        if not config:
            return {}

        project_data = config.get("project", {})
        confidence_scores = project_data.get("confidence_scores", {})

        return confidence_scores

    def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent workspace activity.

        Args:
            limit: Maximum number of activities to return

        Returns:
            List of recent activities
        """
        activities = []

        # Collect reuse checks
        for check in self.loader.load_reuse_checks()[:limit]:
            activities.append(
                {
                    "type": "reuse_check",
                    "timestamp": check.timestamp,
                    "description": f"Reuse analysis: {check.task_name} ({check.reuse_percentage}%)",
                    "data": {
                        "task_name": check.task_name,
                        "reuse_percentage": check.reuse_percentage,
                    },
                }
            )

        # Collect gate completions
        for gate in self.loader.load_phase_gates()[:limit]:
            if all(gate.exit_criteria_met):
                activities.append(
                    {
                        "type": "gate_complete",
                        "timestamp": gate.timestamp,
                        "description": f"Gate completed: {gate.phase} - {gate.task_name}",
                        "data": {"phase": gate.phase, "task_name": gate.task_name},
                    }
                )

        # Sort by timestamp (most recent first)
        activities.sort(key=lambda x: x["timestamp"], reverse=True)

        return activities[:limit]
