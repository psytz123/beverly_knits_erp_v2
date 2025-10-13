"""Phase Gates API endpoints.

Read-only endpoints for accessing phase gate status and information.
"""

from pathlib import Path
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request

from ..services.analytics import MetricsAnalyzer
from ..services.data_loader import WorkspaceDataLoader

gates_bp = Blueprint("gates", __name__, url_prefix="/api/gates")


def _get_loader() -> WorkspaceDataLoader:
    """Get data loader instance.

    Returns:
        WorkspaceDataLoader instance
    """
    return WorkspaceDataLoader(Path.cwd())


@gates_bp.route("/status", methods=["GET"])
def get_status() -> Dict[str, Any]:
    """Get overall phase gate status.

    Returns:
        JSON response with gate status
    """
    try:
        loader = _get_loader()
        analyzer = MetricsAnalyzer(loader)

        gate_progress = analyzer.calculate_gate_progress()

        return jsonify(gate_progress)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@gates_bp.route("/<phase>", methods=["GET"])
def get_phase_details(phase: str) -> Dict[str, Any]:
    """Get details for a specific phase.

    Args:
        phase: Phase name (discovery, design, implementation, verification, integration)

    Returns:
        JSON response with phase details
    """
    try:
        loader = _get_loader()
        task_name = request.args.get("task")

        gates = loader.load_phase_gates(task_name)

        # Filter by phase
        phase_gates = [g for g in gates if g.phase == phase]

        if not phase_gates:
            return jsonify({"phase": phase, "gates": [], "message": "No gates found for this phase"})

        # Convert to dict for JSON serialization
        gates_data = [
            {
                "phase": g.phase,
                "task_name": g.task_name,
                "timestamp": g.timestamp,
                "artifacts": g.artifacts,
                "exit_criteria_met": g.exit_criteria_met,
                "approved_by": g.approved_by,
                "approval_timestamp": g.approval_timestamp,
            }
            for g in phase_gates
        ]

        return jsonify({"phase": phase, "gates": gates_data, "count": len(gates_data)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@gates_bp.route("/list", methods=["GET"])
def list_all_gates() -> Dict[str, Any]:
    """List all phase gates.

    Returns:
        JSON response with all gates
    """
    try:
        loader = _get_loader()
        task_name = request.args.get("task")

        gates = loader.load_phase_gates(task_name)

        # Convert to dict for JSON serialization
        gates_data = [
            {
                "phase": g.phase,
                "task_name": g.task_name,
                "timestamp": g.timestamp,
                "all_criteria_met": all(g.exit_criteria_met),
                "approved": bool(g.approved_by),
            }
            for g in gates
        ]

        return jsonify({"gates": gates_data, "total": len(gates_data)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
