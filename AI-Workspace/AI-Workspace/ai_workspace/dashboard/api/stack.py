"""Stack Detection API endpoints.

Read-only endpoints for accessing detected technology stack information.
"""

from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, jsonify

from ..services.analytics import MetricsAnalyzer
from ..services.data_loader import WorkspaceDataLoader

stack_bp = Blueprint("stack", __name__, url_prefix="/api/stack")


def _get_loader() -> WorkspaceDataLoader:
    """Get data loader instance.

    Returns:
        WorkspaceDataLoader instance
    """
    return WorkspaceDataLoader(Path.cwd())


@stack_bp.route("/detect", methods=["GET"])
def get_detected_stack() -> Dict[str, Any]:
    """Get detected technology stack.

    Returns:
        JSON response with stack information
    """
    try:
        loader = _get_loader()
        config = loader.load_stack_config()

        if not config:
            return jsonify(
                {
                    "detected": False,
                    "message": "No stack configuration found. Run: python .ai-workspace/scripts/detect_stack.py",
                }
            )

        project_data = config.get("project", {})

        return jsonify(
            {
                "detected": True,
                "project_type": project_data.get("project_type", "unknown"),
                "primary_language": project_data.get("primary_language", "unknown"),
                "detected_stack": project_data.get("detected_stack", {}),
                "confidence_scores": project_data.get("confidence_scores", {}),
                "detection_date": project_data.get("detection_date", ""),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@stack_bp.route("/agents", methods=["GET"])
def get_recommended_agents() -> Dict[str, Any]:
    """Get recommended agents based on stack.

    Returns:
        JSON response with agent recommendations
    """
    try:
        loader = _get_loader()
        config = loader.load_stack_config()

        if not config:
            return jsonify(
                {
                    "recommended": False,
                    "message": "No stack configuration found",
                    "agents": {"primary": [], "secondary": [], "optional": []},
                }
            )

        recommended_agents = config.get("recommended_agents", {})

        return jsonify(
            {
                "recommended": True,
                "agents": {
                    "primary": recommended_agents.get("primary", []),
                    "secondary": recommended_agents.get("secondary", []),
                    "optional": recommended_agents.get("optional", []),
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@stack_bp.route("/languages", methods=["GET"])
def get_languages() -> Dict[str, Any]:
    """Get detected programming languages with confidence scores.

    Returns:
        JSON response with language statistics
    """
    try:
        loader = _get_loader()
        analyzer = MetricsAnalyzer(loader)

        language_stats = analyzer.get_language_stats()

        return jsonify(
            {"languages": language_stats, "count": len(language_stats)}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@stack_bp.route("/info", methods=["GET"])
def get_project_info() -> Dict[str, Any]:
    """Get general project information.

    Returns:
        JSON response with project metadata
    """
    try:
        loader = _get_loader()
        project_info = loader.get_project_info()
        workspace_stats = loader.get_workspace_stats()

        return jsonify(
            {
                "project": project_info,
                "workspace": workspace_stats,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
