"""AI Workspace - Universal AI Development System.

This package provides multi-language reuse analysis, automatic enforcement,
and agent orchestration for AI-assisted development.

Version: 1.2.0
"""

__version__ = "1.2.0"
__author__ = "AI Workspace Contributors"

import os
from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent

# Detect installation mode
# Portable mode: bootstrap.py exists in parent directory
# Pip mode: bundled .ai-workspace exists in package
PORTABLE_MODE = (PACKAGE_ROOT.parent / "bootstrap.py").exists()

# Environment variable override (for testing)
if os.getenv("AI_WORKSPACE_PORTABLE_MODE"):
    PORTABLE_MODE = os.getenv("AI_WORKSPACE_PORTABLE_MODE").lower() == "true"

if PORTABLE_MODE:
    # Running from AI-Workspace/ folder (portable installation)
    WORKSPACE_DIR = PACKAGE_ROOT.parent / ".ai-workspace"
    INSTALLATION_MODE = "portable"
else:
    # Running from pip-installed package (bundled mode)
    WORKSPACE_DIR = PACKAGE_ROOT / ".ai-workspace"
    INSTALLATION_MODE = "pip"

# Validate WORKSPACE_DIR exists
if not WORKSPACE_DIR.exists():
    import warnings
    warnings.warn(
        f"AI Workspace data not found at {WORKSPACE_DIR}. "
        f"Installation mode: {INSTALLATION_MODE}. "
        "Package may be incorrectly installed."
    )

# Check if dashboard is available (optional feature)
DASHBOARD_AVAILABLE = False

try:
    # Try to import dashboard dependencies
    import flask
    import flask_cors
    import flask_socketio

    # Try to import dashboard module
    from . import dashboard

    DASHBOARD_AVAILABLE = True
except ImportError:
    # Dashboard dependencies not installed - this is okay
    DASHBOARD_AVAILABLE = False

__all__ = [
    "__version__",
    "__author__",
    "PACKAGE_ROOT",
    "WORKSPACE_DIR",
    "PORTABLE_MODE",
    "INSTALLATION_MODE",
    "DASHBOARD_AVAILABLE",
]
