#!/usr/bin/env python3
"""AI Workspace - Portable Bootstrap CLI.

No pip installation required - just drop and run!

This script provides a portable entry point to AI Workspace without
requiring pip installation. It adds the local ai_workspace package
to the Python path and runs the CLI.

Usage:
    python bootstrap.py init [--preset PRESET] [--interactive]
    python bootstrap.py status
    python bootstrap.py recommend "task description"
    python bootstrap.py info
    python bootstrap.py dashboard

Requirements:
    - Python 3.10+
    - Dependencies: pip install -r requirements.txt
    - Optional (dashboard): pip install -r requirements-dashboard.txt
"""

import sys
from pathlib import Path

# Add local ai_workspace to Python path
PACKAGE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PACKAGE_ROOT))

# Check Python version
if sys.version_info < (3, 10):
    print("[ERROR] Python 3.10+ required")
    print(f"Current version: {sys.version}")
    sys.exit(1)

# Check core dependencies
missing_deps = []
try:
    import yaml
except ImportError:
    missing_deps.append("pyyaml>=6.0")

try:
    import jinja2
except ImportError:
    missing_deps.append("jinja2>=3.1.0")

try:
    import click
except ImportError:
    missing_deps.append("click>=8.1.0")

if missing_deps:
    print("[ERROR] Missing required dependencies:")
    for dep in missing_deps:
        print(f"  - {dep}")
    print("\nPlease install dependencies:")
    print(f"  pip install -r {PACKAGE_ROOT}/requirements.txt")
    print("\nOr install individual packages:")
    for dep in missing_deps:
        print(f"  pip install {dep}")
    sys.exit(1)

# Import and run CLI
try:
    from ai_workspace.cli import main

    if __name__ == "__main__":
        main()
except Exception as e:
    print(f"[ERROR] Failed to run AI Workspace CLI: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print(f"  pip install -r {PACKAGE_ROOT}/requirements.txt")
    sys.exit(1)
