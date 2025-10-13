"""Command-line interface for AI Workspace.

Provides commands for initializing and managing AI Workspace in projects.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

from . import WORKSPACE_DIR, __version__


@click.group()
@click.version_option(version=__version__, prog_name="ai-workspace")
def main() -> None:
    """AI Workspace - Universal AI Development System.

    Multi-language reuse analysis and automatic enforcement for AI development.
    """
    pass


@main.command()
@click.argument("target_dir", type=click.Path(), default=".", required=False)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing .ai-workspace directory",
)
@click.option(
    "--preset",
    type=click.Choice([
        "python-fastapi",
        "typescript-nextjs",
        "django-postgres",
        "rust-actix",
        "microservices"
    ]),
    help="Use a pre-configured stack preset",
)
@click.option(
    "--interactive",
    is_flag=True,
    help="Run interactive setup wizard",
)
def init(target_dir: str, force: bool, preset: Optional[str], interactive: bool) -> None:
    """Initialize AI Workspace in a project directory.

    Args:
        target_dir: Target directory (defaults to current directory)
        force: Overwrite existing installation
        preset: Stack preset to use
        interactive: Run interactive setup wizard
    """
    target_path = Path(target_dir).resolve()
    workspace_target = target_path / ".ai-workspace"

    click.echo(f"Initializing AI Workspace v{__version__}")
    click.echo(f"Target directory: {target_path}")

    # Check if already initialized
    if workspace_target.exists() and not force:
        click.echo("AI Workspace already initialized in this directory.")
        click.echo("   Use --force to overwrite existing installation.")
        sys.exit(1)

    # Verify source directory exists
    if not WORKSPACE_DIR.exists():
        click.echo(f"Error: AI Workspace data not found at {WORKSPACE_DIR}")
        click.echo("   Package may be incorrectly installed.")
        sys.exit(1)

    # Copy AI Workspace to target directory
    try:
        if workspace_target.exists():
            click.echo("Removing existing installation...")
            shutil.rmtree(workspace_target)

        click.echo("Copying AI Workspace files...")
        shutil.copytree(WORKSPACE_DIR, workspace_target)

        click.echo("Running setup script...")
        setup_script = workspace_target / "scripts" / "setup.py"

        if setup_script.exists():
            # Build setup command with options
            setup_cmd = [sys.executable, str(setup_script)]

            if preset:
                setup_cmd.extend(["--preset", preset])

            if interactive:
                setup_cmd.append("--interactive")

            result = subprocess.run(
                setup_cmd,
                cwd=str(target_path),
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                click.echo("AI Workspace initialized successfully!")
                click.echo("\nNext steps:")
                click.echo("   1. Review .ai-workspace/PRINCIPLES.md")
                click.echo("   2. Run: python .ai-workspace/scripts/detect_stack.py")
                click.echo("   3. Check generated CLAUDE.md for agent recommendations")
            else:
                click.echo("Setup script completed with warnings.")
                if result.stdout:
                    click.echo(result.stdout)
                if result.stderr:
                    click.echo(result.stderr, err=True)
        else:
            click.echo("Files copied successfully!")
            click.echo(f"Setup script not found at {setup_script}")

    except Exception as e:
        click.echo(f"Error during initialization: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("target_dir", type=click.Path(exists=True), default=".", required=False)
def status(target_dir: str) -> None:
    """Check AI Workspace installation status.

    Args:
        target_dir: Directory to check (defaults to current directory)
    """
    target_path = Path(target_dir).resolve()
    workspace_dir = target_path / ".ai-workspace"

    click.echo(f"Checking AI Workspace status in: {target_path}")

    if not workspace_dir.exists():
        click.echo("AI Workspace not initialized")
        click.echo("   Run 'ai-workspace init' to set up")
        sys.exit(1)

    # Check key components
    components = {
        "Scripts": workspace_dir / "scripts",
        "Agents": workspace_dir / "agents",
        "Templates": workspace_dir / "templates",
        "Principles": workspace_dir / "PRINCIPLES.md",
    }

    click.echo("AI Workspace is initialized")
    click.echo("\nComponents:")

    for name, path in components.items():
        status_icon = "[OK]" if path.exists() else "[MISSING]"
        click.echo(f"   {status_icon} {name}: {path.relative_to(target_path)}")

    # Check for CLAUDE.md
    claude_md = target_path / "CLAUDE.md"
    if claude_md.exists():
        click.echo(f"\nCLAUDE.md: Found")
    else:
        click.echo(f"\nCLAUDE.md: Not generated")
        click.echo("   Run: python .ai-workspace/scripts/detect_stack.py")


@main.command()
def info() -> None:
    """Display AI Workspace version and installation information."""
    from . import (
        __version__,
        PACKAGE_ROOT,
        WORKSPACE_DIR,
        PORTABLE_MODE,
        INSTALLATION_MODE,
        DASHBOARD_AVAILABLE,
    )

    click.echo(f"AI Workspace v{__version__}")
    click.echo(f"\nInstallation Mode: {INSTALLATION_MODE.upper()}")

    if PORTABLE_MODE:
        click.echo("  Type: Portable (drop-in folder)")
        click.echo(f"  Package: {PACKAGE_ROOT}")
        click.echo(f"  Workspace: {WORKSPACE_DIR}")
    else:
        click.echo("  Type: Pip-installed package")
        click.echo(f"  Package: {PACKAGE_ROOT}")
        click.echo(f"  Workspace: {WORKSPACE_DIR} (bundled)")

    click.echo(f"\nDashboard: {'Available' if DASHBOARD_AVAILABLE else 'Not installed'}")

    # Count agents
    if WORKSPACE_DIR.exists():
        agents_dir = WORKSPACE_DIR / "agents"
        if agents_dir.exists():
            agent_dirs = [d for d in agents_dir.glob("*") if d.is_dir()]
            agent_files = list(agents_dir.rglob("*.md"))
            click.echo(f"\nAgents: {len(agent_files)} agents in {len(agent_dirs)} categories")

    click.echo(f"\nPython: {sys.version}")
    click.echo("\nDocumentation: See README.md")
    click.echo("Issues: https://github.com/your-org/ai-workspace/issues")


@main.command()
@click.argument("task")
@click.option("--top", default=10, help="Number of recommendations (default: 10)")
@click.option("--category", help="Filter by category (e.g., 'development', 'infrastructure')")
@click.argument("target_dir", type=click.Path(exists=True), default=".", required=False)
def recommend(task: str, top: int, category: Optional[str], target_dir: str) -> None:
    """Recommend agents for a specific task.

    Args:
        task: Task description (what you want to accomplish)
        top: Number of recommendations to return
        category: Filter by agent category
        target_dir: Project directory to search
    """
    target_path = Path(target_dir).resolve()
    workspace_dir = target_path / ".ai-workspace"

    # Check if initialized
    if not workspace_dir.exists():
        click.echo("AI Workspace not initialized in this directory.", err=True)
        click.echo("   Run 'ai-workspace init' first")
        sys.exit(1)

    # Run recommendation script
    recommend_script = workspace_dir / "scripts" / "recommend_agents.py"

    if not recommend_script.exists():
        click.echo("Recommendation script not found.", err=True)
        sys.exit(1)

    try:
        cmd = [sys.executable, str(recommend_script), task, "--top", str(top)]

        if category:
            cmd.extend(["--category", category])

        cmd.extend(["--workspace", str(workspace_dir)])

        result = subprocess.run(
            cmd,
            cwd=str(target_path),
            capture_output=False,  # Show output directly
            text=True,
        )

        sys.exit(result.returncode)

    except Exception as e:
        click.echo(f"Error running recommendations: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--port", default=8900, help="Port to run dashboard (default: 8900)")
@click.option("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
@click.option("--debug", is_flag=True, help="Run in debug mode")
def dashboard(port: int, host: str, debug: bool) -> None:
    """Launch AI Workspace dashboard (optional feature).

    Requires dashboard dependencies. Install with:
        pip install ai-workspace[dashboard]

    Args:
        port: Port number for dashboard server
        host: Host address to bind to
        debug: Enable debug mode
    """
    # Check if dashboard dependencies are installed
    try:
        from .dashboard import app, run, DASHBOARD_AVAILABLE
    except ImportError as e:
        click.echo("Dashboard dependencies not installed", err=True)
        click.echo("\nTo use the dashboard, install with:")
        click.echo("  pip install ai-workspace[dashboard]")
        click.echo("\nOr install manually:")
        click.echo("  pip install flask flask-cors flask-socketio python-socketio watchdog")
        sys.exit(1)

    if not DASHBOARD_AVAILABLE:
        click.echo("Dashboard not available", err=True)
        click.echo("Install: pip install ai-workspace[dashboard]")
        sys.exit(1)

    # Launch dashboard
    click.echo(f"Starting AI Workspace Dashboard v{__version__}")
    click.echo(f"Dashboard URL: http://{host}:{port}")
    click.echo("Press Ctrl+C to stop\n")

    try:
        run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped")


if __name__ == "__main__":
    main()
