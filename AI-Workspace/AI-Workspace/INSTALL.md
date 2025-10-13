# AI Workspace v1.2.0 - Installation Guide

## Prerequisites

### Python Version

AI Workspace requires **Python 3.10 or higher**.

Check your Python version:
```bash
python --version
# or
python3 --version
```

If you need to upgrade Python:
- **Windows**: Download from https://www.python.org/downloads/
- **Mac**: `brew install python@3.10` (using Homebrew)
- **Linux**: `sudo apt install python3.10` (Ubuntu/Debian)

### Virtual Environment (Recommended)

Create a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Unix/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

## Installation Steps

### 1. Extract Package

Place the `AI-Workspace/` folder in your project root:

```bash
# Your project structure should look like:
/your-project/
├── AI-Workspace/          # This package
│   ├── bootstrap.py
│   ├── requirements.txt
│   ├── .ai-workspace/
│   └── ai_workspace/
├── src/                   # Your project code
├── tests/
└── README.md
```

### 2. Install Dependencies

Navigate to the AI-Workspace directory and install dependencies:

```bash
cd AI-Workspace/

# Install core dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -E "pyyaml|jinja2|click"
```

**Expected output**:
```
click         8.1.7
Jinja2        3.1.3
PyYAML        6.0.1
```

### 3. Optional: Install Dashboard

If you want to use the interactive web dashboard:

```bash
# Install dashboard dependencies
pip install -r requirements-dashboard.txt

# Verify installation
pip list | grep -E "quart|flask|watchdog"
```

### 4. Verify Installation

Test the bootstrap CLI:

```bash
python bootstrap.py info
```

**Expected output**:
```
AI Workspace v1.2.0

Installation Directory: /path/to/AI-Workspace
Workspace Directory: /path/to/AI-Workspace/.ai-workspace
Package Root: /path/to/AI-Workspace/ai_workspace

Dashboard Available: Yes (if dashboard deps installed)
```

## Initialize Your Project

### Option 1: Interactive Wizard (Recommended)

Run the interactive setup wizard:

```bash
python bootstrap.py init --interactive
```

The wizard will guide you through:
1. Technology stack detection
2. Quality level selection (Strict/Balanced/Relaxed)
3. Optional features (dashboard, git hooks, CI/CD templates)
4. Agent selection

### Option 2: Use Preset

Choose a stack preset:

```bash
# Python FastAPI project
python bootstrap.py init --preset python-fastapi

# TypeScript Next.js project
python bootstrap.py init --preset typescript-nextjs

# Django project
python bootstrap.py init --preset django-postgres

# Rust project
python bootstrap.py init --preset rust-actix

# Microservices architecture
python bootstrap.py init --preset microservices
```

### Option 3: Auto-detect

Let AI Workspace detect your stack automatically:

```bash
python bootstrap.py init
```

## Post-Installation Setup

After initialization, AI Workspace creates:

1. **`.ai-workspace/`** - Configuration and workspace files
2. **`.claude/`** - Symlink to agent definitions (for Claude Code)
3. **`.cursor/`** - Symlink to cursor rules (for Cursor IDE)
4. **`CLAUDE.md`** - AI team guide (custom to your project)
5. **`.ai-workspace-config.yml`** - Configuration file
6. **`.agent-workspace/`** - Local workspace (ADRs, tasks, handoffs)

### Verify Setup

```bash
# Check installation status
python bootstrap.py status
```

**Expected output**:
```
AI Workspace Status: INSTALLED

Core Components:
[OK] .ai-workspace/ exists
[OK] .claude/ symlink exists
[OK] .cursor/ symlink exists
[OK] CLAUDE.md generated
[OK] Git hooks installed

Configuration: .ai-workspace-config.yml
Quality Level: Balanced
```

## Using AI Workspace

### Get Agent Recommendations

Find the best agents for your task:

```bash
# Get recommendations
python bootstrap.py recommend "build REST API with authentication"

# Top 5 results only
python bootstrap.py recommend "optimize database queries" --top 5

# Filter by category
python bootstrap.py recommend "review code" --category quality
```

### Launch Dashboard

Start the interactive web dashboard:

```bash
python bootstrap.py dashboard
```

Open browser to `http://localhost:5000`

### Key Scripts

All scripts are in `.ai-workspace/scripts/`:

- `setup.py` - Master installer
- `search_codebase.py` - Multi-language code search
- `analyze_reuse.py` - Calculate reuse potential
- `enforce_check_before_create.py` - Automatic enforcement
- `create_adr.py` - Create Architecture Decision Records
- `plan_task.py` - Structured task planning
- `validate_gates.py` - Phase gate validation

## Troubleshooting

### Issue: "No module named 'yaml'"

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "Python version too old"

**Solution**: Upgrade to Python 3.10+
```bash
python --version  # Check current version
# Upgrade Python (see Prerequisites section)
```

### Issue: "Permission denied" (Unix/Mac)

**Solution**: Make bootstrap.py executable
```bash
chmod +x bootstrap.py
```

### Issue: Symlinks not working (Windows)

**Solution**: AI Workspace automatically falls back to:
1. Directory junctions (for directories)
2. Hard links (for files)
3. File copying (last resort)

No action needed - setup handles this automatically.

### Issue: Dashboard not available

**Solution**: Install dashboard dependencies
```bash
pip install -r requirements-dashboard.txt
```

### Issue: Git hooks not triggering

**Solution**: Ensure hooks are executable
```bash
chmod +x .git/hooks/pre-commit
```

## Configuration

Edit `.ai-workspace-config.yml` to customize:

```yaml
# Quality levels
complexity_threshold: 10
duplication_threshold: 3.0
coverage_threshold: 85

# Technology stack
languages: [python, typescript]
frameworks: [fastapi, react]
databases: [postgresql]

# Optional features
enable_dashboard: true
enable_git_hooks: true
```

## Uninstallation

To remove AI Workspace:

```bash
# Remove symlinks
rm -rf .claude .cursor

# Remove workspace
rm -rf .ai-workspace

# Remove local workspace
rm -rf .agent-workspace

# Remove generated files
rm CLAUDE.md
rm .ai-workspace-config.yml

# Remove git hooks
rm .git/hooks/pre-commit

# Remove AI-Workspace folder
rm -rf AI-Workspace/
```

## Next Steps

1. Read generated `CLAUDE.md` for project-specific guidance
2. Explore agents in `.ai-workspace/agents/`
3. Run `python bootstrap.py recommend` to find agents for your tasks
4. Create your first ADR: `.ai-workspace/scripts/create_adr.py`
5. Plan tasks: `.ai-workspace/scripts/plan_task.py`

## Support

- Documentation: README.md
- GitHub Issues: https://github.com/your-org/ai-workspace/issues
- License: MIT
