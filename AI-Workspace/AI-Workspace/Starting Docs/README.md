# AI Workspace v1.2.0 - Portable Edition

**Drop-in AI development orchestration** - No pip installation required!

## What is this?

AI Workspace is a universal AI development system providing:
- 159 specialized AI agents across 10 categories
- Multi-language reuse analysis (Python, TypeScript, Rust, Go, Java)
- Automatic enforcement of the 5 Core Principles
- Phase gate reviews and quality automation
- Interactive web dashboard (optional)

This **portable edition** can be dropped into any project root and used immediately.

## Quick Start

### 1. Extract Package

```bash
# Extract AI-Workspace/ into your project root
cp -r AI-Workspace/ /path/to/your/project/
cd /path/to/your/project/AI-Workspace/
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install dashboard dependencies
pip install -r requirements-dashboard.txt
```

### 3. Run Commands

```bash
# Initialize AI Workspace in your project
python bootstrap.py init --interactive

# Check status
python bootstrap.py status

# Get agent recommendations
python bootstrap.py recommend "build REST API"

# Launch dashboard (requires dashboard dependencies)
python bootstrap.py dashboard
```

## Available Commands

### Initialize Workspace

```bash
# Interactive setup wizard
python bootstrap.py init --interactive

# Use preset configuration
python bootstrap.py init --preset python-fastapi
python bootstrap.py init --preset typescript-nextjs
python bootstrap.py init --preset django-postgres
python bootstrap.py init --preset rust-actix
python bootstrap.py init --preset microservices

# Force overwrite existing installation
python bootstrap.py init --force
```

### Check Status

```bash
python bootstrap.py status
```

### Get Recommendations

```bash
# Find best agents for a task
python bootstrap.py recommend "task description"

# Limit to top N results
python bootstrap.py recommend "build API" --top 5

# Filter by category
python bootstrap.py recommend "optimize code" --category quality
```

### Launch Dashboard

```bash
# Start interactive web dashboard
python bootstrap.py dashboard

# Custom port
python bootstrap.py dashboard --port 8080
```

### Show Info

```bash
# Display version and installation info
python bootstrap.py info
```

## Shell Wrappers (Optional)

For convenience, you can use the provided shell wrappers:

**Unix/Mac**:
```bash
chmod +x ai-workspace.sh
./ai-workspace.sh init --interactive
```

**Windows**:
```cmd
ai-workspace.bat init --interactive
```

## The 5 Core Principles

AI Workspace enforces these principles automatically:

1. **Less is More** - Max complexity ≤10, Max duplication <3%
2. **Document Everything** - ADRs required for decisions
3. **Check Before Create** - Mandatory reuse analysis (≥70% auto-approved)
4. **Phase Gate Reviews** - Cannot skip quality gates
5. **Plan Before Act** - Structured planning required

## Stack Presets

Choose from 5 pre-configured setups:

| Preset | Stack | Quality Level |
|--------|-------|--------------|
| `python-fastapi` | FastAPI + PostgreSQL | Strict |
| `typescript-nextjs` | Next.js + React | Balanced |
| `django-postgres` | Django + Celery | Strict |
| `rust-actix` | Actix-web + SQLx | Strict |
| `microservices` | Kubernetes + gRPC | Balanced |

## Agent Categories

159 specialized agents across 10 categories:

- **00-orchestration**: Project setup, coordination (17 agents)
- **01-development**: Fullstack, backend, frontend, mobile (8 agents)
- **02-languages**: Python, TypeScript, Rust, Go, Java (21 agents)
- **03-frameworks**: React, Django, Spring Boot (18 agents)
- **04-infrastructure**: DevOps, Kubernetes, cloud (15 agents)
- **05-quality**: Code review, testing, performance (20 agents)
- **06-data-ai**: Data engineering, ML, LLM (19 agents)
- **07-specialized**: Blockchain, fintech, gaming, IoT (18 agents)
- **08-support**: Documentation, business, management (12 agents)
- **09-utilities**: Research, search, analysis (11 agents)

## Requirements

- Python 3.10+
- Core dependencies: pyyaml, jinja2, click
- Optional (dashboard): quart, flask, watchdog, whoosh

## Support

For issues, questions, or contributions:
- GitHub: https://github.com/your-org/ai-workspace
- Documentation: See INSTALL.md for detailed setup instructions

## License

MIT License - See LICENSE file for details
