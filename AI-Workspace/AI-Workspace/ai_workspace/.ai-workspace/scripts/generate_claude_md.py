#!/usr/bin/env python3
"""
CLAUDE.md Auto-Generator
Generates project-specific CLAUDE.md from detected technology stack.
Includes all available agents from .claude/agents/
"""
import sys
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class ClaudeMdGenerator:
    """Generate CLAUDE.md from stack detection and agent registry."""

    def __init__(self, project_root: Path, ai_dev_kit_path: Path):
        """Initialize generator with project paths."""
        self.project_root = project_root
        self.kit_path = ai_dev_kit_path
        self.config: Dict[str, Any] = {}
        self.agents: Dict[str, List[Dict[str, str]]] = {}

    def load_config(self, config_path: Path) -> None:
        """Load stack detection config."""
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {config_path}\n"
                "Run 'python .ai-workspace/scripts/detect_stack.py' first"
            )
        self.config = yaml.safe_load(config_path.read_text())

    def load_agents(self) -> None:
        """Load all available agents from Claude agents directory."""
        # Check standard locations for agents
        agent_paths = [
            self.kit_path / "agents",  # .ai-workspace/agents/
            self.kit_path / ".claude" / "agents",  # Legacy location
            Path(".claude") / "agents",  # Project .claude/agents/
            self.project_root / ".ai-workspace" / "agents",  # Project workspace
        ]

        for agent_base in agent_paths:
            if agent_base.exists():
                self._scan_agent_directory(agent_base)
                break

    def _scan_agent_directory(self, base_path: Path) -> None:
        """Recursively scan agent directory and categorize agents."""
        categories = {
            "orchestration": [],
            "backend": [],
            "frontend": [],
            "fullstack": [],
            "mobile": [],
            "languages": [],
            "frameworks": [],
            "infrastructure": [],
            "cloud": [],
            "devops": [],
            "security": [],
            "quality": [],
            "testing": [],
            "performance": [],
            "data": [],
            "ai_ml": [],
            "specialized": [],
            "documentation": [],
            "business": [],
            "management": [],
            "utilities": [],
        }

        # Map directory names to categories
        dir_to_category = {
            "00-orchestration": "orchestration",
            "01-development/backend": "backend",
            "01-development/frontend": "frontend",
            "01-development/fullstack": "fullstack",
            "01-development/mobile": "mobile",
            "02-languages": "languages",
            "03-frameworks": "frameworks",
            "04-infrastructure/cloud": "cloud",
            "04-infrastructure/devops": "devops",
            "04-infrastructure/security": "security",
            "05-quality/testing": "testing",
            "05-quality/review": "quality",
            "05-quality/performance": "performance",
            "06-data-ai/data": "data",
            "06-data-ai/ai": "ai_ml",
            "06-data-ai/ml": "ai_ml",
            "07-specialized": "specialized",
            "08-support/documentation": "documentation",
            "08-support/business": "business",
            "08-support/management": "management",
            "09-utilities": "utilities",
        }

        for agent_file in base_path.rglob("*.md"):
            relative_path = agent_file.relative_to(base_path)
            category = self._determine_category(relative_path, dir_to_category)

            agent_name = agent_file.stem
            agent_handle = f"@{agent_name.replace('_', '-')}"

            # Extract description from file (first line after heading)
            description = self._extract_description(agent_file)

            categories[category].append(
                {"name": agent_name, "handle": agent_handle, "description": description}
            )

        self.agents = {k: v for k, v in categories.items() if v}

    def _determine_category(self, rel_path: Path, mapping: Dict[str, str]) -> str:
        """Determine agent category from file path."""
        path_str = str(rel_path).replace("\\", "/")

        for dir_pattern, category in mapping.items():
            if path_str.startswith(dir_pattern):
                return category

        # Default fallback
        parts = rel_path.parts
        if len(parts) > 1:
            return parts[0].replace("-", "_")

        return "utilities"

    def _extract_description(self, agent_file: Path) -> str:
        """Extract agent description from markdown file."""
        try:
            content = agent_file.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Look for description after first heading
            for i, line in enumerate(lines):
                if line.startswith("#") and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith("#"):
                        return next_line[:100]  # First 100 chars

            return "Specialized AI agent"
        except Exception:
            return "AI development agent"

    def generate(self) -> str:
        """Generate complete CLAUDE.md content."""
        stack = self.config["project"]["detected_stack"]
        primary_lang = self.config["project"]["primary_language"]
        project_type = self.config["project"]["project_type"]
        project_name = self.project_root.name.replace("-", " ").replace("_", " ").title()

        md = f"""# {project_name} - AI Development Guide

**Auto-generated by AI Workspace v1.1.0**
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Detection:** Automatic Stack Analysis

---

## 📊 Detected Technology Stack

**Project Type:** {project_type.title()}
**Primary Language:** {primary_lang}

"""

        # Display detected stack
        for category, items in stack.items():
            if items:
                md += f"**{category.title()}:** {', '.join(items)}\n"

        md += """

---

## 🤖 AI Team Configuration

**IMPORTANT: You MUST use specialized subagents when available for tasks.**

AI Workspace has detected your technology stack and configured the following agents:

"""

        # Add recommended agents based on stack
        md += self._generate_agent_assignments()

        # Add operating charter section
        md += """

---

## 📋 Operating Charter

This project follows the **AI Coding Agent Operating Charter** principles:

### Core Principles

1. **Less is More**
   - Smallest, clearest solution
   - Cyclomatic complexity ≤10 per function
   - Code duplication <3%
   - Reuse-first: stdlib → dependencies → internal → new

2. **Document Everything**
   - ADRs for all major decisions
   - Module/file purpose and usage
   - Sample inputs/outputs

3. **Check Before Create** ⚠️ **AUTOMATIC ENFORCEMENT ACTIVE**
   - **BLOCKS code creation** until reuse analysis complete
   - Multi-language support: Python, TypeScript, JavaScript, Rust, Go, Java
   - Mandatory workflow:
     1. Search codebase: `python .ai-workspace/scripts/search_codebase.py "intent"`
     2. Analyze reuse: `python .ai-workspace/scripts/analyze_reuse.py "intent" file.py`
     3. Enforcement check: `python .ai-workspace/scripts/enforce_check_before_create.py --check task "intent"`
   - Auto-approves if reuse ≥70%
   - Requires ADR if reuse <70%
   - 30-minute cache expiry

4. **Phase Gate Reviews**
   - Discovery → Design → Implementation → Verification → Integration
   - Real data validation (not mocks)
   - Exit code 1 on failures

5. **Plan Before Act**
   - Structured planning required
   - Plans stored in `.ai-workspace/handoffs/`
   - Task planner: `python .ai-workspace/scripts/plan_task.py`
   - Every output traces to initial plan

See `.cursor/rules/00-core/operating-charter.md` for complete details.

---

## ⚙️ Quality Standards

"""

        md += self._generate_quality_standards()

        md += """

---

## 🚀 Quick Start Commands

"""

        md += self._generate_quick_start_commands()

        md += """

---

## 🔍 Agent Usage Examples

"""

        md += self._generate_usage_examples()

        md += """

---

## 📚 Available Agents (Complete Registry)

"""

        md += self._generate_complete_agent_registry()

        md += f"""

---

## 🛠️ Development Workflow

1. **Planning Phase**
   - Use `@project-analyst` to understand requirements
   - Document decisions in ADRs
   - Create task breakdown

2. **Implementation Phase**
   - Use language/framework specialists
   - Follow reuse-first workflow
   - Implement with test-first approach

3. **Quality Phase**
   - Run `@code-reviewer` for review
   - Use `@test-automator` for test coverage
   - Verify with `@qa-expert`

4. **Deployment Phase**
   - Configure with `@devops-engineer`
   - Deploy with `@platform-engineer`
   - Monitor with `@sre-engineer`

---

## 📁 Project Structure

This project uses AI Workspace for:
- **Agent Orchestration:** `.agent-workspace/`
- **Quality Rules:** `.cursor/rules/`
- **ADR Documentation:** `.agent-workspace/decisions/`
- **Phase Gates:** `.agent-workspace/phase-gates/`

---

**Powered by AI Workspace v1.1.0**
**Stack Detection Date:** {self.config['project']['detection_date']}
**Last Updated:** {datetime.now().strftime("%Y-%m-%d")}

---

## 🚦 Enforcement System

This project uses **automatic enforcement** of Principle 3 (Check Before Create):

### Before Writing ANY Code:
```bash
# 1. Search for existing code
python .ai-workspace/scripts/search_codebase.py "your functionality"

# 2. Analyze reuse potential
python .ai-workspace/scripts/analyze_reuse.py "your functionality" path/to/existing.py

# 3. Check if approved to create new code
python .ai-workspace/scripts/enforce_check_before_create.py --check task-name "your functionality"
```

### Enforcement Rules:
- ✅ **Auto-records** all searches and analyses
- ✅ **Auto-approves** if reuse ≥70%
- ❌ **Requires ADR** if reuse <70% (manual approval needed)
- ⏱️ **30-min expiry** - old analyses must be refreshed
- 🚫 **Cannot bypass** - agents MUST check before creating

### Languages Supported:
Python, TypeScript, JavaScript, Rust, Go, Java (+ generic fallback)
"""

        return md

    def _generate_agent_assignments(self) -> str:
        """Generate agent assignments based on detected stack."""
        stack = self.config["project"]["detected_stack"]
        recommended = self.config.get("recommended_agents", {})

        md = "### 🎯 Recommended Agents (Based on Your Stack)\n\n"
        md += "| Task Category | Specialist Agent | When to Use |\n"
        md += "|--------------|------------------|-------------|\n"

        # Primary agents
        for agent in recommended.get("primary", []):
            agent_info = self._find_agent_info(agent)
            if agent_info:
                md += f"| **Primary** | `{agent}` | {agent_info['description'][:60]}... |\n"

        # Secondary agents
        for agent in recommended.get("secondary", []):
            agent_info = self._find_agent_info(agent)
            if agent_info:
                md += f"| **Secondary** | `{agent}` | {agent_info['description'][:60]}... |\n"

        md += "\n"
        return md

    def _find_agent_info(self, agent_handle: str) -> Dict[str, str] | None:
        """Find agent information by handle."""
        agent_name = agent_handle.lstrip("@").replace("-", "_")

        for category_agents in self.agents.values():
            for agent in category_agents:
                if agent["name"] == agent_name or agent["handle"] == agent_handle:
                    return agent

        return None

    def _generate_quality_standards(self) -> str:
        """Generate quality standards section."""
        return """
### Code Quality Metrics
- **Cyclomatic Complexity:** ≤10 per function
- **Code Duplication:** <3% across codebase
- **File Size:** ≤500 LOC (soft limit, exceptions allowed)
- **Function Size:** ≤50 LOC recommended
- **Type Coverage:** 100% for public APIs
- **Test Coverage:** ≥85% line coverage

### Reuse Analysis (Multi-Language)
- **Supported Languages:** Python, TypeScript, JavaScript, Rust, Go, Java
- **Search tool:** `search_codebase.py` (semantic search across all 6 languages)
- **Analysis tool:** `analyze_reuse.py` (language-specific analyzers)
- **Enforcement:** `enforce_check_before_create.py` (blocks code creation without analysis)
- **Approval threshold:** ≥70% reuse = auto-approve, <70% = ADR required

### Validation Requirements
- Real data validation (not empty mocks)
- All failures tracked and reported
- Exit code 1 on any failure
- Results before linting
- **Reuse check before ANY code creation**
"""

    def _generate_quick_start_commands(self) -> str:
        """Generate quick start commands based on detected stack."""
        stack = self.config["project"]["detected_stack"]
        commands = ""

        if "Python" in stack.get("languages", []):
            commands += """
### Python Development
```bash
# Install dependencies
uv sync  # or: pip install -r requirements.txt

# Run tests
uv run pytest --cov

# Code quality
uv run ruff check --fix
uv run black .
uv run mypy .
```
"""

        if "TypeScript" in stack.get("languages", []) or "JavaScript" in stack.get("languages", []):
            commands += """
### TypeScript/JavaScript Development
```bash
# Install dependencies
npm install

# Run tests
npm test

# Code quality
npm run lint
npm run type-check
```
"""

        if not commands:
            commands = """
```bash
# Install dependencies
# [Auto-detected from your project]

# Run tests
# [Auto-detected from your project]

# Code quality checks
# [Auto-detected from your project]
```
"""

        return commands

    def _generate_usage_examples(self) -> str:
        """Generate usage examples based on stack."""
        stack = self.config["project"]["detected_stack"]
        examples = ""

        if "Python" in stack.get("languages", []):
            examples += """
```bash
# Backend development
@python-pro implement async endpoint for user registration with Pydantic validation

# Database optimization
@postgres-pro optimize slow query in users table, need <50ms response time
```
"""

        if "FastAPI" in stack.get("frameworks", []):
            examples += """
```bash
# API design
@api-designer design RESTful API for product catalog with versioning

# Microservices architecture
@microservices-architect design event-driven communication between services
```
"""

        if "Docker" in stack.get("tools", []):
            examples += """
```bash
# DevOps
@devops-engineer setup CI/CD pipeline with Docker multi-stage builds

# Platform engineering
@platform-engineer configure Kubernetes deployment with auto-scaling
```
"""

        return examples if examples else "*(Examples will be added based on your stack)*\n"

    def _generate_complete_agent_registry(self) -> str:
        """Generate complete categorized agent registry."""
        md = ""

        category_titles = {
            "orchestration": "🎯 Orchestration & Project Management",
            "backend": "⚙️ Backend Development",
            "frontend": "🎨 Frontend Development",
            "fullstack": "🔄 Full-Stack Development",
            "mobile": "📱 Mobile Development",
            "languages": "🔤 Programming Languages",
            "frameworks": "🏗️ Frameworks",
            "infrastructure": "🏛️ Infrastructure",
            "cloud": "☁️ Cloud Services",
            "devops": "🔧 DevOps & Platform",
            "security": "🔒 Security",
            "quality": "✅ Code Quality & Review",
            "testing": "🧪 Testing",
            "performance": "⚡ Performance",
            "data": "📊 Data Engineering",
            "ai_ml": "🤖 AI & Machine Learning",
            "specialized": "🎯 Specialized Domains",
            "documentation": "📝 Documentation",
            "business": "💼 Business & Support",
            "management": "👔 Project Management",
            "utilities": "🛠️ Utilities & Tools",
        }

        for category, agents in sorted(self.agents.items()):
            if not agents:
                continue

            title = category_titles.get(category, category.replace("_", " ").title())
            md += f"### {title}\n\n"

            for agent in sorted(agents, key=lambda x: x["name"]):
                md += f"- **`{agent['handle']}`** - {agent['description']}\n"

            md += "\n"

        return md


def main() -> None:
    """Main entry point."""
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    config_path = project_root / ".ai-workspace-config.yml"

    print("📝 AI Workspace - CLAUDE.md Generator")
    print("=" * 60)
    print(f"📁 Project: {project_root.absolute()}\n")

    # Find .ai-workspace location
    kit_path = project_root / ".ai-workspace"
    if not kit_path.exists():
        kit_path = Path(__file__).parent.parent

    generator = ClaudeMdGenerator(project_root, kit_path)

    print("⏳ Loading configuration...")
    generator.load_config(config_path)

    print("⏳ Scanning available agents...")
    generator.load_agents()
    print(f"   Found {sum(len(v) for v in generator.agents.values())} agents\n")

    print("⏳ Generating CLAUDE.md...")
    claude_md = generator.generate()

    output_path = project_root / "CLAUDE.md"
    output_path.write_text(claude_md, encoding="utf-8")

    print(f"✅ Generated: {output_path}")
    print(f"   Size: {len(claude_md)} characters")
    print(f"\n📚 Next steps:")
    print(f"   1. Review CLAUDE.md")
    print(f"   2. Customize if needed")
    print(f"   3. Enforcement system is ACTIVE - agents must check before creating code")
    print(f"   4. Multi-language support: Python, TypeScript, JavaScript, Rust, Go, Java")


if __name__ == "__main__":
    main()
