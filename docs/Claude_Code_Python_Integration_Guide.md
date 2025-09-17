# Claude Code Python Excellence Integration Guide

## 🚀 Quick Start: 3 Methods to Integrate Python Best Practices

### Method 1: CLAUDE.md Files (Easiest)
```bash
# Global configuration (applies to all projects)
~/.claude/CLAUDE.md

# Project-specific configuration (overrides global)
/project/CLAUDE.md
```

### Method 2: Hooks & Automation (Most Powerful)
```bash
# Configure hooks in settings
~/.claude/hooks/
├── pre-tool-use.py      # Validate before actions
├── post-tool-use.py     # Enforce standards after
└── user-prompt.py       # Add context to prompts
```

### Method 3: Custom Subagents (Most Scalable)
```json
{
  "subagents": [
    {
      "name": "python-excellence",
      "tools": ["*"],
      "systemPrompt": "path/to/python_charter.md"
    }
  ]
}
```

---

## 📋 Complete Integration Strategies

### 1. CLAUDE.md Configuration (Immediate Impact)

**Global CLAUDE.md** (`~/.claude/CLAUDE.md`):
```markdown
# Python Excellence Standards

## Core Rules
1. ALWAYS use type hints for all functions
2. ALWAYS write tests before implementation (TDD)
3. NEVER use bare except statements
4. NEVER exceed 500 LOC per file
5. ALWAYS handle errors with specific exceptions

## Before Writing Code
- Run: `rg "similar_function"` to check for existing code
- Run: `pytest tests/` to ensure no breaking changes
- Check complexity with: `radon cc -s src/`

## Code Template
Every Python file must follow:
\```python
"""Module purpose and usage."""
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

def function_name(param: type) -> return_type:
    """Docstring with Args, Returns, Raises."""
    try:
        # Implementation
        pass
    except SpecificError as e:
        logger.exception(f"Context: {e}")
        raise
\```

## Validation Commands
After EVERY file change, run:
1. `mypy src/` - Type checking
2. `pytest tests/` - Test suite
3. `ruff check src/` - Linting
4. `black src/` - Formatting
```

**Project CLAUDE.md** (project root):
```markdown
# Project-Specific Python Standards

Extends global standards with:

## Architecture
- Use FastAPI for APIs
- Pydantic for data validation
- SQLAlchemy for database
- pytest for testing

## File Structure
src/
├── api/        # FastAPI routes
├── core/       # Business logic (SOLID)
├── models/     # Pydantic & SQLAlchemy
└── services/   # External integrations

## Required Checks
make lint      # Must pass
make test      # >85% coverage
make typecheck # Zero errors
```

### 2. Hooks for Automated Enforcement

**Pre-Tool Hook** (`~/.claude/hooks/pre_tool_use.py`):
```python
#!/usr/bin/env python3
"""Validate Python code before writing."""

import json
import sys
import ast
import re
from pathlib import Path

def validate_python_code(tool_name: str, params: dict) -> dict:
    """Enforce Python standards before file operations."""

    if tool_name in ["Write", "Edit", "MultiEdit"]:
        file_path = params.get("file_path", "")

        if file_path.endswith(".py"):
            content = params.get("content") or params.get("new_string", "")

            # Check for type hints
            if "def " in content and "->" not in content:
                return {
                    "error": "Missing return type hint. Add -> Type to function",
                    "suggestion": "Use -> None for functions without return"
                }

            # Check for bare except
            if "except:" in content:
                return {
                    "error": "Bare except not allowed. Use specific exceptions",
                    "suggestion": "except (ValueError, TypeError) as e:"
                }

            # Check complexity
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = calculate_complexity(node)
                        if complexity > 10:
                            return {
                                "error": f"Function {node.name} complexity {complexity} > 10",
                                "suggestion": "Break into smaller functions"
                            }
            except SyntaxError as e:
                return {"error": f"Syntax error: {e}"}

            # Check line count
            lines = content.split('\n')
            if len(lines) > 500:
                return {
                    "error": f"File has {len(lines)} lines (max 500)",
                    "suggestion": "Split into multiple modules"
                }

    return {"status": "ok"}

def calculate_complexity(node):
    """Calculate cyclomatic complexity."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity

if __name__ == "__main__":
    tool_data = json.loads(sys.stdin.read())
    result = validate_python_code(tool_data["tool"], tool_data["params"])

    if "error" in result:
        print(f"❌ Python Standards Violation: {result['error']}")
        print(f"💡 Suggestion: {result['suggestion']}")
        sys.exit(1)

    sys.exit(0)
```

**Post-Tool Hook** (`~/.claude/hooks/post_tool_use.py`):
```python
#!/usr/bin/env python3
"""Run quality checks after Python file changes."""

import json
import subprocess
import sys
from pathlib import Path

def run_quality_checks(tool_name: str, params: dict, result: dict) -> None:
    """Automatically run quality tools after file changes."""

    if tool_name in ["Write", "Edit", "MultiEdit"]:
        file_path = params.get("file_path", "")

        if file_path.endswith(".py"):
            print(f"\n🔍 Running Python quality checks on {file_path}...")

            checks = [
                ("Type checking", ["mypy", file_path]),
                ("Linting", ["ruff", "check", file_path]),
                ("Formatting", ["black", "--check", file_path]),
                ("Complexity", ["radon", "cc", "-nc", file_path])
            ]

            for check_name, command in checks:
                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode != 0:
                        print(f"⚠️  {check_name} issues found:")
                        print(result.stdout or result.stderr)
                        print(f"Run: {' '.join(command[:-1])} {file_path}")
                    else:
                        print(f"✅ {check_name} passed")

                except subprocess.TimeoutExpired:
                    print(f"⏱️  {check_name} timed out")
                except FileNotFoundError:
                    print(f"⚠️  {check_name} tool not found. Install with: pip install {command[0]}")

if __name__ == "__main__":
    data = json.loads(sys.stdin.read())
    run_quality_checks(data["tool"], data["params"], data.get("result", {}))
```

**User Prompt Hook** (`~/.claude/hooks/user_prompt_submit.py`):
```python
#!/usr/bin/env python3
"""Enhance prompts with Python context."""

import sys
import os
from pathlib import Path

def enhance_prompt(prompt: str) -> str:
    """Add Python best practices context to prompts."""

    python_keywords = ["python", "function", "class", "test", "api", "debug"]

    if any(keyword in prompt.lower() for keyword in python_keywords):
        context = """
Remember to follow Python Excellence Standards:
- Use type hints for all functions
- Write tests first (TDD)
- Keep functions under 50 LOC
- Handle specific exceptions
- Add logging for debugging
- Document with docstrings
"""

        # Add project-specific context if in Python project
        if Path("pyproject.toml").exists() or Path("requirements.txt").exists():
            context += """
Project uses:
- mypy for type checking
- pytest for testing
- black for formatting
- ruff for linting
Run 'make test' after changes.
"""

        return f"{prompt}\n\n{context}"

    return prompt

if __name__ == "__main__":
    original_prompt = sys.stdin.read()
    enhanced = enhance_prompt(original_prompt)
    print(enhanced)
```

### 3. Custom Python Subagents

**Python Excellence Subagent** (`~/.claude/subagents/python-excellence.json`):
```json
{
  "name": "python-excellence",
  "description": "Python development with enforced best practices",
  "tools": ["*"],
  "systemPrompt": "~/.claude/subagents/python_system_prompt.md",
  "config": {
    "maxTokens": 8192,
    "temperature": 0.3,
    "requireTests": true,
    "enforceTypes": true,
    "maxComplexity": 10
  }
}
```

**Python System Prompt** (`~/.claude/subagents/python_system_prompt.md`):
```markdown
You are a Python Excellence Agent that MUST follow these rules:

## Mandatory Requirements
1. EVERY function has type hints
2. EVERY file has a module docstring
3. EVERY public function has a docstring
4. NO bare except statements
5. NO global variables
6. NO mutable default arguments

## Code Generation Rules
- Generate tests BEFORE implementation
- Use dataclasses for data containers
- Use pathlib for file operations
- Use logging instead of print
- Handle errors explicitly

## Quality Gates
Before considering code complete:
- [ ] mypy reports 0 errors
- [ ] pytest coverage >85%
- [ ] ruff check passes
- [ ] black formatted
- [ ] Complexity <10

## Import Order
1. Standard library
2. Third-party packages
3. Local imports
(One blank line between groups)
```

### 4. VS Code Integration

**`.vscode/settings.json`**:
```json
{
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.rulers": [88],
    "editor.tabSize": 4
  },
  "python.analysis.typeCheckingMode": "strict",
  "python.analysis.autoImportCompletions": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".mypy_cache": true,
    ".ruff_cache": true
  }
}
```

### 5. Makefile for Automation

**`Makefile`**:
```makefile
.PHONY: install lint format test typecheck quality all clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check src/ tests/
	pylint src/

format:
	black src/ tests/
	isort src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

typecheck:
	mypy src/ --strict

quality: lint typecheck test
	@echo "✅ All quality checks passed!"

watch:
	watchdog -w src -w tests -c "clear && make quality"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage

# Auto-fix what can be fixed
fix:
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

# Run before committing
pre-commit: fix quality
	@echo "✅ Ready to commit!"

# Development workflow
dev:
	@echo "Starting development mode..."
	@make install
	@make quality
	@make watch
```

### 6. Pre-commit Configuration

**`.pre-commit-config.yaml`**:
```yaml
default_language_version:
  python: python3.11

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.14
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-pyyaml]
        args: [--strict, --ignore-missing-imports]

  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: [--tb=short, --quiet]
```

---

## 🎯 Implementation Roadmap

### Week 1: Foundation
1. **Day 1-2:** Set up CLAUDE.md files (global + project)
2. **Day 3-4:** Install and configure VS Code settings
3. **Day 5-7:** Set up Makefile and basic tooling

### Week 2: Automation
1. **Day 1-2:** Implement pre/post tool hooks
2. **Day 3-4:** Configure pre-commit hooks
3. **Day 5-7:** Test and refine automation

### Week 3: Advanced
1. **Day 1-3:** Create custom Python subagents
2. **Day 4-5:** Set up CI/CD integration
3. **Day 6-7:** Document and train team

---

## 📊 Measuring Success

### Immediate Metrics (Week 1)
- [ ] Type hint coverage: 100%
- [ ] Linting errors: 0
- [ ] Test coverage: >70%

### Short-term Metrics (Month 1)
- [ ] Cyclomatic complexity: avg <7
- [ ] Code duplication: <3%
- [ ] Bug rate: -50%

### Long-term Metrics (Quarter 1)
- [ ] Development velocity: +30%
- [ ] Production issues: -60%
- [ ] Code review time: -40%

---

## 🔧 Troubleshooting

### Common Issues

**Issue:** Hooks not triggering
```bash
# Check hook permissions
chmod +x ~/.claude/hooks/*.py

# Verify hook configuration
cat ~/.claude/settings.json
```

**Issue:** Type checking too strict
```python
# Add gradual typing
# mypy.ini
[mypy]
warn_return_any = False
warn_unused_ignores = False
allow_untyped_defs = True  # Start permissive
```

**Issue:** Tests failing on Claude Code changes
```bash
# Run tests in isolation
pytest tests/unit -v  # Unit tests only
pytest -m "not integration"  # Skip integration
```

---

## 🚀 Quick Commands

```bash
# One-line setup
curl -L https://your-repo/claude-python-setup.sh | bash

# Validate setup
python -m claude_python_check

# Generate project template
claude-python-init my_project

# Run all checks
make quality

# Auto-fix issues
make fix
```

---

**Version:** 1.0.0
**Last Updated:** January 2025
**Support:** GitHub Issues or Slack #claude-python