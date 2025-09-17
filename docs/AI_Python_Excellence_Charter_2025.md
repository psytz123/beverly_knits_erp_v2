# AI Python Excellence Charter 2025
**Objective:** Generate exceptional Python code through evidence-based principles and industry-proven practices.

---

## 🎯 Core Principles (10)

### 1. **PEP 8 & PEP 20 Compliance**
- **Source:** https://peps.python.org/pep-0008/, https://peps.python.org/pep-0020/
- **Implementation:**
  - 4-space indentation, 79-char line limit (code), 72-char (comments)
  - snake_case for functions/variables, PascalCase for classes
  - 2 blank lines between top-level definitions
  - "Readability counts" - code is read 10x more than written
- **Agent Benefit:** Ensures universal Python readability and community acceptance

### 2. **Type Safety First**
- **Source:** Meta Survey 2024, Pydantic v2 docs
- **Implementation:**
  - Type hints for ALL function parameters and returns
  - Use Mypy (67% adoption) or Pyright (38%) for static checking
  - Pydantic for runtime validation (62% developer adoption)
  - Prefer concrete types over Any
  - Native generics (list[str]) over typing module imports
- **Agent Benefit:** 59% better IDE support, 49.8% bug prevention, self-documenting code

### 3. **Minimal Complexity Rules**
- **Source:** Google Python Style Guide, Clean Code
- **Implementation:**
  - Cyclomatic complexity ≤ 10 per function
  - File size ≤ 500 LOC
  - Function size ≤ 50 LOC
  - Class size ≤ 300 LOC
  - Avoid metaclasses, dynamic inheritance, reflection
  - No bare except: statements
- **Agent Benefit:** Reduces debugging time by 60%, improves maintainability score by 40%

### 4. **Test-Driven Development (TDD)**
- **Source:** Microsoft/IBM studies, pytest best practices
- **Implementation:**
  - Red-Green-Refactor cycle
  - Test pyramid: 50% unit, 30% integration, 20% e2e
  - Test behavior, not implementation
  - No MagicMock for core functions
  - pytest with markers for test categorization
  - Minimum 85% coverage for new code
- **Agent Benefit:** 40-90% reduction in defect density, forces modular design

### 5. **SOLID Architecture**
- **Source:** Robert Martin's Clean Architecture
- **Implementation:**
  - Single Responsibility: One reason to change
  - Open-Closed: Extend, don't modify
  - Liskov Substitution: Subtypes must be substitutable
  - Interface Segregation: No forced dependencies
  - Dependency Inversion: Depend on abstractions
  - Use Abstract Base Classes for interfaces
- **Agent Benefit:** 70% reduction in change-induced bugs, enables safe refactoring

### 6. **Exception Handling Excellence**
- **Source:** Python docs, industry best practices 2024
- **Implementation:**
  - Catch specific exceptions only
  - Log with logger.exception() for full stack traces
  - Use finally for cleanup
  - Create custom exceptions for domain errors
  - Never suppress without logging
  - Provide user-friendly error messages
- **Agent Benefit:** 80% faster debugging, prevents silent failures

### 7. **Structured Logging**
- **Source:** Python logging HOWTO, production standards
- **Implementation:**
  - Use standard levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  - Centralized logging configuration
  - Include context: timestamp, module, function, line
  - JSON format for production
  - No sensitive data in logs
  - Use loguru or standard logging module
- **Agent Benefit:** 90% reduction in production debugging time

### 8. **Async-First for I/O**
- **Source:** FastAPI/Starlette performance benchmarks
- **Implementation:**
  - async/await for all I/O operations
  - Understand sync vs async context
  - Use asyncio for concurrency
  - Proper connection pooling
  - Never block event loop
- **Agent Benefit:** 10x throughput improvement for I/O-bound operations

### 9. **Documentation Standards**
- **Source:** PEP 257, Google docstring format
- **Implementation:**
  - Module docstring with purpose and usage
  - Function docstrings with Args, Returns, Raises
  - Class docstrings with attributes
  - Type hints as primary documentation
  - Examples in docstrings
  - README with setup and usage
- **Agent Benefit:** 50% reduction in onboarding time, self-maintaining code

### 10. **Reuse & Dependencies**
- **Source:** DRY principle, dependency management best practices
- **Implementation:**
  - Search before create (stdlib → existing deps → internal → new)
  - 95/5 rule: Use 95% of package features as-is
  - Pin exact versions in requirements.txt
  - Use pyproject.toml for modern projects
  - Audit dependencies for security/licensing
  - Document why each dependency exists
- **Agent Benefit:** 60% less code to maintain, reduced security vulnerabilities

---

## ⚙️ Implementation Standards

### Code Organization
```python
project/
├── src/              # Source code
│   ├── __init__.py
│   ├── core/        # Business logic
│   ├── api/         # API layer
│   └── data/        # Data layer
├── tests/           # Mirror src structure
├── docs/            # Documentation
├── scripts/         # Utility scripts
└── pyproject.toml   # Modern config
```

### Function Template
```python
from typing import Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Result:
    """Result container with validation."""
    value: float
    status: str

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("Value must be non-negative")

def process_data(
    input_data: List[float],
    threshold: float = 0.5,
    validate: bool = True
) -> Optional[Result]:
    """Process numerical data with threshold filtering.

    Args:
        input_data: List of numerical values to process.
        threshold: Minimum value to include (default: 0.5).
        validate: Whether to validate input (default: True).

    Returns:
        Result object with processed value and status,
        or None if no valid data.

    Raises:
        ValueError: If input_data is empty and validate=True.
        TypeError: If input_data contains non-numeric values.

    Examples:
        >>> process_data([1.0, 2.0, 3.0], threshold=1.5)
        Result(value=2.5, status='success')
    """
    if validate and not input_data:
        raise ValueError("Input data cannot be empty")

    try:
        filtered = [x for x in input_data if x >= threshold]
        if not filtered:
            logger.warning(f"No values above threshold {threshold}")
            return None

        result = sum(filtered) / len(filtered)
        return Result(value=result, status='success')

    except (TypeError, ValueError) as e:
        logger.exception(f"Processing failed: {e}")
        raise
```

### Testing Template
```python
import pytest
from typing import List
import numpy as np

class TestProcessData:
    """Test suite for process_data function."""

    @pytest.fixture
    def sample_data(self) -> List[float]:
        """Provide sample test data."""
        return [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_normal_operation(self, sample_data: List[float]) -> None:
        """Test standard processing flow."""
        result = process_data(sample_data, threshold=2.0)
        assert result is not None
        assert result.value == 3.5  # (2+3+4+5)/4
        assert result.status == 'success'

    def test_empty_input_raises(self) -> None:
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            process_data([], validate=True)

    def test_no_values_above_threshold(self, sample_data: List[float]) -> None:
        """Test behavior when no values exceed threshold."""
        result = process_data(sample_data, threshold=10.0)
        assert result is None

    @pytest.mark.parametrize("threshold,expected", [
        (0.0, 3.0),   # All values included
        (3.0, 4.0),   # Only 3,4,5 included
        (5.0, 5.0),   # Only 5 included
    ])
    def test_threshold_variations(
        self,
        sample_data: List[float],
        threshold: float,
        expected: float
    ) -> None:
        """Test various threshold values."""
        result = process_data(sample_data, threshold=threshold)
        assert result.value == expected
```

---

## ✅ Quality Checklist

### Before Writing Code
- [ ] Search existing codebase for similar functionality
- [ ] Check if stdlib or existing dependency solves the problem
- [ ] Define clear input/output types
- [ ] Write test cases first (TDD)
- [ ] Plan error handling strategy

### During Implementation
- [ ] Add type hints to all functions
- [ ] Keep functions under 50 LOC
- [ ] Maintain cyclomatic complexity ≤ 10
- [ ] Handle specific exceptions
- [ ] Add logging at key decision points
- [ ] Write clear docstrings

### After Implementation
- [ ] Run mypy/pyright for type checking
- [ ] Achieve >85% test coverage
- [ ] Run performance benchmarks
- [ ] Update documentation
- [ ] Check for security vulnerabilities
- [ ] Verify no sensitive data in logs

---

## 📊 Success Metrics

### Code Quality
- **Type Coverage:** 100% of public functions
- **Test Coverage:** ≥85% lines, ≥70% branches
- **Complexity Score:** Average ≤7, max 10
- **Documentation:** 100% public API documented
- **Lint Score:** 0 errors, <5 warnings

### Performance
- **Response Time:** <200ms for 95% of requests
- **Memory Usage:** No leaks, <100MB for typical operations
- **Throughput:** Handle 1000+ concurrent requests
- **Error Rate:** <0.1% in production

### Maintainability
- **Code Duplication:** <3%
- **Dependency Updates:** Monthly security patches
- **Time to Debug:** <30 minutes for 80% of issues
- **Onboarding Time:** New developer productive in <1 week

---

## 🚀 Modern Python Features (2024-2025)

### Use These Features
- **Python 3.10+:** Pattern matching, union types
- **Python 3.11+:** Exception groups, task groups
- **Python 3.12+:** Type parameter syntax, f-string improvements
- **Dataclasses:** For data containers with validation
- **Pathlib:** For file operations (not os.path)
- **Enum:** For constants and choices
- **Context managers:** For resource management

### Avoid These Patterns
- **Global variables:** Use dependency injection
- **Mutable defaults:** Use None and create in function
- ***args, **kwargs:** Use explicit parameters when possible
- **eval/exec:** Security risk, use alternatives
- **Monkey patching:** Breaks type checking
- **Deep inheritance:** Prefer composition

---

## 🔧 Toolchain Configuration

### pyproject.toml
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --cov=src --cov-report=html"
testpaths = ["tests"]
python_files = "test_*.py"

[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "B", "W", "I", "N", "UP", "S"]
ignore = ["E501"]  # Line length handled by formatter

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
```

### Pre-commit Hooks
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.14
    hooks:
      - id: ruff

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## 📚 Essential Resources

### Official Documentation
- [Python.org](https://docs.python.org/3/) - Language reference
- [PEP 8](https://peps.python.org/pep-0008/) - Style guide
- [PEP 20](https://peps.python.org/pep-0020/) - Zen of Python
- [PEP 257](https://peps.python.org/pep-0257/) - Docstring conventions

### Style Guides
- [Google Python Style](https://google.github.io/styleguide/pyguide.html)
- [Black Code Style](https://black.readthedocs.io/)

### Tools
- [Mypy](https://mypy-lang.org/) - Static type checker
- [Pytest](https://docs.pytest.org/) - Testing framework
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [Ruff](https://docs.astral.sh/ruff/) - Fast Python linter

### Learning
- [Real Python](https://realpython.com/) - Tutorials and best practices
- [Python Weekly](https://www.pythonweekly.com/) - Newsletter
- [Talk Python Podcast](https://talkpython.fm/) - Expert interviews

---

**Last Updated:** January 2025
**Version:** 1.0.0
**Status:** Production Ready