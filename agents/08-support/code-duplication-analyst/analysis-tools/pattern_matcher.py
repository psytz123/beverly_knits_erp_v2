"""Pattern Matcher - Boilerplate and Common Pattern Detection.

This tool identifies repeated code patterns and boilerplate across the codebase.
It recognizes common microservice patterns that can be extracted to shared libraries.

Created: 2025-10-05
Modified: 2025-10-05
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PatternOccurrence:
    """A single occurrence of a pattern in a file."""

    file_path: Path
    line_number: int
    code_snippet: str
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": str(self.file_path),
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "context": self.context,
        }


@dataclass
class Pattern:
    """A detected code pattern with all its occurrences."""

    pattern_id: str
    name: str
    description: str
    category: str
    occurrences: list[PatternOccurrence] = field(default_factory=list)
    refactoring_recommendation: str = ""
    delegate_to: str = ""
    estimated_loc_reduction: int = 0

    @property
    def occurrence_count(self) -> int:
        """Number of times this pattern occurs."""
        return len(self.occurrences)

    @property
    def affected_files(self) -> set[Path]:
        """Set of files where this pattern occurs."""
        return {occ.file_path for occ in self.occurrences}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "occurrence_count": self.occurrence_count,
            "affected_files_count": len(self.affected_files),
            "estimated_loc_reduction": self.estimated_loc_reduction,
            "refactoring_recommendation": self.refactoring_recommendation,
            "delegate_to": self.delegate_to,
            "occurrences": [occ.to_dict() for occ in self.occurrences],
        }


@dataclass
class PatternAnalysisResult:
    """Results from pattern matching analysis."""

    analysis_id: str
    timestamp: str
    root_path: Path
    total_files_analyzed: int = 0
    patterns_detected: list[Pattern] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=list)

    @property
    def total_occurrences(self) -> int:
        """Total number of pattern occurrences."""
        return sum(p.occurrence_count for p in self.patterns_detected)

    @property
    def total_estimated_loc_reduction(self) -> int:
        """Total estimated LOC that can be reduced."""
        return sum(p.estimated_loc_reduction for p in self.patterns_detected)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp,
            "root_path": str(self.root_path),
            "total_files_analyzed": self.total_files_analyzed,
            "patterns_detected_count": len(self.patterns_detected),
            "total_occurrences": self.total_occurrences,
            "total_estimated_loc_reduction": self.total_estimated_loc_reduction,
            "file_patterns": self.file_patterns,
            "patterns": [p.to_dict() for p in self.patterns_detected],
        }


class PatternMatcher:
    """Matcher for detecting common code patterns and boilerplate."""

    # Pattern definitions with regex and metadata
    PATTERNS = {
        "health_check": {
            "name": "Health Check Endpoint",
            "category": "api_endpoint",
            "regex": r'@router\.get\(["\']/(health|ready|ping)',
            "description": "Standard health check endpoint pattern",
            "recommendation": "Extract to shared/api/health.py with configurable responses",
            "delegate_to": "refactoring-specialist",
            "estimated_loc": 15,
        },
        "database_session": {
            "name": "Database Session Dependency",
            "category": "dependency_injection",
            "regex": r"def get_db\(\)|async def get_db\(\)",
            "description": "Database session dependency injection pattern",
            "recommendation": "Create unified database session factory in shared/database/",
            "delegate_to": "python-pro",
            "estimated_loc": 20,
        },
        "redis_cache": {
            "name": "Redis Cache Setup",
            "category": "caching",
            "regex": r"def get_redis\(\)|async def get_redis\(\)",
            "description": "Redis cache client initialization pattern",
            "recommendation": "Consolidate to shared/cache/redis.py",
            "delegate_to": "python-pro",
            "estimated_loc": 25,
        },
        "settings_pattern": {
            "name": "Settings Configuration",
            "category": "configuration",
            "regex": r"class Settings\(BaseSettings\):",
            "description": "Pydantic settings configuration pattern",
            "recommendation": "Create base settings class in shared/config/base.py",
            "delegate_to": "backend-developer",
            "estimated_loc": 30,
        },
        "postgres_dsn": {
            "name": "PostgreSQL DSN Builder",
            "category": "database_config",
            "regex": r"PostgresDsn\.build\(",
            "description": "PostgreSQL connection string builder pattern",
            "recommendation": "Extract DSN building logic to shared/database/postgres.py",
            "delegate_to": "backend-developer",
            "estimated_loc": 10,
        },
        "kafka_client": {
            "name": "Kafka Client Setup",
            "category": "messaging",
            "regex": r"class KafkaClient|def get_kafka|async def get_kafka",
            "description": "Kafka client initialization pattern",
            "recommendation": "Create unified Kafka client in shared/messaging/kafka.py",
            "delegate_to": "backend-developer",
            "estimated_loc": 40,
        },
        "fastapi_router": {
            "name": "FastAPI Router Initialization",
            "category": "api_setup",
            "regex": r'router = APIRouter\(.*tags=\[',
            "description": "FastAPI router initialization with tags",
            "recommendation": "Create router factory in shared/api/routers.py",
            "delegate_to": "refactoring-specialist",
            "estimated_loc": 5,
        },
        "cors_middleware": {
            "name": "CORS Middleware Setup",
            "category": "middleware",
            "regex": r"CORSMiddleware.*allow_origins",
            "description": "CORS middleware configuration pattern",
            "recommendation": "Standardize in shared/middleware/cors.py",
            "delegate_to": "backend-developer",
            "estimated_loc": 12,
        },
        "pytest_fixture": {
            "name": "Pytest Async Fixture",
            "category": "testing",
            "regex": r"@pytest\.fixture|@pytest_asyncio\.fixture",
            "description": "Test fixture pattern",
            "recommendation": "Move common fixtures to shared/testing/fixtures.py",
            "delegate_to": "refactoring-specialist",
            "estimated_loc": 20,
        },
        "alembic_env": {
            "name": "Alembic Migration Environment",
            "category": "database_migration",
            "regex": r"def run_migrations_online\(\)",
            "description": "Alembic migration environment setup",
            "recommendation": "Standardize migration env in shared/database/migrations/",
            "delegate_to": "backend-developer",
            "estimated_loc": 50,
        },
        "base_model": {
            "name": "SQLAlchemy Base Model",
            "category": "database_model",
            "regex": r"class Base\(DeclarativeBase\)|DeclarativeBase = declarative_base",
            "description": "SQLAlchemy declarative base pattern",
            "recommendation": "Single base model in shared/models/base.py",
            "delegate_to": "python-pro",
            "estimated_loc": 15,
        },
        "repository_pattern": {
            "name": "Repository Pattern",
            "category": "data_access",
            "regex": r"class \w+Repository:",
            "description": "Repository pattern for data access",
            "recommendation": "Create base repository class in shared/repositories/",
            "delegate_to": "python-pro",
            "estimated_loc": 30,
        },
        "service_pattern": {
            "name": "Service Layer Pattern",
            "category": "business_logic",
            "regex": r"class \w+Service:",
            "description": "Service layer pattern for business logic",
            "recommendation": "Establish base service class in shared/services/",
            "delegate_to": "python-pro",
            "estimated_loc": 25,
        },
        "error_handler": {
            "name": "Exception Handler",
            "category": "error_handling",
            "regex": r"@app\.exception_handler",
            "description": "FastAPI exception handler pattern",
            "recommendation": "Centralize in shared/middleware/error_handler.py",
            "delegate_to": "backend-developer",
            "estimated_loc": 20,
        },
        "logging_setup": {
            "name": "Logging Configuration",
            "category": "observability",
            "regex": r"logging\.getLogger\(__name__\)|structlog\.get_logger",
            "description": "Logger initialization pattern",
            "recommendation": "Unified logging config in shared/utils/logging.py",
            "delegate_to": "refactoring-specialist",
            "estimated_loc": 10,
        },
    }

    DEFAULT_EXCLUDE = [
        "__pycache__",
        "*.pyc",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".pytest_cache",
        "*.egg-info",
    ]

    def __init__(
        self,
        root_path: Path,
        file_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        """Initialize the pattern matcher.

        Args:
            root_path: Root directory to analyze.
            file_patterns: File patterns to include (e.g., ['*.py']).
            exclude_patterns: Patterns to exclude from analysis.
        """
        self.root_path = Path(root_path).resolve()
        self.file_patterns = file_patterns or ["*.py"]
        self.exclude_patterns = exclude_patterns or self.DEFAULT_EXCLUDE

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded.

        Args:
            path: Path to check.

        Returns:
            True if path should be excluded, False otherwise.
        """
        path_str = str(path)
        for pattern in self.exclude_patterns:
            if pattern.startswith("*"):
                if path_str.endswith(pattern[1:]):
                    return True
            elif pattern in path.parts:
                return True
        return False

    def _find_pattern_in_file(
        self, file_path: Path, pattern_id: str, pattern_def: dict[str, Any]
    ) -> list[PatternOccurrence]:
        """Find all occurrences of a pattern in a file.

        Args:
            file_path: Path to the file to search.
            pattern_id: Identifier for the pattern.
            pattern_def: Pattern definition with regex and metadata.

        Returns:
            List of pattern occurrences found.
        """
        occurrences: list[PatternOccurrence] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            regex = re.compile(pattern_def["regex"])

            for line_num, line in enumerate(lines, start=1):
                if regex.search(line):
                    # Get context (3 lines before and after)
                    context_start = max(0, line_num - 4)
                    context_end = min(len(lines), line_num + 3)
                    context_lines = lines[context_start:context_end]
                    context = "".join(context_lines).strip()

                    occurrence = PatternOccurrence(
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip(),
                        context=context,
                    )
                    occurrences.append(occurrence)

        except (UnicodeDecodeError, OSError) as e:
            print(f"Warning: Could not read {file_path}: {e}")

        return occurrences

    def analyze(self) -> PatternAnalysisResult:
        """Analyze codebase for common patterns.

        Returns:
            PatternAnalysisResult containing all detected patterns.
        """
        analysis_id = f"pattern-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        timestamp = datetime.now().isoformat()

        # Collect all files to analyze
        files_to_analyze: list[Path] = []
        for pattern in self.file_patterns:
            for file_path in self.root_path.rglob(pattern):
                if file_path.is_file() and not self._should_exclude(file_path):
                    files_to_analyze.append(file_path)

        total_files = len(files_to_analyze)

        # Search for each pattern in all files
        detected_patterns: list[Pattern] = []

        for pattern_id, pattern_def in self.PATTERNS.items():
            all_occurrences: list[PatternOccurrence] = []

            for file_path in files_to_analyze:
                occurrences = self._find_pattern_in_file(
                    file_path, pattern_id, pattern_def
                )
                all_occurrences.extend(occurrences)

            # Only create pattern if occurrences found
            if all_occurrences:
                # Calculate LOC reduction (pattern occurs N times, save M lines each)
                occurrence_count = len(all_occurrences)
                loc_per_occurrence = pattern_def.get("estimated_loc", 10)
                # We keep one implementation, so reduce by (N-1) * M
                estimated_reduction = max(0, (occurrence_count - 1) * loc_per_occurrence)

                pattern = Pattern(
                    pattern_id=pattern_id,
                    name=pattern_def["name"],
                    description=pattern_def["description"],
                    category=pattern_def["category"],
                    occurrences=all_occurrences,
                    refactoring_recommendation=pattern_def["recommendation"],
                    delegate_to=pattern_def["delegate_to"],
                    estimated_loc_reduction=estimated_reduction,
                )
                detected_patterns.append(pattern)

        # Sort by estimated LOC reduction (highest first)
        detected_patterns.sort(
            key=lambda p: p.estimated_loc_reduction, reverse=True
        )

        return PatternAnalysisResult(
            analysis_id=analysis_id,
            timestamp=timestamp,
            root_path=self.root_path,
            total_files_analyzed=total_files,
            patterns_detected=detected_patterns,
            file_patterns=self.file_patterns,
        )


def generate_markdown_report(result: PatternAnalysisResult) -> str:
    """Generate a markdown report from pattern analysis.

    Args:
        result: Analysis results to report on.

    Returns:
        Markdown formatted report string.
    """
    report = f"""# Code Pattern Analysis Report

**Analysis ID**: {result.analysis_id}
**Timestamp**: {result.timestamp}
**Root Path**: {result.root_path}

## Executive Summary

- **Files Analyzed**: {result.total_files_analyzed:,}
- **Patterns Detected**: {len(result.patterns_detected)}
- **Total Occurrences**: {result.total_occurrences:,}
- **Estimated LOC Reduction**: {result.total_estimated_loc_reduction:,} lines
- **File Patterns**: {', '.join(result.file_patterns)}

## Detected Patterns by Impact

"""

    for pattern in result.patterns_detected:
        report += f"\n### {pattern.name} ({pattern.occurrence_count} occurrences)\n\n"
        report += f"**Category**: {pattern.category}\n\n"
        report += f"**Description**: {pattern.description}\n\n"
        report += f"**Affected Files**: {len(pattern.affected_files)}\n\n"
        report += f"**Estimated LOC Reduction**: {pattern.estimated_loc_reduction} lines\n\n"
        report += f"**Recommendation**: {pattern.refactoring_recommendation}\n\n"
        report += f"**Delegate To**: `{pattern.delegate_to}`\n\n"

        # Show first 5 occurrences as examples
        report += "**Sample Occurrences**:\n\n"
        for occ in pattern.occurrences[:5]:
            report += f"- `{occ.file_path}:{occ.line_number}`\n"
            report += f"  ```python\n  {occ.code_snippet}\n  ```\n\n"

        if len(pattern.occurrences) > 5:
            report += f"*... and {len(pattern.occurrences) - 5} more occurrences*\n\n"

    report += """
## Refactoring Strategy

### Phase 1: High Impact Patterns (>100 LOC Reduction)
1. Extract patterns with highest LOC reduction potential
2. Create shared library implementations
3. Update all services incrementally
4. Validate with comprehensive tests

### Phase 2: Medium Impact Patterns (50-100 LOC)
1. Design flexible abstractions
2. Implement in shared library
3. Gradual migration of services

### Phase 3: Low Impact Patterns (<50 LOC)
1. Standardize implementation patterns
2. Document best practices
3. Apply in new development

## Delegation Plan

**Refactoring Specialist**:
- Health check endpoints
- FastAPI router patterns
- Pytest fixtures
- Logging setup

**Python Pro**:
- Database session factory
- Redis cache setup
- Base model extraction
- Repository pattern base class
- Service pattern base class

**Backend Developer**:
- Settings configuration base
- Kafka client consolidation
- CORS middleware standardization
- Alembic environment template
- Error handler centralization

## Next Steps

1. Review pattern analysis with team
2. Prioritize patterns by business value
3. Create detailed refactoring tasks
4. Assign to specialist agents
5. Track progress and measure impact

---
*Generated by Code Duplication Analyst Agent - Pattern Matcher*
"""

    return report


def main() -> None:
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Analyze codebase for common patterns"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("./new/src"),
        help="Root path to analyze (default: ./new/src)",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.py"],
        help="File patterns to include (default: *.py)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Additional patterns to exclude",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output JSON results to file",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        help="Output markdown report to file",
    )

    args = parser.parse_args()

    # Run pattern matcher
    matcher = PatternMatcher(
        root_path=args.path,
        file_patterns=args.patterns,
        exclude_patterns=args.exclude,
    )

    print(f"Analyzing {args.path} for code patterns...")
    result = matcher.analyze()

    # Output JSON if requested
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"JSON results written to {args.output_json}")

    # Output markdown if requested
    if args.output_md:
        report = generate_markdown_report(result)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Markdown report written to {args.output_md}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Analysis Complete: {result.analysis_id}")
    print(f"{'='*60}")
    print(f"Files Analyzed: {result.total_files_analyzed:,}")
    print(f"Patterns Detected: {len(result.patterns_detected)}")
    print(f"Total Occurrences: {result.total_occurrences:,}")
    print(f"Estimated LOC Reduction: {result.total_estimated_loc_reduction:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
