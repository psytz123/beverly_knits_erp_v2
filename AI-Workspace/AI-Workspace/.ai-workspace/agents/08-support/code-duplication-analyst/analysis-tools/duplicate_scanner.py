"""Duplicate File Scanner - Hash-based Exact Duplicate Detection.

This tool scans a codebase for exact duplicate files using MD5/SHA256 hashing.
It operates in READ-ONLY mode and produces documentation for refactoring teams.

Created: 2025-10-05
Modified: 2025-10-05
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class DuplicateGroup:
    """Group of duplicate files with the same hash."""

    hash_value: str
    algorithm: str
    file_size: int
    files: list[Path] = field(default_factory=list)
    line_count: int = 0
    category: str = "unknown"

    @property
    def duplicate_count(self) -> int:
        """Number of duplicate files in this group."""
        return len(self.files)

    @property
    def total_redundant_loc(self) -> int:
        """Total redundant lines of code (excludes first occurrence)."""
        return self.line_count * (self.duplicate_count - 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hash": self.hash_value,
            "algorithm": self.algorithm,
            "file_size": self.file_size,
            "line_count": self.line_count,
            "duplicate_count": self.duplicate_count,
            "total_redundant_loc": self.total_redundant_loc,
            "category": self.category,
            "files": [str(f) for f in self.files],
        }


@dataclass
class ScanResult:
    """Results from duplicate file scanning."""

    scan_id: str
    timestamp: str
    root_path: Path
    total_files_scanned: int = 0
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)

    @property
    def total_duplicates(self) -> int:
        """Total number of duplicate files found."""
        return sum(g.duplicate_count for g in self.duplicate_groups)

    @property
    def total_redundant_files(self) -> int:
        """Number of redundant files (excludes original)."""
        return sum(g.duplicate_count - 1 for g in self.duplicate_groups)

    @property
    def total_redundant_loc(self) -> int:
        """Total redundant lines of code."""
        return sum(g.total_redundant_loc for g in self.duplicate_groups)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scan_id": self.scan_id,
            "timestamp": self.timestamp,
            "root_path": str(self.root_path),
            "total_files_scanned": self.total_files_scanned,
            "total_duplicate_groups": len(self.duplicate_groups),
            "total_duplicates": self.total_duplicates,
            "total_redundant_files": self.total_redundant_files,
            "total_redundant_loc": self.total_redundant_loc,
            "patterns": self.patterns,
            "exclude_patterns": self.exclude_patterns,
            "duplicate_groups": [g.to_dict() for g in self.duplicate_groups],
        }


class DuplicateScanner:
    """Scanner for detecting exact duplicate files using hashing."""

    DEFAULT_EXCLUDE = [
        "__pycache__",
        "*.pyc",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".pytest_cache",
        "*.egg-info",
        "build",
        "dist",
    ]

    def __init__(
        self,
        root_path: Path,
        patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        algorithm: str = "md5",
    ) -> None:
        """Initialize the duplicate scanner.

        Args:
            root_path: Root directory to scan.
            patterns: File patterns to include (e.g., ['*.py', '*.js']).
            exclude_patterns: Patterns to exclude from scanning.
            algorithm: Hash algorithm to use ('md5' or 'sha256').
        """
        self.root_path = Path(root_path).resolve()
        self.patterns = patterns or ["*.py"]
        self.exclude_patterns = exclude_patterns or self.DEFAULT_EXCLUDE
        self.algorithm = algorithm.lower()

        if self.algorithm not in ("md5", "sha256"):
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _compute_hash(self, file_path: Path) -> str:
        """Compute hash of a file.

        Args:
            file_path: Path to the file.

        Returns:
            Hexadecimal hash string.
        """
        hash_func = hashlib.md5() if self.algorithm == "md5" else hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file.

        Args:
            file_path: Path to the file.

        Returns:
            Number of lines in the file.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except (UnicodeDecodeError, OSError):
            return 0

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

    def _categorize_file(self, file_path: Path) -> str:
        """Categorize a file by its purpose.

        Args:
            file_path: Path to the file.

        Returns:
            Category name.
        """
        name = file_path.name.lower()
        parent = file_path.parent.name.lower()

        if name == "health.py":
            return "health_endpoint"
        elif name == "config.py":
            return "configuration"
        elif name == "database.py":
            return "database_setup"
        elif name == "base.py" and parent == "models":
            return "model_base"
        elif name == "env.py" and parent == "migrations":
            return "migration_env"
        elif name == "conftest.py":
            return "test_fixture"
        elif name.startswith("test_"):
            return "test_file"
        elif name == "__init__.py":
            return "init_file"
        elif name == "dependencies.py":
            return "dependency_injection"
        elif name == "cache.py":
            return "cache_setup"
        else:
            return "other"

    def scan(self) -> ScanResult:
        """Scan for duplicate files.

        Returns:
            ScanResult containing all findings.
        """
        scan_id = f"dup-scan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        timestamp = datetime.now().isoformat()

        file_hashes: dict[str, list[Path]] = defaultdict(list)
        total_files = 0

        # Scan all matching files
        for pattern in self.patterns:
            for file_path in self.root_path.rglob(pattern):
                if not file_path.is_file() or self._should_exclude(file_path):
                    continue

                total_files += 1
                file_hash = self._compute_hash(file_path)
                file_hashes[file_hash].append(file_path)

        # Build duplicate groups (only groups with 2+ files)
        duplicate_groups: list[DuplicateGroup] = []

        for hash_value, files in file_hashes.items():
            if len(files) < 2:
                continue

            # Get file size and line count from first file
            first_file = files[0]
            file_size = first_file.stat().st_size
            line_count = self._count_lines(first_file)
            category = self._categorize_file(first_file)

            group = DuplicateGroup(
                hash_value=hash_value,
                algorithm=self.algorithm,
                file_size=file_size,
                files=sorted(files),
                line_count=line_count,
                category=category,
            )
            duplicate_groups.append(group)

        # Sort by total redundant LOC (highest first)
        duplicate_groups.sort(key=lambda g: g.total_redundant_loc, reverse=True)

        return ScanResult(
            scan_id=scan_id,
            timestamp=timestamp,
            root_path=self.root_path,
            total_files_scanned=total_files,
            duplicate_groups=duplicate_groups,
            patterns=self.patterns,
            exclude_patterns=self.exclude_patterns,
        )


def generate_markdown_report(result: ScanResult) -> str:
    """Generate a markdown report from scan results.

    Args:
        result: Scan results to report on.

    Returns:
        Markdown formatted report string.
    """
    report = f"""# Duplicate File Analysis Report

**Scan ID**: {result.scan_id}
**Timestamp**: {result.timestamp}
**Root Path**: {result.root_path}

## Executive Summary

- **Total Files Scanned**: {result.total_files_scanned:,}
- **Duplicate Groups Found**: {len(result.duplicate_groups)}
- **Total Duplicate Files**: {result.total_duplicates}
- **Redundant Files**: {result.total_redundant_files}
- **Redundant LOC**: {result.total_redundant_loc:,} lines
- **Patterns**: {', '.join(result.patterns)}

## Duplicate Groups by Category

"""

    # Group by category
    by_category: dict[str, list[DuplicateGroup]] = defaultdict(list)
    for group in result.duplicate_groups:
        by_category[group.category].append(group)

    for category, groups in sorted(by_category.items()):
        total_loc = sum(g.total_redundant_loc for g in groups)
        report += f"\n### {category.replace('_', ' ').title()} ({len(groups)} groups, {total_loc:,} redundant LOC)\n\n"

        for group in groups:
            report += f"#### Duplicate Set (Hash: {group.hash_value[:8]}...)\n\n"
            report += f"- **Files**: {group.duplicate_count}\n"
            report += f"- **Size**: {group.file_size:,} bytes\n"
            report += f"- **Lines**: {group.line_count}\n"
            report += f"- **Redundant LOC**: {group.total_redundant_loc}\n\n"
            report += "**Locations**:\n"
            for file_path in group.files:
                report += f"- `{file_path}`\n"
            report += "\n"

    report += """
## Recommendations

### High Priority (Health Endpoints, Config, Database)
1. **Extract to Shared Library**: Move common patterns to `shared/` directory
2. **Update Imports**: Replace duplicates with shared imports
3. **Delegate to**: `refactoring-specialist` for consolidation

### Medium Priority (Test Fixtures, Base Models)
1. **Create Shared Test Utilities**: Consolidate test fixtures
2. **Establish Base Model Library**: Single source for model bases
3. **Delegate to**: `python-pro` for shared library design

### Low Priority (Init Files, Other)
1. **Review and Consolidate**: Case-by-case analysis
2. **Document Patterns**: Establish coding standards
3. **Delegate to**: `backend-developer` for service updates

## Next Steps

1. Review this report with tech lead
2. Create handoff manifest for specialist agents
3. Prioritize refactoring tasks
4. Execute consolidation with appropriate agents
5. Validate improvements and measure LOC reduction

---
*Generated by Code Duplication Analyst Agent*
"""

    return report


def main() -> None:
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Scan for duplicate files in a codebase"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("./new/src"),
        help="Root path to scan (default: ./new/src)",
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
        "--algorithm",
        choices=["md5", "sha256"],
        default="md5",
        help="Hash algorithm (default: md5)",
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

    # Run scanner
    scanner = DuplicateScanner(
        root_path=args.path,
        patterns=args.patterns,
        exclude_patterns=args.exclude,
        algorithm=args.algorithm,
    )

    print(f"Scanning {args.path} for duplicates...")
    result = scanner.scan()

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
    print(f"Scan Complete: {result.scan_id}")
    print(f"{'='*60}")
    print(f"Files Scanned: {result.total_files_scanned:,}")
    print(f"Duplicate Groups: {len(result.duplicate_groups)}")
    print(f"Redundant Files: {result.total_redundant_files}")
    print(f"Redundant LOC: {result.total_redundant_loc:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
