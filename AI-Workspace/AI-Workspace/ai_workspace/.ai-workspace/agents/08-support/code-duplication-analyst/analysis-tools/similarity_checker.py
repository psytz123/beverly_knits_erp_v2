"""Similarity Checker - AST-based Semantic Similarity Detection.

This tool analyzes Python files for semantic similarity using Abstract Syntax Tree
comparison. It identifies files that are functionally similar even if not identical.

Created: 2025-10-05
Modified: 2025-10-05
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional


@dataclass
class ASTSignature:
    """AST-based signature of a Python file."""

    file_path: Path
    imports: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    structure_hash: str = ""
    line_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": str(self.file_path),
            "imports": self.imports,
            "classes": self.classes,
            "functions": self.functions,
            "constants": self.constants,
            "structure_hash": self.structure_hash,
            "line_count": self.line_count,
        }


@dataclass
class SimilarityMatch:
    """A pair of similar files with similarity score."""

    file_a: Path
    file_b: Path
    similarity_score: float
    matching_imports: int = 0
    matching_classes: int = 0
    matching_functions: int = 0
    category: str = "unknown"
    recommendation: str = ""

    @property
    def similarity_percentage(self) -> int:
        """Similarity as percentage (0-100)."""
        return int(self.similarity_score * 100)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_a": str(self.file_a),
            "file_b": str(self.file_b),
            "similarity_score": self.similarity_score,
            "similarity_percentage": self.similarity_percentage,
            "matching_imports": self.matching_imports,
            "matching_classes": self.matching_classes,
            "matching_functions": self.matching_functions,
            "category": self.category,
            "recommendation": self.recommendation,
        }


@dataclass
class SimilarityAnalysisResult:
    """Results from similarity analysis."""

    analysis_id: str
    timestamp: str
    root_path: Path
    threshold: float
    total_files_analyzed: int = 0
    similarity_matches: list[SimilarityMatch] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)

    @property
    def high_similarity_count(self) -> int:
        """Count of matches above threshold."""
        return len(self.similarity_matches)

    @property
    def average_similarity(self) -> float:
        """Average similarity score of all matches."""
        if not self.similarity_matches:
            return 0.0
        return sum(m.similarity_score for m in self.similarity_matches) / len(
            self.similarity_matches
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp,
            "root_path": str(self.root_path),
            "threshold": self.threshold,
            "total_files_analyzed": self.total_files_analyzed,
            "high_similarity_count": self.high_similarity_count,
            "average_similarity": self.average_similarity,
            "patterns": self.patterns,
            "similarity_matches": [m.to_dict() for m in self.similarity_matches],
        }


class ASTAnalyzer(ast.NodeVisitor):
    """AST visitor to extract structural information from Python files."""

    def __init__(self) -> None:
        """Initialize the AST analyzer."""
        self.imports: list[str] = []
        self.classes: list[str] = []
        self.functions: list[str] = []
        self.constants: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements."""
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        self.classes.append(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        self.functions.append(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        self.functions.append(f"async_{node.name}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignments to identify constants."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                self.constants.append(target.id)
        self.generic_visit(node)


class SimilarityChecker:
    """Checker for detecting semantically similar Python files."""

    DEFAULT_EXCLUDE = [
        "__pycache__",
        "*.pyc",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".pytest_cache",
        "*.egg-info",
        "__init__.py",  # Exclude init files from similarity
    ]

    def __init__(
        self,
        root_path: Path,
        threshold: float = 0.85,
        patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        """Initialize the similarity checker.

        Args:
            root_path: Root directory to analyze.
            threshold: Similarity threshold (0.0 to 1.0).
            patterns: File patterns to include (e.g., ['*.py']).
            exclude_patterns: Patterns to exclude from analysis.
        """
        self.root_path = Path(root_path).resolve()
        self.threshold = threshold
        self.patterns = patterns or ["*.py"]
        self.exclude_patterns = exclude_patterns or self.DEFAULT_EXCLUDE

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

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
            elif pattern in path.parts or path.name == pattern:
                return True
        return False

    def _extract_ast_signature(self, file_path: Path) -> Optional[ASTSignature]:
        """Extract AST signature from a Python file.

        Args:
            file_path: Path to the Python file.

        Returns:
            ASTSignature object or None if parsing fails.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)

            analyzer = ASTAnalyzer()
            analyzer.visit(tree)

            # Create structure signature
            structure_parts = [
                f"I:{','.join(sorted(analyzer.imports))}",
                f"C:{','.join(sorted(analyzer.classes))}",
                f"F:{','.join(sorted(analyzer.functions))}",
                f"K:{','.join(sorted(analyzer.constants))}",
            ]
            structure_hash = "|".join(structure_parts)

            return ASTSignature(
                file_path=file_path,
                imports=sorted(analyzer.imports),
                classes=sorted(analyzer.classes),
                functions=sorted(analyzer.functions),
                constants=sorted(analyzer.constants),
                structure_hash=structure_hash,
                line_count=len(content.splitlines()),
            )

        except (SyntaxError, UnicodeDecodeError, OSError) as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return None

    def _calculate_similarity(
        self, sig_a: ASTSignature, sig_b: ASTSignature
    ) -> float:
        """Calculate similarity between two AST signatures.

        Args:
            sig_a: First file signature.
            sig_b: Second file signature.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Use SequenceMatcher on structure hash
        matcher = SequenceMatcher(None, sig_a.structure_hash, sig_b.structure_hash)
        return matcher.ratio()

    def _categorize_similarity(
        self, sig_a: ASTSignature, sig_b: ASTSignature
    ) -> tuple[str, str]:
        """Categorize the type of similarity and provide recommendation.

        Args:
            sig_a: First file signature.
            sig_b: Second file signature.

        Returns:
            Tuple of (category, recommendation).
        """
        file_a_name = sig_a.file_path.name.lower()
        file_b_name = sig_b.file_path.name.lower()

        # Categorize by file names
        if "config" in file_a_name and "config" in file_b_name:
            return (
                "configuration",
                "Extract to shared/config/ with environment-specific overrides",
            )
        elif "database" in file_a_name and "database" in file_b_name:
            return (
                "database_setup",
                "Create shared database connection factory in shared/database/",
            )
        elif "health" in file_a_name and "health" in file_b_name:
            return "health_endpoint", "Consolidate to shared/api/health.py"
        elif "cache" in file_a_name and "cache" in file_b_name:
            return "cache_setup", "Extract to shared/cache/ with unified interface"
        elif file_a_name.startswith("test_") and file_b_name.startswith("test_"):
            return "test_utilities", "Move to shared test fixtures in shared/testing/"
        elif "repository" in file_a_name and "repository" in file_b_name:
            return (
                "repository_pattern",
                "Create base repository class in shared/repositories/",
            )
        elif "service" in file_a_name and "service" in file_b_name:
            return "service_pattern", "Establish base service class in shared/services/"
        else:
            return "other", "Review for potential shared abstraction"

    def analyze(self) -> SimilarityAnalysisResult:
        """Analyze files for semantic similarity.

        Returns:
            SimilarityAnalysisResult containing all matches above threshold.
        """
        analysis_id = f"sim-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        timestamp = datetime.now().isoformat()

        # Extract signatures from all files
        signatures: list[ASTSignature] = []

        for pattern in self.patterns:
            for file_path in self.root_path.rglob(pattern):
                if not file_path.is_file() or self._should_exclude(file_path):
                    continue

                signature = self._extract_ast_signature(file_path)
                if signature:
                    signatures.append(signature)

        total_files = len(signatures)
        similarity_matches: list[SimilarityMatch] = []

        # Compare all pairs
        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                sig_a = signatures[i]
                sig_b = signatures[j]

                similarity = self._calculate_similarity(sig_a, sig_b)

                if similarity >= self.threshold:
                    # Count matching elements
                    matching_imports = len(set(sig_a.imports) & set(sig_b.imports))
                    matching_classes = len(set(sig_a.classes) & set(sig_b.classes))
                    matching_functions = len(
                        set(sig_a.functions) & set(sig_b.functions)
                    )

                    category, recommendation = self._categorize_similarity(sig_a, sig_b)

                    match = SimilarityMatch(
                        file_a=sig_a.file_path,
                        file_b=sig_b.file_path,
                        similarity_score=similarity,
                        matching_imports=matching_imports,
                        matching_classes=matching_classes,
                        matching_functions=matching_functions,
                        category=category,
                        recommendation=recommendation,
                    )
                    similarity_matches.append(match)

        # Sort by similarity (highest first)
        similarity_matches.sort(key=lambda m: m.similarity_score, reverse=True)

        return SimilarityAnalysisResult(
            analysis_id=analysis_id,
            timestamp=timestamp,
            root_path=self.root_path,
            threshold=self.threshold,
            total_files_analyzed=total_files,
            similarity_matches=similarity_matches,
            patterns=self.patterns,
        )


def generate_markdown_report(result: SimilarityAnalysisResult) -> str:
    """Generate a markdown report from similarity analysis.

    Args:
        result: Analysis results to report on.

    Returns:
        Markdown formatted report string.
    """
    report = f"""# Semantic Similarity Analysis Report

**Analysis ID**: {result.analysis_id}
**Timestamp**: {result.timestamp}
**Root Path**: {result.root_path}

## Executive Summary

- **Files Analyzed**: {result.total_files_analyzed:,}
- **Similarity Threshold**: {result.threshold * 100:.0f}%
- **High Similarity Matches**: {result.high_similarity_count}
- **Average Similarity**: {result.average_similarity * 100:.1f}%
- **Patterns**: {', '.join(result.patterns)}

## Similarity Matches by Category

"""

    # Group by category
    from collections import defaultdict

    by_category: dict[str, list[SimilarityMatch]] = defaultdict(list)
    for match in result.similarity_matches:
        by_category[match.category].append(match)

    for category, matches in sorted(by_category.items()):
        avg_similarity = sum(m.similarity_score for m in matches) / len(matches)
        report += f"\n### {category.replace('_', ' ').title()} ({len(matches)} pairs, {avg_similarity*100:.1f}% avg similarity)\n\n"

        for match in matches:
            report += f"#### Similarity: {match.similarity_percentage}%\n\n"
            report += f"**Files**:\n"
            report += f"- `{match.file_a}`\n"
            report += f"- `{match.file_b}`\n\n"
            report += f"**Matching Elements**:\n"
            report += f"- Imports: {match.matching_imports}\n"
            report += f"- Classes: {match.matching_classes}\n"
            report += f"- Functions: {match.matching_functions}\n\n"
            report += f"**Recommendation**: {match.recommendation}\n\n"

    report += """
## Refactoring Recommendations

### High Priority (>90% Similarity)
1. **Immediate Consolidation**: Files with >90% similarity are strong candidates for immediate consolidation
2. **Extract to Shared Library**: Create shared abstractions in `shared/` directory
3. **Delegate to**: `refactoring-specialist` for implementation

### Medium Priority (85-90% Similarity)
1. **Design Shared Abstraction**: Analyze differences and create flexible base classes
2. **Gradual Migration**: Move common functionality incrementally
3. **Delegate to**: `python-pro` for shared library design

### Monitoring
- Track similarity scores over time
- Prevent new duplicates with pre-commit hooks
- Establish coding standards and patterns

## Next Steps

1. Review high-priority matches with development team
2. Create handoff tasks for specialist agents
3. Design shared library architecture
4. Implement consolidation incrementally
5. Validate with comprehensive testing

---
*Generated by Code Duplication Analyst Agent - Similarity Checker*
"""

    return report


def main() -> None:
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Analyze Python files for semantic similarity"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("./new/src"),
        help="Root path to analyze (default: ./new/src)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold 0.0-1.0 (default: 0.85)",
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

    # Run similarity checker
    checker = SimilarityChecker(
        root_path=args.path,
        threshold=args.threshold,
        patterns=args.patterns,
        exclude_patterns=args.exclude,
    )

    print(f"Analyzing {args.path} for semantic similarity...")
    result = checker.analyze()

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
    print(f"Threshold: {result.threshold * 100:.0f}%")
    print(f"High Similarity Matches: {result.high_similarity_count}")
    print(f"Average Similarity: {result.average_similarity * 100:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
