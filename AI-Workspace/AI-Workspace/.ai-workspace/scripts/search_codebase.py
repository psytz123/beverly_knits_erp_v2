#!/usr/bin/env python3
"""
Semantic Code Search for Reuse Analysis
Enforces Principle 3: Check Before Create

Usage:
    python search_codebase.py "email validation"
    python search_codebase.py "user authentication" --path src/
"""

from typing import List
from dataclasses import dataclass
from pathlib import Path
import ast
import re
import sys
import subprocess


@dataclass
class CodeMatch:
    """Represents a code match in the codebase."""
    file_path: str
    function_name: str
    similarity: float
    code_snippet: str
    line_number: int = 0


def semantic_search(intent: str, project_root: str = ".") -> List[CodeMatch]:
    """
    Search codebase for similar implementations.

    Args:
        intent: What functionality you're looking for
        project_root: Root directory to search

    Returns:
        List of code matches with similarity scores

    Example:
        >>> matches = semantic_search("email validation")
        >>> print(f"Found {len(matches)} matches")
        >>> print(f"Best match: {matches[0].similarity*100:.1f}%")
    """
    matches = []
    root_path = Path(project_root)

    # Search Python files
    for py_file in root_path.rglob("*.py"):
        # Skip common exclusions
        if any(skip in str(py_file) for skip in ["venv", "test_", ".test", "__pycache__", "migrations"]):
            continue

        try:
            content = py_file.read_text(encoding='utf-8')
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    similarity = calculate_similarity(
                        intent,
                        node.name,
                        get_docstring(node),
                        [arg.arg for arg in node.args.args]
                    )

                    if similarity > 0.3:  # Threshold for relevance
                        # Get code snippet
                        try:
                            snippet = ast.get_source_segment(content, node)
                            if snippet and len(snippet) > 500:
                                snippet = snippet[:500] + "..."
                        except:
                            snippet = f"def {node.name}(...)"

                        matches.append(CodeMatch(
                            file_path=str(py_file.relative_to(root_path)),
                            function_name=node.name,
                            similarity=similarity,
                            code_snippet=snippet or "",
                            line_number=node.lineno
                        ))

        except Exception as e:
            # Skip files that can't be parsed
            continue

    # Search TypeScript/JavaScript files
    for ts_file in list(root_path.rglob("*.ts")) + list(root_path.rglob("*.tsx")) + \
                   list(root_path.rglob("*.js")) + list(root_path.rglob("*.jsx")):
        if any(skip in str(ts_file) for skip in ["node_modules", "dist", "build", ".test.", "test_"]):
            continue

        try:
            content = ts_file.read_text(encoding='utf-8')
            # Simple regex-based extraction for TS/JS
            function_pattern = r'(?:function|const|let|var)\s+(\w+)\s*(?:=\s*)?(?:\([^)]*\)|<[^>]*>)?\s*(?:=>|{)'
            for match in re.finditer(function_pattern, content):
                func_name = match.group(1)
                similarity = calculate_similarity_simple(intent, func_name)

                if similarity > 0.3:
                    # Extract snippet
                    start = match.start()
                    end = min(start + 500, len(content))
                    snippet = content[start:end]
                    if len(snippet) >= 500:
                        snippet += "..."

                    line_num = content[:start].count('\n') + 1

                    matches.append(CodeMatch(
                        file_path=str(ts_file.relative_to(root_path)),
                        function_name=func_name,
                        similarity=similarity,
                        code_snippet=snippet,
                        line_number=line_num
                    ))

        except Exception:
            continue

    # Search Rust files
    for rust_file in root_path.rglob("*.rs"):
        if any(skip in str(rust_file) for skip in ["target", "test", ".test"]):
            continue

        try:
            content = rust_file.read_text(encoding='utf-8')
            # Rust function pattern: fn, pub fn, async fn, pub async fn
            function_pattern = r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)'
            for match in re.finditer(function_pattern, content):
                func_name = match.group(1)
                similarity = calculate_similarity_simple(intent, func_name)

                if similarity > 0.3:
                    start = match.start()
                    end = min(start + 500, len(content))
                    snippet = content[start:end]
                    if len(snippet) >= 500:
                        snippet += "..."

                    line_num = content[:start].count('\n') + 1

                    matches.append(CodeMatch(
                        file_path=str(rust_file.relative_to(root_path)),
                        function_name=func_name,
                        similarity=similarity,
                        code_snippet=snippet,
                        line_number=line_num
                    ))

        except Exception:
            continue

    # Search Go files
    for go_file in root_path.rglob("*.go"):
        if any(skip in str(go_file) for skip in ["vendor", "test", "_test.go"]):
            continue

        try:
            content = go_file.read_text(encoding='utf-8')
            # Go function pattern: func name, func (receiver) name
            function_pattern = r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\([^)]*\)'
            for match in re.finditer(function_pattern, content):
                func_name = match.group(1)
                similarity = calculate_similarity_simple(intent, func_name)

                if similarity > 0.3:
                    start = match.start()
                    end = min(start + 500, len(content))
                    snippet = content[start:end]
                    if len(snippet) >= 500:
                        snippet += "..."

                    line_num = content[:start].count('\n') + 1

                    matches.append(CodeMatch(
                        file_path=str(go_file.relative_to(root_path)),
                        function_name=func_name,
                        similarity=similarity,
                        code_snippet=snippet,
                        line_number=line_num
                    ))

        except Exception:
            continue

    # Search Java files
    for java_file in root_path.rglob("*.java"):
        if any(skip in str(java_file) for skip in ["target", "build", "test"]):
            continue

        try:
            content = java_file.read_text(encoding='utf-8')
            # Java method pattern: public/private/protected [static] Type methodName(...)
            function_pattern = r'(?:public|private|protected)\s+(?:static\s+)?(?:[\w<>[\]]+\s+)+(\w+)\s*\([^)]*\)'
            for match in re.finditer(function_pattern, content):
                func_name = match.group(1)
                # Skip constructors (same name as class)
                if func_name[0].isupper():
                    continue

                similarity = calculate_similarity_simple(intent, func_name)

                if similarity > 0.3:
                    start = match.start()
                    end = min(start + 500, len(content))
                    snippet = content[start:end]
                    if len(snippet) >= 500:
                        snippet += "..."

                    line_num = content[:start].count('\n') + 1

                    matches.append(CodeMatch(
                        file_path=str(java_file.relative_to(root_path)),
                        function_name=func_name,
                        similarity=similarity,
                        code_snippet=snippet,
                        line_number=line_num
                    ))

        except Exception:
            continue

    return sorted(matches, key=lambda x: x.similarity, reverse=True)


def calculate_similarity(intent: str, func_name: str, docstring: str, args: List[str]) -> float:
    """Calculate semantic similarity between intent and code."""
    # Normalize inputs
    intent_words = set(re.findall(r'\w+', intent.lower()))
    func_words = set(re.findall(r'\w+', func_name.lower()))
    doc_words = set(re.findall(r'\w+', (docstring or "").lower()))
    arg_words = set(word.lower() for word in args)

    # Calculate overlaps
    name_overlap = len(intent_words & func_words) / len(intent_words) if intent_words else 0
    doc_overlap = len(intent_words & doc_words) / len(intent_words) if intent_words else 0
    arg_overlap = len(intent_words & arg_words) / len(intent_words) if intent_words else 0

    # Weighted score
    score = (
        name_overlap * 0.5 +      # Function name is most important
        doc_overlap * 0.3 +        # Documentation is helpful
        arg_overlap * 0.2          # Arguments provide context
    )

    return min(score, 1.0)


def calculate_similarity_simple(intent: str, text: str) -> float:
    """Simple text-based similarity for non-Python files."""
    intent_words = set(re.findall(r'\w+', intent.lower()))
    text_words = set(re.findall(r'\w+', text.lower()))

    if not intent_words:
        return 0.0

    overlap = len(intent_words & text_words) / len(intent_words)
    return min(overlap, 1.0)


def get_docstring(node: ast.FunctionDef) -> str:
    """Extract docstring from function node."""
    if (node.body and
        isinstance(node.body[0], ast.Expr) and
        isinstance(node.body[0].value, ast.Constant) and
        isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value
    return ""


def print_matches(matches: List[CodeMatch], intent: str) -> None:
    """Print formatted search results."""
    print(f"\n🔍 Search Results for: '{intent}'")
    print("=" * 70)

    if not matches:
        print("\n❌ No matches found.")
        print("\n💡 Recommendations:")
        print("   1. Try different search terms")
        print("   2. Search in pattern library: .ai-workspace/cursor/patterns/")
        print("   3. If no reuse possible, proceed with new code")
        print("   4. Document why reuse failed in ADR")
        print("\n📚 Principle 3: Check Before Create")
        return

    print(f"\n✅ Found {len(matches)} potential matches\n")

    # Show top 10 matches
    for i, match in enumerate(matches[:10], 1):
        similarity_pct = match.similarity * 100
        color = "🟢" if similarity_pct >= 70 else "🟡" if similarity_pct >= 50 else "⚪"

        print(f"{color} {i}. {match.function_name} ({similarity_pct:.1f}% match)")
        print(f"   📁 {match.file_path}:{match.line_number}")

        # Show snippet preview (first 2 lines)
        lines = match.code_snippet.split('\n')[:2]
        for line in lines:
            if line.strip():
                print(f"   │ {line[:60]}")

        print()

    # Summary and recommendations
    print("=" * 70)
    best_match = matches[0].similarity * 100

    if best_match >= 90:
        print("✅ Excellent match found (≥90%)!")
        print("   → Recommendation: Use existing code directly")
    elif best_match >= 70:
        print("✅ Good match found (70-89%)")
        print("   → Recommendation: Create wrapper/adapter")
    elif best_match >= 50:
        print("⚠️  Moderate match (50-69%)")
        print("   → Recommendation: Consider wrapper vs new code")
        print("   → Required: Create ADR to justify choice")
    else:
        print("⚠️  Low match (<50%)")
        print("   → Recommendation: Implement new code")
        print("   → Required: Document why reuse failed in ADR")

    print("\n📊 Next steps:")
    print(f"   1. Review top matches: {', '.join([m.file_path for m in matches[:3]])}")
    print("   2. Calculate reuse %: python .ai-workspace/scripts/analyze_reuse.py")
    print("   3. If new code needed: python .ai-workspace/scripts/create_adr.py")

    print("\n💡 Principle 3: Always check before create!")
    print("=" * 70)


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic code search for reuse analysis (Principle 3)",
        epilog="Example: python search_codebase.py 'email validation'"
    )
    parser.add_argument(
        "intent",
        help="What functionality you're searching for"
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Project root path to search (default: current directory)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to show (default: 10)"
    )

    args = parser.parse_args()

    try:
        matches = semantic_search(args.intent, args.path)
        print_matches(matches[:args.limit], args.intent)

        # Auto-record search completion (Principle 3 enforcement)
        try:
            task_name = re.sub(r'[^a-z0-9]+', '-', args.intent.lower()).strip('-')[:30]
            files_found = [m.file_path for m in matches[:10]]

            # Try to record in enforcement system
            enforce_script = Path(__file__).parent / "enforce_check_before_create.py"
            if enforce_script.exists():
                subprocess.run(
                    [sys.executable, str(enforce_script),
                     '--record-search', task_name, ','.join(files_found)],
                    capture_output=True,
                    timeout=5
                )
        except Exception:
            pass  # Don't fail search if recording fails

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Search cancelled")
        return 1

    except Exception as e:
        print(f"\n❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
