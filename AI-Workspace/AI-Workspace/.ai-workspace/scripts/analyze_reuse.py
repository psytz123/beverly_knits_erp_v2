#!/usr/bin/env python3
"""
Reuse Analysis Calculator
Enforces Principles 1 & 3: Less is More + Check Before Create

Usage:
    python analyze_reuse.py "user authentication" existing_auth.py
    python analyze_reuse.py "email validation" --existing src/validators.py --needed "validate email format"
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import ast
import re
import sys
import difflib


@dataclass
class ReusePlan:
    """Represents a reuse analysis result."""
    intent: str
    existing_file: str
    reuse_percentage: float
    recommendation: str
    action_required: str
    similar_functions: List[str]
    missing_features: List[str]
    wrapper_needed: bool
    adr_required: bool


def analyze_reuse(
    intent: str,
    existing_file: str,
    needed_features: Optional[List[str]] = None
) -> ReusePlan:
    """
    Analyze how much of existing code can be reused.

    Args:
        intent: What functionality you need
        existing_file: Path to existing implementation
        needed_features: Optional list of specific features needed

    Returns:
        ReusePlan with reuse percentage and recommendations

    Example:
        >>> plan = analyze_reuse("user authentication", "auth.py")
        >>> print(f"Reuse: {plan.reuse_percentage:.1f}%")
        >>> print(plan.recommendation)
    """
    file_path = Path(existing_file)

    if not file_path.exists():
        return ReusePlan(
            intent=intent,
            existing_file=existing_file,
            reuse_percentage=0.0,
            recommendation="File not found - create new implementation",
            action_required="Create new code + ADR documenting search attempt",
            similar_functions=[],
            missing_features=[intent],
            wrapper_needed=False,
            adr_required=True
        )

    # Determine language and analyze
    if file_path.suffix == '.py':
        return _analyze_python(intent, file_path, needed_features)
    elif file_path.suffix in ['.ts', '.tsx', '.js', '.jsx']:
        return _analyze_typescript(intent, file_path, needed_features)
    elif file_path.suffix == '.rs':
        return _analyze_rust(intent, file_path, needed_features)
    elif file_path.suffix == '.go':
        return _analyze_go(intent, file_path, needed_features)
    elif file_path.suffix == '.java':
        return _analyze_java(intent, file_path, needed_features)
    else:
        return _analyze_generic(intent, file_path, needed_features)


def _analyze_python(
    intent: str,
    file_path: Path,
    needed_features: Optional[List[str]]
) -> ReusePlan:
    """Analyze Python file for reuse potential."""
    try:
        content = file_path.read_text(encoding='utf-8')
        tree = ast.parse(content)
    except Exception as e:
        return _create_error_plan(intent, str(file_path), f"Parse error: {e}")

    # Extract all functions and classes
    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                'name': node.name,
                'docstring': _get_docstring(node),
                'args': [arg.arg for arg in node.args.args],
                'decorators': [_get_decorator_name(d) for d in node.decorator_list],
                'line_count': _count_lines(node)
            })
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes.append({
                'name': node.name,
                'docstring': _get_docstring(node),
                'methods': methods,
                'decorators': [_get_decorator_name(d) for d in node.decorator_list]
            })

    # Calculate reuse based on intent and features
    intent_words = set(re.findall(r'\w+', intent.lower()))

    # Check if needed features are present
    if needed_features:
        feature_words = set()
        for feature in needed_features:
            feature_words.update(re.findall(r'\w+', feature.lower()))
    else:
        feature_words = intent_words

    # Score functions and classes
    matches = []
    for func in functions:
        score = _calculate_function_match(func, intent_words, feature_words)
        if score > 0:
            matches.append((func['name'], score, 'function'))

    for cls in classes:
        score = _calculate_class_match(cls, intent_words, feature_words)
        if score > 0:
            matches.append((cls['name'], score, 'class'))

    # Calculate overall reuse percentage
    if not matches:
        reuse_pct = 0.0
        similar_funcs = []
    else:
        # Weight by best matches
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:5]
        reuse_pct = sum(m[1] for m in top_matches) / len(top_matches) * 100
        similar_funcs = [f"{m[0]} ({m[2]})" for m in top_matches]

    # Determine missing features
    covered_features = set()
    for func in functions:
        func_words = set(re.findall(r'\w+', func['name'].lower()))
        if func['docstring']:
            func_words.update(re.findall(r'\w+', func['docstring'].lower()))
        covered_features.update(feature_words & func_words)

    missing = list(feature_words - covered_features)

    return _create_plan(intent, str(file_path), reuse_pct, similar_funcs, missing)


def _analyze_typescript(
    intent: str,
    file_path: Path,
    needed_features: Optional[List[str]]
) -> ReusePlan:
    """Analyze TypeScript/JavaScript file for reuse potential."""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return _create_error_plan(intent, str(file_path), f"Read error: {e}")

    # Extract functions and classes using regex
    function_pattern = r'(?:export\s+)?(?:async\s+)?(?:function|const|let|var)\s+(\w+)\s*(?:=\s*)?(?:\([^)]*\)|<[^>]*>)?(?:\s*:\s*[^{=]+)?(?:\s*=>|\s*{)'
    class_pattern = r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*{'

    functions = re.findall(function_pattern, content)
    classes = re.findall(class_pattern, content)

    intent_words = set(re.findall(r'\w+', intent.lower()))

    if needed_features:
        feature_words = set()
        for feature in needed_features:
            feature_words.update(re.findall(r'\w+', feature.lower()))
    else:
        feature_words = intent_words

    # Calculate matches
    matches = []
    for func_name in functions:
        func_words = set(re.findall(r'\w+', func_name.lower()))
        overlap = len(feature_words & func_words) / len(feature_words) if feature_words else 0
        if overlap > 0:
            matches.append((func_name, overlap, 'function'))

    for class_name in classes:
        class_words = set(re.findall(r'\w+', class_name.lower()))
        overlap = len(feature_words & class_words) / len(feature_words) if feature_words else 0
        if overlap > 0:
            matches.append((class_name, overlap, 'class'))

    if not matches:
        reuse_pct = 0.0
        similar_funcs = []
    else:
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:5]
        reuse_pct = sum(m[1] for m in top_matches) / len(top_matches) * 100
        similar_funcs = [f"{m[0]} ({m[2]})" for m in top_matches]

    # Determine missing features
    all_names = ' '.join(functions + classes).lower()
    all_words = set(re.findall(r'\w+', all_names))
    missing = list(feature_words - all_words)

    return _create_plan(intent, str(file_path), reuse_pct, similar_funcs, missing)


def _analyze_rust(
    intent: str,
    file_path: Path,
    needed_features: Optional[List[str]]
) -> ReusePlan:
    """Analyze Rust file for reuse potential."""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return _create_error_plan(intent, str(file_path), f"Read error: {e}")

    # Extract functions, structs, and impl blocks using regex
    function_pattern = r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)'
    struct_pattern = r'(?:pub\s+)?struct\s+(\w+)(?:\s*<[^>]*>)?'
    impl_pattern = r'impl(?:\s*<[^>]*>)?\s+(\w+)'

    functions = re.findall(function_pattern, content)
    structs = re.findall(struct_pattern, content)
    impls = re.findall(impl_pattern, content)

    intent_words = set(re.findall(r'\w+', intent.lower()))

    if needed_features:
        feature_words = set()
        for feature in needed_features:
            feature_words.update(re.findall(r'\w+', feature.lower()))
    else:
        feature_words = intent_words

    # Calculate matches
    matches = []
    for func_name in functions:
        func_words = set(re.findall(r'\w+', func_name.lower()))
        overlap = len(feature_words & func_words) / len(feature_words) if feature_words else 0
        if overlap > 0:
            matches.append((func_name, overlap, 'function'))

    for struct_name in structs:
        struct_words = set(re.findall(r'\w+', struct_name.lower()))
        overlap = len(feature_words & struct_words) / len(feature_words) if feature_words else 0
        if overlap > 0:
            matches.append((struct_name, overlap, 'struct'))

    if not matches:
        reuse_pct = 0.0
        similar_funcs = []
    else:
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:5]
        reuse_pct = sum(m[1] for m in top_matches) / len(top_matches) * 100
        similar_funcs = [f"{m[0]} ({m[2]})" for m in top_matches]

    # Determine missing features
    all_names = ' '.join(functions + structs + impls).lower()
    all_words = set(re.findall(r'\w+', all_names))
    missing = list(feature_words - all_words)

    return _create_plan(intent, str(file_path), reuse_pct, similar_funcs, missing)


def _analyze_go(
    intent: str,
    file_path: Path,
    needed_features: Optional[List[str]]
) -> ReusePlan:
    """Analyze Go file for reuse potential."""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return _create_error_plan(intent, str(file_path), f"Read error: {e}")

    # Extract functions, structs, and interfaces using regex
    function_pattern = r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\([^)]*\)'
    struct_pattern = r'type\s+(\w+)\s+struct\s*{'
    interface_pattern = r'type\s+(\w+)\s+interface\s*{'

    functions = re.findall(function_pattern, content)
    structs = re.findall(struct_pattern, content)
    interfaces = re.findall(interface_pattern, content)

    intent_words = set(re.findall(r'\w+', intent.lower()))

    if needed_features:
        feature_words = set()
        for feature in needed_features:
            feature_words.update(re.findall(r'\w+', feature.lower()))
    else:
        feature_words = intent_words

    # Calculate matches
    matches = []
    for func_name in functions:
        func_words = set(re.findall(r'\w+', func_name.lower()))
        overlap = len(feature_words & func_words) / len(feature_words) if feature_words else 0
        if overlap > 0:
            matches.append((func_name, overlap, 'function'))

    for struct_name in structs:
        struct_words = set(re.findall(r'\w+', struct_name.lower()))
        overlap = len(feature_words & struct_words) / len(feature_words) if feature_words else 0
        if overlap > 0:
            matches.append((struct_name, overlap, 'struct'))

    for interface_name in interfaces:
        interface_words = set(re.findall(r'\w+', interface_name.lower()))
        overlap = len(feature_words & interface_words) / len(feature_words) if feature_words else 0
        if overlap > 0:
            matches.append((interface_name, overlap, 'interface'))

    if not matches:
        reuse_pct = 0.0
        similar_funcs = []
    else:
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:5]
        reuse_pct = sum(m[1] for m in top_matches) / len(top_matches) * 100
        similar_funcs = [f"{m[0]} ({m[2]})" for m in top_matches]

    # Determine missing features
    all_names = ' '.join(functions + structs + interfaces).lower()
    all_words = set(re.findall(r'\w+', all_names))
    missing = list(feature_words - all_words)

    return _create_plan(intent, str(file_path), reuse_pct, similar_funcs, missing)


def _analyze_java(
    intent: str,
    file_path: Path,
    needed_features: Optional[List[str]]
) -> ReusePlan:
    """Analyze Java file for reuse potential."""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return _create_error_plan(intent, str(file_path), f"Read error: {e}")

    # Extract methods, classes, and interfaces using regex
    method_pattern = r'(?:public|private|protected)\s+(?:static\s+)?(?:[\w<>[\]]+\s+)+(\w+)\s*\([^)]*\)'
    class_pattern = r'(?:public\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*{'
    interface_pattern = r'(?:public\s+)?interface\s+(\w+)(?:\s+extends\s+[\w,\s]+)?\s*{'

    methods = re.findall(method_pattern, content)
    classes = re.findall(class_pattern, content)
    interfaces = re.findall(interface_pattern, content)

    # Filter out constructors (methods that start with uppercase)
    methods = [m for m in methods if not m[0].isupper()]

    intent_words = set(re.findall(r'\w+', intent.lower()))

    if needed_features:
        feature_words = set()
        for feature in needed_features:
            feature_words.update(re.findall(r'\w+', feature.lower()))
    else:
        feature_words = intent_words

    # Calculate matches
    matches = []
    for method_name in methods:
        method_words = set(re.findall(r'\w+', method_name.lower()))
        overlap = len(feature_words & method_words) / len(feature_words) if feature_words else 0
        if overlap > 0:
            matches.append((method_name, overlap, 'method'))

    for class_name in classes:
        class_words = set(re.findall(r'\w+', class_name.lower()))
        overlap = len(feature_words & class_words) / len(feature_words) if feature_words else 0
        if overlap > 0:
            matches.append((class_name, overlap, 'class'))

    for interface_name in interfaces:
        interface_words = set(re.findall(r'\w+', interface_name.lower()))
        overlap = len(feature_words & interface_words) / len(feature_words) if feature_words else 0
        if overlap > 0:
            matches.append((interface_name, overlap, 'interface'))

    if not matches:
        reuse_pct = 0.0
        similar_funcs = []
    else:
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:5]
        reuse_pct = sum(m[1] for m in top_matches) / len(top_matches) * 100
        similar_funcs = [f"{m[0]} ({m[2]})" for m in top_matches]

    # Determine missing features
    all_names = ' '.join(methods + classes + interfaces).lower()
    all_words = set(re.findall(r'\w+', all_names))
    missing = list(feature_words - all_words)

    return _create_plan(intent, str(file_path), reuse_pct, similar_funcs, missing)


def _analyze_generic(
    intent: str,
    file_path: Path,
    needed_features: Optional[List[str]]
) -> ReusePlan:
    """Generic text-based analysis for unknown file types."""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return _create_error_plan(intent, str(file_path), f"Read error: {e}")

    intent_words = set(re.findall(r'\w+', intent.lower()))
    content_words = set(re.findall(r'\w+', content.lower()))

    # Simple overlap calculation
    overlap = len(intent_words & content_words) / len(intent_words) if intent_words else 0
    reuse_pct = overlap * 100

    # Find similar terms
    similar_funcs = []
    for word in intent_words:
        matches = difflib.get_close_matches(word, content_words, n=3, cutoff=0.6)
        if matches:
            similar_funcs.extend(matches[:2])

    similar_funcs = list(set(similar_funcs))[:5]

    missing = list(intent_words - content_words)

    return _create_plan(intent, str(file_path), reuse_pct, similar_funcs, missing)


def _calculate_function_match(
    func: dict,
    intent_words: set,
    feature_words: set
) -> float:
    """Calculate how well a function matches needed features."""
    func_name_words = set(re.findall(r'\w+', func['name'].lower()))

    doc_words = set()
    if func['docstring']:
        doc_words = set(re.findall(r'\w+', func['docstring'].lower()))

    arg_words = set(word.lower() for word in func['args'])

    # Calculate overlaps
    name_overlap = len(feature_words & func_name_words) / len(feature_words) if feature_words else 0
    doc_overlap = len(feature_words & doc_words) / len(feature_words) if feature_words else 0
    arg_overlap = len(feature_words & arg_words) / len(feature_words) if feature_words else 0

    # Weighted score (same as search_codebase.py)
    score = (
        name_overlap * 0.5 +
        doc_overlap * 0.3 +
        arg_overlap * 0.2
    )

    return min(score, 1.0)


def _calculate_class_match(
    cls: dict,
    intent_words: set,
    feature_words: set
) -> float:
    """Calculate how well a class matches needed features."""
    class_name_words = set(re.findall(r'\w+', cls['name'].lower()))

    doc_words = set()
    if cls['docstring']:
        doc_words = set(re.findall(r'\w+', cls['docstring'].lower()))

    method_words = set()
    for method in cls['methods']:
        method_words.update(re.findall(r'\w+', method.lower()))

    # Calculate overlaps
    name_overlap = len(feature_words & class_name_words) / len(feature_words) if feature_words else 0
    doc_overlap = len(feature_words & doc_words) / len(feature_words) if feature_words else 0
    method_overlap = len(feature_words & method_words) / len(feature_words) if feature_words else 0

    # Weighted score
    score = (
        name_overlap * 0.4 +
        doc_overlap * 0.3 +
        method_overlap * 0.3
    )

    return min(score, 1.0)


def _create_plan(
    intent: str,
    file_path: str,
    reuse_pct: float,
    similar_funcs: List[str],
    missing: List[str]
) -> ReusePlan:
    """Create ReusePlan based on reuse percentage."""

    # Determine recommendation based on reuse %
    if reuse_pct >= 90:
        recommendation = "Excellent reuse opportunity (≥90%)"
        action = "Use existing code directly - import and call"
        wrapper_needed = False
        adr_required = False
    elif reuse_pct >= 70:
        recommendation = "Good reuse opportunity (70-89%)"
        action = "Create thin wrapper/adapter around existing code"
        wrapper_needed = True
        adr_required = False
    elif reuse_pct >= 50:
        recommendation = "Moderate reuse opportunity (50-69%)"
        action = "Evaluate: wrapper vs new code - CREATE ADR to justify choice"
        wrapper_needed = True
        adr_required = True
    else:
        recommendation = "Low reuse potential (<50%)"
        action = "Implement new code - CREATE ADR documenting why reuse failed"
        wrapper_needed = False
        adr_required = True

    return ReusePlan(
        intent=intent,
        existing_file=file_path,
        reuse_percentage=reuse_pct,
        recommendation=recommendation,
        action_required=action,
        similar_functions=similar_funcs,
        missing_features=missing,
        wrapper_needed=wrapper_needed,
        adr_required=adr_required
    )


def _create_error_plan(intent: str, file_path: str, error: str) -> ReusePlan:
    """Create error ReusePlan."""
    return ReusePlan(
        intent=intent,
        existing_file=file_path,
        reuse_percentage=0.0,
        recommendation=f"Analysis failed: {error}",
        action_required="Create new code + ADR documenting analysis failure",
        similar_functions=[],
        missing_features=[intent],
        wrapper_needed=False,
        adr_required=True
    )


def _get_docstring(node) -> str:
    """Extract docstring from AST node."""
    if (node.body and
        isinstance(node.body[0], ast.Expr) and
        isinstance(node.body[0].value, ast.Constant) and
        isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value
    return ""


def _get_decorator_name(decorator) -> str:
    """Extract decorator name from AST node."""
    if isinstance(decorator, ast.Name):
        return decorator.id
    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
        return decorator.func.id
    return ""


def _count_lines(node) -> int:
    """Count lines in AST node."""
    if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
        return node.end_lineno - node.lineno + 1
    return 0


def print_analysis(plan: ReusePlan) -> None:
    """Print formatted reuse analysis."""
    print(f"\n📊 Reuse Analysis: '{plan.intent}'")
    print("=" * 70)

    # Reuse percentage with color coding
    pct = plan.reuse_percentage
    color = "🟢" if pct >= 70 else "🟡" if pct >= 50 else "🔴"

    print(f"\n{color} **Reuse Potential: {pct:.1f}%**")
    print(f"📁 Existing File: {plan.existing_file}")

    # Similar functions
    if plan.similar_functions:
        print(f"\n✅ Found {len(plan.similar_functions)} matching components:")
        for func in plan.similar_functions:
            print(f"   • {func}")
    else:
        print("\n❌ No matching components found")

    # Missing features
    if plan.missing_features:
        print(f"\n⚠️  Missing {len(plan.missing_features)} features:")
        for feature in plan.missing_features[:5]:
            print(f"   • {feature}")
        if len(plan.missing_features) > 5:
            print(f"   • ... and {len(plan.missing_features) - 5} more")

    # Recommendation
    print("\n" + "=" * 70)
    print(f"💡 {plan.recommendation}")
    print(f"\n🎯 Action Required:")
    print(f"   {plan.action_required}")

    if plan.wrapper_needed:
        print("\n🔧 Wrapper Pattern Recommended:")
        print("   1. Import existing code")
        print("   2. Create adapter/wrapper class")
        print("   3. Add missing functionality")
        print("   4. Test wrapper thoroughly")

    if plan.adr_required:
        print("\n📝 ADR Required:")
        print("   Create architectural decision record documenting:")
        print("   • Why reuse percentage is suboptimal")
        print("   • What approach was chosen (wrapper vs new)")
        print("   • Trade-offs and alternatives considered")
        print(f"\n   Run: python .ai-workspace/scripts/create_adr.py reuse-{plan.intent.replace(' ', '-')}")

    print("\n📚 Principles Enforced:")
    print("   ✅ Principle 1: Less is More (reuse-first)")
    print("   ✅ Principle 3: Check Before Create")
    if plan.adr_required:
        print("   ✅ Principle 2: Document Everything (ADR required)")

    print("=" * 70)


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze code reuse potential (Principles 1 & 3)",
        epilog="Example: python analyze_reuse.py 'user auth' existing_auth.py"
    )
    parser.add_argument(
        "intent",
        help="What functionality you need"
    )
    parser.add_argument(
        "existing_file",
        help="Path to existing implementation to analyze"
    )
    parser.add_argument(
        "--features",
        nargs='+',
        help="Specific features needed (optional)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    try:
        plan = analyze_reuse(args.intent, args.existing_file, args.features)

        if args.json:
            import json
            print(json.dumps({
                'intent': plan.intent,
                'existing_file': plan.existing_file,
                'reuse_percentage': plan.reuse_percentage,
                'recommendation': plan.recommendation,
                'action_required': plan.action_required,
                'similar_functions': plan.similar_functions,
                'missing_features': plan.missing_features,
                'wrapper_needed': plan.wrapper_needed,
                'adr_required': plan.adr_required
            }, indent=2))
        else:
            print_analysis(plan)

        # Auto-record reuse analysis (Principle 3 enforcement)
        try:
            import subprocess
            task_name = re.sub(r'[^a-z0-9]+', '-', args.intent.lower()).strip('-')[:30]
            files_analyzed = [plan.existing_file] + plan.similar_functions

            # Try to record in enforcement system
            enforce_script = Path(__file__).parent / "enforce_check_before_create.py"
            if enforce_script.exists():
                subprocess.run(
                    [sys.executable, str(enforce_script),
                     '--record-analysis', task_name,
                     str(plan.reuse_percentage), ','.join(files_analyzed)],
                    capture_output=True,
                    timeout=5
                )
        except Exception:
            pass  # Don't fail analysis if recording fails

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis cancelled")
        return 1

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
