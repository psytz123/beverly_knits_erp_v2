#!/usr/bin/env python3
"""
ADR (Architectural Decision Record) Creator
Enforces Principle 2: Document Everything

Interactive wizard for creating architectural decision records.
Auto-populates from reuse analysis and gate validations.

Usage:
    python create_adr.py                        # Interactive wizard
    python create_adr.py --title "Use PostgreSQL for data storage"
    python create_adr.py --from-reuse analyze_results.json
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import sys
import re


@dataclass
class ADRMetadata:
    """Metadata for an ADR."""
    number: int
    title: str
    status: str  # proposed, accepted, rejected, deprecated, superseded
    date: str
    context: str
    decision: str
    consequences: str
    alternatives: List[Dict[str, str]]
    related_decisions: List[str]
    author: str


class ADRCreator:
    """Interactive ADR creation wizard."""

    def __init__(self, workspace_path: str = ".agent-workspace"):
        """Initialize ADR creator."""
        self.workspace = Path(workspace_path)
        self.decisions_dir = self.workspace / "decisions"
        self.template_path = self.workspace / "templates" / "adr.template.md"

        # Create directories if needed
        self.decisions_dir.mkdir(parents=True, exist_ok=True)

    def create_interactive(self) -> str:
        """Interactive ADR creation wizard."""
        print("\n📝 ADR Creator - Interactive Wizard")
        print("=" * 70)
        print("Enforcing Principle 2: Document Everything")
        print("=" * 70)

        # Get next ADR number
        adr_number = self._get_next_adr_number()
        print(f"\n📋 Creating ADR-{adr_number:03d}")

        # Collect information
        title = self._prompt("Decision Title", required=True)
        status = self._prompt_choice(
            "Status",
            ['proposed', 'accepted', 'rejected', 'deprecated'],
            default='proposed'
        )

        print("\n📖 Context:")
        print("   (Describe the issue/problem that needs a decision)")
        context = self._prompt_multiline("Context")

        print("\n✅ Decision:")
        print("   (Describe what was decided and why)")
        decision = self._prompt_multiline("Decision")

        print("\n⚖️  Consequences:")
        print("   (Describe the impact - positive and negative)")
        consequences = self._prompt_multiline("Consequences")

        # Alternatives
        print("\n🔄 Alternatives Considered:")
        alternatives = self._collect_alternatives()

        # Related decisions
        print("\n🔗 Related Decisions:")
        related = self._collect_related_decisions()

        author = self._prompt("Author", default="AI Assistant")

        # Create ADR
        metadata = ADRMetadata(
            number=adr_number,
            title=title,
            status=status,
            date=datetime.now().strftime("%Y-%m-%d"),
            context=context,
            decision=decision,
            consequences=consequences,
            alternatives=alternatives,
            related_decisions=related,
            author=author
        )

        file_path = self._write_adr(metadata)

        print(f"\n✅ ADR created successfully!")
        print(f"📁 Path: {file_path}")
        print(f"\n💡 Next steps:")
        print(f"   1. Review and edit: {file_path}")
        print(f"   2. Commit to version control")
        print(f"   3. Share with team for review")

        return str(file_path)

    def create_from_title(
        self,
        title: str,
        status: str = "proposed",
        author: str = "AI Assistant"
    ) -> str:
        """Create ADR from title (minimal mode)."""
        adr_number = self._get_next_adr_number()

        metadata = ADRMetadata(
            number=adr_number,
            title=title,
            status=status,
            date=datetime.now().strftime("%Y-%m-%d"),
            context="TODO: Describe the context and problem",
            decision="TODO: Describe the decision and rationale",
            consequences="TODO: Describe the consequences",
            alternatives=[],
            related_decisions=[],
            author=author
        )

        file_path = self._write_adr(metadata)

        print(f"\n✅ ADR template created!")
        print(f"📁 Path: {file_path}")
        print(f"\n⚠️  Template created with TODOs - please edit before committing")
        print(f"\n💡 Edit now:")
        print(f"   code {file_path}")

        return str(file_path)

    def create_from_reuse_analysis(
        self,
        reuse_result_file: str,
        author: str = "AI Assistant"
    ) -> str:
        """Create ADR from reuse analysis results."""
        try:
            with open(reuse_result_file, 'r') as f:
                reuse_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read reuse analysis: {e}")

        adr_number = self._get_next_adr_number()

        # Extract from reuse analysis
        intent = reuse_data.get('intent', 'Unknown')
        reuse_pct = reuse_data.get('reuse_percentage', 0)
        recommendation = reuse_data.get('recommendation', '')
        action = reuse_data.get('action_required', '')

        # Build context
        context = f"""## Problem
New functionality needed: {intent}

## Reuse Analysis (Multi-Language Support)
- **Supported Languages:** Python, TypeScript, JavaScript, Rust, Go, Java
- Existing code analyzed: {reuse_data.get('existing_file', 'N/A')}
- Reuse potential: {reuse_pct:.1f}%
- Similar components found: {len(reuse_data.get('similar_functions', []))}
- Missing features: {len(reuse_data.get('missing_features', []))}

## Analysis Result
{recommendation}

## Enforcement Note
This ADR is required because reuse percentage is <70%. Automatic enforcement system
will block code creation until this ADR is reviewed and approved.
"""

        # Build decision based on reuse percentage
        if reuse_pct >= 90:
            decision = f"""## Decision: Reuse Existing Code

We will directly use the existing implementation with minimal or no modifications.

### Rationale
- Reuse percentage: {reuse_pct:.1f}% (Excellent match)
- Principle 1: Less is More - maximize reuse
- Principle 3: Check Before Create - existing solution found

### Implementation
- Import and call existing functions
- No wrapper needed
"""
        elif reuse_pct >= 70:
            decision = f"""## Decision: Create Wrapper/Adapter

We will create a thin wrapper around the existing implementation to meet our specific needs.

### Rationale
- Reuse percentage: {reuse_pct:.1f}% (Good match)
- Core functionality exists
- Adaptation needed for specific requirements
- Principle 1: Less is More - maximize reuse through adaptation

### Implementation
- Wrapper class/function around existing code
- Add missing features: {', '.join(reuse_data.get('missing_features', [])[:3])}
"""
        else:
            decision = f"""## Decision: Implement New Code

We will implement new code rather than reusing existing implementation.

### Rationale
- Reuse percentage: {reuse_pct:.1f}% (Low match)
- Existing code does not meet requirements
- Cost of adaptation > cost of new implementation
- Missing critical features: {', '.join(reuse_data.get('missing_features', [])[:5])}

### Implementation
- New implementation from scratch
- Learn from existing patterns where applicable
"""

        # Build consequences
        consequences = f"""## Positive
- Decision based on quantitative reuse analysis
- {action}

## Negative
- {"Additional wrapper maintenance" if reuse_pct >= 70 else "No direct reuse of existing code"}
- {"Potential coupling to existing implementation" if reuse_pct >= 70 else "Duplication of functionality"}

## Risks
- {self._get_risk_for_reuse_level(reuse_pct)}
"""

        # Build alternatives
        alternatives = []
        if reuse_pct >= 70:
            alternatives.append({
                'name': 'Implement from scratch',
                'pros': 'Full control, no dependencies',
                'cons': 'More time, potential duplication',
                'rejected_because': f'Existing code provides {reuse_pct:.1f}% match'
            })
        if reuse_pct < 90:
            alternatives.append({
                'name': 'Direct reuse without modifications',
                'pros': 'Minimal effort, maximum reuse',
                'cons': f'Missing {len(reuse_data.get("missing_features", []))} required features',
                'rejected_because': 'Does not fully meet requirements'
            })

        metadata = ADRMetadata(
            number=adr_number,
            title=f"Reuse Strategy: {intent}",
            status="accepted",
            date=datetime.now().strftime("%Y-%m-%d"),
            context=context,
            decision=decision,
            consequences=consequences,
            alternatives=alternatives,
            related_decisions=[],
            author=author
        )

        file_path = self._write_adr(metadata)

        print(f"\n✅ ADR created from reuse analysis!")
        print(f"📁 Path: {file_path}")
        print(f"📊 Reuse: {reuse_pct:.1f}%")
        print(f"🎯 Strategy: {'Direct use' if reuse_pct >= 90 else 'Wrapper' if reuse_pct >= 70 else 'New code'}")

        return str(file_path)

    def _get_next_adr_number(self) -> int:
        """Get next ADR number by scanning existing ADRs."""
        if not self.decisions_dir.exists():
            return 1

        existing = list(self.decisions_dir.glob("ADR-*.md"))
        if not existing:
            return 1

        # Extract numbers from filenames
        numbers = []
        for adr_file in existing:
            match = re.match(r'ADR-(\d+)', adr_file.name)
            if match:
                numbers.append(int(match.group(1)))

        return max(numbers) + 1 if numbers else 1

    def _write_adr(self, metadata: ADRMetadata) -> Path:
        """Write ADR to file."""
        # Load template if exists
        if self.template_path.exists():
            template = self.template_path.read_text(encoding='utf-8')
        else:
            template = self._get_default_template()

        # Replace variables
        content = template.replace('{{NUMBER}}', f'{metadata.number:03d}')
        content = content.replace('{{TITLE}}', metadata.title)
        content = content.replace('{{STATUS}}', metadata.status)
        content = content.replace('{{DATE}}', metadata.date)
        content = content.replace('{{CONTEXT}}', metadata.context)
        content = content.replace('{{DECISION}}', metadata.decision)
        content = content.replace('{{CONSEQUENCES}}', metadata.consequences)
        content = content.replace('{{AUTHOR}}', metadata.author)

        # Build alternatives section
        if metadata.alternatives:
            alt_section = "\n## Alternatives Considered\n\n"
            for i, alt in enumerate(metadata.alternatives, 1):
                alt_section += f"### Alternative {i}: {alt['name']}\n\n"
                alt_section += f"**Pros:** {alt['pros']}\n\n"
                alt_section += f"**Cons:** {alt['cons']}\n\n"
                alt_section += f"**Rejected because:** {alt['rejected_because']}\n\n"
            content = content.replace('{{ALTERNATIVES}}', alt_section)
        else:
            content = content.replace('{{ALTERNATIVES}}', '')

        # Build related decisions section
        if metadata.related_decisions:
            related_section = "\n## Related Decisions\n\n"
            for decision in metadata.related_decisions:
                related_section += f"- {decision}\n"
            content = content.replace('{{RELATED}}', related_section)
        else:
            content = content.replace('{{RELATED}}', '')

        # Generate filename
        title_slug = re.sub(r'[^a-z0-9]+', '-', metadata.title.lower())
        title_slug = title_slug.strip('-')[:50]  # Max 50 chars
        filename = f"ADR-{metadata.number:03d}-{title_slug}.md"
        file_path = self.decisions_dir / filename

        # Write file
        file_path.write_text(content, encoding='utf-8')

        return file_path

    def _prompt(
        self,
        field_name: str,
        default: Optional[str] = None,
        required: bool = False
    ) -> str:
        """Prompt user for input."""
        prompt = f"{field_name}"
        if default:
            prompt += f" [{default}]"
        prompt += ": "

        while True:
            value = input(prompt).strip()

            if not value and default:
                return default

            if not value and required:
                print("   ⚠️  This field is required. Please enter a value.")
                continue

            if not value:
                return ""

            return value

    def _prompt_choice(
        self,
        field_name: str,
        choices: List[str],
        default: Optional[str] = None
    ) -> str:
        """Prompt user to select from choices."""
        print(f"\n{field_name}:")
        for i, choice in enumerate(choices, 1):
            marker = " (default)" if choice == default else ""
            print(f"   {i}. {choice}{marker}")

        while True:
            selection = input("Select: ").strip()

            if not selection and default:
                return default

            if selection.isdigit():
                idx = int(selection) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]

            print("   ⚠️  Invalid selection. Try again.")

    def _prompt_multiline(self, field_name: str) -> str:
        """Prompt for multiline input."""
        print(f"{field_name} (empty line to finish):")
        lines = []

        while True:
            line = input("   ")
            if not line:
                break
            lines.append(line)

        return '\n'.join(lines)

    def _collect_alternatives(self) -> List[Dict[str, str]]:
        """Collect alternatives considered."""
        alternatives = []

        while True:
            add_more = input("\nAdd alternative? (y/n): ").strip().lower()
            if add_more != 'y':
                break

            name = self._prompt("Alternative name", required=True)
            pros = self._prompt("Pros")
            cons = self._prompt("Cons")
            rejected = self._prompt("Rejected because")

            alternatives.append({
                'name': name,
                'pros': pros,
                'cons': cons,
                'rejected_because': rejected
            })

        return alternatives

    def _collect_related_decisions(self) -> List[str]:
        """Collect related ADR references."""
        related = []

        while True:
            add_more = input("\nAdd related ADR? (y/n): ").strip().lower()
            if add_more != 'y':
                break

            ref = self._prompt("ADR reference (e.g., ADR-001)")
            if ref:
                related.append(ref)

        return related

    def _get_risk_for_reuse_level(self, reuse_pct: float) -> str:
        """Get risk assessment based on reuse percentage."""
        if reuse_pct >= 90:
            return "Low - direct reuse with high confidence"
        elif reuse_pct >= 70:
            return "Medium - wrapper may introduce bugs, test thoroughly"
        else:
            return "Medium - new code requires comprehensive testing"

    def _get_default_template(self) -> str:
        """Get default ADR template."""
        return """# ADR-{{NUMBER}}: {{TITLE}}

**Status:** {{STATUS}}
**Date:** {{DATE}}
**Author:** {{AUTHOR}}

---

## Context

{{CONTEXT}}

---

## Decision

{{DECISION}}

---

## Consequences

{{CONSEQUENCES}}

---

{{ALTERNATIVES}}

{{RELATED}}

---

## References

- Operating Charter: `.ai-workspace/OPERATING_CHARTER.md`
- Principle 2 Documentation: `.ai-workspace/PRINCIPLES.md`
- Multi-language reuse analysis: Python, TypeScript, JavaScript, Rust, Go, Java
- Enforcement system: `.ai-workspace/scripts/enforce_check_before_create.py`

---

*This ADR was created to enforce Principle 2: Document Everything*
*All architectural decisions must be documented and version controlled*
*For reuse <70%, ADR is required before code creation (automatic enforcement)*
"""


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create ADR (Architectural Decision Record) - Principle 2",
        epilog="Example: python create_adr.py --title 'Use PostgreSQL'"
    )
    parser.add_argument(
        '--title',
        help='ADR title for quick creation'
    )
    parser.add_argument(
        '--from-reuse',
        help='Create ADR from reuse analysis JSON file'
    )
    parser.add_argument(
        '--workspace',
        default='.agent-workspace',
        help='Workspace path (default: .agent-workspace)'
    )
    parser.add_argument(
        '--author',
        default='AI Assistant',
        help='Author name (default: AI Assistant)'
    )

    args = parser.parse_args()

    try:
        creator = ADRCreator(args.workspace)

        # Create from reuse analysis
        if args.from_reuse:
            file_path = creator.create_from_reuse_analysis(
                args.from_reuse,
                args.author
            )
            return 0

        # Create from title
        if args.title:
            file_path = creator.create_from_title(
                args.title,
                author=args.author
            )
            return 0

        # Interactive mode
        file_path = creator.create_interactive()
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  ADR creation cancelled")
        return 1

    except Exception as e:
        print(f"\n❌ ADR creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
