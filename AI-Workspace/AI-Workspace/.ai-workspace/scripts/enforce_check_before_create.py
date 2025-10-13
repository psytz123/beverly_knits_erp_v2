#!/usr/bin/env python3
"""
Automatic Check-Before-Create Enforcer
Enforces Principle 3: MUST search for existing code before writing new code

This script acts as a guard - preventing new code creation without reuse analysis.
"""

from typing import Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
import subprocess


@dataclass
class ReuseCheck:
    """Track reuse analysis for a task."""
    task_name: str
    search_completed: bool
    reuse_analyzed: bool
    reuse_percentage: float
    timestamp: str
    files_checked: List[str]
    approved_to_create: bool


class CheckBeforeCreateEnforcer:
    """Enforce mandatory reuse checking before code creation."""

    def __init__(self, workspace_path: str = ".ai-workspace"):
        """Initialize enforcer."""
        self.workspace = Path(workspace_path)
        self.cache_dir = self.workspace / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checks_file = self.cache_dir / "reuse_checks.json"

        # Load existing checks
        self.checks: Dict[str, ReuseCheck] = self._load_checks()

    def _load_checks(self) -> Dict[str, ReuseCheck]:
        """Load existing reuse checks from cache."""
        if not self.checks_file.exists():
            return {}

        try:
            with open(self.checks_file, 'r') as f:
                data = json.load(f)

            checks = {}
            for task_name, check_data in data.items():
                checks[task_name] = ReuseCheck(**check_data)

            return checks
        except Exception:
            return {}

    def _save_checks(self) -> None:
        """Save reuse checks to cache."""
        data = {}
        for task_name, check in self.checks.items():
            data[task_name] = {
                'task_name': check.task_name,
                'search_completed': check.search_completed,
                'reuse_analyzed': check.reuse_analyzed,
                'reuse_percentage': check.reuse_percentage,
                'timestamp': check.timestamp,
                'files_checked': check.files_checked,
                'approved_to_create': check.approved_to_create
            }

        with open(self.checks_file, 'w') as f:
            json.dump(data, f, indent=2)

    def can_create_new_code(
        self,
        task_name: str,
        intent: str,
        max_age_minutes: int = 30
    ) -> tuple[bool, str]:
        """
        Check if new code creation is allowed.

        Returns:
            (allowed: bool, reason: str)
        """
        # Check if reuse analysis exists for this task
        check = self.checks.get(task_name)

        if not check:
            return (False,
                    f"❌ BLOCKED: No reuse analysis found for '{task_name}'\n"
                    f"   → REQUIRED: Run search_codebase.py first\n"
                    f"   → Command: python .ai-workspace/scripts/search_codebase.py \"{intent}\""
            )

        # Check if analysis is recent (within max_age_minutes)
        check_time = datetime.fromisoformat(check.timestamp)
        age = datetime.now() - check_time

        if age > timedelta(minutes=max_age_minutes):
            return (False,
                    f"❌ BLOCKED: Reuse analysis is stale ({age.total_seconds() / 60:.0f} min old)\n"
                    f"   → REQUIRED: Re-run reuse analysis (max age: {max_age_minutes} min)\n"
                    f"   → Command: python .ai-workspace/scripts/search_codebase.py \"{intent}\""
            )

        # Check if search was completed
        if not check.search_completed:
            return (False,
                    f"❌ BLOCKED: Code search not completed for '{task_name}'\n"
                    f"   → REQUIRED: Complete search workflow\n"
                    f"   → Command: python .ai-workspace/scripts/search_codebase.py \"{intent}\""
            )

        # Check if reuse analysis was done
        if not check.reuse_analyzed:
            return (False,
                    f"❌ BLOCKED: Reuse analysis not completed for '{task_name}'\n"
                    f"   → REQUIRED: Analyze reuse potential\n"
                    f"   → Command: python .ai-workspace/scripts/analyze_reuse.py \"{intent}\" [file]"
            )

        # Check reuse percentage and ADR requirement
        if check.reuse_percentage < 70 and not check.approved_to_create:
            return (False,
                    f"❌ BLOCKED: Low reuse ({check.reuse_percentage:.1f}%) - ADR required\n"
                    f"   → REQUIRED: Create ADR documenting why reuse failed\n"
                    f"   → Command: python .ai-workspace/scripts/create_adr.py --title \"Reuse: {task_name}\"\n"
                    f"   → After ADR: Run: python .ai-workspace/scripts/enforce_check_before_create.py --approve {task_name}"
            )

        # All checks passed!
        return (True,
                f"✅ APPROVED: Reuse analysis complete ({check.reuse_percentage:.1f}% reuse)\n"
                f"   → You may proceed with {'wrapper/adapter' if check.reuse_percentage >= 70 else 'new code'}\n"
                f"   → Analysis age: {age.total_seconds() / 60:.0f} minutes"
        )

    def record_search(
        self,
        task_name: str,
        files_found: List[str]
    ) -> None:
        """Record that search was completed."""
        check = self.checks.get(task_name, ReuseCheck(
            task_name=task_name,
            search_completed=False,
            reuse_analyzed=False,
            reuse_percentage=0.0,
            timestamp=datetime.now().isoformat(),
            files_checked=[],
            approved_to_create=False
        ))

        check.search_completed = True
        check.files_checked = files_found
        check.timestamp = datetime.now().isoformat()

        self.checks[task_name] = check
        self._save_checks()

        print(f"✅ Search recorded for: {task_name}")
        print(f"   Files found: {len(files_found)}")

    def record_analysis(
        self,
        task_name: str,
        reuse_percentage: float,
        files_analyzed: List[str]
    ) -> None:
        """Record that reuse analysis was completed."""
        check = self.checks.get(task_name, ReuseCheck(
            task_name=task_name,
            search_completed=True,
            reuse_analyzed=False,
            reuse_percentage=0.0,
            timestamp=datetime.now().isoformat(),
            files_checked=[],
            approved_to_create=False
        ))

        check.reuse_analyzed = True
        check.reuse_percentage = reuse_percentage
        check.timestamp = datetime.now().isoformat()

        # Auto-approve if reuse >= 70%
        if reuse_percentage >= 70:
            check.approved_to_create = True

        self.checks[task_name] = check
        self._save_checks()

        print(f"✅ Reuse analysis recorded for: {task_name}")
        print(f"   Reuse percentage: {reuse_percentage:.1f}%")
        if reuse_percentage >= 70:
            print(f"   ✅ Auto-approved (≥70% reuse)")
        else:
            print(f"   ⚠️  ADR required (<70% reuse)")

    def approve_new_code(
        self,
        task_name: str,
        adr_path: str
    ) -> None:
        """Manually approve new code creation after ADR is created."""
        check = self.checks.get(task_name)

        if not check:
            print(f"❌ No reuse analysis found for: {task_name}")
            return

        if not Path(adr_path).exists():
            print(f"❌ ADR not found: {adr_path}")
            return

        check.approved_to_create = True
        check.timestamp = datetime.now().isoformat()

        self.checks[task_name] = check
        self._save_checks()

        print(f"✅ New code creation approved for: {task_name}")
        print(f"   ADR: {adr_path}")

    def run_automatic_check(
        self,
        task_name: str,
        intent: str
    ) -> bool:
        """
        Run automatic check workflow.

        Returns:
            True if checks pass, False otherwise
        """
        print(f"\n🔍 Automatic Check-Before-Create Workflow")
        print("=" * 70)
        print(f"Task: {task_name}")
        print(f"Intent: {intent}")
        print("=" * 70)

        # Step 1: Check if allowed
        allowed, reason = self.can_create_new_code(task_name, intent)

        print(f"\n{reason}")

        if not allowed:
            print("\n📚 Principle 3: Check Before Create")
            print("   You MUST search for existing code before creating new code")
            print("=" * 70)
            return False

        print("\n✅ All checks passed - you may proceed with coding")
        print("=" * 70)
        return True


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enforce Principle 3: Check Before Create",
        epilog="Example: python enforce_check_before_create.py --check 'user-auth' 'user authentication'"
    )

    parser.add_argument(
        '--check',
        nargs=2,
        metavar=('TASK', 'INTENT'),
        help='Check if new code creation is allowed'
    )
    parser.add_argument(
        '--record-search',
        nargs=2,
        metavar=('TASK', 'FILES'),
        help='Record that search was completed'
    )
    parser.add_argument(
        '--record-analysis',
        nargs=3,
        metavar=('TASK', 'PERCENTAGE', 'FILES'),
        help='Record reuse analysis result'
    )
    parser.add_argument(
        '--approve',
        nargs=2,
        metavar=('TASK', 'ADR_PATH'),
        help='Approve new code after ADR created'
    )
    parser.add_argument(
        '--workspace',
        default='.ai-workspace',
        help='Workspace path (default: .ai-workspace)'
    )

    args = parser.parse_args()

    try:
        enforcer = CheckBeforeCreateEnforcer(args.workspace)

        # Check if code creation is allowed
        if args.check:
            task_name, intent = args.check
            success = enforcer.run_automatic_check(task_name, intent)
            return 0 if success else 1

        # Record search completion
        if args.record_search:
            task_name, files_str = args.record_search
            files = files_str.split(',') if files_str else []
            enforcer.record_search(task_name, files)
            return 0

        # Record analysis completion
        if args.record_analysis:
            task_name, percentage_str, files_str = args.record_analysis
            percentage = float(percentage_str)
            files = files_str.split(',') if files_str else []
            enforcer.record_analysis(task_name, percentage, files)
            return 0

        # Approve new code
        if args.approve:
            task_name, adr_path = args.approve
            enforcer.approve_new_code(task_name, adr_path)
            return 0

        # No arguments - show help
        parser.print_help()
        return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
