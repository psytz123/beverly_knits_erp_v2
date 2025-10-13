#!/usr/bin/env python3
"""
Task Planning Wizard
Enforces Principle 5: Plan Before Act

Creates structured task plans following the 5-phase gate workflow.
Integrates reuse analysis and phase gate validation.

Usage:
    python plan_task.py                          # Interactive wizard
    python plan_task.py --task "Add user authentication"
    python plan_task.py --quick "Fix email bug"
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import sys
import subprocess


@dataclass
class TaskPlan:
    """Represents a structured task plan."""
    task_id: str
    title: str
    description: str
    priority: str  # critical, high, medium, low
    estimated_effort: str  # hours or story points
    phases: Dict[str, dict]
    reuse_analysis: Optional[dict]
    success_criteria: List[str]
    risks: List[str]
    dependencies: List[str]
    created_at: str
    created_by: str


class TaskPlanner:
    """Interactive task planning wizard."""

    def __init__(self, workspace_path: str = ".agent-workspace"):
        """Initialize task planner."""
        self.workspace = Path(workspace_path)
        self.handoffs_dir = self.workspace / "handoffs"
        self.scripts_dir = self.workspace / "scripts"

        # Create directories
        self.handoffs_dir.mkdir(parents=True, exist_ok=True)

        # Phase template
        self.phases = {
            'discovery': {
                'name': 'Discovery',
                'status': 'pending',
                'tasks': [],
                'deliverables': [],
                'notes': ''
            },
            'design': {
                'name': 'Design',
                'status': 'pending',
                'tasks': [],
                'deliverables': [],
                'notes': ''
            },
            'implementation': {
                'name': 'Implementation',
                'status': 'pending',
                'tasks': [],
                'deliverables': [],
                'notes': ''
            },
            'verification': {
                'name': 'Verification',
                'status': 'pending',
                'tasks': [],
                'deliverables': [],
                'notes': ''
            },
            'integration': {
                'name': 'Integration',
                'status': 'pending',
                'tasks': [],
                'deliverables': [],
                'notes': ''
            }
        }

    def create_interactive_plan(self) -> str:
        """Interactive planning wizard."""
        print("\n📋 Task Planning Wizard")
        print("=" * 70)
        print("Enforcing Principle 5: Plan Before Act")
        print("=" * 70)

        # Basic information
        title = self._prompt("Task Title", required=True)
        description = self._prompt_multiline("Description")

        priority = self._prompt_choice(
            "Priority",
            ['critical', 'high', 'medium', 'low'],
            default='medium'
        )

        effort = self._prompt("Estimated Effort (hours/points)", default="TBD")

        # Reuse analysis
        print("\n🔍 Step 1: Reuse Analysis (Principle 3: Check Before Create)")
        print("-" * 70)

        reuse_analysis = None
        do_reuse = input("Run reuse analysis? (y/n): ").strip().lower()

        if do_reuse == 'y':
            reuse_analysis = self._run_reuse_analysis(title)

        # Phase planning
        print("\n📝 Step 2: Phase Planning")
        print("-" * 70)
        print("Plan tasks for each phase gate:")

        phases = self._plan_phases(title, reuse_analysis)

        # Success criteria
        print("\n✅ Step 3: Success Criteria")
        print("-" * 70)
        success_criteria = self._collect_success_criteria()

        # Risks
        print("\n⚠️  Step 4: Risk Assessment")
        print("-" * 70)
        risks = self._collect_risks()

        # Dependencies
        print("\n🔗 Step 5: Dependencies")
        print("-" * 70)
        dependencies = self._collect_dependencies()

        # Create task plan
        task_id = self._generate_task_id(title)

        plan = TaskPlan(
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            estimated_effort=effort,
            phases=phases,
            reuse_analysis=reuse_analysis,
            success_criteria=success_criteria,
            risks=risks,
            dependencies=dependencies,
            created_at=datetime.now().isoformat(),
            created_by="AI Assistant"
        )

        # Save plan
        file_path = self._save_plan(plan)

        # Print summary
        self._print_plan_summary(plan, file_path)

        return str(file_path)

    def create_quick_plan(self, task_title: str) -> str:
        """Create quick plan with defaults."""
        print(f"\n⚡ Quick Plan: {task_title}")
        print("=" * 70)

        # Generate default phases
        phases = self._generate_default_phases(task_title)

        task_id = self._generate_task_id(task_title)

        plan = TaskPlan(
            task_id=task_id,
            title=task_title,
            description="TODO: Add detailed description",
            priority="medium",
            estimated_effort="TBD",
            phases=phases,
            reuse_analysis=None,
            success_criteria=["TODO: Define success criteria"],
            risks=["TODO: Assess risks"],
            dependencies=[],
            created_at=datetime.now().isoformat(),
            created_by="AI Assistant"
        )

        file_path = self._save_plan(plan)

        print(f"\n✅ Quick plan created!")
        print(f"📁 Path: {file_path}")
        print(f"\n⚠️  Template created with TODOs - please complete planning")
        print(f"\n💡 Edit now:")
        print(f"   code {file_path}")

        return str(file_path)

    def _run_reuse_analysis(self, intent: str) -> Optional[dict]:
        """
        Run reuse analysis using search_codebase.py.

        Supports: Python, TypeScript, JavaScript, Rust, Go, Java
        """
        search_script = self.scripts_dir / "search_codebase.py"

        if not search_script.exists():
            print("   ⚠️  search_codebase.py not found - skipping reuse analysis")
            return None

        print(f"   🔍 Searching for similar code: '{intent}'")
        print(f"   📚 Multi-language support: Python, TypeScript, JavaScript, Rust, Go, Java")

        try:
            result = subprocess.run(
                [sys.executable, str(search_script), intent, "--limit", "5"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("   ✅ Reuse analysis complete - check output above")
                return {
                    'intent': intent,
                    'search_completed': True,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                print(f"   ⚠️  Search failed: {result.stderr}")
                return None

        except Exception as e:
            print(f"   ⚠️  Reuse analysis error: {e}")
            return None

    def _plan_phases(
        self,
        task_title: str,
        reuse_analysis: Optional[dict]
    ) -> Dict[str, dict]:
        """Plan tasks for each phase."""
        phases = {}

        for phase_key in ['discovery', 'design', 'implementation', 'verification', 'integration']:
            phase_name = self.phases[phase_key]['name']

            print(f"\n▶ {phase_name} Phase:")

            # Default tasks based on phase
            default_tasks = self._get_default_phase_tasks(phase_key, task_title, reuse_analysis)

            print("   Suggested tasks:")
            for i, task in enumerate(default_tasks, 1):
                print(f"     {i}. {task}")

            customize = input("   Customize tasks? (y/n): ").strip().lower()

            if customize == 'y':
                tasks = self._collect_phase_tasks(default_tasks)
            else:
                tasks = default_tasks

            deliverables = self._collect_deliverables(phase_key)

            phases[phase_key] = {
                'name': phase_name,
                'status': 'pending',
                'tasks': tasks,
                'deliverables': deliverables,
                'notes': ''
            }

        return phases

    def _get_default_phase_tasks(
        self,
        phase: str,
        task_title: str,
        reuse_analysis: Optional[dict]
    ) -> List[str]:
        """Get default tasks for a phase."""
        defaults = {
            'discovery': [
                'Define problem statement and requirements',
                'Identify constraints and assumptions',
                'Research existing solutions',
                'Define success criteria'
            ],
            'design': [
                'Create architectural design',
                'Design API contracts/interfaces',
                'Create data models',
                'Complete reuse analysis (6 languages)' if not reuse_analysis else '✅ Reuse analysis complete',
                'Run enforcement check: enforce_check_before_create.py',
                'Create ADR for key decisions (required if reuse <70%)'
            ],
            'implementation': [
                'Implement core functionality',
                'Write unit tests (≥85% coverage)',
                'Write integration tests',
                'Add documentation (docstrings/comments)',
                'Code review'
            ],
            'verification': [
                '⚠️  Test with REAL production data (not mocks)',
                'Performance testing and benchmarking',
                'Security review',
                'Test edge cases and error scenarios',
                'User acceptance testing (if applicable)'
            ],
            'integration': [
                'Integrate with existing systems',
                'Deploy to staging environment',
                'Setup monitoring and alerts',
                'Create rollback plan',
                'Production deployment'
            ]
        }

        return defaults.get(phase, [])

    def _generate_default_phases(self, task_title: str) -> Dict[str, dict]:
        """Generate default phases with standard tasks."""
        phases = {}

        for phase_key in ['discovery', 'design', 'implementation', 'verification', 'integration']:
            tasks = self._get_default_phase_tasks(phase_key, task_title, None)

            phases[phase_key] = {
                'name': self.phases[phase_key]['name'],
                'status': 'pending',
                'tasks': tasks,
                'deliverables': [],
                'notes': 'TODO: Add phase-specific notes'
            }

        return phases

    def _collect_phase_tasks(self, defaults: List[str]) -> List[str]:
        """Collect custom phase tasks."""
        print("   Enter tasks (empty line to finish, 'd' to use defaults):")

        tasks = []
        while True:
            task = input("   • ").strip()

            if not task:
                break

            if task.lower() == 'd':
                return defaults

            tasks.append(task)

        return tasks if tasks else defaults

    def _collect_deliverables(self, phase: str) -> List[str]:
        """Collect phase deliverables."""
        defaults = {
            'discovery': ['Requirements document', 'Problem statement'],
            'design': ['Architecture diagram', 'ADR', 'Reuse analysis report'],
            'implementation': ['Working code', 'Tests', 'Documentation'],
            'verification': ['Test results', 'Performance benchmarks'],
            'integration': ['Deployed system', 'Monitoring dashboard']
        }

        default_dels = defaults.get(phase, [])

        print(f"   Default deliverables: {', '.join(default_dels)}")
        use_defaults = input("   Use defaults? (y/n): ").strip().lower()

        if use_defaults == 'y':
            return default_dels

        deliverables = []
        print("   Enter deliverables (empty line to finish):")

        while True:
            deliverable = input("   • ").strip()
            if not deliverable:
                break
            deliverables.append(deliverable)

        return deliverables if deliverables else default_dels

    def _collect_success_criteria(self) -> List[str]:
        """Collect success criteria."""
        print("Define success criteria (what does 'done' look like?):")
        criteria = []

        while True:
            criterion = input("   • ").strip()
            if not criterion:
                break
            criteria.append(criterion)

        if not criteria:
            criteria = ["TODO: Define success criteria"]

        return criteria

    def _collect_risks(self) -> List[str]:
        """Collect potential risks."""
        print("Identify potential risks:")
        risks = []

        while True:
            risk = input("   • ").strip()
            if not risk:
                break
            risks.append(risk)

        if not risks:
            risks = ["TODO: Assess risks"]

        return risks

    def _collect_dependencies(self) -> List[str]:
        """Collect task dependencies."""
        print("List dependencies (other tasks, systems, resources):")
        dependencies = []

        while True:
            dependency = input("   • ").strip()
            if not dependency:
                break
            dependencies.append(dependency)

        return dependencies

    def _generate_task_id(self, title: str) -> str:
        """Generate task ID from title."""
        import re

        # Clean title
        task_id = re.sub(r'[^a-z0-9]+', '-', title.lower())
        task_id = task_id.strip('-')[:30]

        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        return f"task-{task_id}-{timestamp}"

    def _save_plan(self, plan: TaskPlan) -> Path:
        """Save task plan to file."""
        filename = f"{plan.task_id}.json"
        file_path = self.handoffs_dir / filename

        # Convert to dict
        plan_dict = asdict(plan)

        # Write JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(plan_dict, f, indent=2)

        return file_path

    def _print_plan_summary(self, plan: TaskPlan, file_path: Path) -> None:
        """Print task plan summary."""
        print("\n" + "=" * 70)
        print("📋 Task Plan Summary")
        print("=" * 70)

        print(f"\n📌 Task: {plan.title}")
        print(f"🆔 ID: {plan.task_id}")
        print(f"🎯 Priority: {plan.priority}")
        print(f"⏱️  Effort: {plan.estimated_effort}")

        print(f"\n📝 Phases ({len(plan.phases)}):")
        for i, (phase_key, phase) in enumerate(plan.phases.items(), 1):
            task_count = len(phase['tasks'])
            deliverable_count = len(phase['deliverables'])
            print(f"   {i}. {phase['name']}: {task_count} tasks, {deliverable_count} deliverables")

        if plan.reuse_analysis:
            print(f"\n✅ Reuse analysis: Complete")

        print(f"\n🎯 Success Criteria ({len(plan.success_criteria)}):")
        for criterion in plan.success_criteria[:3]:
            print(f"   • {criterion}")

        if len(plan.success_criteria) > 3:
            print(f"   ... and {len(plan.success_criteria) - 3} more")

        if plan.risks:
            print(f"\n⚠️  Risks ({len(plan.risks)}):")
            for risk in plan.risks[:3]:
                print(f"   • {risk}")

        print(f"\n📁 Plan saved: {file_path}")

        print("\n💡 Next steps:")
        print("   1. Review and refine plan")
        print("   2. Start with Discovery phase")
        print(f"   3. Before coding:")
        print(f"      - Search: python .ai-workspace/scripts/search_codebase.py 'intent'")
        print(f"      - Analyze: python .ai-workspace/scripts/analyze_reuse.py 'intent' file.py")
        print(f"      - Check: python .ai-workspace/scripts/enforce_check_before_create.py --check task 'intent'")
        print(f"   4. Track progress: python .ai-workspace/scripts/validate_gates.py --check-all")
        print("   5. Create gate files as you complete each phase")

        print("\n📚 Principles Enforced:")
        print("   ✅ Principle 5: Plan Before Act (structured planning)")
        print("   ✅ Principle 3: Check Before Create (automatic enforcement, 6 languages)")
        print("   ✅ Principle 4: Phase Gate Reviews (5-phase workflow)")
        print("   ✅ Principle 2: Document Everything (ADR required if reuse <70%)")

        print("=" * 70)

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
                print("   ⚠️  This field is required.")
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
        """Prompt for choice selection."""
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

            print("   ⚠️  Invalid selection.")

    def _prompt_multiline(self, field_name: str) -> str:
        """Prompt for multiline input."""
        print(f"{field_name} (empty line to finish):")
        lines = []

        while True:
            line = input("   ")
            if not line:
                break
            lines.append(line)

        return '\n'.join(lines) if lines else "TODO: Add description"


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Task Planning Wizard - Principle 5: Plan Before Act",
        epilog="Example: python plan_task.py --task 'Add user authentication'"
    )
    parser.add_argument(
        '--task',
        help='Task title for quick planning'
    )
    parser.add_argument(
        '--quick',
        help='Create quick plan (alias for --task)'
    )
    parser.add_argument(
        '--workspace',
        default='.agent-workspace',
        help='Workspace path (default: .agent-workspace)'
    )

    args = parser.parse_args()

    try:
        planner = TaskPlanner(args.workspace)

        # Quick plan
        if args.task or args.quick:
            task_title = args.task or args.quick
            file_path = planner.create_quick_plan(task_title)
            return 0

        # Interactive plan
        file_path = planner.create_interactive_plan()
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Planning cancelled")
        return 1

    except Exception as e:
        print(f"\n❌ Planning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
