#!/usr/bin/env python3
"""
Phase Gate Validator
Enforces Principle 4: Phase Gate Reviews

Validates completion of required phase gates before proceeding.
Gates: Discovery → Design → Implementation → Verification → Integration

Usage:
    python validate_gates.py --phase design
    python validate_gates.py --phase verification --task feature/user-auth
    python validate_gates.py --check-all
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import sys


# Phase gate definitions
PHASE_GATES = {
    'discovery': {
        'order': 1,
        'name': 'Discovery',
        'required_artifacts': [
            'problem_statement',
            'requirements',
            'constraints',
            'success_criteria'
        ],
        'exit_criteria': [
            'Problem clearly defined',
            'Requirements documented',
            'Constraints identified',
            'Success criteria established'
        ]
    },
    'design': {
        'order': 2,
        'name': 'Design',
        'required_artifacts': [
            'architecture_decision',
            'api_design',
            'data_model',
            'reuse_analysis'
        ],
        'exit_criteria': [
            'Architecture documented (ADR created)',
            'API contracts defined',
            'Data models specified',
            'Reuse analysis completed (≥0% documented)'
        ]
    },
    'implementation': {
        'order': 3,
        'name': 'Implementation',
        'required_artifacts': [
            'code_complete',
            'unit_tests',
            'integration_tests',
            'documentation'
        ],
        'exit_criteria': [
            'Code implemented and reviewed',
            'Unit tests passing (≥85% coverage)',
            'Integration tests passing',
            'Code documented (docstrings/comments)'
        ]
    },
    'verification': {
        'order': 4,
        'name': 'Verification',
        'required_artifacts': [
            'real_data_test',
            'performance_test',
            'security_review',
            'edge_cases_tested'
        ],
        'exit_criteria': [
            'Tested with REAL production data (not mocks)',
            'Performance benchmarks met',
            'Security review completed',
            'Edge cases validated'
        ]
    },
    'integration': {
        'order': 5,
        'name': 'Integration',
        'required_artifacts': [
            'integration_complete',
            'deployment_verified',
            'monitoring_setup',
            'rollback_plan'
        ],
        'exit_criteria': [
            'Integrated with existing systems',
            'Deployed to staging/production',
            'Monitoring and alerts configured',
            'Rollback plan documented'
        ]
    }
}


@dataclass
class GateValidation:
    """Represents validation result for a phase gate."""
    phase: str
    passed: bool
    missing_artifacts: List[str]
    missing_criteria: List[str]
    warnings: List[str]
    gate_file: Optional[str]
    timestamp: Optional[str]


class PhaseGateValidator:
    """Validator for phase gate compliance."""

    def __init__(self, workspace_path: str = ".agent-workspace"):
        """Initialize validator."""
        self.workspace = Path(workspace_path)
        self.handoffs_dir = self.workspace / "handoffs"
        self.decisions_dir = self.workspace / "decisions"
        self.errors = []
        self.warnings = []

    def validate_phase(
        self,
        phase: str,
        task_name: Optional[str] = None
    ) -> GateValidation:
        """
        Validate completion of a specific phase gate.

        Args:
            phase: Phase to validate (discovery, design, etc.)
            task_name: Optional task identifier

        Returns:
            GateValidation result
        """
        if phase not in PHASE_GATES:
            raise ValueError(f"Unknown phase: {phase}. Valid phases: {', '.join(PHASE_GATES.keys())}")

        gate_config = PHASE_GATES[phase]

        # Check if previous gates are complete
        prev_incomplete = self._check_previous_gates(phase, task_name)
        if prev_incomplete:
            return GateValidation(
                phase=phase,
                passed=False,
                missing_artifacts=[],
                missing_criteria=[],
                warnings=[f"Cannot proceed to {phase}: previous gate(s) incomplete: {', '.join(prev_incomplete)}"],
                gate_file=None,
                timestamp=None
            )

        # Find gate file
        gate_file = self._find_gate_file(phase, task_name)

        if not gate_file:
            return GateValidation(
                phase=phase,
                passed=False,
                missing_artifacts=gate_config['required_artifacts'],
                missing_criteria=gate_config['exit_criteria'],
                warnings=[f"No gate file found for {phase}"],
                gate_file=None,
                timestamp=None
            )

        # Validate gate file contents
        try:
            with open(gate_file, 'r', encoding='utf-8') as f:
                gate_data = json.load(f)
        except Exception as e:
            return GateValidation(
                phase=phase,
                passed=False,
                missing_artifacts=gate_config['required_artifacts'],
                missing_criteria=gate_config['exit_criteria'],
                warnings=[f"Failed to parse gate file: {e}"],
                gate_file=str(gate_file),
                timestamp=None
            )

        # Check artifacts
        missing_artifacts = []
        for artifact in gate_config['required_artifacts']:
            if artifact not in gate_data.get('artifacts', {}):
                missing_artifacts.append(artifact)

        # Check exit criteria
        missing_criteria = []
        completed_criteria = gate_data.get('exit_criteria_met', [])
        for i, criterion in enumerate(gate_config['exit_criteria']):
            if i >= len(completed_criteria) or not completed_criteria[i]:
                missing_criteria.append(criterion)

        # Check for warnings
        warnings = []

        # Special validation for verification phase - must use real data
        if phase == 'verification':
            if not gate_data.get('artifacts', {}).get('real_data_test', {}).get('used_real_data', False):
                warnings.append("⚠️  CRITICAL: Verification must use REAL production data, not mocks!")

        passed = len(missing_artifacts) == 0 and len(missing_criteria) == 0

        return GateValidation(
            phase=phase,
            passed=passed,
            missing_artifacts=missing_artifacts,
            missing_criteria=missing_criteria,
            warnings=warnings,
            gate_file=str(gate_file),
            timestamp=gate_data.get('timestamp')
        )

    def validate_all_gates(
        self,
        task_name: Optional[str] = None
    ) -> Dict[str, GateValidation]:
        """Validate all phase gates for a task."""
        results = {}

        for phase in PHASE_GATES.keys():
            results[phase] = self.validate_phase(phase, task_name)

        return results

    def _check_previous_gates(
        self,
        current_phase: str,
        task_name: Optional[str]
    ) -> List[str]:
        """Check if all previous gates are complete."""
        current_order = PHASE_GATES[current_phase]['order']
        incomplete = []

        for phase, config in PHASE_GATES.items():
            if config['order'] < current_order:
                validation = self.validate_phase(phase, task_name)
                if not validation.passed:
                    incomplete.append(config['name'])

        return incomplete

    def _find_gate_file(
        self,
        phase: str,
        task_name: Optional[str]
    ) -> Optional[Path]:
        """Find gate file for phase and task."""
        if not self.handoffs_dir.exists():
            return None

        # Pattern: gate-{phase}-{task_name}.json or gate-{phase}.json
        if task_name:
            pattern = f"gate-{phase}-{task_name}.json"
            specific_file = self.handoffs_dir / pattern
            if specific_file.exists():
                return specific_file

        # Try generic gate file
        generic_file = self.handoffs_dir / f"gate-{phase}.json"
        if generic_file.exists():
            return generic_file

        # Search for any gate file matching phase
        for gate_file in self.handoffs_dir.glob(f"gate-{phase}-*.json"):
            return gate_file

        return None

    def create_gate_template(
        self,
        phase: str,
        task_name: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """Create template gate file for a phase."""
        if phase not in PHASE_GATES:
            raise ValueError(f"Unknown phase: {phase}")

        gate_config = PHASE_GATES[phase]

        # Create template
        template = {
            'phase': phase,
            'task_name': task_name or 'default',
            'timestamp': datetime.now().isoformat(),
            'artifacts': {
                artifact: {
                    'completed': False,
                    'path': '',
                    'notes': ''
                } for artifact in gate_config['required_artifacts']
            },
            'exit_criteria_met': [False] * len(gate_config['exit_criteria']),
            'exit_criteria': gate_config['exit_criteria'],
            'notes': '',
            'approved_by': '',
            'approval_timestamp': None
        }

        # Special fields for verification phase
        if phase == 'verification':
            template['artifacts']['real_data_test']['used_real_data'] = False
            template['artifacts']['real_data_test']['data_source'] = ''
            template['artifacts']['real_data_test']['sample_size'] = 0

        # Determine output path
        if output_path:
            file_path = Path(output_path)
        else:
            self.handoffs_dir.mkdir(parents=True, exist_ok=True)
            filename = f"gate-{phase}-{task_name or 'default'}.json"
            file_path = self.handoffs_dir / filename

        # Write template
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)

        return str(file_path)


def print_validation(validation: GateValidation) -> None:
    """Print formatted validation result."""
    print(f"\n🚦 Phase Gate Validation: {PHASE_GATES[validation.phase]['name']}")
    print("=" * 70)

    if validation.passed:
        print(f"\n✅ Gate PASSED - {validation.phase.upper()} complete")
        print(f"📁 Gate file: {validation.gate_file}")
        print(f"⏱️  Timestamp: {validation.timestamp}")
    else:
        print(f"\n❌ Gate FAILED - {validation.phase.upper()} incomplete")

        if validation.missing_artifacts:
            print(f"\n⚠️  Missing {len(validation.missing_artifacts)} required artifacts:")
            for artifact in validation.missing_artifacts:
                print(f"   • {artifact}")

        if validation.missing_criteria:
            print(f"\n⚠️  Incomplete exit criteria ({len(validation.missing_criteria)}):")
            for criterion in validation.missing_criteria:
                print(f"   • {criterion}")

    if validation.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in validation.warnings:
            print(f"   {warning}")

    print("\n📚 Principle 4: Phase Gate Reviews")
    print("   • Cannot skip gates (sequential enforcement)")
    print("   • Real data validation required in verification phase")
    print("   • All exit criteria must be met")

    print("=" * 70)


def print_all_validations(validations: Dict[str, GateValidation]) -> None:
    """Print summary of all gate validations."""
    print("\n🚦 Phase Gate Review - Complete Status")
    print("=" * 70)

    total_gates = len(validations)
    passed_gates = sum(1 for v in validations.values() if v.passed)
    current_phase = None

    print(f"\n📊 Overall Progress: {passed_gates}/{total_gates} gates passed")
    print()

    for phase in ['discovery', 'design', 'implementation', 'verification', 'integration']:
        validation = validations[phase]
        status = "✅" if validation.passed else "❌"
        gate_name = PHASE_GATES[phase]['name']

        print(f"{status} {gate_name:<20} ", end='')

        if validation.passed:
            print(f"COMPLETE ({validation.timestamp})")
        else:
            missing_count = len(validation.missing_artifacts) + len(validation.missing_criteria)
            print(f"INCOMPLETE ({missing_count} items)")

            if not current_phase and not validation.passed:
                current_phase = phase

    if current_phase:
        print(f"\n🎯 Current Phase: {PHASE_GATES[current_phase]['name']}")
        print(f"   Next: Complete {current_phase} gate to proceed")

    print("\n📋 Gate Workflow:")
    print("   1. Discovery  → Define problem and requirements")
    print("   2. Design     → Create architecture and reuse analysis")
    print("   3. Implementation → Build and test code")
    print("   4. Verification   → Validate with REAL data")
    print("   5. Integration    → Deploy and monitor")

    print("\n💡 To create gate file:")
    print(f"   python .ai-workspace/scripts/validate_gates.py --create {current_phase or 'discovery'}")

    print("=" * 70)


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate phase gate completion (Principle 4)",
        epilog="Example: python validate_gates.py --phase verification"
    )
    parser.add_argument(
        '--phase',
        choices=list(PHASE_GATES.keys()),
        help='Specific phase to validate'
    )
    parser.add_argument(
        '--task',
        help='Task identifier (optional)'
    )
    parser.add_argument(
        '--check-all',
        action='store_true',
        help='Validate all gates'
    )
    parser.add_argument(
        '--create',
        choices=list(PHASE_GATES.keys()),
        help='Create gate template for phase'
    )
    parser.add_argument(
        '--workspace',
        default='.agent-workspace',
        help='Path to workspace (default: .agent-workspace)'
    )

    args = parser.parse_args()

    try:
        validator = PhaseGateValidator(args.workspace)

        # Create template
        if args.create:
            file_path = validator.create_gate_template(args.create, args.task)
            print(f"✅ Created gate template: {file_path}")
            print(f"\n📝 Next steps:")
            print(f"   1. Edit {file_path}")
            print(f"   2. Fill in artifact paths and completion status")
            print(f"   3. Mark exit criteria as met")
            print(f"   4. Re-run validation: python validate_gates.py --phase {args.create}")
            return 0

        # Validate all gates
        if args.check_all:
            validations = validator.validate_all_gates(args.task)
            print_all_validations(validations)

            # Return non-zero if any gate failed
            if not all(v.passed for v in validations.values()):
                return 1
            return 0

        # Validate specific phase
        if args.phase:
            validation = validator.validate_phase(args.phase, args.task)
            print_validation(validation)

            if not validation.passed:
                print(f"\n💡 To fix:")
                print(f"   1. Create gate file: python validate_gates.py --create {args.phase}")
                print(f"   2. Complete required artifacts")
                print(f"   3. Re-validate: python validate_gates.py --phase {args.phase}")

            return 0 if validation.passed else 1

        # No arguments - show usage
        parser.print_help()
        return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Validation cancelled")
        return 1

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
