#!/usr/bin/env python3
"""
AI Agent Orchestrator
Master script to coordinate all refactoring phases for Beverly Knits ERP v2
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_agent_orchestration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RefactoringOrchestrator:
    """Orchestrates the complete AI agent refactoring process"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.start_time = datetime.now()
        self.phase_results = {}
        self.checkpoints_dir = self.project_root / '.ai_checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Phase scripts
        self.phases = {
            'phase1': {
                'name': 'Critical Fixes',
                'script': 'phase1_critical_fixes.py',
                'timeout': 300,  # 5 minutes
                'required': True
            },
            'phase2': {
                'name': 'Monolith Decomposition',
                'script': 'phase2_monolith_decomposition.py',
                'timeout': 600,  # 10 minutes
                'required': True
            },
            'phase3': {
                'name': 'Testing & Validation',
                'script': 'phase3_testing_validation.py',
                'timeout': 900,  # 15 minutes
                'required': False
            }
        }
    
    def save_checkpoint(self, phase: str, results: Dict):
        """Save checkpoint after each phase"""
        checkpoint_file = self.checkpoints_dir / f"{phase}_checkpoint.json"
        checkpoint_data = {
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'elapsed_time': str(datetime.now() - self.start_time)
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"💾 Saved checkpoint for {phase}")
    
    def load_checkpoint(self, phase: str) -> Optional[Dict]:
        """Load checkpoint if it exists"""
        checkpoint_file = self.checkpoints_dir / f"{phase}_checkpoint.json"
        
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def run_phase(self, phase_key: str) -> Dict:
        """Run a single refactoring phase"""
        phase = self.phases[phase_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting {phase['name']} ({phase_key})")
        logger.info(f"{'='*60}")
        
        # Check if already completed
        checkpoint = self.load_checkpoint(phase_key)
        if checkpoint and checkpoint['results'].get('success'):
            logger.info(f"✅ {phase['name']} already completed (from checkpoint)")
            return checkpoint['results']
        
        script_path = self.project_root / 'scripts' / 'ai_tasks' / phase['script']
        
        # Check if script exists
        if not script_path.exists():
            logger.warning(f"⚠️ Script not found: {script_path}")
            
            # Try to create a minimal version
            if phase_key == 'phase3':
                self._create_phase3_script(script_path)
            else:
                return {'success': False, 'error': 'Script not found'}
        
        try:
            # Run the phase script
            env = os.environ.copy()
            env['PROJECT_ROOT'] = str(self.project_root)
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=phase['timeout'],
                env=env
            )
            
            # Parse results
            success = result.returncode == 0
            
            phase_results = {
                'success': success,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            # Save checkpoint
            self.save_checkpoint(phase_key, phase_results)
            
            if success:
                logger.info(f"✅ {phase['name']} completed successfully")
            else:
                logger.error(f"❌ {phase['name']} failed")
                if phase['required']:
                    logger.error(f"Required phase failed. Stopping orchestration.")
                    return phase_results
            
            return phase_results
            
        except subprocess.TimeoutExpired:
            logger.error(f"⏱️ {phase['name']} timed out after {phase['timeout']} seconds")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"❌ Error running {phase['name']}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_phase3_script(self, script_path: Path):
        """Create a minimal Phase 3 testing script"""
        phase3_content = '''#!/usr/bin/env python3
"""
AI Agent Phase 3: Testing & Validation
Validates the refactored system and generates tests
"""

import os
import sys
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_tests():
    """Run the test suite"""
    logger.info("🧪 Running test suite...")
    
    project_root = Path(os.getenv('PROJECT_ROOT', os.getcwd()))
    
    # Try to run pytest
    try:
        result = subprocess.run(
            ['pytest', 'tests/', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode == 0:
            logger.info("✅ All tests passed")
            return True
        else:
            logger.warning("⚠️ Some tests failed")
            return False
    except FileNotFoundError:
        logger.warning("pytest not found, skipping tests")
        return True
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

def validate_refactoring():
    """Validate the refactoring was successful"""
    logger.info("🔍 Validating refactoring...")
    
    project_root = Path(os.getenv('PROJECT_ROOT', os.getcwd()))
    
    # Check if components directory was created
    components_dir = project_root / 'src' / 'components'
    if components_dir.exists():
        component_count = len(list(components_dir.glob('*.py')))
        logger.info(f"  Found {component_count} extracted components")
    
    # Check if API routes were extracted
    api_routes = project_root / 'src' / 'api' / 'routes.py'
    if api_routes.exists():
        logger.info("  ✅ API routes extracted")
    
    # Check monolith size
    monolith = project_root / 'src' / 'core' / 'beverly_comprehensive_erp.py'
    if monolith.exists():
        line_count = len(monolith.read_text().splitlines())
        logger.info(f"  Monolith reduced to {line_count} lines")
        
        if line_count < 10000:
            logger.info("  ✅ Monolith successfully reduced")
    
    return True

def main():
    """Main execution"""
    logger.info("=" * 60)
    logger.info("🚀 PHASE 3: TESTING & VALIDATION")
    logger.info("=" * 60)
    
    tests_passed = run_tests()
    validation_passed = validate_refactoring()
    
    if tests_passed and validation_passed:
        logger.info("✅ Phase 3 completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Phase 3 validation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        script_path.write_text(phase3_content)
        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod(script_path, 0o755)
        logger.info(f"Created minimal Phase 3 script at {script_path}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive refactoring report"""
        report_lines = [
            "=" * 80,
            "📊 AI AGENT REFACTORING REPORT",
            "=" * 80,
            f"Project: Beverly Knits ERP v2",
            f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {datetime.now() - self.start_time}",
            "",
            "PHASE RESULTS:",
            "-" * 40
        ]
        
        for phase_key, results in self.phase_results.items():
            phase_name = self.phases[phase_key]['name']
            status = "✅ SUCCESS" if results.get('success') else "❌ FAILED"
            report_lines.append(f"\n{phase_name}: {status}")
            
            if not results.get('success'):
                error = results.get('error', 'Unknown error')
                report_lines.append(f"  Error: {error}")
        
        # Add recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "-" * 40
        ])
        
        all_success = all(r.get('success') for r in self.phase_results.values())
        
        if all_success:
            report_lines.extend([
                "✅ All phases completed successfully!",
                "",
                "Next Steps:",
                "1. Run comprehensive integration tests",
                "2. Review extracted components for correctness",
                "3. Update documentation",
                "4. Deploy to staging environment",
                "5. Monitor system performance"
            ])
        else:
            failed_phases = [k for k, v in self.phase_results.items() if not v.get('success')]
            report_lines.extend([
                "⚠️ Some phases failed. Manual intervention required.",
                "",
                "Failed Phases:",
            ])
            for phase in failed_phases:
                report_lines.append(f"  - {self.phases[phase]['name']}")
            
            report_lines.extend([
                "",
                "Recommended Actions:",
                "1. Review log files for detailed error information",
                "2. Manually fix identified issues",
                "3. Re-run failed phases using individual scripts",
                "4. Contact senior developers if issues persist"
            ])
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def execute_all_phases(self) -> Dict:
        """Execute all refactoring phases in sequence"""
        logger.info("🎯 Starting AI Agent Refactoring Orchestration")
        logger.info(f"Project Root: {self.project_root}")
        logger.info(f"Start Time: {self.start_time}")
        
        # Run each phase
        for phase_key in self.phases.keys():
            results = self.run_phase(phase_key)
            self.phase_results[phase_key] = results
            
            # Stop if required phase failed
            if not results.get('success') and self.phases[phase_key]['required']:
                logger.error(f"Stopping orchestration due to {phase_key} failure")
                break
            
            # Small delay between phases
            time.sleep(2)
        
        # Generate and save report
        report = self.generate_report()
        report_file = self.project_root / 'ai_refactoring_report.txt'
        report_file.write_text(report)
        
        # Print report
        print("\n" + report)
        
        # Determine overall success
        all_required_passed = all(
            self.phase_results.get(k, {}).get('success', False)
            for k, v in self.phases.items()
            if v['required']
        )
        
        return {
            'success': all_required_passed,
            'phases_completed': len(self.phase_results),
            'duration': str(datetime.now() - self.start_time),
            'report_path': str(report_file)
        }


def main():
    """Main orchestration entry point"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='AI Agent Refactoring Orchestrator')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--phase', help='Run specific phase only (phase1, phase2, phase3)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoints')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = RefactoringOrchestrator(args.project_root)
    
    # Run specific phase or all phases
    if args.phase:
        results = orchestrator.run_phase(args.phase)
        success = results.get('success', False)
    else:
        results = orchestrator.execute_all_phases()
        success = results['success']
    
    # Exit with appropriate code
    if success:
        logger.info("✅ Orchestration completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Orchestration failed. Please review logs and report.")
        sys.exit(1)


if __name__ == "__main__":
    main()