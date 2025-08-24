#!/usr/bin/env python3
"""
Comprehensive Test Runner for Beverly Knits ERP v2
Executes all tests and generates coverage report
"""

import subprocess
import sys
import os
from datetime import datetime
import json

# Set up proper Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.environ['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), 'src')

class TestRunner:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_suites': {},
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0
            }
        }
    
    def run_test_suite(self, name, test_path, markers=None):
        """Run a specific test suite"""
        print(f"\n{'='*60}")
        print(f"Running {name} Tests")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            test_path,
            '-v',
            '--tb=short',
            '--json-report',
            '--json-report-file=/tmp/test_report.json'
        ]
        
        if markers:
            cmd.extend(['-m', markers])
        
        # Add coverage if available
        try:
            import pytest_cov
            cmd.extend(['--cov=src', '--cov-report=term-missing'])
        except ImportError:
            pass
        
        # Run tests
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse results
        suite_results = self.parse_results(result, name)
        self.results['test_suites'][name] = suite_results
        
        # Update summary
        self.update_summary(suite_results)
        
        return suite_results
    
    def parse_results(self, result, suite_name):
        """Parse test results from pytest output"""
        output_lines = result.stdout.split('\n')
        
        # Look for summary line
        summary = {
            'suite': suite_name,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'duration': 0
        }
        
        for line in output_lines:
            if 'passed' in line and 'failed' in line:
                # Parse summary line
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'passed' in part:
                        try:
                            summary['passed'] = int(parts[i-1])
                        except:
                            pass
                    elif 'failed' in part:
                        try:
                            summary['failed'] = int(parts[i-1])
                        except:
                            pass
                    elif 'error' in part:
                        try:
                            summary['errors'] = int(parts[i-1])
                        except:
                            pass
                    elif 'skipped' in part:
                        try:
                            summary['skipped'] = int(parts[i-1])
                        except:
                            pass
            elif 'in' in line and 's' in line and '=' in line:
                # Parse duration
                try:
                    duration_str = line.split('in')[1].strip().replace('s', '')
                    summary['duration'] = float(duration_str)
                except:
                    pass
        
        # Add raw output for debugging
        summary['output'] = result.stdout if result.returncode != 0 else ""
        summary['errors_output'] = result.stderr
        
        return summary
    
    def update_summary(self, suite_results):
        """Update overall summary with suite results"""
        self.results['summary']['passed'] += suite_results.get('passed', 0)
        self.results['summary']['failed'] += suite_results.get('failed', 0)
        self.results['summary']['errors'] += suite_results.get('errors', 0)
        self.results['summary']['skipped'] += suite_results.get('skipped', 0)
        
        total = (suite_results.get('passed', 0) + 
                suite_results.get('failed', 0) + 
                suite_results.get('errors', 0) + 
                suite_results.get('skipped', 0))
        self.results['summary']['total_tests'] += total
    
    def run_all_tests(self):
        """Run all test suites"""
        print("\n" + "="*60)
        print("Beverly Knits ERP v2 - Comprehensive Test Execution")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Define test suites
        test_suites = [
            ('Unit - Critical Paths', 'tests/unit/test_critical_paths.py'),
            ('Unit - Inventory Analyzer', 'tests/unit/test_inventory_analyzer.py'),
            ('Unit - Sales Forecasting', 'tests/unit/test_sales_forecasting_engine.py'),
            ('Unit - Capacity Planning', 'tests/unit/test_capacity_planning_engine.py'),
            ('Unit - Service Tests', 'tests/unit/test_inventory_service.py'),
            ('Integration - API Endpoints', 'tests/integration/test_api_endpoints.py'),
            ('Integration - Workflows', 'tests/integration/test_critical_workflows.py'),
            ('E2E - Critical Workflows', 'tests/e2e/test_critical_workflows.py'),
            ('Performance', 'tests/performance/'),
        ]
        
        # Run each test suite
        for suite_name, test_path in test_suites:
            if os.path.exists(test_path):
                self.run_test_suite(suite_name, test_path)
            else:
                print(f"⚠️  Skipping {suite_name}: Path not found")
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate final test report"""
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        
        # Overall statistics
        total = self.results['summary']['total_tests']
        passed = self.results['summary']['passed']
        failed = self.results['summary']['failed']
        errors = self.results['summary']['errors']
        skipped = self.results['summary']['skipped']
        
        if total > 0:
            pass_rate = (passed / total) * 100
        else:
            pass_rate = 0
        
        print(f"\n📊 Overall Results:")
        print(f"  Total Tests: {total}")
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  ⚠️  Errors: {errors}")
        print(f"  ⏭️  Skipped: {skipped}")
        print(f"  📈 Pass Rate: {pass_rate:.1f}%")
        
        # Suite-by-suite breakdown
        print(f"\n📋 Test Suite Breakdown:")
        for suite_name, results in self.results['test_suites'].items():
            status = "✅" if results['failed'] == 0 and results['errors'] == 0 else "❌"
            print(f"  {status} {suite_name}:")
            print(f"      Passed: {results['passed']}, Failed: {results['failed']}, "
                  f"Errors: {results['errors']}, Duration: {results.get('duration', 0):.2f}s")
        
        # Coverage report (if available)
        self.generate_coverage_report()
        
        # Save results to file
        self.save_results()
        
        # Exit code based on results
        if failed > 0 or errors > 0:
            print(f"\n❌ Tests FAILED - {failed} failures, {errors} errors")
            return 1
        else:
            print(f"\n✅ All tests PASSED!")
            return 0
    
    def generate_coverage_report(self):
        """Generate coverage report if available"""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'coverage', 'report'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"\n📊 Code Coverage Report:")
                # Parse coverage output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'TOTAL' in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            coverage = parts[-1]
                            print(f"  Overall Coverage: {coverage}")
                            break
        except:
            pass
    
    def save_results(self):
        """Save test results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'test_results_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main entry point"""
    runner = TestRunner()
    
    # Check if specific test suite requested
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        suite_name = os.path.basename(test_path)
        runner.run_test_suite(suite_name, test_path)
        return runner.generate_report()
    else:
        # Run all tests
        runner.run_all_tests()
        return runner.generate_report()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)