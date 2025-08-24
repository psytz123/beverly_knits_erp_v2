#!/usr/bin/env python3
"""
Complete Test Execution Suite for Beverly Knits ERP v2
Runs all tests and generates comprehensive coverage report
"""

import subprocess
import sys
import os
from datetime import datetime
import json
from pathlib import Path
import time

class CompleteTestExecutor:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.results = {
            'execution_date': datetime.now().isoformat(),
            'test_categories': {},
            'overall_stats': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0,
                'duration': 0
            }
        }
        
        # Set Python path
        os.environ['PYTHONPATH'] = str(self.base_path / 'src')
    
    def run_all_tests(self):
        """Execute all test categories"""
        print("="*80)
        print("BEVERLY KNITS ERP v2 - COMPLETE TEST EXECUTION")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        start_time = time.time()
        
        # Define test categories
        test_categories = [
            {
                'name': 'Unit Tests - Core Business Logic',
                'path': 'tests/unit/test_critical_paths.py',
                'expected_tests': 17,
                'importance': 'CRITICAL'
            },
            {
                'name': 'Unit Tests - Inventory Analyzer',
                'path': 'tests/unit/test_inventory_analyzer.py',
                'expected_tests': 15,
                'importance': 'HIGH'
            },
            {
                'name': 'Unit Tests - Sales Forecasting',
                'path': 'tests/unit/test_sales_forecasting_engine.py',
                'expected_tests': 19,
                'importance': 'HIGH'
            },
            {
                'name': 'Unit Tests - Capacity Planning',
                'path': 'tests/unit/test_capacity_planning_engine.py',
                'expected_tests': 15,
                'importance': 'HIGH'
            },
            {
                'name': 'Integration Tests - API Endpoints',
                'path': 'tests/integration/test_api_endpoints_comprehensive.py',
                'expected_tests': 30,
                'importance': 'CRITICAL'
            },
            {
                'name': 'Performance Tests',
                'path': 'tests/performance/test_load_and_performance.py',
                'expected_tests': 15,
                'importance': 'MEDIUM'
            },
            {
                'name': 'Integration Tests - Critical Workflows',
                'path': 'tests/integration/test_critical_workflows.py',
                'expected_tests': 20,
                'importance': 'HIGH'
            }
        ]
        
        # Run each category
        for category in test_categories:
            self.run_test_category(category)
        
        # Calculate total duration
        self.results['overall_stats']['duration'] = time.time() - start_time
        
        # Generate final report
        self.generate_final_report()
    
    def run_test_category(self, category):
        """Run a specific test category"""
        print(f"\n🧪 Testing: {category['name']}")
        print(f"   Priority: {category['importance']}")
        print("-"*60)
        
        test_path = self.base_path / category['path']
        
        if not test_path.exists():
            print(f"   ⚠️  Test file not found, creating placeholder results")
            self.results['test_categories'][category['name']] = {
                'status': 'NOT_FOUND',
                'importance': category['importance'],
                'expected': category['expected_tests'],
                'executed': 0
            }
            return
        
        # Run pytest
        cmd = [
            sys.executable, '-m', 'pytest',
            str(test_path),
            '--tb=short',
            '--quiet',
            '-v'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Parse results
            self.parse_results(category['name'], result.stdout, category)
            
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  Timeout after 120 seconds")
            self.results['test_categories'][category['name']] = {
                'status': 'TIMEOUT',
                'importance': category['importance']
            }
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results['test_categories'][category['name']] = {
                'status': 'ERROR',
                'importance': category['importance'],
                'error': str(e)
            }
    
    def parse_results(self, name, output, category):
        """Parse pytest output"""
        lines = output.split('\n')
        
        passed = failed = errors = skipped = 0
        
        for line in lines:
            if 'passed' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'passed' in part:
                        try:
                            passed = int(parts[i-1])
                        except:
                            pass
            if 'failed' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'failed' in part:
                        try:
                            failed = int(parts[i-1])
                        except:
                            pass
            if 'error' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'error' in part.lower():
                        try:
                            errors = int(parts[i-1])
                        except:
                            pass
        
        # Update totals
        self.results['overall_stats']['passed'] += passed
        self.results['overall_stats']['failed'] += failed
        self.results['overall_stats']['errors'] += errors
        self.results['overall_stats']['skipped'] += skipped
        self.results['overall_stats']['total_tests'] += (passed + failed + errors + skipped)
        
        # Store category results
        total = passed + failed + errors + skipped
        status = 'PASS' if failed == 0 and errors == 0 else 'FAIL'
        
        self.results['test_categories'][name] = {
            'status': status,
            'importance': category['importance'],
            'expected': category['expected_tests'],
            'executed': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'pass_rate': (passed / total * 100) if total > 0 else 0
        }
        
        # Print summary
        if total > 0:
            icon = "✅" if status == 'PASS' else "❌"
            print(f"   {icon} Executed: {total}/{category['expected_tests']} tests")
            print(f"   📊 Results: {passed} passed, {failed} failed, {errors} errors")
            print(f"   📈 Pass Rate: {self.results['test_categories'][name]['pass_rate']:.1f}%")
        else:
            print(f"   ⚠️  No tests executed")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("FINAL TEST EXECUTION REPORT")
        print("="*80)
        
        # Overall statistics
        total = self.results['overall_stats']['total_tests']
        passed = self.results['overall_stats']['passed']
        failed = self.results['overall_stats']['failed']
        errors = self.results['overall_stats']['errors']
        
        overall_pass_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Tests: {total}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⚠️  Errors: {errors}")
        print(f"   📈 Overall Pass Rate: {overall_pass_rate:.1f}%")
        print(f"   ⏱️  Total Duration: {self.results['overall_stats']['duration']:.2f}s")
        
        # Category breakdown
        print(f"\n📋 CATEGORY BREAKDOWN:")
        
        for category_name, results in self.results['test_categories'].items():
            if results['status'] == 'NOT_FOUND':
                print(f"\n   ⚠️  {category_name}")
                print(f"      Status: NOT FOUND")
            else:
                icon = "✅" if results['status'] == 'PASS' else "❌"
                print(f"\n   {icon} {category_name}")
                print(f"      Importance: {results['importance']}")
                print(f"      Coverage: {results.get('executed', 0)}/{results.get('expected', 0)} tests")
                if 'pass_rate' in results:
                    print(f"      Pass Rate: {results['pass_rate']:.1f}%")
        
        # Critical test analysis
        print(f"\n🎯 CRITICAL TEST ANALYSIS:")
        critical_tests = [
            (name, res) for name, res in self.results['test_categories'].items()
            if res.get('importance') == 'CRITICAL'
        ]
        
        critical_passed = sum(1 for _, r in critical_tests if r.get('status') == 'PASS')
        critical_total = len(critical_tests)
        
        print(f"   Critical Test Suites: {critical_passed}/{critical_total} passing")
        
        if critical_passed == critical_total:
            print(f"   ✅ All critical tests passing!")
        else:
            print(f"   ❌ Critical test failures detected!")
        
        # Coverage estimation
        print(f"\n📊 COVERAGE ESTIMATION:")
        
        expected_total = sum(
            cat.get('expected', 0) 
            for cat in self.results['test_categories'].values()
        )
        
        coverage_pct = (total / expected_total * 100) if expected_total > 0 else 0
        
        print(f"   Expected Tests: {expected_total}")
        print(f"   Executed Tests: {total}")
        print(f"   Coverage: {coverage_pct:.1f}%")
        print(f"   Target Coverage: 80%")
        print(f"   Gap to Target: {max(0, 80 - coverage_pct):.1f}%")
        
        # Quality gates
        print(f"\n🚦 QUALITY GATES:")
        
        gates = {
            'Overall Pass Rate >= 95%': overall_pass_rate >= 95,
            'Critical Tests Pass': critical_passed == critical_total,
            'Coverage >= 80%': coverage_pct >= 80,
            'No Errors': errors == 0
        }
        
        for gate, passed in gates.items():
            icon = "✅" if passed else "❌"
            print(f"   {icon} {gate}")
        
        all_gates_passed = all(gates.values())
        
        # Final verdict
        print(f"\n🏁 FINAL VERDICT:")
        
        if all_gates_passed:
            print(f"   ✅ ALL QUALITY GATES PASSED - System ready for production!")
        elif overall_pass_rate >= 80 and critical_passed == critical_total:
            print(f"   🟡 CONDITIONAL PASS - Minor issues to address")
        else:
            print(f"   ❌ QUALITY GATES FAILED - Significant issues need resolution")
        
        # Recommendations
        print(f"\n📝 RECOMMENDATIONS:")
        
        if failed > 0 or errors > 0:
            print(f"   1. Fix {failed} failing tests and {errors} test errors")
        
        if coverage_pct < 80:
            print(f"   2. Increase test coverage by {80 - coverage_pct:.1f}%")
        
        if critical_passed < critical_total:
            print(f"   3. Priority: Fix critical test failures immediately")
        
        print(f"   4. Consider adding:")
        print(f"      - Security testing suite")
        print(f"      - UI/Dashboard testing")
        print(f"      - Database migration tests")
        print(f"      - Stress testing scenarios")
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("END OF TEST EXECUTION REPORT")
        print("="*80)
    
    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'test_execution_report_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main entry point"""
    executor = CompleteTestExecutor()
    executor.run_all_tests()

if __name__ == "__main__":
    main()