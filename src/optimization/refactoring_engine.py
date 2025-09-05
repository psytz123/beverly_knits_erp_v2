"""
Beverly Knits ERP - Automated Refactoring Engine
Intelligently refactors monolithic code into modular services
"""

import ast
import os
import re
import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class FunctionComplexity:
    """Function complexity metrics"""
    name: str
    file_path: str
    line_number: int
    line_count: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    parameters: int
    dependencies: List[str]
    calls_to: List[str]
    called_by: List[str]
    complexity_grade: str  # A-F grade

@dataclass
class RefactoringCandidate:
    """Candidate for refactoring"""
    type: str  # 'function', 'class', 'module'
    name: str
    file_path: str
    line_range: Tuple[int, int]
    complexity: int
    priority: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    suggested_actions: List[str]
    estimated_effort: str  # 'hours', 'days', 'weeks'

class CodeAnalyzer(ast.NodeVisitor):
    """AST-based code analyzer for complexity metrics"""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.current_class = None
        self.complexity_stack = []
        
    def visit_FunctionDef(self, node):
        """Analyze function definitions"""
        complexity = self._calculate_cyclomatic_complexity(node)
        cognitive = self._calculate_cognitive_complexity(node)
        
        func_info = {
            'name': node.name,
            'line_number': node.lineno,
            'end_line': node.end_lineno or node.lineno,
            'line_count': (node.end_lineno or node.lineno) - node.lineno + 1,
            'cyclomatic_complexity': complexity,
            'cognitive_complexity': cognitive,
            'parameters': len(node.args.args),
            'class': self.current_class,
            'decorators': [d.id if isinstance(d, ast.Name) else None for d in node.decorator_list],
            'docstring': ast.get_docstring(node) is not None
        }
        
        self.functions.append(func_info)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Analyze class definitions"""
        self.current_class = node.name
        class_info = {
            'name': node.name,
            'line_number': node.lineno,
            'end_line': node.end_lineno or node.lineno,
            'line_count': (node.end_lineno or node.lineno) - node.lineno + 1,
            'methods': [],
            'attributes': [],
            'base_classes': [base.id if isinstance(base, ast.Name) else None for base in node.bases]
        }
        
        self.classes.append(class_info)
        self.generic_visit(node)
        self.current_class = None
    
    def visit_Import(self, node):
        """Track imports"""
        for alias in node.names:
            self.imports.append({
                'module': alias.name,
                'alias': alias.asname,
                'line': node.lineno
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Track from imports"""
        for alias in node.names:
            self.imports.append({
                'module': f"{node.module}.{alias.name}" if node.module else alias.name,
                'alias': alias.asname,
                'line': node.lineno,
                'from_import': True
            })
        self.generic_visit(node)
    
    def _calculate_cyclomatic_complexity(self, node):
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, node):
        """Calculate cognitive complexity (mental effort to understand)"""
        cognitive = 0
        nesting_level = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                cognitive += 1 + nesting_level
                nesting_level += 1
            elif isinstance(child, ast.Break):
                cognitive += 1
            elif isinstance(child, ast.Continue):
                cognitive += 1
            elif isinstance(child, ast.BoolOp):
                cognitive += 1
            elif isinstance(child, ast.Lambda):
                cognitive += 1
        
        return cognitive

class RefactoringEngine:
    """Automated refactoring engine for Beverly Knits ERP"""
    
    # Complexity thresholds
    COMPLEXITY_THRESHOLDS = {
        'cyclomatic': {
            'A': 5,
            'B': 10,
            'C': 20,
            'D': 30,
            'E': 50,
            'F': float('inf')
        },
        'lines': {
            'A': 20,
            'B': 50,
            'C': 100,
            'D': 200,
            'E': 500,
            'F': float('inf')
        }
    }
    
    def __init__(self, project_root: str):
        """
        Initialize refactoring engine
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.analysis_results = {}
        self.refactoring_candidates = []
        self.service_extraction_plan = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Python file for refactoring opportunities
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Analysis results
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read())
                analyzer = CodeAnalyzer()
                analyzer.visit(tree)
                
                results = {
                    'file_path': file_path,
                    'line_count': sum(1 for _ in open(file_path, 'r', encoding='utf-8')),
                    'functions': analyzer.functions,
                    'classes': analyzer.classes,
                    'imports': analyzer.imports,
                    'complexity_summary': self._calculate_file_complexity(analyzer)
                }
                
                self.analysis_results[file_path] = results
                return results
                
            except SyntaxError as e:
                logger.error(f"Syntax error in {file_path}: {e}")
                return {'error': str(e)}
    
    def _calculate_file_complexity(self, analyzer: CodeAnalyzer) -> Dict[str, Any]:
        """Calculate overall file complexity"""
        if not analyzer.functions:
            return {'average_complexity': 0, 'max_complexity': 0, 'grade': 'A'}
        
        complexities = [f['cyclomatic_complexity'] for f in analyzer.functions]
        avg_complexity = sum(complexities) / len(complexities)
        max_complexity = max(complexities)
        
        # Determine grade
        grade = 'F'
        for level, threshold in self.COMPLEXITY_THRESHOLDS['cyclomatic'].items():
            if max_complexity < threshold:
                grade = level
                break
        
        return {
            'average_complexity': avg_complexity,
            'max_complexity': max_complexity,
            'total_functions': len(analyzer.functions),
            'complex_functions': sum(1 for c in complexities if c > 10),
            'very_complex_functions': sum(1 for c in complexities if c > 20),
            'grade': grade
        }
    
    def identify_refactoring_candidates(self) -> List[RefactoringCandidate]:
        """Identify all refactoring candidates in analyzed files"""
        candidates = []
        
        for file_path, analysis in self.analysis_results.items():
            if 'error' in analysis:
                continue
            
            # Check functions
            for func in analysis.get('functions', []):
                if func['cyclomatic_complexity'] > 10:
                    priority = self._determine_priority(func['cyclomatic_complexity'])
                    
                    candidate = RefactoringCandidate(
                        type='function',
                        name=func['name'],
                        file_path=file_path,
                        line_range=(func['line_number'], func['end_line']),
                        complexity=func['cyclomatic_complexity'],
                        priority=priority,
                        suggested_actions=self._suggest_function_refactoring(func),
                        estimated_effort=self._estimate_effort(func)
                    )
                    candidates.append(candidate)
            
            # Check classes
            for cls in analysis.get('classes', []):
                if cls['line_count'] > 200:
                    candidate = RefactoringCandidate(
                        type='class',
                        name=cls['name'],
                        file_path=file_path,
                        line_range=(cls['line_number'], cls['end_line']),
                        complexity=cls['line_count'],
                        priority='HIGH' if cls['line_count'] > 500 else 'MEDIUM',
                        suggested_actions=['Extract service', 'Split responsibilities'],
                        estimated_effort='days' if cls['line_count'] > 500 else 'hours'
                    )
                    candidates.append(candidate)
            
            # Check file size
            if analysis['line_count'] > 1000:
                candidate = RefactoringCandidate(
                    type='module',
                    name=Path(file_path).name,
                    file_path=file_path,
                    line_range=(1, analysis['line_count']),
                    complexity=analysis['line_count'],
                    priority='CRITICAL' if analysis['line_count'] > 5000 else 'HIGH',
                    suggested_actions=['Break into multiple modules', 'Extract services'],
                    estimated_effort='weeks' if analysis['line_count'] > 5000 else 'days'
                )
                candidates.append(candidate)
        
        self.refactoring_candidates = sorted(
            candidates, 
            key=lambda x: (x.priority == 'CRITICAL', x.priority == 'HIGH', x.complexity),
            reverse=True
        )
        
        return self.refactoring_candidates
    
    def _determine_priority(self, complexity: int) -> str:
        """Determine refactoring priority based on complexity"""
        if complexity > 50:
            return 'CRITICAL'
        elif complexity > 30:
            return 'HIGH'
        elif complexity > 20:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _suggest_function_refactoring(self, func: Dict[str, Any]) -> List[str]:
        """Suggest refactoring actions for a function"""
        suggestions = []
        
        if func['cyclomatic_complexity'] > 50:
            suggestions.append('Break into multiple functions')
            suggestions.append('Extract complex conditionals')
            suggestions.append('Consider Strategy pattern')
        elif func['cyclomatic_complexity'] > 30:
            suggestions.append('Extract helper functions')
            suggestions.append('Simplify conditional logic')
        elif func['cyclomatic_complexity'] > 20:
            suggestions.append('Extract validation logic')
            suggestions.append('Reduce nesting levels')
        
        if func['parameters'] > 5:
            suggestions.append('Use parameter object pattern')
        
        if func['line_count'] > 50:
            suggestions.append('Split into smaller functions')
        
        if not func.get('docstring'):
            suggestions.append('Add documentation')
        
        return suggestions
    
    def _estimate_effort(self, func: Dict[str, Any]) -> str:
        """Estimate refactoring effort"""
        if func['cyclomatic_complexity'] > 50 or func['line_count'] > 200:
            return 'days'
        elif func['cyclomatic_complexity'] > 30 or func['line_count'] > 100:
            return 'hours'
        else:
            return 'minutes'
    
    def generate_service_extraction_plan(self, file_path: str) -> Dict[str, Any]:
        """
        Generate plan to extract services from monolithic file
        
        Args:
            file_path: Path to monolithic file
            
        Returns:
            Service extraction plan
        """
        if file_path not in self.analysis_results:
            self.analyze_file(file_path)
        
        analysis = self.analysis_results.get(file_path, {})
        if 'error' in analysis:
            return {'error': analysis['error']}
        
        # Group related functionality
        services = self._identify_service_boundaries(analysis)
        
        plan = {
            'source_file': file_path,
            'total_lines': analysis['line_count'],
            'services': services,
            'migration_steps': self._generate_migration_steps(services),
            'estimated_effort': self._estimate_total_effort(services),
            'benefits': [
                'Improved maintainability',
                'Better testability',
                'Scalable architecture',
                'Clear separation of concerns'
            ]
        }
        
        self.service_extraction_plan = plan
        return plan
    
    def _identify_service_boundaries(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify logical service boundaries in code"""
        services = []
        
        # Group functions by patterns in their names
        function_groups = defaultdict(list)
        
        for func in analysis.get('functions', []):
            # Extract service hint from function name
            name = func['name']
            
            # Common patterns
            if 'inventory' in name.lower() or 'yarn' in name.lower():
                function_groups['inventory_service'].append(func)
            elif 'forecast' in name.lower() or 'ml' in name.lower():
                function_groups['forecasting_service'].append(func)
            elif 'production' in name.lower() or 'planning' in name.lower():
                function_groups['production_service'].append(func)
            elif 'api' in name.lower() or 'route' in name.lower():
                function_groups['api_layer'].append(func)
            elif 'cache' in name.lower() or 'optimize' in name.lower():
                function_groups['optimization_service'].append(func)
            else:
                function_groups['core_service'].append(func)
        
        # Create service definitions
        for service_name, functions in function_groups.items():
            if functions:
                services.append({
                    'name': service_name,
                    'functions': [f['name'] for f in functions],
                    'function_count': len(functions),
                    'total_complexity': sum(f['cyclomatic_complexity'] for f in functions),
                    'suggested_path': f"src/services/{service_name}.py"
                })
        
        # Group classes similarly
        for cls in analysis.get('classes', []):
            name = cls['name'].lower()
            
            if 'inventory' in name:
                services.append({
                    'name': f"{cls['name']}_service",
                    'class': cls['name'],
                    'line_count': cls['line_count'],
                    'suggested_path': f"src/services/{cls['name'].lower()}_service.py"
                })
        
        return services
    
    def _generate_migration_steps(self, services: List[Dict[str, Any]]) -> List[str]:
        """Generate step-by-step migration plan"""
        steps = [
            "1. Create service directory structure (src/services/)",
            "2. Extract data models to src/models/",
            "3. Create base service classes",
        ]
        
        for i, service in enumerate(services, start=4):
            steps.append(f"{i}. Extract {service['name']} to {service['suggested_path']}")
        
        steps.extend([
            f"{len(services) + 4}. Update imports in main file",
            f"{len(services) + 5}. Create service registry",
            f"{len(services) + 6}. Implement dependency injection",
            f"{len(services) + 7}. Add comprehensive tests",
            f"{len(services) + 8}. Validate functionality",
            f"{len(services) + 9}. Remove extracted code from monolith"
        ])
        
        return steps
    
    def _estimate_total_effort(self, services: List[Dict[str, Any]]) -> str:
        """Estimate total refactoring effort"""
        if len(services) > 10:
            return "2-3 weeks"
        elif len(services) > 5:
            return "1-2 weeks"
        else:
            return "3-5 days"
    
    def generate_refactoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive refactoring report"""
        report = {
            'project_root': str(self.project_root),
            'files_analyzed': len(self.analysis_results),
            'total_refactoring_candidates': len(self.refactoring_candidates),
            'critical_issues': sum(1 for c in self.refactoring_candidates if c.priority == 'CRITICAL'),
            'high_priority_issues': sum(1 for c in self.refactoring_candidates if c.priority == 'HIGH'),
            'candidates': [asdict(c) for c in self.refactoring_candidates[:20]],  # Top 20
            'service_extraction_plan': self.service_extraction_plan,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check for critical issues
        critical = [c for c in self.refactoring_candidates if c.priority == 'CRITICAL']
        if critical:
            recommendations.append(f"URGENT: Address {len(critical)} critical complexity issues immediately")
        
        # Check for monolithic files
        large_files = [a for a in self.analysis_results.values() if a.get('line_count', 0) > 1000]
        if large_files:
            recommendations.append(f"Break down {len(large_files)} monolithic files into services")
        
        # General recommendations
        recommendations.extend([
            "Implement automated code quality checks in CI/CD",
            "Set complexity thresholds for new code",
            "Establish refactoring sprints every quarter",
            "Create coding standards documentation",
            "Implement comprehensive test coverage before refactoring"
        ])
        
        return recommendations
    
    def export_refactoring_plan(self, output_path: str):
        """Export refactoring plan to JSON file"""
        report = self.generate_refactoring_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Refactoring plan exported to {output_path}")
        return output_path


def analyze_beverly_knits_erp():
    """Analyze the Beverly Knits ERP codebase"""
    project_root = "D:\\AI\\Workspaces\\efab.ai\\beverly_knits_erp_v2"
    engine = RefactoringEngine(project_root)
    
    # Analyze main monolithic file
    main_file = os.path.join(project_root, "src", "core", "beverly_comprehensive_erp.py")
    
    print("Analyzing Beverly Knits ERP codebase...")
    print(f"Main file: {main_file}")
    
    # Analyze the file
    results = engine.analyze_file(main_file)
    
    if 'error' not in results:
        print(f"\nFile Analysis:")
        print(f"  Lines of code: {results['line_count']}")
        print(f"  Functions: {len(results['functions'])}")
        print(f"  Classes: {len(results['classes'])}")
        print(f"  Complexity Grade: {results['complexity_summary']['grade']}")
        
        # Identify refactoring candidates
        candidates = engine.identify_refactoring_candidates()
        print(f"\nRefactoring Candidates: {len(candidates)}")
        
        # Show top critical issues
        critical = [c for c in candidates if c.priority == 'CRITICAL']
        if critical:
            print(f"\nCRITICAL Issues ({len(critical)}):")
            for c in critical[:5]:
                print(f"  - {c.name}: Complexity {c.complexity} at line {c.line_range[0]}")
                print(f"    Actions: {', '.join(c.suggested_actions[:2])}")
        
        # Generate service extraction plan
        plan = engine.generate_service_extraction_plan(main_file)
        if 'services' in plan:
            print(f"\nService Extraction Plan:")
            print(f"  Proposed services: {len(plan['services'])}")
            for service in plan['services'][:5]:
                print(f"    - {service['name']}: {service.get('function_count', 0)} functions")
        
        # Export report
        output_file = os.path.join(project_root, "refactoring_report.json")
        engine.export_refactoring_plan(output_file)
        print(f"\nRefactoring report saved to: {output_file}")
    else:
        print(f"Error analyzing file: {results['error']}")
    
    return engine


if __name__ == "__main__":
    analyze_beverly_knits_erp()