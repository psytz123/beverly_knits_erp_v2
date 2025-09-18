#!/usr/bin/env python3
"""
Find and report duplicate code between monolith and service modules
Identifies classes that exist in both the monolith and extracted services
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

def parse_file_classes(filepath: str) -> Dict[str, ast.ClassDef]:
    """Parse a Python file and extract all class definitions"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node
        return classes
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return {}

def find_duplicate_classes() -> Dict[str, List[str]]:
    """Find classes that exist in both monolith and services"""

    # Known mappings of duplicate classes
    known_duplicates = {
        'InventoryAnalyzer': 'services/inventory_analyzer_service.py',
        'SalesForecastingEngine': 'services/sales_forecasting_service.py',
        'CapacityPlanningEngine': 'services/capacity_planning_service.py',
        'InventoryManagementPipeline': 'services/inventory_pipeline_service.py',
        'YarnRequirementCalculator': 'services/yarn_requirement_service.py',
        'MultiStageInventoryTracker': 'services/inventory_pipeline_service.py',
        'ProductionScheduler': 'services/capacity_planning_service.py',
        'TimePhasedMRP': 'production/time_phased_planning.py',
        'ManufacturingSupplyChainAI': 'production/six_phase_planning_engine.py'
    }

    monolith_path = Path("src/core/beverly_comprehensive_erp.py")
    services_dir = Path("src/services")
    production_dir = Path("src/production")

    duplicates = {}

    if monolith_path.exists():
        monolith_classes = parse_file_classes(str(monolith_path))
        print(f"\nFound {len(monolith_classes)} classes in monolith:")
        for class_name in monolith_classes.keys():
            print(f"  - {class_name}")

        # Check for known duplicates
        print("\n" + "="*60)
        print("DUPLICATE CLASSES FOUND:")
        print("="*60)

        for class_name, service_file in known_duplicates.items():
            if class_name in monolith_classes:
                service_path = Path("src") / service_file
                if service_path.exists():
                    duplicates[class_name] = str(service_path)
                    print(f"\n✗ {class_name}")
                    print(f"  Monolith: src/core/beverly_comprehensive_erp.py")
                    print(f"  Service:  {service_path}")

    return duplicates

def find_unused_functions(filepath: str) -> Set[str]:
    """Find functions that are defined but never called"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)

        # Get all function definitions
        functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip special methods
                if not node.name.startswith('__'):
                    functions.add(node.name)

        # Find function calls
        called_functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_functions.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_functions.add(node.func.attr)

        # Also check for functions used in decorators/routes
        for line in content.split('\n'):
            if '@app.route' in line:
                # Extract function name from next line
                continue
            if 'def ' in line and any(func in line for func in functions):
                # Mark as used if it's a route handler
                for func in functions:
                    if func in line:
                        called_functions.add(func)

        # Find unused functions
        unused = functions - called_functions

        # Filter out Flask route handlers (they're called by Flask)
        route_handlers = set()
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '@app.route' in line and i+1 < len(lines):
                next_line = lines[i+1]
                if 'def ' in next_line:
                    func_name = next_line.split('def ')[1].split('(')[0]
                    route_handlers.add(func_name)

        unused = unused - route_handlers

        return unused
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return set()

def analyze_monolith_size():
    """Analyze the size and complexity of the monolith"""
    monolith_path = "src/core/beverly_comprehensive_erp.py"

    if not Path(monolith_path).exists():
        print(f"Monolith file not found: {monolith_path}")
        return

    with open(monolith_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Count different types of definitions
    classes = 0
    functions = 0
    routes = 0
    imports = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('class '):
            classes += 1
        elif stripped.startswith('def '):
            functions += 1
        elif '@app.route' in stripped:
            routes += 1
        elif stripped.startswith('import ') or stripped.startswith('from '):
            imports += 1

    print("\n" + "="*60)
    print("MONOLITH ANALYSIS:")
    print("="*60)
    print(f"Total lines: {len(lines):,}")
    print(f"Classes: {classes}")
    print(f"Functions: {functions}")
    print(f"Routes: {routes}")
    print(f"Imports: {imports}")

def generate_removal_script(duplicates: Dict[str, str]):
    """Generate a script to remove duplicate classes"""
    print("\n" + "="*60)
    print("RECOMMENDED ACTIONS:")
    print("="*60)

    print("\n1. Replace duplicate classes with imports:")
    print("-" * 40)

    print("# Add these imports at the top of beverly_comprehensive_erp.py:")
    print("from services.inventory_analyzer_service import InventoryAnalyzerService")
    print("from services.sales_forecasting_service import SalesForecastingService")
    print("from services.capacity_planning_service import CapacityPlanningService")
    print("from services.inventory_pipeline_service import InventoryManagementPipelineService")
    print("from services.yarn_requirement_service import YarnRequirementService")

    print("\n# Create backward-compatible aliases:")
    print("InventoryAnalyzer = InventoryAnalyzerService")
    print("SalesForecastingEngine = SalesForecastingService")
    print("CapacityPlanningEngine = CapacityPlanningService")

    print("\n2. Remove the duplicate class definitions from the monolith")
    print("-" * 40)
    for class_name in duplicates.keys():
        print(f"  - Remove class {class_name}")

    print("\n3. Test that everything still works:")
    print("-" * 40)
    print("  pytest tests/unit/ -v")
    print("  pytest tests/integration/ -v")

def main():
    """Main execution"""
    print("="*60)
    print("DUPLICATE CODE ANALYSIS")
    print("="*60)

    # Analyze monolith size
    analyze_monolith_size()

    # Find duplicate classes
    duplicates = find_duplicate_classes()

    # Find unused functions
    print("\n" + "="*60)
    print("POTENTIALLY UNUSED FUNCTIONS:")
    print("="*60)

    monolith_path = "src/core/beverly_comprehensive_erp.py"
    unused = find_unused_functions(monolith_path)

    if unused:
        print(f"\nFound {len(unused)} potentially unused functions:")
        for func in sorted(unused):
            if func not in ['clean_html_from_string', 'find_column', 'find_column_value']:
                print(f"  - {func}")
    else:
        print("No obviously unused functions found")

    # Generate recommendations
    if duplicates:
        generate_removal_script(duplicates)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nTotal duplicate classes to remove: {len(duplicates)}")
    print(f"Estimated lines to be removed: ~{len(duplicates) * 200}")
    print(f"This could reduce the monolith by approximately {len(duplicates) * 200 / 18662 * 100:.1f}%")

if __name__ == "__main__":
    main()