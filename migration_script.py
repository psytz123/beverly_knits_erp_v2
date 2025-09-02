#!/usr/bin/env python3
"""
Migration Script for Beverly Knits ERP Modularization
This script helps identify and replace duplicate code with existing modules
"""
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# File paths
MAIN_FILE = Path('/mnt/c/finalee/beverly_knits_erp_v2/src/core/beverly_comprehensive_erp.py')
BACKUP_FILE = Path('/mnt/c/finalee/beverly_knits_erp_v2/src/core/beverly_comprehensive_erp_backup.py')

# Mapping of duplicate classes to their modular replacements
CLASS_REPLACEMENTS = {
    'InventoryAnalyzer': {
        'lines': (675, 812),
        'replacement': 'from services.inventory_analyzer_service import InventoryAnalyzerService as InventoryAnalyzer',
        'module': 'services/inventory_analyzer_service.py',
        'description': 'Inventory analysis with safety stock calculations'
    },
    'InventoryManagementPipeline': {
        'lines': (813, 980),
        'replacement': 'from services.inventory_pipeline_service import InventoryManagementPipelineService as InventoryManagementPipeline',
        'module': 'services/inventory_pipeline_service.py',
        'description': 'Complete inventory management pipeline'
    },
    'SalesForecastingEngine': {
        'lines': (981, 2039),
        'replacement': 'from services.sales_forecasting_service import SalesForecastingService as SalesForecastingEngine',
        'module': 'services/sales_forecasting_service.py',
        'description': 'ML-powered sales forecasting'
    },
    'CapacityPlanningEngine': {
        'lines': (2040, 2473),
        'replacement': 'from services.capacity_planning_service import CapacityPlanningService as CapacityPlanningEngine',
        'module': 'services/capacity_planning_service.py',
        'description': 'Production capacity planning'
    },
    'YarnRequirementCalculator': {
        'lines': (1340, 1531),
        'replacement': 'from services.yarn_requirement_service import YarnRequirementService as YarnRequirementCalculator',
        'module': 'services/yarn_requirement_service.py',
        'description': 'Yarn requirement calculations'
    }
}

# API endpoints that should be moved to blueprints
API_BLUEPRINT_MAPPING = {
    'inventory': [
        '/api/inventory-analysis',
        '/api/inventory-intelligence-enhanced',
        '/api/inventory-netting',
        '/api/real-time-inventory',
        '/api/multi-stage-inventory',
        '/api/inventory-overview',
        '/api/inventory-analysis/complete',
        '/api/inventory-analysis/yarn-shortages',
        '/api/inventory-analysis/stock-risks',
        '/api/inventory-analysis/forecast-vs-stock',
        '/api/inventory-analysis/action-items',
        '/api/inventory-analysis/dashboard-data'
    ],
    'production': [
        '/api/production-planning',
        '/api/production-pipeline',
        '/api/production-suggestions',
        '/api/po-risk-analysis',
        '/api/fabric/yarn-requirements',
        '/api/execute-planning',
        '/api/planning-status',
        '/api/planning-phases'
    ],
    'forecasting': [
        '/api/ml-forecast-report',
        '/api/ml-forecast-detailed',
        '/api/ml-forecasting',
        '/api/sales-forecast-analysis',
        '/api/ml-validation-summary',
        '/api/backtest/fabric-comprehensive',
        '/api/backtest/yarn-comprehensive',
        '/api/backtest/full-report'
    ],
    'yarn': [
        '/api/yarn-intelligence',
        '/api/yarn-data',
        '/api/yarn-shortage-analysis',
        '/api/yarn-alternatives',
        '/api/yarn-substitution-intelligent',
        '/api/yarn-aggregation',
        '/api/yarn-forecast-shortages'
    ],
    'system': [
        '/api/health',
        '/api/debug-data',
        '/api/cache-stats',
        '/api/cache-clear',
        '/api/reload-data',
        '/api/consolidation-metrics'
    ]
}


def analyze_file():
    """Analyze the main file for modularization opportunities"""
    print("=" * 70)
    print(" MODULARIZATION ANALYSIS ")
    print("=" * 70)
    
    if not MAIN_FILE.exists():
        print(f"❌ Main file not found: {MAIN_FILE}")
        return False
    
    with open(MAIN_FILE, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"\n📊 File Statistics:")
    print(f"  Total lines: {total_lines:,}")
    print(f"  File size: {MAIN_FILE.stat().st_size / 1024:.1f} KB")
    
    # Calculate lines that can be removed
    removable_lines = 0
    for class_name, info in CLASS_REPLACEMENTS.items():
        start, end = info['lines']
        removable_lines += (end - start + 1)
    
    print(f"\n🔄 Modularization Potential:")
    print(f"  Lines that can be removed: {removable_lines:,}")
    print(f"  Percentage reduction: {removable_lines / total_lines * 100:.1f}%")
    print(f"  Final size estimate: {total_lines - removable_lines:,} lines")
    
    return True


def list_duplicate_classes():
    """List all duplicate classes that can be replaced"""
    print("\n" + "=" * 70)
    print(" DUPLICATE CLASSES TO REPLACE ")
    print("=" * 70)
    
    total_lines = 0
    for i, (class_name, info) in enumerate(CLASS_REPLACEMENTS.items(), 1):
        start, end = info['lines']
        lines_count = end - start + 1
        total_lines += lines_count
        
        print(f"\n{i}. {class_name}")
        print(f"   Lines: {start}-{end} ({lines_count} lines)")
        print(f"   Module: {info['module']}")
        print(f"   Purpose: {info['description']}")
    
    print(f"\n📊 Total duplicate lines: {total_lines}")
    print(f"   These can be replaced with simple imports!")


def list_api_endpoints():
    """List API endpoints that should be moved to blueprints"""
    print("\n" + "=" * 70)
    print(" API ENDPOINTS TO MIGRATE ")
    print("=" * 70)
    
    total_endpoints = 0
    for blueprint, endpoints in API_BLUEPRINT_MAPPING.items():
        print(f"\n📁 {blueprint.upper()} Blueprint ({len(endpoints)} endpoints):")
        for endpoint in endpoints[:5]:  # Show first 5
            print(f"   • {endpoint}")
        if len(endpoints) > 5:
            print(f"   ... and {len(endpoints) - 5} more")
        total_endpoints += len(endpoints)
    
    print(f"\n📊 Total endpoints to migrate: {total_endpoints}")


def generate_import_statements():
    """Generate the import statements needed"""
    print("\n" + "=" * 70)
    print(" IMPORT STATEMENTS TO ADD ")
    print("=" * 70)
    
    print("\n# Add these imports at the top of the file:")
    print("# (After line ~226 where other service imports are)")
    print()
    
    # Service imports
    print("# Service imports (to replace duplicate classes)")
    for class_name, info in CLASS_REPLACEMENTS.items():
        print(info['replacement'])
    
    print("\n# ServiceManager for orchestration")
    print("from services.service_manager import ServiceManager")
    
    print("\n# Data loader (already imported at line 90)")
    print("# from data_loaders.unified_data_loader import ConsolidatedDataLoader")
    
    print("\n# API Blueprints")
    print("from api.blueprints import (")
    print("    inventory_bp, production_bp, forecasting_bp,")
    print("    yarn_bp, planning_bp, system_bp")
    print(")")


def generate_initialization_code():
    """Generate initialization code for services"""
    print("\n" + "=" * 70)
    print(" INITIALIZATION CODE ")
    print("=" * 70)
    
    print("\n# Replace direct class instantiation with ServiceManager")
    print("# (Around line 3200 where classes are initialized)")
    print("""
# Initialize ServiceManager
service_config = {
    'data_path': data_path,
    'safety_stock_multiplier': 1.5,
    'lead_time_days': 30,
    'forecast_horizon': 90,
    'target_accuracy': 85.0
}

# Create service manager
service_manager = ServiceManager(service_config)

# Get services from manager
analyzer = service_manager.get_service('inventory')
forecasting_engine = service_manager.get_service('forecasting')
capacity_engine = service_manager.get_service('capacity')
pipeline = service_manager.get_service('pipeline')

# Data loader (keep existing)
data_loader = ConsolidatedDataLoader(data_path, max_workers=5)
""")


def generate_blueprint_registration():
    """Generate blueprint registration code"""
    print("\n" + "=" * 70)
    print(" BLUEPRINT REGISTRATION ")
    print("=" * 70)
    
    print("\n# Register all blueprints (after app initialization)")
    print("# Replace individual @app.route decorators with:")
    print("""
# Initialize blueprints with services
from api.blueprints import init_inventory_bp, init_production_bp, init_forecasting_bp

init_inventory_bp(analyzer, pipeline, data_loader)
init_production_bp(capacity_engine, pipeline, data_loader)
init_forecasting_bp(forecasting_engine, data_loader)

# Register blueprints
app.register_blueprint(inventory_bp, url_prefix='/api')
app.register_blueprint(production_bp, url_prefix='/api')
app.register_blueprint(forecasting_bp, url_prefix='/api')
app.register_blueprint(yarn_bp, url_prefix='/api')
app.register_blueprint(planning_bp, url_prefix='/api')
app.register_blueprint(system_bp, url_prefix='/api')
""")


def create_backup():
    """Create a backup of the main file"""
    print("\n" + "=" * 70)
    print(" CREATING BACKUP ")
    print("=" * 70)
    
    if MAIN_FILE.exists():
        import shutil
        shutil.copy2(MAIN_FILE, BACKUP_FILE)
        print(f"✅ Backup created: {BACKUP_FILE}")
        return True
    else:
        print(f"❌ Main file not found: {MAIN_FILE}")
        return False


def generate_rollback_script():
    """Generate a rollback script"""
    print("\n" + "=" * 70)
    print(" ROLLBACK PLAN ")
    print("=" * 70)
    
    print("\n# If anything goes wrong, rollback with:")
    print(f"cp {BACKUP_FILE} {MAIN_FILE}")
    print("pkill -f 'python3.*beverly'")
    print("python3 src/core/beverly_comprehensive_erp.py")
    
    print("\n# Or use feature flags (in config/feature_flags.py):")
    print("""
FEATURE_FLAGS = {
    'use_service_manager': False,  # Disable ServiceManager
    'use_blueprints': False,        # Disable blueprint routing
    'api_consolidation_enabled': True  # Keep API consolidation
}
""")


def generate_testing_commands():
    """Generate commands for testing"""
    print("\n" + "=" * 70)
    print(" TESTING COMMANDS ")
    print("=" * 70)
    
    print("\n# Test existing modules:")
    print("python3 test_existing_modules.py")
    
    print("\n# Test modular app (runs on port 5007):")
    print("python3 modular_app_example.py")
    
    print("\n# Run unit tests:")
    print("pytest tests/unit/test_inventory_analyzer.py -v")
    print("pytest tests/unit/test_sales_forecasting_engine.py -v")
    
    print("\n# Test API endpoints:")
    print("curl http://localhost:5006/api/inventory-analysis")
    print("curl http://localhost:5006/api/consolidation-metrics")
    
    print("\n# Check server status:")
    print("curl http://localhost:5006/api/debug-data | python3 -m json.tool")


def main():
    """Main migration script"""
    print("\n" + "=" * 80)
    print(" BEVERLY KNITS ERP - MODULARIZATION MIGRATION SCRIPT ")
    print("=" * 80)
    
    # Step 1: Analyze current state
    if not analyze_file():
        return 1
    
    # Step 2: Show what can be replaced
    list_duplicate_classes()
    list_api_endpoints()
    
    # Step 3: Generate code snippets
    generate_import_statements()
    generate_initialization_code()
    generate_blueprint_registration()
    
    # Step 4: Backup and rollback plan
    create_backup()
    generate_rollback_script()
    
    # Step 5: Testing commands
    generate_testing_commands()
    
    # Summary
    print("\n" + "=" * 80)
    print(" MIGRATION SUMMARY ")
    print("=" * 80)
    
    print("\n✅ Migration Plan Generated!")
    print("\n📋 Next Steps:")
    print("1. Review the changes above")
    print("2. Apply imports and initialization code")
    print("3. Remove duplicate class definitions")
    print("4. Test with modular_app_example.py")
    print("5. Run full test suite")
    
    print("\n⏱️ Estimated Time: 2-3 hours")
    print("📉 Code Reduction: ~3,000+ lines")
    print("✨ Result: Clean, modular, maintainable codebase")
    
    print("\n💡 Tip: Start with one class at a time and test after each change!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())