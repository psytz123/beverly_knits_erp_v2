#!/usr/bin/env python3
"""
Apply Day 0 Emergency Fixes to Beverly Knits ERP
This script patches the main ERP file to use the emergency fixes
Created: 2025-09-02
"""

import sys
import os
from pathlib import Path
import shutil
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

def create_backup(file_path):
    """Create timestamped backup of file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = Path(file_path).parent / f"{Path(file_path).stem}_backup_{timestamp}{Path(file_path).suffix}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def apply_imports_patch():
    """Add Day 0 emergency fixes imports to the main ERP file"""
    erp_file = Path('/mnt/c/finalee/beverly_knits_erp_v2/src/core/beverly_comprehensive_erp.py')
    
    if not erp_file.exists():
        print(f"ERROR: ERP file not found: {erp_file}")
        return False
    
    # Read the file
    with open(erp_file, 'r') as f:
        lines = f.readlines()
    
    # Find the import section (after initial comments/docstrings)
    import_index = None
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_index = i
            break
    
    if import_index is None:
        print("ERROR: Could not find import section")
        return False
    
    # Check if already patched
    for line in lines:
        if 'day0_emergency_fixes' in line:
            print("File already patched with Day 0 fixes")
            return True
    
    # Add Day 0 emergency fixes import
    day0_imports = [
        "\n# Day 0 Emergency Fixes - Added 2025-09-02\n",
        "try:\n",
        "    from scripts.day0_emergency_fixes import (\n",
        "        DynamicPathResolver,\n",
        "        ColumnAliasSystem,\n",
        "        PriceStringParser,\n",
        "        RealKPICalculator,\n",
        "        MultiLevelBOMNetting,\n",
        "        EmergencyFixManager\n",
        "    )\n",
        "    DAY0_FIXES_AVAILABLE = True\n",
        "    print('[DAY0] Emergency fixes loaded successfully')\n",
        "except ImportError as e:\n",
        "    print(f'[DAY0] Emergency fixes not available: {e}')\n",
        "    DAY0_FIXES_AVAILABLE = False\n",
        "\n"
    ]
    
    # Insert the imports after the first import statement
    lines[import_index:import_index] = day0_imports
    
    # Write back
    with open(erp_file, 'w') as f:
        f.writelines(lines)
    
    print(f"✓ Added Day 0 imports to {erp_file}")
    return True

def apply_path_resolution_patch():
    """Patch the load_all_data method to use dynamic path resolution"""
    erp_file = Path('/mnt/c/finalee/beverly_knits_erp_v2/src/core/beverly_comprehensive_erp.py')
    
    with open(erp_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'DynamicPathResolver()' in content:
        print("Path resolution already patched")
        return True
    
    # Find and patch the load_all_data method
    patch = """
            # Use Day 0 Dynamic Path Resolution if available
            if DAY0_FIXES_AVAILABLE:
                try:
                    path_resolver = DynamicPathResolver()
                    
                    # Resolve all data files dynamically
                    yarn_file = path_resolver.resolve_file('yarn_inventory')
                    if yarn_file:
                        self.raw_materials_data = pd.read_excel(yarn_file) if yarn_file.suffix == '.xlsx' else pd.read_csv(yarn_file)
                        print(f"[DAY0] Loaded yarn inventory from: {yarn_file}")
                    
                    bom_file = path_resolver.resolve_file('bom')
                    if bom_file:
                        self.bom_data = pd.read_csv(bom_file)
                        print(f"[DAY0] Loaded BOM from: {bom_file}")
                    
                    sales_file = path_resolver.resolve_file('sales')
                    if sales_file:
                        self.sales_data = pd.read_csv(sales_file)
                        print(f"[DAY0] Loaded sales from: {sales_file}")
                    
                    knit_orders_file = path_resolver.resolve_file('knit_orders')
                    if knit_orders_file:
                        self.knit_orders = pd.read_excel(knit_orders_file) if knit_orders_file.suffix == '.xlsx' else pd.read_csv(knit_orders_file)
                        print(f"[DAY0] Loaded knit orders from: {knit_orders_file}")
                        
                except Exception as e:
                    print(f"[DAY0] Path resolution failed, using fallback: {e}")
"""
    
    # Find the load_all_data method
    if 'def load_all_data(self):' in content:
        # Insert the patch after the method definition
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'def load_all_data(self):' in line:
                # Find the try block
                for j in range(i, min(i+10, len(lines))):
                    if 'try:' in lines[j]:
                        # Insert after the try
                        lines.insert(j+1, patch)
                        content = '\n'.join(lines)
                        break
                break
        
        with open(erp_file, 'w') as f:
            f.write(content)
        
        print("✓ Applied path resolution patch")
        return True
    
    print("WARNING: Could not find load_all_data method")
    return False

def apply_column_alias_patch():
    """Add column alias system to data loading"""
    erp_file = Path('/mnt/c/finalee/beverly_knits_erp_v2/src/core/beverly_comprehensive_erp.py')
    
    with open(erp_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'ColumnAliasSystem()' in content:
        print("Column alias system already patched")
        return True
    
    patch = """
                # Apply Day 0 Column Alias System if available
                if DAY0_FIXES_AVAILABLE:
                    try:
                        column_system = ColumnAliasSystem()
                        
                        if hasattr(self, 'raw_materials_data') and self.raw_materials_data is not None:
                            self.raw_materials_data = column_system.standardize_dataframe(self.raw_materials_data)
                            print(f"[DAY0] Standardized {len(column_system.applied_mappings)} columns in yarn inventory")
                        
                        if hasattr(self, 'bom_data') and self.bom_data is not None:
                            self.bom_data = column_system.standardize_dataframe(self.bom_data)
                            print(f"[DAY0] Standardized BOM columns")
                            
                        if hasattr(self, 'sales_data') and self.sales_data is not None:
                            self.sales_data = column_system.standardize_dataframe(self.sales_data)
                            print(f"[DAY0] Standardized sales columns")
                            
                    except Exception as e:
                        print(f"[DAY0] Column standardization failed: {e}")
"""
    
    # Find where to insert (after data loading)
    if 'print(f"Loaded BOM data:' in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'print(f"Loaded BOM data:' in line:
                # Insert after the next few lines
                for j in range(i, min(i+20, len(lines))):
                    if 'print(' in lines[j] and 'BOM' in lines[j]:
                        lines.insert(j+1, patch)
                        content = '\n'.join(lines)
                        break
                break
        
        with open(erp_file, 'w') as f:
            f.write(content)
        
        print("✓ Applied column alias patch")
        return True
    
    print("WARNING: Could not find appropriate location for column alias patch")
    return False

def apply_kpi_calculation_patch():
    """Patch KPI calculations to use real data"""
    erp_file = Path('/mnt/c/finalee/beverly_knits_erp_v2/src/core/beverly_comprehensive_erp.py')
    
    with open(erp_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'RealKPICalculator()' in content:
        print("KPI calculations already patched")
        return True
    
    # Find the comprehensive-kpis endpoint
    if '@app.route(\'/api/comprehensive-kpis\')' in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '@app.route(\'/api/comprehensive-kpis\')' in line:
                # Find the function definition
                for j in range(i, min(i+5, len(lines))):
                    if 'def ' in lines[j]:
                        # Insert patch at beginning of function
                        patch = """
    # Use Day 0 Real KPI Calculator if available
    if DAY0_FIXES_AVAILABLE:
        try:
            kpi_calc = RealKPICalculator()
            
            # Load all data sources
            kpi_calc.load_data({
                'yarn_inventory': inventory_analyzer.raw_materials_data if inventory_analyzer else None,
                'bom': inventory_analyzer.bom_data if inventory_analyzer else None,
                'sales': inventory_analyzer.sales_data if inventory_analyzer else None,
                'knit_orders': inventory_analyzer.knit_orders if inventory_analyzer else None
            })
            
            # Calculate real KPIs
            real_kpis = kpi_calc.calculate_all_kpis()
            
            # Merge with existing response
            if real_kpis and real_kpis.get('status') == 'success':
                return jsonify(real_kpis)
                
        except Exception as e:
            print(f"[DAY0] Real KPI calculation failed: {e}")
"""
                        # Find the start of the function body
                        for k in range(j+1, min(j+10, len(lines))):
                            if lines[k].strip() and not lines[k].strip().startswith('#'):
                                lines.insert(k, patch)
                                content = '\n'.join(lines)
                                break
                        break
                break
        
        with open(erp_file, 'w') as f:
            f.write(content)
        
        print("✓ Applied KPI calculation patch")
        return True
    
    print("WARNING: Could not find comprehensive-kpis endpoint")
    return False

def main():
    """Apply all Day 0 emergency fixes"""
    print("="*60)
    print("Beverly Knits ERP - Day 0 Emergency Fix Application")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create backup first
    erp_file = Path('/mnt/c/finalee/beverly_knits_erp_v2/src/core/beverly_comprehensive_erp.py')
    if erp_file.exists():
        backup = create_backup(erp_file)
        print(f"Backup created: {backup}")
        print()
    
    # Apply patches
    print("Applying patches...")
    results = []
    
    results.append(("Import statements", apply_imports_patch()))
    results.append(("Path resolution", apply_path_resolution_patch()))
    results.append(("Column alias system", apply_column_alias_patch()))
    results.append(("KPI calculations", apply_kpi_calculation_patch()))
    
    print()
    print("="*60)
    print("Patch Application Summary")
    print("="*60)
    
    success_count = 0
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}: {'Applied' if success else 'Failed'}")
        if success:
            success_count += 1
    
    print()
    print(f"Total: {success_count}/{len(results)} patches applied successfully")
    
    if success_count == len(results):
        print()
        print("SUCCESS: All Day 0 emergency fixes have been applied!")
        print()
        print("Next steps:")
        print("1. Restart the ERP server:")
        print("   pkill -f 'python3.*beverly'")
        print("   python3 src/core/beverly_comprehensive_erp.py")
        print()
        print("2. Validate the fixes:")
        print("   python3 scripts/day0_emergency_fixes.py --validate")
        print()
        print("3. Check system health:")
        print("   curl http://localhost:5006/api/health-check")
    else:
        print()
        print("WARNING: Not all patches were applied successfully.")
        print("You may need to apply some fixes manually.")
        print(f"Backup available at: {backup}")
    
    return success_count == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)