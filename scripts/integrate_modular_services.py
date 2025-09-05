#!/usr/bin/env python3
"""
Script to integrate modular services into the main ERP file
Implements Phase 2 modularization by replacing embedded classes with service imports
"""

import sys
import re
from pathlib import Path
from datetime import datetime
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ServiceIntegrator:
    def __init__(self):
        self.erp_path = Path('src/core/beverly_comprehensive_erp.py')
        self.backup_dir = Path('backups/service_integration')
        self.backup_path = None
        
    def create_backup(self):
        """Create backup before making changes"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_path = self.backup_dir / f'beverly_comprehensive_erp_backup_{timestamp}.py'
        shutil.copy2(self.erp_path, self.backup_path)
        logger.info(f"Backup created: {self.backup_path}")
        
    def integrate_services(self):
        """Main integration logic"""
        logger.info("Starting service integration...")
        
        # Create backup first
        self.create_backup()
        
        # Read the file
        with open(self.erp_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Track changes
        changes_made = 0
        original_content = content
        
        # First, add the service imports if not already present
        if 'from src.services.erp_service_manager import ERPServiceManager' not in content:
            # Find where to insert imports (after DAY0 imports or after sys.path setup)
            import_location = content.find('DAY0_FIXES_AVAILABLE = False')
            if import_location > 0:
                import_location = content.find('\n', import_location) + 1
                
                service_imports = """
# Phase 2 Modularization - Import extracted services
try:
    from src.services.erp_service_manager import ERPServiceManager
    from src.services.inventory_analyzer_service import InventoryAnalyzer as InventoryAnalyzerService
    from src.services.inventory_analyzer_service import InventoryManagementPipeline as InventoryManagementPipelineService
    from src.services.sales_forecasting_service import SalesForecastingEngine as SalesForecastingEngineService
    from src.services.capacity_planning_service import CapacityPlanningEngine as CapacityPlanningEngineService
    MODULAR_SERVICES_AVAILABLE = True
    print('[PHASE2] Modular services loaded successfully')
except ImportError as e:
    print(f'[PHASE2] Modular services not available: {e}')
    MODULAR_SERVICES_AVAILABLE = False
"""
                content = content[:import_location] + service_imports + content[import_location:]
                logger.info("Added service imports")
                changes_made += 1
        
        # Find where each class starts
        inventory_analyzer_match = re.search(r'^class InventoryAnalyzer:', content, re.MULTILINE)
        inventory_pipeline_match = re.search(r'^class InventoryManagementPipeline:', content, re.MULTILINE)
        forecasting_engine_match = re.search(r'^class SalesForecastingEngine:', content, re.MULTILINE)
        capacity_planning_match = re.search(r'^class CapacityPlanningEngine:', content, re.MULTILINE)
        
        # Process each class in reverse order (bottom to top) to preserve offsets
        classes_to_wrap = []
        
        if capacity_planning_match:
            classes_to_wrap.append({
                'name': 'CapacityPlanningEngine',
                'service': 'CapacityPlanningEngineService',
                'start': capacity_planning_match.start(),
                'line': content[:capacity_planning_match.start()].count('\n') + 1
            })
            
        if forecasting_engine_match:
            classes_to_wrap.append({
                'name': 'SalesForecastingEngine', 
                'service': 'SalesForecastingEngineService',
                'start': forecasting_engine_match.start(),
                'line': content[:forecasting_engine_match.start()].count('\n') + 1
            })
            
        if inventory_pipeline_match:
            classes_to_wrap.append({
                'name': 'InventoryManagementPipeline',
                'service': 'InventoryManagementPipelineService',
                'start': inventory_pipeline_match.start(),
                'line': content[:inventory_pipeline_match.start()].count('\n') + 1
            })
            
        if inventory_analyzer_match:
            classes_to_wrap.append({
                'name': 'InventoryAnalyzer',
                'service': 'InventoryAnalyzerService',
                'start': inventory_analyzer_match.start(),
                'line': content[:inventory_analyzer_match.start()].count('\n') + 1
            })
            
        # Sort by position (reverse order)
        classes_to_wrap.sort(key=lambda x: x['start'], reverse=True)
        
        # Apply wrapping to each class
        for class_info in classes_to_wrap:
            logger.info(f"Wrapping class {class_info['name']} at line {class_info['line']}")
            
            # Find the end of the class (next class or end of file)
            class_start = class_info['start']
            
            # Find next class definition or end of file
            remaining_content = content[class_start:]
            next_class_match = re.search(r'\n^class \w+:', remaining_content[1:], re.MULTILINE)
            
            if next_class_match:
                class_end = class_start + next_class_match.start() + 1
            else:
                # No next class, go to end of file
                class_end = len(content)
                
            # Extract the class definition
            class_definition = content[class_start:class_end]
            
            # Create the wrapped version
            wrapper = f"""
# Use modular service if available for {class_info['name']}
if MODULAR_SERVICES_AVAILABLE:
    {class_info['name']} = {class_info['service']}
    print('[PHASE2] Using modular {class_info['name']} from services')
else:
    # Embedded class definition follows
    pass  # Class definition will follow below

# Original embedded class (used if modular services not available)
"""
            
            # Replace in content
            content = content[:class_start] + wrapper + class_definition
            changes_made += 1
            
        # Write back the modified content
        if changes_made > 0:
            with open(self.erp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Successfully wrapped {changes_made} classes")
        else:
            logger.warning("No classes found to wrap")
            
        return changes_made > 0
        
    def verify_integration(self):
        """Verify the integration was successful"""
        logger.info("Verifying integration...")
        
        with open(self.erp_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Debug: save content to file for inspection
        debug_file = Path('debug_integrated_content.txt')
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Debug content saved to {debug_file}")
            
        checks = {
            'imports': 'from src.services.erp_service_manager import ERPServiceManager' in content,
            'inventory_analyzer': 'InventoryAnalyzer = InventoryAnalyzerService' in content,
            'inventory_pipeline': 'InventoryManagementPipeline = InventoryManagementPipelineService' in content,
            'forecasting': 'SalesForecastingEngine = SalesForecastingEngineService' in content,
            'capacity': 'CapacityPlanningEngine = CapacityPlanningEngineService' in content or 
                       'CapacityPlanningEngine = CapacityPlanningEngineService' in content  # Try both
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            if passed:
                logger.info(f"✓ {check_name} integration found")
            else:
                logger.warning(f"✗ {check_name} integration missing")
                all_passed = False
                
        return all_passed
        
    def rollback(self):
        """Rollback to backup if needed"""
        if self.backup_path and self.backup_path.exists():
            shutil.copy2(self.backup_path, self.erp_path)
            logger.info(f"Rolled back to {self.backup_path}")
            return True
        else:
            logger.error("No backup found for rollback")
            return False

def main():
    integrator = ServiceIntegrator()
    
    try:
        # Integrate services
        success = integrator.integrate_services()
        
        if success:
            # Verify integration
            verified = integrator.verify_integration()
            
            if verified:
                logger.info("="*60)
                logger.info("SERVICE INTEGRATION SUCCESSFUL")
                logger.info("="*60)
                logger.info("\nNext steps:")
                logger.info("1. Restart the ERP application")
                logger.info("2. Run validation: python scripts/validate_endpoints.py")
                logger.info("3. Test all endpoints are working")
                logger.info("\nIf issues arise, run rollback:")
                logger.info(f"  cp {integrator.backup_path} {integrator.erp_path}")
                return 0
            else:
                logger.error("Integration verification failed")
                logger.info("Rolling back changes...")
                integrator.rollback()
                return 1
        else:
            logger.error("Integration failed")
            return 1
            
    except Exception as e:
        logger.error(f"Integration error: {e}")
        if integrator.backup_path:
            logger.info("Rolling back due to error...")
            integrator.rollback()
        return 1

if __name__ == "__main__":
    sys.exit(main())