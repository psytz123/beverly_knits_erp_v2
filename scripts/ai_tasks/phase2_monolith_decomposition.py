#!/usr/bin/env python3
"""
AI Agent Phase 2: Monolith Decomposition
Breaks down the 18,000-line beverly_comprehensive_erp.py into modular components
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monolith_decomposition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MonolithDecomposer:
    """AI Agent for decomposing the monolithic ERP file"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.monolith_path = self.project_root / 'src' / 'core' / 'beverly_comprehensive_erp.py'
        self.components_dir = self.project_root / 'src' / 'components'
        self.api_dir = self.project_root / 'src' / 'api'
        self.extracted_components = []
        
        # Create directories if they don't exist
        self.components_dir.mkdir(parents=True, exist_ok=True)
        self.api_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_monolith(self) -> Dict:
        """Analyze the monolithic file to identify components"""
        logger.info("🔍 Analyzing monolithic file structure...")
        
        if not self.monolith_path.exists():
            logger.error(f"Monolith file not found at {self.monolith_path}")
            return {}
        
        content = self.monolith_path.read_text()
        lines = content.split('\n')
        
        analysis = {
            'total_lines': len(lines),
            'classes': self._find_classes(content),
            'routes': self._find_routes(content),
            'functions': self._find_functions(content),
            'imports': self._find_imports(content),
        }
        
        logger.info(f"📊 Monolith Analysis:")
        logger.info(f"  Total Lines: {analysis['total_lines']}")
        logger.info(f"  Classes: {len(analysis['classes'])}")
        logger.info(f"  Routes: {len(analysis['routes'])}")
        logger.info(f"  Functions: {len(analysis['functions'])}")
        
        return analysis
    
    def _find_classes(self, content: str) -> List[Dict]:
        """Find all class definitions in the file"""
        classes = []
        pattern = r'^class\s+(\w+).*?(?=^class\s+|\Z)'
        
        for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
            class_name = match.group(1)
            class_content = match.group(0)
            line_start = content[:match.start()].count('\n') + 1
            line_end = content[:match.end()].count('\n') + 1
            
            classes.append({
                'name': class_name,
                'line_start': line_start,
                'line_end': line_end,
                'lines_count': line_end - line_start,
                'content': class_content[:200] + '...' if len(class_content) > 200 else class_content
            })
        
        return classes
    
    def _find_routes(self, content: str) -> List[Dict]:
        """Find all Flask routes in the file"""
        routes = []
        pattern = r'@app\.route\((.*?)\)\s*\ndef\s+(\w+)'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            route_path = match.group(1)
            function_name = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            
            routes.append({
                'path': route_path,
                'function': function_name,
                'line': line_num
            })
        
        return routes
    
    def _find_functions(self, content: str) -> List[Dict]:
        """Find all function definitions"""
        functions = []
        pattern = r'^def\s+(\w+)\s*\([^)]*\):'
        
        for match in re.finditer(pattern, content, re.MULTILINE):
            func_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            functions.append({
                'name': func_name,
                'line': line_num
            })
        
        return functions
    
    def _find_imports(self, content: str) -> List[str]:
        """Find all import statements"""
        imports = []
        lines = content.split('\n')
        
        for line in lines:
            if line.strip().startswith(('import ', 'from ')):
                imports.append(line.strip())
        
        return imports
    
    def extract_component(self, class_name: str, line_start: int, line_end: int) -> bool:
        """Extract a specific component to its own file"""
        logger.info(f"📦 Extracting component: {class_name}")
        
        try:
            # Read the monolith content
            content = self.monolith_path.read_text()
            lines = content.split('\n')
            
            # Extract the class content
            class_lines = lines[line_start-1:line_end]
            class_content = '\n'.join(class_lines)
            
            # Find required imports for this class
            required_imports = self._identify_required_imports(class_content)
            
            # Create the new component file
            component_file = self.components_dir / f"{self._camel_to_snake(class_name)}.py"
            
            # Generate the component module
            module_content = self._generate_component_module(
                class_name, 
                class_content, 
                required_imports
            )
            
            # Write the new component file
            component_file.write_text(module_content)
            
            # Update the extracted components list
            self.extracted_components.append({
                'class': class_name,
                'file': str(component_file),
                'lines_extracted': line_end - line_start
            })
            
            logger.info(f"  ✅ Extracted {class_name} to {component_file}")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Failed to extract {class_name}: {e}")
            return False
    
    def _camel_to_snake(self, name: str) -> str:
        """Convert CamelCase to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _identify_required_imports(self, class_content: str) -> List[str]:
        """Identify imports required for a class"""
        # Common imports that most components need
        base_imports = [
            "import pandas as pd",
            "import numpy as np",
            "from typing import Dict, List, Optional, Tuple, Any",
            "from datetime import datetime, timedelta",
            "import logging",
            ""
        ]
        
        # Check for specific requirements
        if 'cache' in class_content.lower():
            base_imports.append("from src.utils.cache_manager import UnifiedCacheManager")
        
        if 'data_loader' in class_content.lower() or 'DataLoader' in class_content:
            base_imports.append("from src.data_loaders.optimized_data_loader import OptimizedDataLoader")
        
        if 'sklearn' in class_content or 'xgboost' in class_content:
            base_imports.append("from sklearn.ensemble import RandomForestRegressor")
            base_imports.append("import xgboost as xgb")
        
        if 'prophet' in class_content.lower():
            base_imports.append("from prophet import Prophet")
        
        return base_imports
    
    def _generate_component_module(self, class_name: str, class_content: str, imports: List[str]) -> str:
        """Generate a complete component module"""
        module_template = f'''"""
{class_name} Component
Extracted from beverly_comprehensive_erp.py during monolith decomposition
Generated by AI Agent
"""

{chr(10).join(imports)}

logger = logging.getLogger(__name__)


{class_content}


# Component Registration
def get_component():
    """Factory function to get component instance"""
    return {class_name}()


if __name__ == "__main__":
    # Test the component independently
    component = get_component()
    logger.info(f"{{class_name}} component loaded successfully")
'''
        return module_template
    
    def extract_api_routes(self) -> bool:
        """Extract all API routes to separate module"""
        logger.info("🌐 Extracting API routes...")
        
        try:
            content = self.monolith_path.read_text()
            
            # Find all routes
            route_pattern = r'(@app\.route\([^)]+\)[\s\S]*?)(?=@app\.route|def\s+\w+(?!.*@app\.route)|$)'
            routes = re.findall(route_pattern, content)
            
            if not routes:
                logger.warning("  No routes found to extract")
                return False
            
            # Generate routes module
            routes_module = self._generate_routes_module(routes)
            
            # Write routes file
            routes_file = self.api_dir / 'routes.py'
            routes_file.write_text(routes_module)
            
            logger.info(f"  ✅ Extracted {len(routes)} routes to {routes_file}")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Failed to extract routes: {e}")
            return False
    
    def _generate_routes_module(self, routes: List[str]) -> str:
        """Generate the API routes module"""
        module_template = f'''"""
API Routes Module
Extracted from beverly_comprehensive_erp.py during monolith decomposition
Generated by AI Agent
"""

from flask import Blueprint, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json
import io

# Import components (these will be created during extraction)
from src.components.inventory_analyzer import InventoryAnalyzer
from src.components.sales_forecasting_engine import SalesForecastingEngine
from src.components.capacity_planning_engine import CapacityPlanningEngine

logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint('api', __name__)

# Initialize components
inventory_analyzer = InventoryAnalyzer()
sales_forecasting = SalesForecastingEngine()
capacity_planning = CapacityPlanningEngine()


# Extracted Routes
{chr(10).join(routes)}


def register_routes(app):
    """Register the API blueprint with the Flask app"""
    app.register_blueprint(api_bp)
    logger.info(f"Registered {{len(api_bp.deferred_functions)}} API routes")
'''
        return module_template
    
    def update_main_file(self) -> bool:
        """Update the main file to use extracted components"""
        logger.info("🔄 Updating main file to use extracted components...")
        
        try:
            content = self.monolith_path.read_text()
            
            # Add imports for extracted components
            import_lines = [
                "# Extracted Components (AI Agent Refactoring)",
            ]
            
            for component in self.extracted_components:
                class_name = component['class']
                module_name = self._camel_to_snake(class_name)
                import_lines.append(f"from src.components.{module_name} import {class_name}")
            
            if self.api_dir.joinpath('routes.py').exists():
                import_lines.append("from src.api.routes import register_routes")
            
            # Add imports after existing imports
            import_section = '\n'.join(import_lines) + '\n\n'
            
            # Find the right place to insert imports (after the last import)
            lines = content.split('\n')
            last_import_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    last_import_idx = i
            
            lines.insert(last_import_idx + 1, import_section)
            
            # Write back
            self.monolith_path.write_text('\n'.join(lines))
            
            logger.info("  ✅ Updated main file with component imports")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Failed to update main file: {e}")
            return False
    
    def create_component_registry(self) -> bool:
        """Create a registry to track all extracted components"""
        logger.info("📝 Creating component registry...")
        
        registry_file = self.project_root / 'src' / 'core' / 'component_registry.py'
        
        registry_content = f'''"""
Component Registry
Tracks all components extracted from the monolithic ERP file
Generated by AI Agent
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Central registry for all decomposed components"""
    
    def __init__(self):
        self.components = {{}}
        self.extraction_date = datetime.now()
        self.original_file_lines = 18000
        
        # Register extracted components
        self._register_all_components()
    
    def _register_all_components(self):
        """Register all extracted components"""
        components = {str(self.extracted_components)}
        
        for component in components:
            self.register_component(
                name=component['class'],
                module_path=component['file'],
                lines_extracted=component['lines_extracted']
            )
    
    def register_component(self, name: str, module_path: str, lines_extracted: int = 0):
        """Register an extracted component"""
        self.components[name] = {{
            'module': module_path,
            'extracted_date': datetime.now().isoformat(),
            'lines_extracted': lines_extracted,
            'status': 'active'
        }}
        logger.info(f"Registered component: {{name}}")
    
    def get_component(self, name: str) -> Optional[Dict]:
        """Get component information"""
        return self.components.get(name)
    
    def get_extraction_status(self) -> Dict:
        """Get current extraction progress"""
        total_lines_extracted = sum(
            comp.get('lines_extracted', 0) 
            for comp in self.components.values()
        )
        
        return {{
            'total_components': len(self.components),
            'components': list(self.components.keys()),
            'original_file_lines': self.original_file_lines,
            'lines_extracted': total_lines_extracted,
            'remaining_lines': self.original_file_lines - total_lines_extracted,
            'extraction_date': self.extraction_date.isoformat(),
            'completion_percentage': (total_lines_extracted / self.original_file_lines) * 100
        }}
    
    def validate_components(self) -> Dict[str, bool]:
        """Validate that all components are working"""
        validation_results = {{}}
        
        for name, info in self.components.items():
            try:
                # Try to import the component
                module_path = info['module'].replace('/', '.').replace('.py', '')
                exec(f"from {{module_path}} import {{name}}")
                validation_results[name] = True
                logger.info(f"✅ Component {{name}} validated successfully")
            except Exception as e:
                validation_results[name] = False
                logger.error(f"❌ Component {{name}} validation failed: {{e}}")
        
        return validation_results


# Singleton instance
_registry = None

def get_registry() -> ComponentRegistry:
    """Get the singleton registry instance"""
    global _registry
    if _registry is None:
        _registry = ComponentRegistry()
    return _registry


if __name__ == "__main__":
    # Test the registry
    registry = get_registry()
    status = registry.get_extraction_status()
    
    print("Component Registry Status:")
    print(f"  Total Components: {{status['total_components']}}")
    print(f"  Lines Extracted: {{status['lines_extracted']}}")
    print(f"  Completion: {{status['completion_percentage']:.1f}}%")
    
    # Validate components
    validation = registry.validate_components()
    print(f"\\nValidation Results:")
    for component, valid in validation.items():
        status = "✅" if valid else "❌"
        print(f"  {{status}} {{component}}")
'''
        
        registry_file.write_text(registry_content)
        logger.info(f"  ✅ Created component registry at {registry_file}")
        return True
    
    def execute_decomposition(self, target_components: List[str] = None) -> Dict:
        """Execute the full monolith decomposition"""
        logger.info("=" * 60)
        logger.info("🚀 PHASE 2: MONOLITH DECOMPOSITION - STARTING")
        logger.info("=" * 60)
        
        # Analyze the monolith
        analysis = self.analyze_monolith()
        
        if not analysis:
            logger.error("Failed to analyze monolith")
            return {'success': False, 'error': 'Analysis failed'}
        
        # Default target components if not specified
        if not target_components:
            target_components = [
                'InventoryAnalyzer',
                'SalesForecastingEngine',
                'CapacityPlanningEngine',
                'InventoryManagementPipeline',
                'YarnSubstitutionEngine',
            ]
        
        # Extract components
        extraction_results = {}
        for class_info in analysis['classes']:
            if class_info['name'] in target_components:
                success = self.extract_component(
                    class_info['name'],
                    class_info['line_start'],
                    class_info['line_end']
                )
                extraction_results[class_info['name']] = success
        
        # Extract API routes
        routes_extracted = self.extract_api_routes()
        
        # Update main file
        main_updated = self.update_main_file()
        
        # Create component registry
        registry_created = self.create_component_registry()
        
        # Summary
        logger.info("=" * 60)
        logger.info("📊 DECOMPOSITION SUMMARY:")
        logger.info(f"  Components Extracted: {len(self.extracted_components)}")
        logger.info(f"  Routes Extracted: {routes_extracted}")
        logger.info(f"  Main File Updated: {main_updated}")
        logger.info(f"  Registry Created: {registry_created}")
        
        total_lines_extracted = sum(comp['lines_extracted'] for comp in self.extracted_components)
        logger.info(f"  Total Lines Extracted: {total_lines_extracted}")
        
        logger.info("=" * 60)
        
        return {
            'success': True,
            'components_extracted': len(self.extracted_components),
            'lines_extracted': total_lines_extracted,
            'routes_extracted': routes_extracted,
            'main_updated': main_updated,
            'registry_created': registry_created
        }


def main():
    """Main execution for AI agents"""
    import sys
    
    # Get project root
    project_root = os.getenv('PROJECT_ROOT', os.getcwd())
    
    # Initialize decomposer
    decomposer = MonolithDecomposer(project_root)
    
    # Execute decomposition
    results = decomposer.execute_decomposition()
    
    if results['success']:
        logger.info("✅ Phase 2 completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Phase 2 failed. Please review logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()