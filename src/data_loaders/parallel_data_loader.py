"""
Parallel Data Loader for Beverly Knits ERP
Implements concurrent data loading for 4x faster startup
"""

import concurrent.futures
import pandas as pd
import logging
from typing import Dict, Any, Optional
import time
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class ParallelDataLoader:
    """
    Load multiple data sources concurrently for improved performance
    """
    
    def __init__(self, data_path: str = None, max_workers: int = 5):
        """Initialize parallel data loader"""
        self.data_path = data_path or "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5/"
        self.max_workers = max_workers
        self.results = {}
        
    def load_yarn_inventory(self) -> pd.DataFrame:
        """Load yarn inventory data"""
        try:
            # Look for the latest yarn_inventory file
            import glob
            pattern = os.path.join(self.data_path, "yarn_inventory*.xlsx")
            files = glob.glob(pattern)
            
            if not files:
                pattern = os.path.join(self.data_path, "yarn_inventory*.csv")
                files = glob.glob(pattern)
            
            if files:
                # Get the most recent file
                latest_file = max(files, key=os.path.getmtime)
                logger.info(f"Loading yarn inventory from: {latest_file}")
                
                if latest_file.endswith('.xlsx'):
                    return pd.read_excel(latest_file, engine='openpyxl')
                else:
                    return pd.read_csv(latest_file)
            else:
                logger.warning("No yarn inventory file found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading yarn inventory: {e}")
            return pd.DataFrame()
    
    def load_style_bom(self) -> pd.DataFrame:
        """Load Style BOM data - prioritize BOM_updated.csv"""
        try:
            # First try BOM_updated.csv
            bom_updated_file = os.path.join(self.data_path, "BOM_updated.csv")
            if os.path.exists(bom_updated_file):
                logger.info(f"Using BOM_updated.csv from {self.data_path}")
                return pd.read_csv(bom_updated_file)
            
            # Fallback to Style_BOM.csv
            bom_file = os.path.join(self.data_path, "Style_BOM.csv")
            if os.path.exists(bom_file):
                logger.info(f"Using Style_BOM.csv from {self.data_path}")
                return pd.read_csv(bom_file)
            else:
                logger.warning("Neither BOM_updated.csv nor Style_BOM.csv found")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading Style BOM: {e}")
            return pd.DataFrame()
    
    def load_sales_activity(self) -> pd.DataFrame:
        """Load sales activity data"""
        try:
            # Look for Sales Activity Report
            import glob
            pattern = os.path.join(self.data_path, "Sales Activity*.csv")
            files = glob.glob(pattern)
            
            if files:
                latest_file = max(files, key=os.path.getmtime)
                logger.info(f"Loading sales activity from: {latest_file}")
                return pd.read_csv(latest_file)
            else:
                logger.warning("No Sales Activity Report found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading sales activity: {e}")
            return pd.DataFrame()
    
    def load_knit_orders(self) -> pd.DataFrame:
        """Load knit orders data"""
        try:
            import glob
            pattern = os.path.join(self.data_path, "eFab_Knit_Orders*.xlsx")
            files = glob.glob(pattern)
            
            if files:
                latest_file = max(files, key=os.path.getmtime)
                logger.info(f"Loading knit orders from: {latest_file}")
                return pd.read_excel(latest_file, engine='openpyxl')
            else:
                logger.warning("No Knit Orders file found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading knit orders: {e}")
            return pd.DataFrame()
    
    def load_inventory_stages(self) -> Dict[str, pd.DataFrame]:
        """Load all inventory stage files (F01, G00, G02, I01)"""
        stages = {}
        stage_names = ['F01', 'G00', 'G02', 'I01']
        
        for stage in stage_names:
            try:
                import glob
                pattern = os.path.join(self.data_path, f"eFab_Inventory_{stage}*.xlsx")
                files = glob.glob(pattern)
                
                if files:
                    latest_file = max(files, key=os.path.getmtime)
                    logger.info(f"Loading {stage} inventory from: {latest_file}")
                    stages[stage] = pd.read_excel(latest_file, engine='openpyxl')
                else:
                    logger.warning(f"No {stage} inventory file found")
                    stages[stage] = pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"Error loading {stage} inventory: {e}")
                stages[stage] = pd.DataFrame()
        
        return stages
    
    def load_styles_mapping(self) -> pd.DataFrame:
        """Load eFab styles mapping"""
        try:
            import glob
            pattern = os.path.join(self.data_path, "eFab_Styles*.xlsx")
            files = glob.glob(pattern)
            
            if files:
                latest_file = max(files, key=os.path.getmtime)
                logger.info(f"Loading styles mapping from: {latest_file}")
                return pd.read_excel(latest_file, engine='openpyxl')
            else:
                logger.warning("No eFab_Styles file found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading styles mapping: {e}")
            return pd.DataFrame()
    
    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all data sources in parallel
        Returns dict with all loaded datasets
        """
        start_time = time.time()
        logger.info("Starting parallel data loading...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all loading tasks
            futures = {
                'yarn_inventory': executor.submit(self.load_yarn_inventory),
                'style_bom': executor.submit(self.load_style_bom),
                'sales_activity': executor.submit(self.load_sales_activity),
                'knit_orders': executor.submit(self.load_knit_orders),
                'inventory_stages': executor.submit(self.load_inventory_stages),
                'styles_mapping': executor.submit(self.load_styles_mapping)
            }
            
            # Collect results as they complete
            results = {}
            for key, future in futures.items():
                try:
                    results[key] = future.result(timeout=30)
                    logger.info(f"Loaded {key} successfully")
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout loading {key}")
                    results[key] = pd.DataFrame() if key != 'inventory_stages' else {}
                except Exception as e:
                    logger.error(f"Error loading {key}: {e}")
                    results[key] = pd.DataFrame() if key != 'inventory_stages' else {}
        
        # Log summary statistics
        load_time = time.time() - start_time
        logger.info(f"Parallel data loading completed in {load_time:.2f} seconds")
        
        # Report loaded data statistics
        for key, data in results.items():
            if key == 'inventory_stages':
                for stage, df in data.items():
                    if not df.empty:
                        logger.info(f"  {stage}: {len(df)} rows")
            elif hasattr(data, 'shape'):
                if not data.empty:
                    logger.info(f"  {key}: {data.shape[0]} rows, {data.shape[1]} columns")
        
        self.results = results
        return results
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data"""
        summary = {
            'total_datasets': len(self.results),
            'datasets': {}
        }
        
        for key, data in self.results.items():
            if key == 'inventory_stages':
                summary['datasets'][key] = {
                    stage: {'rows': len(df), 'columns': len(df.columns)} 
                    for stage, df in data.items()
                }
            elif hasattr(data, 'shape'):
                summary['datasets'][key] = {
                    'rows': data.shape[0],
                    'columns': data.shape[1],
                    'empty': data.empty
                }
        
        return summary
    
    def validate_data(self) -> Dict[str, bool]:
        """Validate that critical data was loaded successfully"""
        validation = {
            'yarn_inventory': not self.results.get('yarn_inventory', pd.DataFrame()).empty,
            'style_bom': not self.results.get('style_bom', pd.DataFrame()).empty,
            'sales_activity': not self.results.get('sales_activity', pd.DataFrame()).empty,
            'has_inventory_stages': any(
                not df.empty for df in self.results.get('inventory_stages', {}).values()
            )
        }
        
        validation['all_critical_loaded'] = all([
            validation['yarn_inventory'],
            validation['style_bom']
        ])
        
        return validation


# Helper function for quick parallel loading
def quick_parallel_load(data_path: str = None) -> Dict[str, Any]:
    """Quick helper to load all data in parallel"""
    loader = ParallelDataLoader(data_path)
    return loader.load_all_data()


if __name__ == "__main__":
    # Test parallel loading
    logging.basicConfig(level=logging.INFO)
    
    loader = ParallelDataLoader()
    data = loader.load_all_data()
    
    print("\nData Loading Summary:")
    print("=" * 50)
    summary = loader.get_data_summary()
    for key, info in summary['datasets'].items():
        print(f"{key}: {info}")
    
    print("\nValidation Results:")
    print("=" * 50)
    validation = loader.validate_data()
    for check, result in validation.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}: {result}")