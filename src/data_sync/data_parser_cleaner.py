#!/usr/bin/env python3
"""
Data Parser and Cleaner for Beverly Knits ERP
Ensures all data files have correct headings, formats, and data types
Runs automatically after each SharePoint sync
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional

# Import fabric mapping configuration
try:
    from fabric_mapping_config import FabricMappingConfig, apply_fabric_mapping
    FABRIC_MAPPING_AVAILABLE = True
except ImportError:
    FABRIC_MAPPING_AVAILABLE = False
    logger.warning("Fabric mapping configuration not available")

# Import yarn mapping configuration
try:
    from yarn_mapping_config import YarnMappingConfig, apply_yarn_mapping
    YARN_MAPPING_AVAILABLE = True
except ImportError:
    YARN_MAPPING_AVAILABLE = False
    logger.warning("Yarn mapping configuration not available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataParserCleaner:
    """Cleans and standardizes data files from SharePoint"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.validation_report = {}
        self.cleaned_dir = self.data_dir / "cleaned"
        self.cleaned_dir.mkdir(exist_ok=True)
        
        # Define expected column mappings for each file type
        self.column_mappings = self._define_column_mappings()
        
        # Define data type specifications
        self.data_types = self._define_data_types()
        
        # Track corrections made
        self.corrections_log = []
    
    def _define_column_mappings(self) -> Dict[str, Dict[str, str]]:
        """Define correct column names for each file type"""
        return {
            "yarn_inventory": {
                # Common variations -> Correct name
                "Desc#": "Desc#",
                "Desc #": "Desc#",
                "Description": "Desc#",
                "Yarn_ID": "Desc#",
                "YarnID": "Desc#",
                "yarn_id": "Desc#",
                
                "Planning_Balance": "Planning_Balance",
                "Planning_Ballance": "Planning_Balance",  # Common typo
                "Planning Balance": "Planning_Balance",
                "PlanningBalance": "Planning_Balance",
                
                "Theoretical_Balance": "Theoretical_Balance",
                "Theoretical Balance": "Theoretical_Balance",
                "Theo_Balance": "Theoretical_Balance",
                "TheoreticalBalance": "Theoretical_Balance",
                
                "Allocated": "Allocated",
                "Allocation": "Allocated",
                "Alloc": "Allocated",
                
                "On_Order": "On_Order",
                "On Order": "On_Order",
                "OnOrder": "On_Order",
                "On-Order": "On_Order",
                
                "Consumed": "Consumed",
                "Consumption": "Consumed",
                "Used": "Consumed",
                
                "Unit": "Unit",
                "Units": "Unit",
                "UOM": "Unit",
                "Unit of Measure": "Unit"
            },
            
            "style_bom": {
                "Style#": "Style#",
                "Style #": "Style#",
                "StyleNumber": "Style#",
                "Style_Number": "Style#",
                "style_number": "Style#",
                
                "Desc#": "Desc#",
                "Yarn_ID": "Desc#",
                "Component": "Desc#",
                "Material": "Desc#",
                
                "Qty": "Qty",
                "Quantity": "Qty",
                "Amount": "Qty",
                "Usage": "Qty",
                
                "Unit": "Unit",
                "UOM": "Unit",
                "Units": "Unit"
            },
            
            "sales_activity": {
                "Style#": "Style#",
                "Style Number": "Style#",
                "StyleNumber": "Style#",
                
                "Customer": "Customer",
                "Customer Name": "Customer",
                "CustomerName": "Customer",
                "Cust": "Customer",
                
                "Order_Date": "Order_Date",
                "Order Date": "Order_Date",
                "OrderDate": "Order_Date",
                "Date": "Order_Date",
                
                "Qty": "Qty",
                "Quantity": "Qty",
                "Order_Qty": "Qty",
                "OrderQty": "Qty",
                
                "Unit": "Unit",
                "UOM": "Unit",
                "Units": "Unit"
            },
            
            "efab_inventory": {
                # Note: Fabric mapping handles style columns based on specific inventory type
                # This is for additional columns not covered by fabric mapping
                
                "Qty": "Qty",
                "Quantity": "Qty",
                "Inventory": "Qty",
                "Stock": "Qty",
                
                "Location": "Location",
                "Stage": "Location",
                "Warehouse": "Location",
                
                "Unit": "Unit",
                "UOM": "Unit"
            },
            
            "efab_styles": {
                "fStyle#": "fStyle#",
                "fStyle #": "fStyle#",
                "Fabric Style": "fStyle#",
                
                "Style#": "Style#",
                "Style #": "Style#",
                "Production Style": "Style#",
                
                "Description": "Description",
                "Desc": "Description",
                "Style Description": "Description"
            },
            
            "knit_orders": {
                "Order#": "Order#",
                "Order #": "Order#",
                "OrderNumber": "Order#",
                "KO#": "Order#",
                "KO Number": "Order#",
                
                "Style#": "Style#",
                "Style Number": "Style#",
                
                "Qty": "Qty",
                "Quantity": "Qty",
                "Order Qty": "Qty",
                
                "Status": "Status",
                "Order Status": "Status",
                "OrderStatus": "Status"
            }
        }
    
    def _define_data_types(self) -> Dict[str, Dict[str, type]]:
        """Define expected data types for critical columns"""
        return {
            "numeric": ["Planning_Balance", "Theoretical_Balance", "Allocated", 
                       "On_Order", "Consumed", "Qty", "Stock"],
            "date": ["Order_Date", "Due_Date", "Ship_Date"],
            "text": ["Desc#", "Style#", "fStyle#", "Customer", "Description"],
            "unit": ["Unit", "UOM"]
        }
    
    def identify_file_type(self, file_path: Path) -> Optional[str]:
        """Identify the type of data file based on name and content"""
        filename = file_path.name.lower()
        
        # Check filename patterns
        if 'yarn_inventory' in filename:
            return 'yarn_inventory'
        elif 'bom' in filename or 'style_bom' in filename:
            return 'style_bom'
        elif 'sales' in filename and 'activity' in filename:
            return 'sales_activity'
        elif 'efab_inventory' in filename:
            return 'efab_inventory'
        elif 'efab_styles' in filename:
            return 'efab_styles'
        elif 'knit_order' in filename or 'efab_knit' in filename:
            return 'knit_orders'
        
        # If can't identify by name, check content
        try:
            df = pd.read_csv(file_path, nrows=5) if file_path.suffix == '.csv' else pd.read_excel(file_path, nrows=5)
            columns = df.columns.tolist()
            
            # Check for characteristic columns
            if any('planning' in col.lower() and 'balance' in col.lower() for col in columns):
                return 'yarn_inventory'
            elif any('style' in col.lower() for col in columns) and any('desc' in col.lower() for col in columns):
                return 'style_bom'
            
        except Exception:
            pass
        
        return None
    
    def clean_column_names(self, df: pd.DataFrame, file_type: str) -> Tuple[pd.DataFrame, List[str]]:
        """Clean and standardize column names"""
        corrections = []
        
        if file_type not in self.column_mappings:
            return df, corrections
        
        mapping = self.column_mappings[file_type]
        new_columns = {}
        
        for col in df.columns:
            # Remove extra spaces and normalize
            clean_col = ' '.join(col.split()).strip()
            
            # Check if this column needs to be renamed
            if clean_col in mapping:
                new_name = mapping[clean_col]
                if clean_col != new_name:
                    new_columns[col] = new_name
                    corrections.append(f"Renamed column '{col}' to '{new_name}'")
            else:
                # Keep original if not in mapping
                new_columns[col] = col
        
        # Apply renaming
        if new_columns:
            df = df.rename(columns=new_columns)
        
        return df, corrections
    
    def clean_data_types(self, df: pd.DataFrame, file_type: str) -> Tuple[pd.DataFrame, List[str]]:
        """Clean and convert data types"""
        corrections = []
        
        for col in df.columns:
            # Numeric columns
            if col in self.data_types["numeric"]:
                try:
                    # Clean numeric values
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')
                    nulls = df[col].isna().sum()
                    if nulls > 0:
                        df[col] = df[col].fillna(0)
                        corrections.append(f"Filled {nulls} null values in '{col}' with 0")
                except Exception as e:
                    corrections.append(f"Warning: Could not convert '{col}' to numeric: {e}")
            
            # Date columns
            elif col in self.data_types["date"]:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    nulls = df[col].isna().sum()
                    if nulls > 0:
                        corrections.append(f"Found {nulls} invalid dates in '{col}'")
                except Exception as e:
                    corrections.append(f"Warning: Could not convert '{col}' to date: {e}")
            
            # Text columns - clean whitespace
            elif col in self.data_types["text"]:
                try:
                    df[col] = df[col].astype(str).str.strip()
                    # Remove 'nan' strings
                    df[col] = df[col].replace('nan', '')
                except Exception:
                    pass
            
            # Unit columns - standardize
            elif col in self.data_types["unit"]:
                try:
                    df[col] = df[col].astype(str).str.upper().str.strip()
                    # Standardize common units
                    unit_mapping = {
                        'LB': 'LBS',
                        'POUND': 'LBS',
                        'POUNDS': 'LBS',
                        'YD': 'YDS',
                        'YARD': 'YDS',
                        'YARDS': 'YDS',
                        'PC': 'PCS',
                        'PIECE': 'PCS',
                        'PIECES': 'PCS'
                    }
                    df[col] = df[col].replace(unit_mapping)
                except Exception:
                    pass
        
        return df, corrections
    
    def validate_critical_columns(self, df: pd.DataFrame, file_type: str) -> List[str]:
        """Validate that critical columns exist and have valid data"""
        issues = []
        
        # Define critical columns for each file type
        critical_columns = {
            'yarn_inventory': ['Desc#', 'Planning_Balance', 'Unit'],
            'style_bom': ['Style#', 'Desc#', 'Qty'],
            'sales_activity': ['Style#', 'Order_Date', 'Qty'],
            'efab_inventory': ['fStyle#', 'Qty'],
            'efab_styles': ['fStyle#', 'Style#'],
            'knit_orders': ['Order#', 'Style#', 'Qty']
        }
        
        if file_type in critical_columns:
            for col in critical_columns[file_type]:
                if col not in df.columns:
                    issues.append(f"Critical column '{col}' missing")
                elif df[col].isna().all():
                    issues.append(f"Critical column '{col}' is entirely empty")
        
        return issues
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file - clean, validate, and save"""
        logger.info(f"Processing: {file_path.name}")
        
        result = {
            'file': file_path.name,
            'status': 'pending',
            'corrections': [],
            'issues': [],
            'rows_before': 0,
            'rows_after': 0
        }
        
        try:
            # Identify file type
            file_type = self.identify_file_type(file_path)
            if not file_type:
                logger.warning(f"Could not identify file type for: {file_path.name}")
                file_type = 'unknown'
            
            result['file_type'] = file_type
            
            # Read file
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, dtype=str)
            else:
                df = pd.read_excel(file_path, dtype=str)
            
            result['rows_before'] = len(df)
            
            # Apply fabric mapping FIRST (before other column cleaning)
            if FABRIC_MAPPING_AVAILABLE:
                df, fabric_corrections = apply_fabric_mapping(df, file_path.name)
                result['corrections'].extend(fabric_corrections)
            
            # Apply yarn mapping (also before general column cleaning)
            if YARN_MAPPING_AVAILABLE:
                df, yarn_corrections = apply_yarn_mapping(df, file_path.name)
                result['corrections'].extend(yarn_corrections)
            
            # Clean column names
            df, col_corrections = self.clean_column_names(df, file_type)
            result['corrections'].extend(col_corrections)
            
            # Clean data types
            df, type_corrections = self.clean_data_types(df, file_type)
            result['corrections'].extend(type_corrections)
            
            # Validate critical columns
            issues = self.validate_critical_columns(df, file_type)
            result['issues'].extend(issues)
            
            # Remove empty rows
            before_drop = len(df)
            df = df.dropna(how='all')
            if len(df) < before_drop:
                result['corrections'].append(f"Removed {before_drop - len(df)} empty rows")
            
            result['rows_after'] = len(df)
            
            # Save cleaned file
            output_path = self.cleaned_dir / file_path.name
            if file_path.suffix == '.csv':
                df.to_csv(output_path, index=False)
            else:
                df.to_excel(output_path, index=False, engine='openpyxl')
            
            result['status'] = 'success' if not issues else 'warnings'
            result['output_path'] = str(output_path)
            
            logger.info(f"Cleaned {file_path.name}: {len(col_corrections)} column fixes, {len(type_corrections)} type fixes")
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"Error processing {file_path.name}: {e}")
        
        return result
    
    def process_all_files(self) -> Dict[str, Any]:
        """Process all data files in the directory"""
        logger.info(f"Starting data cleaning for: {self.data_dir}")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': str(self.data_dir),
            'files_processed': [],
            'summary': {
                'total_files': 0,
                'successful': 0,
                'warnings': 0,
                'errors': 0,
                'total_corrections': 0
            }
        }
        
        # Process all Excel and CSV files
        files = list(self.data_dir.glob("*.xlsx")) + list(self.data_dir.glob("*.csv"))
        report['summary']['total_files'] = len(files)
        
        for file_path in files:
            # Skip if already in cleaned directory
            if 'cleaned' in str(file_path):
                continue
            
            result = self.process_file(file_path)
            report['files_processed'].append(result)
            
            # Update summary
            if result['status'] == 'success':
                report['summary']['successful'] += 1
            elif result['status'] == 'warnings':
                report['summary']['warnings'] += 1
            else:
                report['summary']['errors'] += 1
            
            report['summary']['total_corrections'] += len(result['corrections'])
        
        # Save report
        report_path = self.data_dir / f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Data cleaning complete. Report saved to: {report_path}")
        logger.info(f"Summary: {report['summary']}")
        
        return report
    
    def create_data_quality_summary(self) -> str:
        """Create a human-readable summary of data quality"""
        report = self.process_all_files()
        
        summary = f"""
DATA QUALITY REPORT
==================
Timestamp: {report['timestamp']}
Directory: {report['data_dir']}

SUMMARY
-------
Total Files: {report['summary']['total_files']}
✅ Successful: {report['summary']['successful']}
⚠️  Warnings: {report['summary']['warnings']}
❌ Errors: {report['summary']['errors']}
🔧 Total Corrections: {report['summary']['total_corrections']}

FILE DETAILS
------------
"""
        
        for file_result in report['files_processed']:
            status_emoji = {'success': '✅', 'warnings': '⚠️', 'error': '❌'}.get(file_result['status'], '?')
            summary += f"\n{status_emoji} {file_result['file']} (Type: {file_result.get('file_type', 'unknown')})"
            summary += f"\n   Rows: {file_result['rows_before']} → {file_result['rows_after']}"
            
            if file_result['corrections']:
                summary += f"\n   Corrections ({len(file_result['corrections'])}):"
                for correction in file_result['corrections'][:3]:  # Show first 3
                    summary += f"\n     - {correction}"
                if len(file_result['corrections']) > 3:
                    summary += f"\n     ... and {len(file_result['corrections']) - 3} more"
            
            if file_result['issues']:
                summary += f"\n   ⚠️  Issues:"
                for issue in file_result['issues']:
                    summary += f"\n     - {issue}"
            
            if file_result.get('error'):
                summary += f"\n   ❌ Error: {file_result['error']}"
        
        summary += "\n\n✅ Cleaned files saved to: " + str(self.cleaned_dir)
        
        return summary


# Integration function
def clean_sharepoint_data(data_dir: Path) -> bool:
    """Clean and validate SharePoint data after sync"""
    try:
        cleaner = DataParserCleaner(data_dir)
        report = cleaner.process_all_files()
        
        # Check if cleaning was successful
        if report['summary']['errors'] == 0:
            logger.info("Data cleaning completed successfully")
            
            # Move cleaned files back to main directory
            cleaned_dir = data_dir / "cleaned"
            for cleaned_file in cleaned_dir.glob("*"):
                if cleaned_file.is_file():
                    target = data_dir / cleaned_file.name
                    # Backup original
                    if target.exists():
                        backup = data_dir / f"{target.stem}_original{target.suffix}"
                        target.rename(backup)
                    # Move cleaned file
                    cleaned_file.rename(target)
            
            logger.info("Cleaned files moved to main directory")
            return True
        else:
            logger.error(f"Data cleaning encountered {report['summary']['errors']} errors")
            return False
            
    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean and validate ERP data files')
    parser.add_argument('--dir', default='C:/finalee/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/sharepoint_sync',
                       help='Data directory to clean')
    parser.add_argument('--report', action='store_true', help='Generate detailed report')
    
    args = parser.parse_args()
    
    cleaner = DataParserCleaner(Path(args.dir))
    
    if args.report:
        summary = cleaner.create_data_quality_summary()
        print(summary)
    else:
        report = cleaner.process_all_files()
        print(f"\nCleaning complete:")
        print(f"Files: {report['summary']['total_files']}")
        print(f"Success: {report['summary']['successful']}")
        print(f"Warnings: {report['summary']['warnings']}")
        print(f"Errors: {report['summary']['errors']}")
        print(f"Total corrections: {report['summary']['total_corrections']}")