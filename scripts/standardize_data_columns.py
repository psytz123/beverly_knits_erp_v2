#!/usr/bin/env python3
"""
Data Column Standardization Script
Phase 2 Implementation - Comprehensive System Fix
Ensures consistent column naming across all data files
Created: 2025-09-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import shutil
import json
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataStandardizer:
    def __init__(self):
        # Use project root for consistent path resolution
        project_root = Path(__file__).parent.parent
        self.data_path = project_root / 'data' / 'production' / '5'
        self.erp_path = self.data_path / 'ERP Data'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.changes_made = []
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'columns_standardized': 0,
            'errors': 0
        }
        
        # Define standard column mappings
        self.column_mappings = {
            'yarn_inventory': {
                'Planning Balance': 'Planning_Balance',
                'Theoretical Balance': 'Theoretical_Balance',
                'On Order': 'On_Order',
                'Beginning Balance': 'Beginning_Balance',
                'Cost/Pound': 'Cost_Per_Pound',
                'Total Cost': 'Total_Cost',
                'Reconcile Date': 'Reconcile_Date',
                'QS Cust': 'QS_Customer'
            },
            'yarn_demand': {
                'Total Demand': 'Total_Demand',
                'Total Receipt': 'Total_Receipt',
                'Monday Inventory': 'Monday_Inventory',
                'Past Due Receipts': 'Past_Due_Receipts'
            },
            'sales_orders': {
                'SO #': 'SO_Number',
                'Unit Price': 'Unit_Price',
                'Quoted Date': 'Quoted_Date',
                'PO #': 'PO_Number',
                'Ship Date': 'Ship_Date',
                'On Hold': 'On_Hold',
                'fStyle#': 'Style#',  # Important mapping for sales data
                'Invoice Date': 'Invoice_Date'
            },
            'knit_orders': {
                'Style #': 'Style_Number',
                'Style#': 'Style_Number',  # Handle both variations
                'Order #': 'Order_Number',
                'Start Date': 'Start_Date',
                'Quoted Date': 'Quoted_Date',
                'Qty Ordered (lbs)': 'Qty_Ordered_Lbs',
                'G00 (lbs)': 'G00_Lbs',
                'Shipped (lbs)': 'Shipped_Lbs',
                'Balance (lbs)': 'Balance_Lbs',
                'Seconds (lbs)': 'Seconds_Lbs',
                'BKI #s': 'BKI_Numbers'
            },
            'bom': {
                'Style#': 'Style_Number',
                'Style #': 'Style_Number',
                'Desc#': 'Desc_Number',
                'BOM Percentage': 'BOM_Percentage',
                'BOM_Percentage': 'BOM_Percentage'
            }
        }
    
    def backup_files(self):
        """Create backups before standardization"""
        logger.info("Creating backups...")
        
        backup_dir = self.data_path / f'backups/column_standardization_{self.timestamp}'
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all CSV and Excel files
        files_to_backup = []
        
        # Check multiple locations
        search_paths = [
            self.erp_path,
            self.erp_path / '8-28-2025',
            self.data_path
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                csv_files = list(search_path.glob('*.csv'))
                xlsx_files = list(search_path.glob('*.xlsx'))
                files_to_backup.extend(csv_files + xlsx_files)
        
        # Backup each file
        backed_up = 0
        for file in files_to_backup:
            try:
                relative_path = file.relative_to(self.data_path)
                backup_path = backup_dir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, backup_path)
                backed_up += 1
            except Exception as e:
                logger.warning(f"Could not backup {file}: {e}")
            
        logger.info(f"Backed up {backed_up} files to {backup_dir}")
        return backup_dir
    
    def detect_file_type(self, file_path):
        """Detect the type of data file based on name and content"""
        file_name = file_path.name.lower()
        
        if 'yarn_inventory' in file_name:
            return 'yarn_inventory'
        elif 'yarn_demand' in file_name:
            return 'yarn_demand'
        elif 'so_list' in file_name or 'sales' in file_name:
            return 'sales_orders'
        elif 'knit_order' in file_name:
            return 'knit_orders'
        elif 'bom' in file_name:
            return 'bom'
        else:
            # Try to detect by reading columns
            try:
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path, nrows=1)
                else:
                    df = pd.read_excel(file_path, nrows=1)
                
                columns = df.columns.tolist()
                
                # Check for specific columns
                if 'Planning Balance' in columns or 'Planning_Balance' in columns:
                    return 'yarn_inventory'
                elif 'fStyle#' in columns or 'Invoice Date' in columns:
                    return 'sales_orders'
                elif 'Style #' in columns or 'Qty Ordered (lbs)' in columns:
                    return 'knit_orders'
                elif 'BOM_Percentage' in columns:
                    return 'bom'
                    
            except Exception:
                pass
        
        return None
    
    def standardize_file(self, file_path, file_type=None):
        """Standardize columns in a single file"""
        try:
            # Detect file type if not provided
            if file_type is None:
                file_type = self.detect_file_type(file_path)
            
            if file_type is None:
                logger.debug(f"Could not detect file type for {file_path.name}")
                return False, 0
            
            # Read the file
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            original_columns = df.columns.tolist()
            columns_changed = 0
            
            if file_type in self.column_mappings:
                mapping = self.column_mappings[file_type]
                
                # Create new columns with standardized names (keep both old and new)
                for old_name, new_name in mapping.items():
                    if old_name in df.columns and new_name not in df.columns:
                        df[new_name] = df[old_name]
                        columns_changed += 1
                        self.stats['columns_standardized'] += 1
                        
                        self.changes_made.append({
                            'file': file_path.name,
                            'type': file_type,
                            'old_column': old_name,
                            'new_column': new_name
                        })
                        
                        logger.info(f"  {file_path.name}: Added '{new_name}' (from '{old_name}')")
                
                # Save the file if changes were made
                if columns_changed > 0:
                    if file_path.suffix == '.csv':
                        df.to_csv(file_path, index=False)
                    else:
                        df.to_excel(file_path, index=False)
                    
                    self.stats['files_modified'] += 1
                    return True, columns_changed
            
            return False, 0
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats['errors'] += 1
            return False, 0
    
    def standardize_all_files(self):
        """Standardize all data files"""
        logger.info("Starting column standardization...")
        logger.info("Note: Original columns are preserved, standardized columns are added")
        
        # Process each file type
        results = {}
        
        # Define file patterns for each type
        file_patterns = {
            'yarn_inventory': ['yarn_inventory*.csv', 'yarn_inventory*.xlsx'],
            'yarn_demand': ['Yarn_Demand*.csv', 'yarn_demand*.csv'],
            'sales_orders': ['eFab_SO_List*.csv', '*Sales*.csv', 'sales*.csv'],
            'knit_orders': ['eFab_Knit_Orders*.csv', 'eFab_Knit_Orders*.xlsx'],
            'bom': ['BOM*.csv', 'bom*.csv']
        }
        
        # Search in multiple locations
        search_paths = [
            self.erp_path / '8-28-2025',
            self.erp_path,
            self.data_path
        ]
        
        for file_type, patterns in file_patterns.items():
            logger.info(f"\nProcessing {file_type} files...")
            
            for search_path in search_paths:
                if not search_path.exists():
                    continue
                    
                for pattern in patterns:
                    files = list(search_path.glob(pattern))
                    
                    for file in files:
                        self.stats['files_processed'] += 1
                        success, count = self.standardize_file(file, file_type)
                        
                        if file.name not in results:
                            results[file.name] = {
                                'success': success,
                                'columns_added': count,
                                'type': file_type
                            }
        
        return results
    
    def update_column_standardizer_class(self):
        """Update the ColumnStandardizer class with new mappings"""
        standardizer_path = Path('/mnt/c/finalee/beverly_knits_erp_v2/src/utils/column_standardization.py')
        
        if standardizer_path.exists():
            logger.info("\nChecking ColumnStandardizer class...")
            
            # Read current file
            with open(standardizer_path, 'r') as f:
                content = f.read()
            
            # Check if key mappings are present
            missing_mappings = []
            
            if 'Planning_Balance' not in content:
                missing_mappings.append('Planning_Balance')
            if 'fStyle#' not in content:
                missing_mappings.append('fStyle# -> Style#')
            
            if missing_mappings:
                logger.warning(f"ColumnStandardizer may need update for: {', '.join(missing_mappings)}")
                logger.info("Manual update recommended to maintain consistency")
            else:
                logger.info("ColumnStandardizer appears to have necessary mappings")
    
    def generate_report(self, results, backup_dir):
        """Generate standardization report"""
        report_path = self.data_path / f'reports/column_standardization_{self.timestamp}.json'
        report_path.parent.mkdir(exist_ok=True)
        
        # Create detailed report
        report = {
            'timestamp': self.timestamp,
            'backup_location': str(backup_dir),
            'statistics': self.stats,
            'files_processed': results,
            'changes_made': self.changes_made,
            'column_mappings_used': self.column_mappings,
            'recommendations': [
                "Original columns are preserved for backward compatibility",
                "New standardized columns have been added",
                "Update application code to use standardized column names",
                "Test thoroughly before removing original columns"
            ]
        }
        
        # Save JSON report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create text summary
        summary_path = self.data_path / f'reports/column_standardization_summary_{self.timestamp}.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DATA COLUMN STANDARDIZATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Backup Location: {backup_dir}\n\n")
            
            f.write("Statistics:\n")
            f.write("-"*40 + "\n")
            f.write(f"Files processed: {self.stats['files_processed']}\n")
            f.write(f"Files modified: {self.stats['files_modified']}\n")
            f.write(f"Columns standardized: {self.stats['columns_standardized']}\n")
            f.write(f"Errors: {self.stats['errors']}\n\n")
            
            if self.changes_made:
                f.write("Changes Made:\n")
                f.write("-"*40 + "\n")
                for change in self.changes_made[:20]:  # Show first 20 changes
                    f.write(f"{change['file']}: {change['old_column']} → {change['new_column']}\n")
                
                if len(self.changes_made) > 20:
                    f.write(f"... and {len(self.changes_made) - 20} more changes\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("Standardization Complete!\n")
            f.write("Original columns preserved, standardized columns added.\n")
            f.write("="*60 + "\n")
        
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Summary saved to: {summary_path}")
        
        return report_path, summary_path
    
    def run(self):
        """Execute standardization process"""
        logger.info("="*60)
        logger.info("Data Column Standardization Process")
        logger.info("="*60)
        
        try:
            # Backup files
            backup_dir = self.backup_files()
            
            # Standardize columns
            results = self.standardize_all_files()
            
            # Update column standardizer class
            self.update_column_standardizer_class()
            
            # Generate report
            report_path, summary_path = self.generate_report(results, backup_dir)
            
            logger.info("="*60)
            logger.info("Standardization Complete!")
            logger.info(f"Backup: {backup_dir}")
            logger.info(f"Report: {report_path}")
            logger.info("="*60)
            
            # Print summary
            print("\n" + "="*60)
            print("COLUMN STANDARDIZATION SUMMARY")
            print("="*60)
            print(f"Files processed: {self.stats['files_processed']}")
            print(f"Files modified: {self.stats['files_modified']}")
            print(f"Columns standardized: {self.stats['columns_standardized']}")
            print(f"Errors: {self.stats['errors']}")
            print("\nKey Standardizations Applied:")
            print("  • Planning Balance → Planning_Balance")
            print("  • fStyle# → Style#")
            print("  • Style # → Style_Number")
            print("\n✓ Original columns preserved for compatibility")
            print("✓ Standardized columns added for consistency")
            print("\nTo rollback if needed:")
            print(f"  cp -r {backup_dir}/* {self.data_path}/")
            print("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during standardization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    standardizer = DataStandardizer()
    success = standardizer.run()
    
    sys.exit(0 if success else 1)