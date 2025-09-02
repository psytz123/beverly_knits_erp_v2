#!/usr/bin/env python3
"""
BOM Orphan Detection and Cleanup Script
Phase 2 Implementation - Comprehensive System Fix
Identifies and resolves yarn references in BOM that don't exist in inventory
Created: 2025-09-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2/src')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bom_cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BOMCleaner:
    def __init__(self, data_path='/mnt/c/finalee/beverly_knits_erp_v2/data/production/5'):
        self.data_path = Path(data_path)
        self.erp_path = self.data_path / 'ERP Data'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.stats = {
            'total_bom_entries': 0,
            'total_orphans': 0,
            'orphans_fixed': 0,
            'orphans_removed': 0,
            'typos_corrected': 0
        }
        
    def load_data(self):
        """Load BOM and inventory data"""
        logger.info("Loading data files...")
        
        # Try multiple locations for BOM
        bom_paths = [
            self.data_path / 'BOM_updated.csv',
            self.data_path / 'BOM.csv',
            self.erp_path / 'BOM_updated.csv',
            self.erp_path / 'BOM.csv'
        ]
        
        bom_loaded = False
        for bom_path in bom_paths:
            if bom_path.exists():
                self.bom = pd.read_csv(bom_path)
                logger.info(f"Loaded BOM from {bom_path}: {len(self.bom)} entries")
                self.bom_path = bom_path
                bom_loaded = True
                break
        
        if not bom_loaded:
            raise FileNotFoundError(f"BOM file not found in any expected location")
        
        self.stats['total_bom_entries'] = len(self.bom)
        
        # Try multiple locations for yarn inventory
        yarn_paths = [
            self.erp_path / '8-28-2025' / 'yarn_inventory.csv',
            self.erp_path / 'yarn_inventory.csv',
            self.data_path / 'yarn_inventory.csv',
            self.data_path / 'yarn_inventory.xlsx'
        ]
        
        yarn_loaded = False
        for yarn_path in yarn_paths:
            if yarn_path.exists():
                if yarn_path.suffix == '.csv':
                    self.yarn_inv = pd.read_csv(yarn_path)
                else:
                    self.yarn_inv = pd.read_excel(yarn_path)
                logger.info(f"Loaded inventory from {yarn_path}: {len(self.yarn_inv)} yarns")
                yarn_loaded = True
                break
        
        if not yarn_loaded:
            raise FileNotFoundError(f"Yarn inventory file not found in any expected location")
        
        # Ensure Desc# is string type for comparison
        if 'Desc#' in self.bom.columns:
            self.bom['Desc#'] = self.bom['Desc#'].astype(str)
        if 'Desc#' in self.yarn_inv.columns:
            self.yarn_inv['Desc#'] = self.yarn_inv['Desc#'].astype(str)
            
        # Load yarn master if exists (for future yarns)
        yarn_master_path = self.data_path / 'Yarn_ID_Master.csv'
        if not yarn_master_path.exists():
            yarn_master_path = self.erp_path / 'Yarn_ID_Master.csv'
            
        if yarn_master_path.exists():
            self.yarn_master = pd.read_csv(yarn_master_path)
            logger.info(f"Loaded yarn master: {len(self.yarn_master)} entries")
        else:
            self.yarn_master = None
            logger.warning("Yarn master file not found - skipping future yarn detection")
            
    def identify_orphans(self):
        """Identify orphaned yarn references"""
        logger.info("Identifying orphaned references...")
        
        # Find unique yarns in each dataset
        bom_yarns = set(self.bom['Desc#'].unique())
        inv_yarns = set(self.yarn_inv['Desc#'].unique())
        
        # Remove NaN values from sets
        bom_yarns = {y for y in bom_yarns if pd.notna(y) and y != 'nan'}
        inv_yarns = {y for y in inv_yarns if pd.notna(y) and y != 'nan'}
        
        # Identify orphans
        self.orphans = bom_yarns - inv_yarns
        self.stats['total_orphans'] = len(self.orphans)
        
        logger.info(f"Found {len(self.orphans)} orphaned yarn references")
        
        # Create detailed orphan report
        orphan_details = []
        for yarn in self.orphans:
            bom_entries = self.bom[self.bom['Desc#'] == yarn]
            orphan_details.append({
                'Yarn_ID': yarn,
                'BOM_Count': len(bom_entries),
                'Affected_Styles': bom_entries['Style#'].nunique() if 'Style#' in bom_entries.columns else 0,
                'Style_List': ', '.join(str(s) for s in bom_entries['Style#'].unique()[:5]) if 'Style#' in bom_entries.columns else '',
                'Total_Percentage': bom_entries['BOM_Percentage'].sum() if 'BOM_Percentage' in bom_entries.columns else 0,
                'Avg_Percentage': bom_entries['BOM_Percentage'].mean() if 'BOM_Percentage' in bom_entries.columns else 0
            })
        
        self.orphan_report = pd.DataFrame(orphan_details)
        if not self.orphan_report.empty:
            self.orphan_report = self.orphan_report.sort_values('BOM_Count', ascending=False)
        
    def categorize_orphans(self):
        """Categorize orphans for appropriate handling"""
        logger.info("Categorizing orphans...")
        
        categories = {
            'discontinued': [],
            'future': [],
            'typo': [],
            'unknown': []
        }
        
        for yarn in self.orphans:
            yarn_str = str(yarn)
            
            # Check if it's a typo (similar to existing yarn)
            is_typo = False
            for inv_yarn in self.yarn_inv['Desc#'].unique():
                if pd.notna(inv_yarn) and self._similarity(yarn_str, str(inv_yarn)) > 0.9:
                    categories['typo'].append((yarn, inv_yarn))
                    is_typo = True
                    self.stats['typos_corrected'] += 1
                    break
            
            if not is_typo:
                # Check if in yarn master (future yarn)
                if self.yarn_master is not None:
                    if 'Yarn_ID' in self.yarn_master.columns:
                        if yarn in self.yarn_master['Yarn_ID'].astype(str).values:
                            categories['future'].append(yarn)
                        else:
                            categories['unknown'].append(yarn)
                    else:
                        categories['unknown'].append(yarn)
                else:
                    categories['unknown'].append(yarn)
        
        self.orphan_categories = categories
        
        # Log categorization results
        for category, yarns in categories.items():
            if category == 'typo':
                logger.info(f"{category.capitalize()}: {len(yarns)} yarn typos found")
            else:
                logger.info(f"{category.capitalize()}: {len(yarns)} yarns")
    
    def _similarity(self, s1, s2):
        """Calculate string similarity (simple Levenshtein ratio)"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()
    
    def create_cleanup_options(self):
        """Create different cleanup strategies"""
        logger.info("Creating cleanup options...")
        
        # Option 1: Remove all orphans (aggressive)
        bom_clean_aggressive = self.bom[~self.bom['Desc#'].isin(self.orphans)].copy()
        
        # Option 2: Fix typos, remove unknown (moderate) - RECOMMENDED
        typo_map = dict(self.orphan_categories['typo'])
        bom_clean_moderate = self.bom.copy()
        
        # Fix typos
        if typo_map:
            bom_clean_moderate['Desc#'] = bom_clean_moderate['Desc#'].replace(typo_map)
            logger.info(f"Fixed {len(typo_map)} typo mappings")
        
        # Remove unknown yarns
        unknown_yarns = self.orphan_categories['unknown']
        if unknown_yarns:
            bom_clean_moderate = bom_clean_moderate[~bom_clean_moderate['Desc#'].isin(unknown_yarns)]
            self.stats['orphans_removed'] = len(unknown_yarns)
        
        # Option 3: Keep future yarns, fix typos (conservative)
        bom_clean_conservative = self.bom.copy()
        if typo_map:
            bom_clean_conservative['Desc#'] = bom_clean_conservative['Desc#'].replace(typo_map)
        
        self.cleanup_options = {
            'aggressive': bom_clean_aggressive,
            'moderate': bom_clean_moderate,
            'conservative': bom_clean_conservative
        }
        
        # Report cleanup impact
        for option, df in self.cleanup_options.items():
            removed = len(self.bom) - len(df)
            logger.info(f"{option.capitalize()} cleanup: {removed} entries removed ({removed/len(self.bom)*100:.1f}%)")
    
    def save_results(self):
        """Save all results and reports"""
        logger.info("Saving results...")
        
        # Create reports directory
        reports_dir = self.data_path / 'reports' / f'bom_cleanup_{self.timestamp}'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save orphan report
        if not self.orphan_report.empty:
            self.orphan_report.to_csv(reports_dir / 'orphan_report.csv', index=False)
            logger.info(f"Saved orphan report: {len(self.orphan_report)} orphans")
        
        # Save categorization
        with open(reports_dir / 'orphan_categories.json', 'w') as f:
            # Convert to serializable format
            categories_serializable = {
                k: [str(item) for item in v] if k != 'typo' 
                else [(str(a), str(b)) for a, b in v]
                for k, v in self.orphan_categories.items()
            }
            json.dump(categories_serializable, f, indent=2)
        
        # Save cleanup options
        for option, df in self.cleanup_options.items():
            output_file = reports_dir / f'BOM_cleaned_{option}.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {option} cleanup: {len(df)} entries")
        
        # Create summary report
        summary = {
            'timestamp': self.timestamp,
            'original_bom_entries': self.stats['total_bom_entries'],
            'total_orphans': self.stats['total_orphans'],
            'typos_corrected': self.stats['typos_corrected'],
            'orphans_removed': self.stats['orphans_removed'],
            'orphan_categories': {k: len(v) for k, v in self.orphan_categories.items()},
            'cleanup_impact': {
                option: {
                    'entries_after': len(df),
                    'entries_removed': len(self.bom) - len(df),
                    'percentage_removed': (len(self.bom) - len(df)) / len(self.bom) * 100
                }
                for option, df in self.cleanup_options.items()
            }
        }
        
        with open(reports_dir / 'cleanup_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save statistics
        with open(reports_dir / 'statistics.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"All results saved to: {reports_dir}")
        
        return reports_dir
    
    def apply_recommended_cleanup(self):
        """Apply the recommended (moderate) cleanup option"""
        logger.info("Applying recommended cleanup (moderate option)...")
        
        # Backup original BOM
        backup_path = self.bom_path.parent / f"BOM_backup_{self.timestamp}.csv"
        self.bom.to_csv(backup_path, index=False)
        logger.info(f"Created backup: {backup_path}")
        
        # Apply moderate cleanup
        cleaned_bom = self.cleanup_options['moderate']
        
        # Save cleaned BOM
        cleaned_bom.to_csv(self.bom_path, index=False)
        logger.info(f"Updated BOM saved to: {self.bom_path}")
        logger.info(f"Entries: {len(self.bom)} -> {len(cleaned_bom)} ({len(self.bom) - len(cleaned_bom)} removed)")
        
        self.stats['orphans_fixed'] = self.stats['typos_corrected'] + self.stats['orphans_removed']
        
        return True
    
    def run(self, auto_apply=False):
        """Execute full cleanup process"""
        logger.info("="*60)
        logger.info("BOM Orphan Cleanup Process Started")
        logger.info("="*60)
        
        try:
            # Load data
            self.load_data()
            
            # Identify orphans
            self.identify_orphans()
            
            if self.stats['total_orphans'] == 0:
                logger.info("No orphaned yarns found! BOM is clean.")
                return None
            
            # Categorize orphans
            self.categorize_orphans()
            
            # Create cleanup options
            self.create_cleanup_options()
            
            # Save results
            reports_dir = self.save_results()
            
            # Apply cleanup if requested
            if auto_apply:
                self.apply_recommended_cleanup()
            
            logger.info("="*60)
            logger.info("BOM Cleanup Complete!")
            logger.info(f"Reports saved to: {reports_dir}")
            logger.info("="*60)
            
            # Print summary
            print("\n" + "="*60)
            print("BOM ORPHAN CLEANUP SUMMARY")
            print("="*60)
            print(f"Total BOM entries: {self.stats['total_bom_entries']}")
            print(f"Orphaned yarns found: {self.stats['total_orphans']}")
            print(f"Typos corrected: {self.stats['typos_corrected']}")
            print(f"Orphans removed: {self.stats['orphans_removed']}")
            print(f"Total fixed: {self.stats['orphans_fixed']}")
            print("="*60)
            
            if not auto_apply:
                print("\nRECOMMENDATIONS:")
                print("-" * 40)
                print("1. Review the orphan_report.csv to understand impact")
                print("2. Check orphan_categories.json for categorization")
                print("3. Choose appropriate cleanup strategy:")
                print("   - Conservative: Keeps future yarns (safest)")
                print("   - Moderate: Fixes typos, removes unknown (RECOMMENDED)")
                print("   - Aggressive: Removes all orphans (cleanest)")
                print("\nTo apply moderate cleanup (recommended):")
                print(f"  python3 {__file__} --apply")
                print("\nTo apply specific cleanup manually:")
                print(f"  cp {reports_dir}/BOM_cleaned_[option].csv {self.bom_path}")
            else:
                print("\n✓ Moderate cleanup has been applied!")
                print(f"  Backup saved to: BOM_backup_{self.timestamp}.csv")
            
            return reports_dir
            
        except Exception as e:
            logger.error(f"Error during BOM cleanup: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

if __name__ == "__main__":
    # Check for command line arguments
    auto_apply = '--apply' in sys.argv or '--auto' in sys.argv
    
    cleaner = BOMCleaner()
    cleaner.run(auto_apply=auto_apply)