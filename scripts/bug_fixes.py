#!/usr/bin/env python3
"""
Beverly Knits ERP Critical Bug Fixes
Phase 1 Day 7-8: Fix critical bugs in Planning Balance calculations
CRITICAL: These fixes patch the main beverly_comprehensive_erp.py file
"""

import os
import sys
import re
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import gc
import shutil
from typing import Dict, Any, Optional, List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BugFixer:
    """Fix critical bugs in Beverly Knits ERP"""
    
    def __init__(self):
        self.erp_path = Path("src/core/beverly_comprehensive_erp.py")
        self.backup_path = None
        self.fixes_applied = []
        self.verification_results = {}
        
    def create_backup(self):
        """Create backup of the main ERP file before applying fixes"""
        
        try:
            backup_dir = Path("backups/bug_fixes")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_path = backup_dir / f"beverly_comprehensive_erp_backup_{timestamp}.py"
            
            if self.erp_path.exists():
                shutil.copy2(self.erp_path, self.backup_path)
                logger.info(f"Backup created: {self.backup_path}")
                
                self.fixes_applied.append({
                    "action": "backup_created",
                    "path": str(self.backup_path),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                logger.error(f"ERP file not found at {self.erp_path}")
                raise FileNotFoundError(f"ERP file not found at {self.erp_path}")
                
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def fix_planning_balance_formula(self):
        """
        Fix the Planning Balance calculation formula
        CORRECT: Planning_Balance = Theoretical_Balance + Allocated + On_Order
        Note: Allocated is ALREADY NEGATIVE in the data files
        """
        
        logger.info("Fixing Planning Balance formula...")
        
        try:
            with open(self.erp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Pattern 1: Fix incorrect formula that subtracts allocated
            # WRONG: planning_balance = theoretical_balance - allocated + on_order
            # RIGHT: planning_balance = theoretical_balance + allocated + on_order
            
            patterns_to_fix = [
                # Pattern with subtraction
                (r"planning_balance\s*=\s*theoretical_balance\s*-\s*allocated\s*\+\s*on_order",
                 "planning_balance = theoretical_balance + allocated + on_order"),
                
                # Pattern with abs(allocated)
                (r"planning_balance\s*=\s*theoretical_balance\s*-\s*abs\(allocated\)\s*\+\s*on_order",
                 "planning_balance = theoretical_balance + allocated + on_order"),
                
                # DataFrame column calculations
                (r"df\['Planning.Balance'\]\s*=\s*df\['Theoretical.Balance'\]\s*-\s*df\['Allocated'\]\s*\+\s*df\['On.Order'\]",
                 "df['Planning_Balance'] = df['Theoretical_Balance'] + df['Allocated'] + df['On_Order']"),
                
                # With underscores
                (r"df\['Planning_Balance'\]\s*=\s*df\['Theoretical_Balance'\]\s*-\s*df\['Allocated'\]\s*\+\s*df\['On_Order'\]",
                 "df['Planning_Balance'] = df['Theoretical_Balance'] + df['Allocated'] + df['On_Order']"),
                
                # Fix negative allocated handling
                (r"allocated\s*=\s*-abs\(.*?\['Allocated'\].*?\)",
                 "allocated = row['Allocated']  # Already negative in data"),
                
                # Fix Planning_Ballance typo
                (r"Planning_Ballance", "Planning_Balance"),
                (r"planning_ballance", "planning_balance")
            ]
            
            fixes_made = 0
            for pattern, replacement in patterns_to_fix:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                    fixes_made += len(matches)
                    logger.info(f"Fixed {len(matches)} instances of pattern: {pattern[:50]}...")
            
            # Add comment explaining the formula
            if fixes_made > 0:
                formula_comment = """
# CRITICAL: Planning Balance Formula
# Planning_Balance = Theoretical_Balance + Allocated + On_Order
# Note: Allocated values are ALREADY NEGATIVE in the source data files
# Do NOT subtract or apply abs() to Allocated values
"""
                # Add comment near imports if not already present
                if "CRITICAL: Planning Balance Formula" not in content:
                    content = content.replace("import pandas as pd", f"import pandas as pd\n{formula_comment}")
            
            # Write fixed content back
            with open(self.erp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Planning Balance formula fixed: {fixes_made} instances corrected")
            
            self.fixes_applied.append({
                "fix": "planning_balance_formula",
                "instances_fixed": fixes_made,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to fix Planning Balance formula: {e}")
            raise
    
    def fix_memory_leaks(self):
        """Fix memory leaks in data loading and processing"""
        
        logger.info("Fixing memory leaks...")
        
        try:
            with open(self.erp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add garbage collection after large operations
            gc_additions = []
            
            # Pattern 1: After DataFrame operations
            df_operations = [
                "pd.read_csv",
                "pd.read_excel",
                "pd.DataFrame",
                "df.merge",
                "df.groupby",
                "df.pivot"
            ]
            
            for op in df_operations:
                if op in content:
                    gc_additions.append(op)
            
            # Add memory management utilities
            memory_utils = """
# Memory Management Utilities
import gc
import tracemalloc

def clear_memory():
    \"\"\"Clear memory and run garbage collection\"\"\"
    gc.collect()
    
def limit_dataframe_size(df, max_rows=1000000):
    \"\"\"Limit DataFrame size to prevent memory overflow\"\"\"
    if len(df) > max_rows:
        logger.warning(f"DataFrame has {len(df)} rows, limiting to {max_rows}")
        return df.head(max_rows)
    return df

def periodic_memory_cleanup():
    \"\"\"Periodic memory cleanup for long-running operations\"\"\"
    import psutil
    process = psutil.Process()
    mem_usage = process.memory_info().rss / 1024 / 1024  # MB
    
    if mem_usage > 2000:  # If using more than 2GB
        logger.warning(f"High memory usage: {mem_usage:.1f} MB. Running garbage collection...")
        gc.collect()
        return True
    return False
"""
            
            # Add memory utilities if not present
            if "def clear_memory()" not in content:
                # Add after imports
                import_end = content.find("\nclass ")
                if import_end > 0:
                    content = content[:import_end] + "\n" + memory_utils + content[import_end:]
            
            # Add gc.collect() calls after large operations
            patterns_to_add_gc = [
                # After loading large files
                (r"(df = pd\.read_.*?\(.*?\))", r"\1\n        gc.collect()  # Clean up after loading"),
                
                # After merge operations
                (r"(df.*?\.merge\(.*?\))", r"\1\n        gc.collect()  # Clean up after merge"),
                
                # In loops processing DataFrames
                (r"(for.*?in.*?:.*?\n.*?df.*?=.*?)\n", r"\1\n            if i % 100 == 0:\n                gc.collect()  # Periodic cleanup\n")
            ]
            
            gc_added = 0
            for pattern, replacement in patterns_to_add_gc[:2]:  # Limit to avoid over-adding
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content, count=3)  # Limit replacements
                    gc_added += 1
            
            # Write fixed content back
            with open(self.erp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Memory leak fixes applied: {gc_added} garbage collection points added")
            
            self.fixes_applied.append({
                "fix": "memory_leaks",
                "gc_points_added": gc_added,
                "utilities_added": "def clear_memory()" not in content,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to fix memory leaks: {e}")
            raise
    
    def add_timeout_handling(self):
        """Add timeout handling to all API endpoints"""
        
        logger.info("Adding timeout handling to API endpoints...")
        
        try:
            with open(self.erp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add timeout decorator
            timeout_decorator = """
# Timeout decorator for API endpoints
from functools import wraps
import signal
import time

class TimeoutError(Exception):
    pass

def timeout(seconds=30):
    \"\"\"Decorator to add timeout to functions\"\"\"
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set the timeout handler
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                logger.error(f"Timeout in {func.__name__}")
                return jsonify({"error": f"Request timeout after {seconds} seconds"}), 504
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Disable alarm
            
            return result
        return wrapper
    return decorator
"""
            
            # Add timeout decorator if not present
            if "def timeout(" not in content:
                # Add after imports
                import_end = content.find("\napp = Flask")
                if import_end > 0:
                    content = content[:import_end] + "\n" + timeout_decorator + content[import_end:]
            
            # Add @timeout decorator to slow endpoints
            slow_endpoints = [
                "@app.route('/api/yarn-intelligence'",
                "@app.route('/api/execute-planning'",
                "@app.route('/api/ml-forecast-detailed'",
                "@app.route('/api/six-phase-supply-chain'",
                "@app.route('/api/production-planning'"
            ]
            
            timeouts_added = 0
            for endpoint in slow_endpoints:
                if endpoint in content:
                    # Add timeout decorator before route
                    pattern = f"({re.escape(endpoint)})"
                    replacement = f"@timeout(30)\n{endpoint}"
                    
                    # Check if timeout not already added
                    check_pattern = f"@timeout.*?\n.*?{re.escape(endpoint)}"
                    if not re.search(check_pattern, content):
                        content = re.sub(pattern, replacement, content)
                        timeouts_added += 1
            
            # Write fixed content back
            with open(self.erp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Timeout handling added: {timeouts_added} endpoints protected")
            
            self.fixes_applied.append({
                "fix": "timeout_handling",
                "endpoints_protected": timeouts_added,
                "decorator_added": "def timeout(" not in content,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to add timeout handling: {e}")
            raise
    
    def fix_column_name_handling(self):
        """Fix column name variations and typos"""
        
        logger.info("Fixing column name handling...")
        
        try:
            with open(self.erp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add column name standardization
            column_standardizer = """
# Column Name Standardization
COLUMN_MAPPINGS = {
    # Planning Balance variations
    'Planning_Ballance': 'Planning_Balance',
    'Planning Ballance': 'Planning_Balance',
    'planning_ballance': 'Planning_Balance',
    'Planning Balance': 'Planning_Balance',
    
    # Theoretical Balance variations
    'Theoratical_Balance': 'Theoretical_Balance',
    'Theoretical Balance': 'Theoretical_Balance',
    
    # Yarn ID variations
    'Yarn_ID': 'Desc#',
    'YarnID': 'Desc#',
    'Yarn ID': 'Desc#',
    'Desc': 'Desc#',
    'desc_num': 'Desc#',
    
    # Style variations
    'Style#': 'Style_Number',
    'fStyle#': 'Style_Number',
    'style_num': 'Style_Number',
    
    # Balance variations
    'Balance (lbs)': 'Balance_lbs',
    'Balance(lbs)': 'Balance_lbs',
}

def standardize_columns(df):
    \"\"\"Standardize column names in DataFrame\"\"\"
    if df is None or df.empty:
        return df
    
    # Apply mappings
    df.columns = [COLUMN_MAPPINGS.get(col, col) for col in df.columns]
    
    # Remove extra spaces
    df.columns = df.columns.str.strip()
    
    # Handle numeric columns with commas
    numeric_columns = ['Balance_lbs', 'Planning_Balance', 'Theoretical_Balance', 'Allocated', 'On_Order']
    for col in numeric_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float, errors='ignore')
    
    return df
"""
            
            # Add standardizer if not present
            if "def standardize_columns(" not in content:
                # Add after imports
                import_end = content.find("\napp = Flask")
                if import_end > 0:
                    content = content[:import_end] + "\n" + column_standardizer + content[import_end:]
            
            # Add standardization calls after DataFrame loads
            patterns_to_add_standardization = [
                (r"(df = pd\.read_csv\(.*?\))", r"\1\n        df = standardize_columns(df)"),
                (r"(df = pd\.read_excel\(.*?\))", r"\1\n        df = standardize_columns(df)"),
                (r"(yarn_df = .*?load.*?\(.*?\))", r"\1\n        yarn_df = standardize_columns(yarn_df)")
            ]
            
            standardizations_added = 0
            for pattern, replacement in patterns_to_add_standardization:
                if re.search(pattern, content):
                    # Check if standardization not already added
                    check_pattern = pattern + r".*?standardize_columns"
                    if not re.search(check_pattern, content, re.DOTALL):
                        content = re.sub(pattern, replacement, content, count=2)
                        standardizations_added += 1
            
            # Write fixed content back
            with open(self.erp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Column name handling fixed: {standardizations_added} standardization calls added")
            
            self.fixes_applied.append({
                "fix": "column_name_handling",
                "standardizations_added": standardizations_added,
                "mappings_added": 0,  # Fixed reference
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to fix column name handling: {e}")
            raise
    
    def verify_fixes(self):
        """Verify that fixes were applied correctly"""
        
        logger.info("Verifying fixes...")
        
        try:
            with open(self.erp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            verifications = {
                "planning_balance_formula": {
                    "check": "theoretical_balance + allocated + on_order" in content.lower(),
                    "description": "Planning Balance formula corrected"
                },
                "memory_management": {
                    "check": "gc.collect()" in content,
                    "description": "Garbage collection added"
                },
                "timeout_handling": {
                    "check": "def timeout(" in content,
                    "description": "Timeout decorator added"
                },
                "column_standardization": {
                    "check": "def standardize_columns(" in content,
                    "description": "Column standardization added"
                },
                "no_planning_ballance_typo": {
                    "check": "planning_ballance" not in content.lower(),
                    "description": "Planning_Ballance typo removed"
                }
            }
            
            all_passed = True
            for fix_name, verification in verifications.items():
                passed = verification["check"]
                self.verification_results[fix_name] = {
                    "passed": passed,
                    "description": verification["description"]
                }
                
                if passed:
                    logger.info(f"✓ {verification['description']}")
                else:
                    logger.warning(f"✗ {verification['description']} - FAILED")
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Failed to verify fixes: {e}")
            return False
    
    def create_rollback_script(self):
        """Create script to rollback fixes if needed"""
        
        rollback_script = f"""#!/usr/bin/env python3
# Rollback script for bug fixes

import shutil
from pathlib import Path

def rollback():
    backup_path = Path("{self.backup_path}")
    target_path = Path("{self.erp_path}")
    
    if backup_path.exists():
        shutil.copy2(backup_path, target_path)
        print("Rolled back successfully")
    else:
        print(f"Backup not found: {{backup_path}}")

if __name__ == "__main__":
    rollback()
"""
        
        rollback_path = Path("scripts/rollback_bug_fixes.py")
        rollback_path.write_text(rollback_script)
        rollback_path.chmod(0o755)
        
        logger.info(f"Rollback script created: {rollback_path}")
    
    def save_fix_report(self):
        """Save report of applied fixes"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "backup_path": str(self.backup_path),
            "fixes_applied": self.fixes_applied,
            "verification_results": self.verification_results
        }
        
        report_path = Path("docs/reports/bug_fixes_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Fix report saved: {report_path}")
    
    def apply_all_fixes(self):
        """Apply all critical bug fixes"""
        
        logger.info("Applying all critical bug fixes...")
        logger.info("=" * 60)
        
        try:
            # Step 1: Create backup
            logger.info("Step 1: Creating backup...")
            self.create_backup()
            
            # Step 2: Fix Planning Balance formula
            logger.info("\nStep 2: Fixing Planning Balance formula...")
            self.fix_planning_balance_formula()
            
            # Step 3: Fix memory leaks
            logger.info("\nStep 3: Fixing memory leaks...")
            self.fix_memory_leaks()
            
            # Step 4: Add timeout handling
            logger.info("\nStep 4: Adding timeout handling...")
            self.add_timeout_handling()
            
            # Step 5: Fix column name handling
            logger.info("\nStep 5: Fixing column name handling...")
            self.fix_column_name_handling()
            
            # Step 6: Verify fixes
            logger.info("\nStep 6: Verifying fixes...")
            all_passed = self.verify_fixes()
            
            # Step 7: Create rollback script
            logger.info("\nStep 7: Creating rollback script...")
            self.create_rollback_script()
            
            # Step 8: Save report
            logger.info("\nStep 8: Saving fix report...")
            self.save_fix_report()
            
            logger.info("\n" + "=" * 60)
            if all_passed:
                logger.info("ALL FIXES APPLIED SUCCESSFULLY")
            else:
                logger.warning("SOME FIXES MAY NOT BE COMPLETE - Review verification results")
            logger.info("=" * 60)
            
            logger.info("\nNext steps:")
            logger.info("1. Restart the ERP application")
            logger.info("2. Test Planning Balance calculations")
            logger.info("3. Monitor memory usage")
            logger.info("4. Verify API timeouts are working")
            logger.info("5. If issues arise, run: python scripts/rollback_bug_fixes.py")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Failed to apply fixes: {e}")
            logger.info("Run rollback script if needed: python scripts/rollback_bug_fixes.py")
            raise

def main():
    """Run the bug fixer"""
    
    fixer = BugFixer()
    success = fixer.apply_all_fixes()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())