#!/usr/bin/env python3
# Rollback script for bug fixes

import shutil
from pathlib import Path

def rollback():
    backup_path = Path("backups\bug_fixes\beverly_comprehensive_erp_backup_20250905_013104.py")
    target_path = Path("src\core\beverly_comprehensive_erp.py")
    
    if backup_path.exists():
        shutil.copy2(backup_path, target_path)
        print("Rolled back successfully")
    else:
        print(f"Backup not found: {backup_path}")

if __name__ == "__main__":
    rollback()
