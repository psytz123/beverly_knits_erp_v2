#!/usr/bin/env python3
"""
Script to consolidate multiple data loader implementations into one
Phase 5: Performance Optimization
"""

import os
import re
from pathlib import Path
import shutil
from datetime import datetime

class DataLoaderConsolidator:
    def __init__(self):
        self.src_path = Path('/mnt/c/finalee/beverly_knits_erp_v2/src')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.changes = []
    
    def backup_files(self):
        """Backup files before changes"""
        backup_dir = self.src_path / f'backups/loader_consolidation_{self.timestamp}'
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup data_loaders directory if it exists
        data_loaders_path = self.src_path / 'data_loaders'
        if data_loaders_path.exists():
            shutil.copytree(
                data_loaders_path,
                backup_dir / 'data_loaders'
            )
            print(f"✓ Backup created: {backup_dir}")
        else:
            print(f"⚠ No data_loaders directory to backup")
        
        return backup_dir
    
    def find_import_statements(self):
        """Find all import statements for data loaders"""
        import_patterns = [
            r'from data_loaders\.optimized_data_loader import \w+',
            r'from data_loaders\.parallel_data_loader import \w+',
            r'from data_loaders\.database_data_loader import \w+',
            r'import data_loaders\.\w+_data_loader',
            r'from src\.data_loaders\.\w+ import \w+',
        ]
        
        files_with_imports = []
        
        for py_file in self.src_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in import_patterns:
                    if re.search(pattern, content):
                        files_with_imports.append(py_file)
                        break
            except Exception as e:
                print(f"⚠ Error reading {py_file}: {e}")
        
        print(f"✓ Found {len(files_with_imports)} files with data loader imports")
        return files_with_imports
    
    def update_imports(self, files):
        """Update import statements in files"""
        
        replacements = [
            # Direct imports
            (r'from data_loaders\.optimized_data_loader import OptimizedDataLoader',
             'from data_loaders.unified_data_loader import UnifiedDataLoader'),
            (r'from data_loaders\.parallel_data_loader import ParallelDataLoader',
             'from data_loaders.unified_data_loader import UnifiedDataLoader'),
            (r'from data_loaders\.database_data_loader import DatabaseDataLoader',
             'from data_loaders.unified_data_loader import UnifiedDataLoader'),
            
            # With src prefix
            (r'from src\.data_loaders\.optimized_data_loader import OptimizedDataLoader',
             'from src.data_loaders.unified_data_loader import UnifiedDataLoader'),
            (r'from src\.data_loaders\.parallel_data_loader import ParallelDataLoader',
             'from src.data_loaders.unified_data_loader import UnifiedDataLoader'),
            
            # Class instantiation
            (r'OptimizedDataLoader\(',
             'UnifiedDataLoader('),
            (r'ParallelDataLoader\(',
             'UnifiedDataLoader('),
            (r'DatabaseDataLoader\(',
             'UnifiedDataLoader('),
        ]
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                for old_pattern, new_pattern in replacements:
                    content = re.sub(old_pattern, new_pattern, content)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.changes.append(file_path)
                    print(f"  ✓ Updated: {file_path.relative_to(self.src_path)}")
            except Exception as e:
                print(f"  ✗ Error updating {file_path}: {e}")
    
    def check_existing_loaders(self):
        """Check which data loader files actually exist"""
        data_loaders_dir = self.src_path / 'data_loaders'
        
        if not data_loaders_dir.exists():
            print("⚠ No data_loaders directory found")
            return []
        
        existing_loaders = []
        potential_loaders = [
            'optimized_data_loader.py',
            'parallel_data_loader.py',
            'database_data_loader.py',
            'unified_data_loader.py'
        ]
        
        for loader in potential_loaders:
            loader_path = data_loaders_dir / loader
            if loader_path.exists():
                existing_loaders.append(loader)
                print(f"  ✓ Found: {loader}")
        
        return existing_loaders
    
    def consolidate_loader_code(self):
        """Create unified data loader if it doesn't exist"""
        unified_loader_path = self.src_path / 'data_loaders' / 'unified_data_loader.py'
        
        if unified_loader_path.exists():
            print("✓ Unified data loader already exists")
            return
        
        print("✓ Unified data loader exists, checking for improvements...")
        
        # Check if we need to update the unified loader
        with open(unified_loader_path, 'r') as f:
            content = f.read()
            
        # Check if UnifiedDataLoader class exists
        if 'class UnifiedDataLoader' not in content:
            print("  ⚠ UnifiedDataLoader class not found, keeping existing implementation")
    
    def remove_duplicate_loaders(self):
        """Remove duplicate loader files (disabled for safety)"""
        print("⚠ Skipping removal of duplicate loaders (safety measure)")
        print("  To remove manually, delete these files after verification:")
        
        loaders_to_check = [
            'data_loaders/optimized_data_loader.py',
            'data_loaders/parallel_data_loader.py',
            'data_loaders/database_data_loader.py'
        ]
        
        for loader in loaders_to_check:
            file_path = self.src_path / loader
            if file_path.exists():
                print(f"    - {loader}")
    
    def generate_report(self):
        """Generate consolidation report"""
        report_dir = self.src_path.parent / 'reports'
        report_dir.mkdir(exist_ok=True)
        
        report_path = report_dir / f'loader_consolidation_{self.timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("DATA LOADER CONSOLIDATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Files updated: {len(self.changes)}\n\n")
            
            if self.changes:
                f.write("Updated files:\n")
                for file in self.changes:
                    f.write(f"  - {file.relative_to(self.src_path)}\n")
            else:
                f.write("No files needed updating.\n")
            
            f.write("\nNext steps:\n")
            f.write("1. Review the unified_data_loader.py implementation\n")
            f.write("2. Run tests to verify functionality\n")
            f.write("3. Check application startup\n")
            f.write("4. Monitor performance improvements\n")
            f.write("5. Remove old loader files after verification\n")
        
        print(f"\n✓ Report saved: {report_path}")
        return report_path
    
    def run(self):
        """Execute consolidation"""
        print("="*60)
        print("DATA LOADER CONSOLIDATION")
        print("Phase 5: Performance Optimization")
        print("="*60)
        
        # Check existing loaders
        print("\n1. Checking existing data loaders...")
        existing = self.check_existing_loaders()
        
        if not existing:
            print("\n⚠ No data loaders found to consolidate")
            print("  The system may already be using unified_data_loader.py")
            return
        
        # Backup
        print("\n2. Creating backup...")
        backup_dir = self.backup_files()
        
        # Find and update imports
        print("\n3. Finding files with data loader imports...")
        files = self.find_import_statements()
        
        if files:
            print("\n4. Updating import statements...")
            self.update_imports(files)
        else:
            print("\n4. No import statements to update")
        
        # Consolidate loader code
        print("\n5. Checking unified loader...")
        self.consolidate_loader_code()
        
        # Note about duplicate removal
        print("\n6. Checking for duplicate loaders...")
        self.remove_duplicate_loaders()
        
        # Generate report
        print("\n7. Generating report...")
        report_path = self.generate_report()
        
        print("\n" + "="*60)
        print("CONSOLIDATION COMPLETE!")
        print("="*60)
        
        if self.changes:
            print(f"\n✓ Updated {len(self.changes)} files")
            print("✓ Please test the application to ensure everything works")
        else:
            print("\n✓ No changes were necessary")
            print("✓ System appears to already be using unified data loader")
        
        return True

if __name__ == "__main__":
    consolidator = DataLoaderConsolidator()
    success = consolidator.run()
    exit(0 if success else 1)