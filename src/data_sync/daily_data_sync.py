#!/usr/bin/env python3
"""
Daily Data Sync Manager for Beverly Knits ERP
Automatically downloads the newest data from SharePoint on each startup
Old versions are deleted from SharePoint, so we always need the latest
"""

import os
import json
import requests
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import webbrowser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyDataSync:
    """Manages daily data synchronization from SharePoint"""
    
    def __init__(self):
        # SharePoint ERP Data folder (updated daily)
        self.sharepoint_url = "https://beverlyknits-my.sharepoint.com/:f:/r/personal/psytz_beverlyknits_com/Documents/ERP%20Data?csf=1&web=1&e=ByOLem"
        
        # Local data directory
        self.local_data_dir = Path("C:/finalee/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/sharepoint_sync")
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download directory (where browser saves files)
        self.download_dir = Path.home() / "Downloads"
        
        # Sync status file
        self.sync_status_file = self.local_data_dir / ".daily_sync_status.json"
        self.sync_status = self._load_sync_status()
    
    def _load_sync_status(self) -> Dict:
        """Load sync status from file"""
        if self.sync_status_file.exists():
            with open(self.sync_status_file, 'r') as f:
                return json.load(f)
        return {
            "last_sync": None,
            "last_sync_file": None,
            "sync_count": 0
        }
    
    def _save_sync_status(self):
        """Save sync status to file"""
        with open(self.sync_status_file, 'w') as f:
            json.dump(self.sync_status, f, indent=2)
    
    def needs_sync(self) -> bool:
        """Check if data needs to be synced (daily update)"""
        if not self.sync_status["last_sync"]:
            return True
        
        last_sync = datetime.fromisoformat(self.sync_status["last_sync"])
        
        # Check if last sync was today
        if last_sync.date() < datetime.now().date():
            logger.info(f"Last sync was on {last_sync.date()}, need to sync today's data")
            return True
        
        # Also check if data directory is empty
        data_files = list(self.local_data_dir.glob("*.xlsx")) + list(self.local_data_dir.glob("*.csv"))
        if len(data_files) == 0:
            logger.warning("No data files found locally, sync required")
            return True
        
        logger.info(f"Data already synced today at {last_sync.strftime('%H:%M:%S')}")
        return False
    
    def find_latest_download(self) -> Optional[Path]:
        """Find the most recent ERP data download in Downloads folder"""
        # Look for files downloaded in the last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        potential_files = []
        patterns = ["*ERP*Data*.zip", "*ERP_Data*.zip", "*SharePoint*.zip", "Documents.zip"]
        
        for pattern in patterns:
            for file_path in self.download_dir.glob(pattern):
                if file_path.stat().st_mtime > one_hour_ago.timestamp():
                    potential_files.append(file_path)
        
        if potential_files:
            # Return the most recent file
            return max(potential_files, key=lambda p: p.stat().st_mtime)
        
        return None
    
    def clean_old_data(self):
        """Remove old data files before extracting new ones"""
        logger.info("Cleaning old data files...")
        
        # Backup current data first
        if any(self.local_data_dir.iterdir()):
            backup_dir = self.local_data_dir.parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(exist_ok=True)
            
            for file_path in self.local_data_dir.iterdir():
                if file_path.is_file() and not file_path.name.startswith('.'):
                    shutil.copy2(file_path, backup_dir)
            
            logger.info(f"Backed up old data to: {backup_dir}")
        
        # Remove old files
        for file_path in self.local_data_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                file_path.unlink()
    
    def extract_and_sync(self, zip_path: Path) -> bool:
        """Extract ZIP file and sync data"""
        try:
            logger.info(f"Extracting data from: {zip_path}")
            
            # Clean old data first (SharePoint deletes old versions)
            self.clean_old_data()
            
            # Extract new data
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List contents
                file_list = zip_ref.namelist()
                logger.info(f"Found {len(file_list)} files in ZIP")
                
                # Extract all files
                for file_name in file_list:
                    # Skip directories and system files
                    if file_name.endswith('/') or file_name.startswith('__'):
                        continue
                    
                    # Extract to local data directory
                    target_path = self.local_data_dir / Path(file_name).name
                    
                    with zip_ref.open(file_name) as source:
                        with open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                    
                    logger.info(f"Extracted: {target_path.name}")
            
            # CLEAN AND VALIDATE DATA
            logger.info("\n[DATA CLEANING] Starting data validation and cleaning...")
            try:
                from data_parser_cleaner import clean_sharepoint_data
                
                if clean_sharepoint_data(self.local_data_dir):
                    logger.info("[DATA CLEANING] ✅ Data cleaned and validated successfully")
                else:
                    logger.warning("[DATA CLEANING] ⚠️ Data cleaning encountered issues")
                
            except Exception as e:
                logger.error(f"[DATA CLEANING] Error during data cleaning: {e}")
                logger.warning("[DATA CLEANING] Proceeding with uncleaned data")
            
            # Update sync status
            self.sync_status["last_sync"] = datetime.now().isoformat()
            self.sync_status["last_sync_file"] = zip_path.name
            self.sync_status["sync_count"] += 1
            self._save_sync_status()
            
            logger.info("Data sync completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return False
    
    def auto_download_and_sync(self) -> bool:
        """Automatically download and sync the latest data"""
        if not self.needs_sync():
            return True  # Already synced today
        
        logger.info("="*60)
        logger.info("DAILY DATA SYNC REQUIRED")
        logger.info("="*60)
        logger.info("SharePoint data is updated daily. Downloading newest version...")
        
        # Open SharePoint in browser
        logger.info(f"\nOpening SharePoint folder in browser...")
        logger.info(f"Please download the data when the page loads")
        webbrowser.open(self.sharepoint_url)
        
        # Wait for user to download
        logger.info("\nWaiting for download to complete...")
        logger.info("The file will be saved to your Downloads folder")
        
        # Monitor downloads folder
        import time
        start_time = time.time()
        timeout = 300  # 5 minutes timeout
        
        while time.time() - start_time < timeout:
            latest_download = self.find_latest_download()
            
            if latest_download:
                logger.info(f"\nFound downloaded file: {latest_download.name}")
                
                # Wait a bit to ensure download is complete
                time.sleep(2)
                
                # Extract and sync
                if self.extract_and_sync(latest_download):
                    logger.info("\n✅ Daily data sync completed successfully!")
                    return True
                else:
                    logger.error("Failed to extract data")
                    return False
            
            time.sleep(5)  # Check every 5 seconds
        
        logger.error("Timeout waiting for download")
        return False
    
    def get_data_status(self) -> Dict:
        """Get current data status"""
        data_files = list(self.local_data_dir.glob("*.xlsx")) + list(self.local_data_dir.glob("*.csv"))
        
        status = {
            "total_files": len(data_files),
            "xlsx_files": len([f for f in data_files if f.suffix == '.xlsx']),
            "csv_files": len([f for f in data_files if f.suffix == '.csv']),
            "last_sync": self.sync_status.get("last_sync"),
            "sync_count": self.sync_status.get("sync_count", 0),
            "needs_sync": self.needs_sync()
        }
        
        if data_files:
            # Get newest file modification time
            newest_file = max(data_files, key=lambda p: p.stat().st_mtime)
            status["newest_file_date"] = datetime.fromtimestamp(newest_file.stat().st_mtime).isoformat()
        
        return status


# Integration with ERP
def ensure_daily_data_sync():
    """Ensure daily data is synced before ERP starts"""
    syncer = DailyDataSync()
    
    status = syncer.get_data_status()
    logger.info(f"Data status: {status['total_files']} files, last sync: {status['last_sync']}")
    
    if status['needs_sync']:
        logger.warning("Daily data sync required!")
        success = syncer.auto_download_and_sync()
        
        if not success:
            logger.error("Failed to sync daily data")
            logger.info("\nMANUAL SYNC REQUIRED:")
            logger.info(f"1. Open: {syncer.sharepoint_url}")
            logger.info("2. Download the data (it will be a ZIP file)")
            logger.info(f"3. Run: python daily_data_sync.py --file <downloaded_file.zip>")
            return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Daily SharePoint data sync')
    parser.add_argument('--file', help='Path to downloaded ZIP file (for manual sync)')
    parser.add_argument('--status', action='store_true', help='Check sync status')
    parser.add_argument('--auto', action='store_true', help='Run automatic sync')
    
    args = parser.parse_args()
    
    syncer = DailyDataSync()
    
    if args.status:
        status = syncer.get_data_status()
        print(f"\nData Sync Status:")
        print(f"Total files: {status['total_files']} ({status['xlsx_files']} Excel, {status['csv_files']} CSV)")
        print(f"Last sync: {status['last_sync'] or 'Never'}")
        print(f"Sync count: {status['sync_count']}")
        print(f"Needs sync: {'YES' if status['needs_sync'] else 'NO'}")
        
    elif args.file:
        # Manual sync with provided file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
        else:
            if syncer.extract_and_sync(file_path):
                print("✅ Manual sync completed successfully!")
            else:
                print("❌ Manual sync failed")
                
    else:
        # Auto sync
        if syncer.auto_download_and_sync():
            print("\n✅ Daily data is up to date!")
        else:
            print("\n❌ Please download data manually from SharePoint")