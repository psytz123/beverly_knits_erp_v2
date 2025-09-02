#!/usr/bin/env python3
"""
Exclusive Data Source Configuration for Beverly Knits ERP
This module ensures data is ONLY loaded from the designated SharePoint folder
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExclusiveDataConfig:
    """Manages exclusive data source configuration"""
    
    # SINGLE SOURCE OF TRUTH - Integrated PostgreSQL Database and centralized data
    DATABASE_PRIMARY = True  # Use database as primary source
    SHAREPOINT_URL = "https://beverlyknits-my.sharepoint.com/:f:/r/personal/psytz_beverlyknits_com/Documents/ERP%20Data?csf=1&web=1&e=ByOLem"
    
    # Primary data path - now points to centralized sc data location
    EXCLUSIVE_DATA_PATH = Path("/mnt/c/Users/psytz/sc data/ERP Data")
    
    # Database configuration path
    DATABASE_CONFIG_PATH = Path(__file__).parent.parent / 'database' / 'database_config.json'
    
    # Lock file to ensure exclusive access
    LOCK_FILE = EXCLUSIVE_DATA_PATH / ".data_source_lock"
    
    @classmethod
    def initialize(cls):
        """Initialize exclusive data configuration"""
        # Create exclusive data directory
        cls.EXCLUSIVE_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        # Create lock file with configuration
        lock_data = {
            "data_source": "SHAREPOINT_ONLY",
            "sharepoint_url": cls.SHAREPOINT_URL,
            "created": datetime.now().isoformat(),
            "warning": "DATA MUST ONLY BE LOADED FROM THE SHAREPOINT ERP DATA FOLDER",
            "local_path": str(cls.EXCLUSIVE_DATA_PATH)
        }
        
        with open(cls.LOCK_FILE, 'w') as f:
            json.dump(lock_data, f, indent=2)
        
        logger.info(f"Exclusive data source configured: {cls.SHAREPOINT_URL}")
        logger.info(f"Local sync path: {cls.EXCLUSIVE_DATA_PATH}")
        
        # Clear any old data from other sources
        cls._cleanup_old_data_sources()
        
        return cls.EXCLUSIVE_DATA_PATH
    
    @classmethod
    def _cleanup_old_data_sources(cls):
        """Remove data from non-authorized sources"""
        old_paths = [
            Path("C:/finalee/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5"),
            Path("C:/finalee/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/4"),
            Path("C:/finalee/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/prompts/5")
        ]
        
        for old_path in old_paths:
            if old_path.exists() and old_path != cls.EXCLUSIVE_DATA_PATH:
                # Create backup before removing
                backup_path = old_path.parent / f"{old_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.warning(f"Moving old data source to backup: {old_path} -> {backup_path}")
                try:
                    shutil.move(str(old_path), str(backup_path))
                except Exception as e:
                    logger.error(f"Could not move old data: {e}")
    
    @classmethod
    def get_data_path(cls) -> Path:
        """Get the exclusive data path - ONLY valid source"""
        if not cls.LOCK_FILE.exists():
            cls.initialize()
        
        # Verify lock file
        with open(cls.LOCK_FILE, 'r') as f:
            lock_data = json.load(f)
            
        if lock_data.get("data_source") != "SHAREPOINT_ONLY":
            raise ValueError("Invalid data source configuration!")
        
        logger.info(f"Using EXCLUSIVE data source: {cls.EXCLUSIVE_DATA_PATH}")
        return cls.EXCLUSIVE_DATA_PATH
    
    @classmethod
    def validate_data_source(cls, path: Path) -> bool:
        """Validate that the given path is the authorized data source"""
        authorized_path = cls.get_data_path()
        
        # Resolve paths for comparison
        path = Path(path).resolve()
        authorized_path = authorized_path.resolve()
        
        # Check if path is the authorized path or a subdirectory
        try:
            path.relative_to(authorized_path)
            return True
        except ValueError:
            logger.error(f"UNAUTHORIZED DATA SOURCE: {path}")
            logger.error(f"Data MUST be loaded from: {authorized_path}")
            logger.error(f"SharePoint source: {cls.SHAREPOINT_URL}")
            return False
    
    @classmethod
    def enforce_exclusive_source(cls, data_path: Path) -> Path:
        """Enforce exclusive data source - redirect to authorized path"""
        if not cls.validate_data_source(data_path):
            logger.warning(f"Redirecting from {data_path} to authorized source")
            return cls.get_data_path()
        return data_path


# Integration function for ERP
def configure_exclusive_data_source(erp_instance):
    """Configure ERP to use ONLY SharePoint data source"""
    from exclusive_data_config import ExclusiveDataConfig
    
    # Get exclusive data path
    exclusive_path = ExclusiveDataConfig.get_data_path()
    
    # Update ERP data path
    old_path = erp_instance.data_path
    erp_instance.data_path = exclusive_path.parent  # Set to parent to maintain /5 structure
    
    # Create symlink or copy data if needed
    target_path = exclusive_path.parent / "5"
    if not target_path.exists():
        target_path.symlink_to(exclusive_path)
    
    logger.info(f"ERP configured to use EXCLUSIVE data source")
    logger.info(f"Old path: {old_path}")
    logger.info(f"New path: {exclusive_path}")
    logger.info(f"SharePoint: {ExclusiveDataConfig.SHAREPOINT_URL}")
    
    return exclusive_path


if __name__ == "__main__":
    # Initialize exclusive configuration
    print("Configuring exclusive SharePoint data source...")
    
    config = ExclusiveDataConfig()
    data_path = config.initialize()
    
    print(f"\nConfiguration complete!")
    print(f"SharePoint URL: {config.SHAREPOINT_URL}")
    print(f"Local sync path: {data_path}")
    print(f"\nIMPORTANT: Data will ONLY be loaded from the SharePoint ERP Data folder")
    print(f"All other data sources have been disabled")
    
    # Check current data
    if data_path.exists():
        files = list(data_path.glob("*"))
        print(f"\nCurrent files in exclusive data directory: {len(files)}")
        if not files:
            print("\nNo data files found. Please sync from SharePoint:")
            print(f"1. Open: {config.SHAREPOINT_URL}")
            print(f"2. Download all files")
            print(f"3. Extract to: {data_path}")