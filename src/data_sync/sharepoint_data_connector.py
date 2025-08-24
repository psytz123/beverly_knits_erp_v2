#!/usr/bin/env python3
"""
SharePoint Data Connector for Beverly Knits ERP
Handles authentication and data synchronization from SharePoint to local filesystem
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import requests
from urllib.parse import urlparse, unquote

# Try to import SharePoint libraries
try:
    from office365.runtime.auth.authentication_context import AuthenticationContext
    from office365.sharepoint.client_context import ClientContext
    from office365.runtime.auth.user_credential import UserCredential
    SHAREPOINT_LIBS_AVAILABLE = True
except ImportError:
    SHAREPOINT_LIBS_AVAILABLE = False
    print("SharePoint libraries not available. Install with: pip install Office365-REST-Python-Client")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SharePointDataConnector:
    """Connects to SharePoint and syncs data files to local filesystem"""
    
    def __init__(self, site_url: str, folder_url: str, local_data_path: str):
        """
        Initialize SharePoint connector
        
        Args:
            site_url: SharePoint site URL
            folder_url: Full SharePoint folder URL
            local_data_path: Local directory to sync files to
        """
        self.site_url = site_url
        self.folder_url = folder_url
        self.local_data_path = Path(local_data_path)
        self.local_data_path.mkdir(parents=True, exist_ok=True)
        
        # Cache directory for downloaded files
        self.cache_dir = self.local_data_path / ".sharepoint_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Metadata file to track sync status
        self.metadata_file = self.cache_dir / "sync_metadata.json"
        self.metadata = self._load_metadata()
        
        # Authentication context
        self.ctx = None
        
    def _load_metadata(self) -> Dict:
        """Load sync metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "last_sync": None,
            "files": {},
            "sync_interval_minutes": 60
        }
    
    def _save_metadata(self):
        """Save sync metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def authenticate_interactive(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Authenticate to SharePoint interactively
        
        Args:
            username: SharePoint username (email)
            password: SharePoint password
        """
        if not SHAREPOINT_LIBS_AVAILABLE:
            raise ImportError("SharePoint libraries not installed. Run: pip install Office365-REST-Python-Client")
        
        # If credentials not provided, try environment variables
        if not username:
            username = os.environ.get('SHAREPOINT_USERNAME')
        if not password:
            password = os.environ.get('SHAREPOINT_PASSWORD')
        
        # If still not available, prompt user
        if not username:
            username = input("Enter SharePoint username (email): ")
        if not password:
            import getpass
            password = getpass.getpass("Enter SharePoint password: ")
        
        try:
            # Create authentication context
            ctx_auth = AuthenticationContext(self.site_url)
            if ctx_auth.acquire_token_for_user(username, password):
                self.ctx = ClientContext(self.site_url, ctx_auth)
                logger.info("Successfully authenticated to SharePoint")
                return True
            else:
                logger.error("Authentication failed")
                return False
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def authenticate_app_only(self, client_id: str, client_secret: str):
        """
        Authenticate using app-only credentials (for automated sync)
        
        Args:
            client_id: Azure AD app client ID
            client_secret: Azure AD app client secret
        """
        if not SHAREPOINT_LIBS_AVAILABLE:
            raise ImportError("SharePoint libraries not installed")
        
        try:
            ctx_auth = AuthenticationContext(self.site_url)
            if ctx_auth.acquire_token_for_app(client_id, client_secret):
                self.ctx = ClientContext(self.site_url, ctx_auth)
                logger.info("Successfully authenticated with app credentials")
                return True
            return False
        except Exception as e:
            logger.error(f"App authentication error: {e}")
            return False
    
    def list_files(self, file_extensions: List[str] = ['.xlsx', '.csv']) -> List[Dict]:
        """
        List files in the SharePoint folder
        
        Args:
            file_extensions: List of file extensions to filter
            
        Returns:
            List of file information dictionaries
        """
        if not self.ctx:
            raise RuntimeError("Not authenticated. Call authenticate_* first.")
        
        try:
            # Parse folder path from URL
            parsed = urlparse(self.folder_url)
            folder_path = unquote(parsed.path)
            
            # Get folder
            folder = self.ctx.web.get_folder_by_server_relative_url(folder_path)
            files = folder.files
            self.ctx.load(files)
            self.ctx.execute_query()
            
            # Filter and return file info
            file_list = []
            for file in files:
                if any(file.properties['Name'].endswith(ext) for ext in file_extensions):
                    file_list.append({
                        'name': file.properties['Name'],
                        'size': file.properties['Length'],
                        'modified': file.properties['TimeLastModified'],
                        'url': file.properties['ServerRelativeUrl']
                    })
            
            return file_list
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def download_file(self, file_name: str, server_relative_url: str) -> Optional[Path]:
        """
        Download a file from SharePoint
        
        Args:
            file_name: Name of the file
            server_relative_url: Server relative URL of the file
            
        Returns:
            Path to downloaded file or None if failed
        """
        if not self.ctx:
            raise RuntimeError("Not authenticated")
        
        try:
            local_file_path = self.local_data_path / file_name
            
            # Download file
            file = self.ctx.web.get_file_by_server_relative_url(server_relative_url)
            self.ctx.load(file)
            self.ctx.execute_query()
            
            # Save to local
            with open(local_file_path, 'wb') as local_file:
                file.download(local_file)
                self.ctx.execute_query()
            
            logger.info(f"Downloaded: {file_name}")
            return local_file_path
        except Exception as e:
            logger.error(f"Error downloading {file_name}: {e}")
            return None
    
    def sync_folder(self, force: bool = False) -> Dict[str, int]:
        """
        Sync all files from SharePoint folder to local
        
        Args:
            force: Force sync even if within sync interval
            
        Returns:
            Dictionary with sync statistics
        """
        stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}
        
        # Check if sync needed
        if not force and self.metadata['last_sync']:
            last_sync = datetime.fromisoformat(self.metadata['last_sync'])
            if datetime.now() - last_sync < timedelta(minutes=self.metadata['sync_interval_minutes']):
                logger.info(f"Skipping sync - last sync was {(datetime.now() - last_sync).seconds // 60} minutes ago")
                return stats
        
        # List and download files
        files = self.list_files()
        for file_info in files:
            file_name = file_info['name']
            
            # Check if file needs download
            if file_name in self.metadata['files']:
                if self.metadata['files'][file_name]['modified'] == file_info['modified']:
                    logger.debug(f"Skipping {file_name} - not modified")
                    stats['skipped'] += 1
                    continue
            
            # Download file
            if self.download_file(file_name, file_info['url']):
                self.metadata['files'][file_name] = {
                    'modified': file_info['modified'],
                    'size': file_info['size'],
                    'downloaded': datetime.now().isoformat()
                }
                stats['downloaded'] += 1
            else:
                stats['failed'] += 1
        
        # Update metadata
        self.metadata['last_sync'] = datetime.now().isoformat()
        self._save_metadata()
        
        logger.info(f"Sync complete: {stats}")
        return stats
    
    def setup_auto_sync(self, interval_minutes: int = 60):
        """
        Set up automatic synchronization interval
        
        Args:
            interval_minutes: Minutes between sync attempts
        """
        self.metadata['sync_interval_minutes'] = interval_minutes
        self._save_metadata()
        logger.info(f"Auto-sync interval set to {interval_minutes} minutes")
    
    @staticmethod
    def create_from_url(sharepoint_url: str, local_data_path: str) -> 'SharePointDataConnector':
        """
        Create connector from SharePoint folder URL
        
        Args:
            sharepoint_url: Full SharePoint folder URL
            local_data_path: Local directory path
            
        Returns:
            Configured SharePointDataConnector instance
        """
        # Parse SharePoint URL
        parsed = urlparse(sharepoint_url)
        site_url = f"{parsed.scheme}://{parsed.netloc}"
        
        return SharePointDataConnector(site_url, sharepoint_url, local_data_path)


# Integration function for Beverly Knits ERP
def integrate_sharepoint_with_erp(erp_instance, sharepoint_url: str, username: str = None, password: str = None):
    """
    Integrate SharePoint data source with Beverly Knits ERP
    
    Args:
        erp_instance: Instance of ManufacturingSupplyChainAI
        sharepoint_url: SharePoint folder URL
        username: SharePoint username (optional)
        password: SharePoint password (optional)
    """
    # Create connector
    connector = SharePointDataConnector.create_from_url(
        sharepoint_url,
        str(erp_instance.data_path / "5")  # Use existing data directory
    )
    
    # Authenticate
    if connector.authenticate_interactive(username, password):
        # Sync files
        stats = connector.sync_folder()
        logger.info(f"SharePoint sync completed: {stats}")
        
        # Store connector reference
        erp_instance.sharepoint_connector = connector
        
        # Set up auto-sync
        connector.setup_auto_sync(60)  # Sync every hour
        
        return True
    return False


# Standalone sync utility
def main():
    """Standalone SharePoint sync utility"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sync SharePoint data for Beverly Knits ERP')
    parser.add_argument('--url', default='https://beverlyknits-my.sharepoint.com/:f:/p/psytz/EjOE8ifm4IpGpot4KsLgFy4B9H_dO-vtPFuA4ergdB__Og?e=9GncTp',
                       help='SharePoint folder URL')
    parser.add_argument('--local-path', default='C:/finalee/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5',
                       help='Local data directory')
    parser.add_argument('--username', help='SharePoint username')
    parser.add_argument('--password', help='SharePoint password')
    parser.add_argument('--force', action='store_true', help='Force sync even if recently synced')
    
    args = parser.parse_args()
    
    # Create and configure connector
    connector = SharePointDataConnector.create_from_url(args.url, args.local_path)
    
    # Authenticate
    if connector.authenticate_interactive(args.username, args.password):
        # Sync files
        stats = connector.sync_folder(force=args.force)
        print(f"Sync completed: Downloaded {stats['downloaded']}, Skipped {stats['skipped']}, Failed {stats['failed']}")
    else:
        print("Authentication failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())