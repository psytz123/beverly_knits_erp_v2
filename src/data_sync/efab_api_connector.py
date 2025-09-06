#!/usr/bin/env python3
"""
eFab API Connector for Beverly Knits ERP
Integrates with eFab.bklapps.com API for real-time data synchronization
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
from pathlib import Path
import time
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class eFabAPIConnector:
    """Connector for eFab API integration"""
    
    def __init__(self, base_url: str = "https://efab.bklapps.com", 
                 session_cookie: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize eFab API connector
        
        Args:
            base_url: Base URL for eFab API
            session_cookie: Session cookie for authentication
            username: Username for login authentication
            password: Password for login authentication
            config_path: Path to configuration file with credentials
        """
        # Store initial parameters
        self._initial_base_url = base_url
        self.base_url = base_url
        self.session = requests.Session()
        self.session_cookie = session_cookie
        self.username = username or "psytz"  # From .env.efab.example
        self.password = password or "big$cat"  # From .env.efab.example
        self.config_path = config_path or "/mnt/c/finalee/beverly_knits_erp_v2/config/efab_config.json"
        
        # Load configuration (but don't override explicit parameters)
        self._load_config()
        
        # Setup session with headers (use the final base_url)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': f'{self.base_url}/fabrics/so/list',
            'Origin': self.base_url,
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin'
        })
        
        # Set authentication
        if self.session_cookie:
            self._set_authentication()
    
    def _load_config(self):
        """Load configuration from file if exists"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Only load session cookie from config if not provided
                    if not self.session_cookie:
                        self.session_cookie = config.get('session_cookie')
                    # Only use config base_url if no explicit URL was provided
                    if self._initial_base_url == "https://efab.bklapps.com":
                        self.base_url = config.get('base_url', self.base_url)
                    logger.info(f"Configuration loaded from {self.config_path}, using base_url: {self.base_url}")
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    def _set_authentication(self):
        """Set authentication cookies in session"""
        if self.session_cookie:
            # Extract domain from base_url
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            domain = parsed.hostname or 'efab.bklapps.com'
            
            # Set the dancer.session cookie
            # For localhost, don't set domain explicitly
            if domain == 'localhost' or '127.0.0.1' in domain:
                self.session.cookies.set(
                    'dancer.session',
                    self.session_cookie,
                    path='/'
                )
            else:
                self.session.cookies.set(
                    'dancer.session',
                    self.session_cookie,
                    domain=domain,
                    path='/'
                )
            logger.info(f"Authentication cookies set for domain: {domain}")
        elif self.username and self.password:
            # Try to login with username/password
            self._login()
    
    def _login(self) -> bool:
        """
        Login to eFab using username and password
        
        Returns:
            True if login successful, False otherwise
        """
        try:
            login_url = urljoin(self.base_url, '/login')
            login_data = {
                'username': self.username,
                'password': self.password
            }
            
            # Attempt login
            response = self.session.post(login_url, data=login_data, allow_redirects=True)
            
            # Check if login was successful by looking for session cookie
            if 'dancer.session' in self.session.cookies:
                self.session_cookie = self.session.cookies.get('dancer.session')
                logger.info(f"Login successful for user: {self.username}")
                
                # Save the session cookie to config
                self._save_session_cookie()
                return True
            else:
                logger.error("Login failed - no session cookie received")
                return False
                
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def _save_session_cookie(self):
        """Save session cookie to config file"""
        try:
            config = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            
            config['session_cookie'] = self.session_cookie
            config['last_login'] = datetime.now().isoformat()
            
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            logger.info(f"Session cookie saved to {self.config_path}")
        except Exception as e:
            logger.warning(f"Could not save session cookie: {e}")
    
    def _make_request(self, endpoint: str, method: str = 'GET', 
                     params: Optional[Dict] = None, 
                     data: Optional[Dict] = None,
                     retries: int = 3) -> Optional[Dict]:
        """Make API request with retry logic"""
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(retries):
            try:
                if method == 'GET':
                    response = self.session.get(url, params=params, timeout=30)
                elif method == 'POST':
                    response = self.session.post(url, json=data, timeout=30)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                response.raise_for_status()
                
                # Check if response is JSON
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    return response.json()
                else:
                    logger.warning(f"Non-JSON response: {response.text[:200]}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        return None
    
    def get_sales_order_plan_list(self) -> Optional[pd.DataFrame]:
        """
        Fetch sales order plan list from eFab API
        
        Returns:
            DataFrame with sales order planning data
        """
        logger.info("Fetching sales order plan list...")
        
        try:
            data = self._make_request('/api/sales-order/plan/list')
            
            if data:
                # Convert to DataFrame
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict) and 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
                
                logger.info(f"Fetched {len(df)} sales order records")
                return df
            else:
                logger.warning("No data received from API")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching sales orders: {e}")
            return None
    
    def get_knit_orders(self) -> Optional[pd.DataFrame]:
        """
        Fetch knit orders from eFab API
        
        Returns:
            DataFrame with knit order data
        """
        logger.info("Fetching knit orders...")
        
        try:
            # Try multiple possible endpoints
            endpoints = [
                '/api/knit-orders',
                '/api/production/knit-orders',
                '/api/orders/knit'
            ]
            
            for endpoint in endpoints:
                try:
                    data = self._make_request(endpoint)
                    if data:
                        df = pd.DataFrame(data if isinstance(data, list) else data.get('data', [data]))
                        logger.info(f"Fetched {len(df)} knit orders from {endpoint}")
                        return df
                except:
                    continue
            
            logger.warning("Could not fetch knit orders from any endpoint")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching knit orders: {e}")
            return None
    
    def get_inventory_data(self, warehouse: str = 'all') -> Optional[pd.DataFrame]:
        """
        Fetch inventory data from eFab API
        
        Args:
            warehouse: Warehouse code (F01, G00, G02, I01) or 'all'
        
        Returns:
            DataFrame with inventory data
        """
        logger.info(f"Fetching inventory data for warehouse: {warehouse}")
        
        try:
            if warehouse == 'all':
                warehouses = ['F01', 'G00', 'G02', 'I01']
                all_data = []
                
                for wh in warehouses:
                    endpoint = f'/api/inventory/{wh}'
                    data = self._make_request(endpoint)
                    if data:
                        df = pd.DataFrame(data if isinstance(data, list) else data.get('data', [data]))
                        df['warehouse'] = wh
                        all_data.append(df)
                
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    logger.info(f"Fetched {len(combined_df)} total inventory records")
                    return combined_df
            else:
                endpoint = f'/api/inventory/{warehouse}'
                data = self._make_request(endpoint)
                if data:
                    df = pd.DataFrame(data if isinstance(data, list) else data.get('data', [data]))
                    df['warehouse'] = warehouse
                    logger.info(f"Fetched {len(df)} inventory records for {warehouse}")
                    return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching inventory: {e}")
            return None
    
    def sync_to_csv(self, data: pd.DataFrame, filename: str, 
                    output_dir: str = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/"):
        """
        Save DataFrame to CSV file for integration with existing system
        
        Args:
            data: DataFrame to save
            filename: Output filename
            output_dir: Output directory path
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Add timestamp to filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = filename.replace('.csv', '')
            output_path = os.path.join(output_dir, f"{base_name}_efab_{timestamp}.csv")
            
            # Save to CSV
            data.to_csv(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
            
            # Also save as latest version
            latest_path = os.path.join(output_dir, filename)
            data.to_csv(latest_path, index=False)
            logger.info(f"Latest data saved to {latest_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            return None
    
    def sync_all_data(self) -> Dict[str, str]:
        """
        Sync all available data from eFab API
        
        Returns:
            Dictionary with paths to saved files
        """
        logger.info("Starting full data synchronization...")
        saved_files = {}
        
        # Sync sales orders
        sales_orders = self.get_sales_order_plan_list()
        if sales_orders is not None and not sales_orders.empty:
            path = self.sync_to_csv(sales_orders, 'eFab_SO_List.csv')
            if path:
                saved_files['sales_orders'] = path
        
        # Sync knit orders
        knit_orders = self.get_knit_orders()
        if knit_orders is not None and not knit_orders.empty:
            path = self.sync_to_csv(knit_orders, 'eFab_Knit_Orders.csv')
            if path:
                saved_files['knit_orders'] = path
        
        # Sync inventory for all warehouses
        for warehouse in ['F01', 'G00', 'G02', 'I01']:
            inventory = self.get_inventory_data(warehouse)
            if inventory is not None and not inventory.empty:
                path = self.sync_to_csv(inventory, f'eFab_Inventory_{warehouse}.csv')
                if path:
                    saved_files[f'inventory_{warehouse}'] = path
        
        logger.info(f"Synchronization complete. Saved {len(saved_files)} files")
        return saved_files
    
    def test_connection(self) -> bool:
        """
        Test the API connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing API connection...")
            data = self._make_request('/api/sales-order/plan/list')
            
            if data is not None:
                logger.info("✅ Connection successful!")
                return True
            else:
                logger.error("❌ Connection failed - no data received")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    connector = eFabAPIConnector(
        session_cookie="aLfEsRKatML6uTMdgQEvwQchdl6c3LyRbm"  # From the screenshot
    )
    
    # Test connection
    if connector.test_connection():
        print("\n🔄 Starting data synchronization...")
        
        # Sync all data
        saved_files = connector.sync_all_data()
        
        print("\n✅ Synchronization complete!")
        print("\nSaved files:")
        for key, path in saved_files.items():
            print(f"  - {key}: {path}")