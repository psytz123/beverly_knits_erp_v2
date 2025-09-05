"""
QuadS API Client for Beverly Knits ERP
Fetches style to work center mappings from QuadS API
"""

import requests
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from functools import lru_cache
import os

logger = logging.getLogger(__name__)

class QuadSAPIClient:
    """Client for interacting with QuadS API"""
    
    def __init__(self, base_url: str = "https://quads.bkiapps.com/api", 
                 api_key: Optional[str] = None,
                 timeout: int = 30):
        """
        Initialize QuadS API client
        
        Args:
            base_url: Base URL for QuadS API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.environ.get('QUADS_API_KEY')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set up headers
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
        
        # Cache settings
        self._cache = {}
        self._cache_ttl = timedelta(minutes=15)
        self._cache_timestamps = {}
        
        logger.info(f"Initialized QuadS API client for {self.base_url}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        
        age = datetime.now() - self._cache_timestamps[cache_key]
        return age < self._cache_ttl
    
    def _make_request(self, endpoint: str, method: str = 'GET', 
                     params: Optional[Dict] = None,
                     data: Optional[Dict] = None) -> Optional[Any]:
        """
        Make HTTP request to QuadS API
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            params: Query parameters
            data: Request body data
            
        Returns:
            Response data or None if error
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # Handle different response types
            content_type = response.headers.get('Content-Type', '')
            
            if 'application/json' in content_type:
                return response.json()
            elif 'text/csv' in content_type:
                # Parse CSV response
                import io
                return pd.read_csv(io.StringIO(response.text))
            else:
                # Return raw text
                return response.text
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout calling {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling {url}: {e}")
            return None
    
    def get_active_greige_styles(self) -> Optional[pd.DataFrame]:
        """
        Fetch active greige styles with work center mappings
        
        Returns:
            DataFrame with style to work center mappings or None if error
        """
        cache_key = 'active_greige_styles'
        
        # Check cache
        if self._is_cache_valid(cache_key):
            logger.debug("Returning cached greige styles")
            return self._cache[cache_key]
        
        # Fetch from API
        logger.info("Fetching active greige styles from QuadS API")
        
        data = self._make_request('styles/greige/active')
        
        if data is None:
            logger.error("Failed to fetch greige styles")
            return None
        
        # Convert to DataFrame
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Handle wrapped response
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                elif 'styles' in data:
                    df = pd.DataFrame(data['styles'])
                else:
                    df = pd.DataFrame([data])
            elif isinstance(data, str):
                # Try to parse as JSON if it's a string
                try:
                    import json
                    parsed = json.loads(data)
                    if isinstance(parsed, list):
                        df = pd.DataFrame(parsed)
                    else:
                        df = pd.DataFrame([parsed])
                except:
                    logger.error(f"Could not parse string response as JSON")
                    return None
            else:
                logger.error(f"Unexpected response type: {type(data)}")
                return None
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Cache the result
            self._cache[cache_key] = df
            self._cache_timestamps[cache_key] = datetime.now()
            
            logger.info(f"Fetched {len(df)} active greige styles")
            return df
            
        except Exception as e:
            logger.error(f"Error processing greige styles response: {e}")
            return None
    
    def get_style_work_center_mapping(self, style_id: Optional[str] = None) -> Dict[str, str]:
        """
        Get style to work center mapping
        
        Args:
            style_id: Optional specific style to look up
            
        Returns:
            Dictionary mapping style IDs to work center IDs
        """
        # Get all active styles
        df = self.get_active_greige_styles()
        
        if df is None or df.empty:
            logger.warning("No style data available")
            return {}
        
        # Build mapping dictionary
        mapping = {}
        
        # Look for style and work center columns
        # Try exact matches first for the QuadS API
        style_col = self._find_column(df, ['ref_style', 'base', 'style_id', 'style', 'style_num', 'fabric'])
        wc_col = self._find_column(df, ['work_center', 'wc', 'workcenter', 'machine_group'])
        
        if not style_col or not wc_col:
            logger.error(f"Could not identify required columns. Available: {df.columns.tolist()}")
            return {}
        
        # Build mapping
        for _, row in df.iterrows():
            style = str(row[style_col]).strip()
            wc = str(row[wc_col]).strip()
            
            if style and wc and style != 'nan' and wc != 'nan':
                mapping[style] = wc
        
        logger.info(f"Built style to work center mapping with {len(mapping)} entries")
        
        # Return specific style or all mappings
        if style_id:
            return {style_id: mapping.get(style_id)}
        
        return mapping
    
    def get_work_center_details(self, work_center_id: str) -> Optional[Dict]:
        """
        Get details for a specific work center
        
        Args:
            work_center_id: Work center ID
            
        Returns:
            Work center details or None
        """
        # This could be extended to fetch from a work centers endpoint
        # For now, parse from the work center ID pattern
        
        # Work center pattern: x.xx.xx.X (e.g., 9.38.20.F)
        parts = work_center_id.split('.')
        
        if len(parts) == 4:
            return {
                'work_center_id': work_center_id,
                'construction': parts[0],
                'diameter': parts[1],
                'needle': parts[2],
                'type': parts[3],
                'pattern': work_center_id
            }
        
        return {'work_center_id': work_center_id, 'pattern': work_center_id}
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names in DataFrame"""
        if df is None or df.empty:
            return df
        
        # Common column name mappings
        column_mappings = {
            'Style': 'style_id',
            'Style#': 'style_id',
            'Style #': 'style_id',
            'fStyle#': 'style_id',
            'Fabric': 'style_id',
            'Work Center': 'work_center',
            'WorkCenter': 'work_center',
            'WC': 'work_center',
            'Machine Group': 'work_center'
        }
        
        # Apply mappings
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        return df
    
    def _find_column(self, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        """Find column matching any of the patterns"""
        for col in df.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern.lower() in col_lower:
                    return col
        return None
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            # Try a simple endpoint
            response = self._make_request('health', method='GET')
            
            if response is None:
                # Try the styles endpoint directly
                response = self._make_request('styles/greige/active', method='GET')
            
            return response is not None
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Singleton instance
_quads_client = None

def get_quads_client(api_key: Optional[str] = None) -> QuadSAPIClient:
    """Get singleton QuadS API client instance"""
    global _quads_client
    
    if _quads_client is None:
        _quads_client = QuadSAPIClient(api_key=api_key)
    
    return _quads_client