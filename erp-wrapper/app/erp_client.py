from typing import Any, Dict, Optional, List
from .session import SessionManager
from .config import settings
import logging

logger = logging.getLogger(__name__)

# Global session manager instance
session_mgr = SessionManager()

def erp_get_json(api_path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Calls ERP upstream JSON endpoint (e.g., '/api/styles') and returns parsed JSON.
    Automatically refreshes session if expired.
    """
    if not api_path.startswith(settings.ERP_API_PREFIX):
        # guardrail: only allow calls under the configured API prefix
        raise ValueError(f"Upstream path must start with {settings.ERP_API_PREFIX}")
    
    r = session_mgr.call_with_refresh("GET", api_path, params=params)
    
    if "application/json" in r.headers.get("Content-Type", ""):
        return r.json()
    
    # Try to parse as JSON anyway
    try:
        return r.json()
    except:
        return r.text  # fallback if endpoint sends other content types

def erp_post_json(api_path: str, json_data: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> Any:
    """
    POST to ERP upstream JSON endpoint.
    """
    if not api_path.startswith(settings.ERP_API_PREFIX):
        raise ValueError(f"Upstream path must start with {settings.ERP_API_PREFIX}")
    
    kwargs = {}
    if json_data is not None:
        kwargs['json'] = json_data
    if data is not None:
        kwargs['data'] = data
    
    r = session_mgr.call_with_refresh("POST", api_path, **kwargs)
    
    if "application/json" in r.headers.get("Content-Type", ""):
        return r.json()
    
    try:
        return r.json()
    except:
        return r.text

# Specific ERP API methods

def get_sales_orders() -> List[Dict[str, Any]]:
    """Fetch sales order plan list from eFab."""
    try:
        data = erp_get_json("/api/sales-order/plan/list")
        if isinstance(data, list):
            logger.info(f"Fetched {len(data)} sales orders")
            return data
        elif isinstance(data, dict) and 'data' in data:
            logger.info(f"Fetched {len(data['data'])} sales orders")
            return data['data']
        else:
            logger.warning(f"Unexpected sales order response type: {type(data)}")
            return []
    except Exception as e:
        logger.error(f"Error fetching sales orders: {e}")
        raise

def get_knit_orders() -> List[Dict[str, Any]]:
    """Fetch knit orders from eFab."""
    try:
        # Try multiple possible endpoints
        endpoints = [
            "/api/knit-orders",
            "/api/production/knit-orders",
            "/api/orders/knit"
        ]
        
        for endpoint in endpoints:
            try:
                data = erp_get_json(endpoint)
                if data:
                    if isinstance(data, list):
                        logger.info(f"Fetched {len(data)} knit orders from {endpoint}")
                        return data
                    elif isinstance(data, dict) and 'data' in data:
                        logger.info(f"Fetched {len(data['data'])} knit orders from {endpoint}")
                        return data['data']
            except Exception as e:
                logger.debug(f"Failed to fetch from {endpoint}: {e}")
                continue
        
        logger.warning("Could not fetch knit orders from any endpoint")
        return []
        
    except Exception as e:
        logger.error(f"Error fetching knit orders: {e}")
        raise

def get_inventory(warehouse: str = "all") -> List[Dict[str, Any]]:
    """Fetch inventory data for specific warehouse or all warehouses."""
    try:
        if warehouse == "all":
            warehouses = ["F01", "G00", "G02", "I01"]
            all_inventory = []
            
            for wh in warehouses:
                try:
                    data = erp_get_json(f"/api/inventory/{wh}")
                    if isinstance(data, list):
                        for item in data:
                            item['warehouse'] = wh
                        all_inventory.extend(data)
                        logger.info(f"Fetched {len(data)} items from warehouse {wh}")
                except Exception as e:
                    logger.warning(f"Failed to fetch inventory for {wh}: {e}")
            
            return all_inventory
        else:
            data = erp_get_json(f"/api/inventory/{warehouse}")
            if isinstance(data, list):
                for item in data:
                    item['warehouse'] = warehouse
                logger.info(f"Fetched {len(data)} items from warehouse {warehouse}")
                return data
            return []
            
    except Exception as e:
        logger.error(f"Error fetching inventory: {e}")
        raise

def get_styles() -> List[Dict[str, Any]]:
    """Fetch style data from eFab."""
    try:
        data = erp_get_json("/api/styles")
        if isinstance(data, list):
            logger.info(f"Fetched {len(data)} styles")
            return data
        elif isinstance(data, dict) and 'data' in data:
            logger.info(f"Fetched {len(data['data'])} styles")
            return data['data']
        return []
    except Exception as e:
        logger.error(f"Error fetching styles: {e}")
        raise

def test_connection() -> bool:
    """Test if we can connect to the ERP API."""
    try:
        # Try to fetch a simple endpoint
        data = erp_get_json("/api/sales-order/plan/list")
        return data is not None
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False