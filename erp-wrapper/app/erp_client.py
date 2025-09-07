from typing import Any, Dict, Optional, List
from .session import SessionManager
from .config import settings
import logging
import httpx

logger = logging.getLogger(__name__)

# Global session manager instances
session_mgr = SessionManager()

class QuadSSessionManager:
    """Session manager specifically for QuadS domain."""
    
    def __init__(self):
        self.client: Optional[httpx.Client] = None
        self.cookie_cache: Optional[Dict[str, str]] = None
        self._ensure_client()
    
    def _ensure_client(self) -> None:
        if self.client is None:
            self.client = httpx.Client(
                base_url=str(settings.QUADS_BASE_URL),
                timeout=settings.REQUEST_TIMEOUT,
                verify=settings.VERIFY_SSL,
                headers={
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "User-Agent": "efab-wrapper/1.0 (+internal)",
                    "X-Requested-With": "XMLHttpRequest",
                },
                follow_redirects=True
            )
            self.login(force=False)
    
    def login(self, force: bool = False) -> None:
        """Login to QuadS using same credentials as ERP."""
        if not force and self._load_cookie_from_disk():
            self._attach_cookie_to_client()
            # Quick probe to test if cookie is still valid
            try:
                probe_path = f"{settings.QUADS_API_PREFIX}/styles/greige/active"
                probe = self.client.get(self._quads_url(probe_path))
                if probe.status_code < 400:
                    logger.info("Using cached QuadS session cookie")
                    return
            except Exception as e:
                logger.debug(f"QuadS cookie probe failed: {e}")
        
        logger.info("Performing fresh QuadS login...")
        self._form_login()
        self._save_cookie_to_disk()
    
    def call_with_refresh(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Perform a request against QuadS, refreshing the session on 401."""
        url = self._quads_url(path)
        
        # Remove verify from kwargs if present, use settings
        kwargs.pop('verify', None)
        
        try:
            r = self.client.request(method, url, **kwargs)
            
            if r.status_code in (401, 403):
                # session likely expired -> relogin and retry once
                logger.info(f"QuadS got {r.status_code}, refreshing session...")
                self.login(force=True)
                r = self.client.request(method, url, **kwargs)
            
            r.raise_for_status()
            return r
            
        except httpx.RequestError as e:
            logger.error(f"QuadS request error: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"QuadS HTTP error: {e.response.status_code} - {e.response.text[:200]}")
            raise
    
    def _quads_url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        # normalize duplicate slashes
        if not path.startswith("/"):
            path = "/" + path
        base = str(settings.QUADS_BASE_URL).rstrip('/')
        return f"{base}{path}"
    
    def _load_cookie_from_disk(self) -> bool:
        try:
            import json
            with open(settings.QUADS_SESSION_STATE_PATH, "r") as f:
                data = json.load(f)
            self.cookie_cache = data
            return settings.SESSION_COOKIE_NAME in data
        except FileNotFoundError:
            logger.debug("No saved QuadS session found")
            return False
        except Exception as e:
            logger.error(f"Error loading QuadS session: {e}")
            return False

    def _save_cookie_to_disk(self) -> None:
        if not self.cookie_cache:
            return
        try:
            import json
            with open(settings.QUADS_SESSION_STATE_PATH, "w") as f:
                json.dump(self.cookie_cache, f)
            logger.debug("QuadS session saved to disk")
        except Exception as e:
            logger.error(f"Error saving QuadS session: {e}")

    def _attach_cookie_to_client(self) -> None:
        if not self.cookie_cache:
            return
        cookie_value = self.cookie_cache.get(settings.SESSION_COOKIE_NAME)
        if cookie_value:
            self.client.cookies.set(
                settings.SESSION_COOKIE_NAME,
                cookie_value,
                domain=httpx.URL(str(settings.QUADS_BASE_URL)).host,
                path="/",
            )
            logger.debug("QuadS cookie attached to client")

    def _form_login(self) -> None:
        """Perform form-based login to QuadS to get session cookie."""
        # Step 1: GET login page to pick up CSRF and initial cookies
        try:
            r = self.client.get(str(settings.QUADS_LOGIN_URL))
            r.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to load QuadS login page: {e}")
            raise
        
        csrf_value = None
        if settings.ERP_CSRF_QUERY_SELECTOR:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(r.text, "html.parser")
                el = soup.select_one(settings.ERP_CSRF_QUERY_SELECTOR)
                if el and el.has_attr("value"):
                    csrf_value = el["value"]
                    logger.debug(f"Found QuadS CSRF token: {csrf_value[:10]}...")
            except Exception as e:
                logger.debug(f"No QuadS CSRF token found: {e}")

        # Step 2: POST credentials
        form = {
            settings.ERP_USER_FIELD: settings.ERP_USERNAME,
            settings.ERP_PASS_FIELD: settings.ERP_PASSWORD,
        }
        if csrf_value:
            form[settings.ERP_CSRF_FIELD_NAME] = csrf_value

        logger.info(f"Logging into QuadS as {settings.ERP_USERNAME}...")
        
        try:
            r2 = self.client.post(
                str(settings.QUADS_LOGIN_URL), 
                data=form, 
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            # Check if login was successful
            if r2.status_code >= 400:
                logger.error(f"QuadS login failed with status {r2.status_code}")
                logger.debug(f"Response: {r2.text[:500]}")
                raise RuntimeError(f"QuadS login failed: {r2.status_code}")
                
        except Exception as e:
            logger.error(f"QuadS login request failed: {e}")
            raise

        # Extract session cookie
        cookie_val = self.client.cookies.get(settings.SESSION_COOKIE_NAME)
        if not cookie_val:
            # Sometimes the cookie is in Set-Cookie headers
            for cookie in r2.cookies:
                if cookie.name == settings.SESSION_COOKIE_NAME:
                    cookie_val = cookie.value
                    break
        
        if not cookie_val:
            logger.error("QuadS login succeeded but session cookie not found.")
            logger.debug(f"Cookies: {self.client.cookies}")
            raise RuntimeError("QuadS login succeeded but session cookie not found.")
        
        logger.info(f"QuadS login successful, got session cookie: {cookie_val[:10]}...")
        self.cookie_cache = {settings.SESSION_COOKIE_NAME: cookie_val}

# Global QuadS session manager instance
quads_session_mgr = QuadSSessionManager()

def quads_get_json(api_path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Calls QuadS upstream JSON endpoint and returns parsed JSON.
    Automatically refreshes session if expired.
    """
    if not api_path.startswith(settings.QUADS_API_PREFIX):
        # guardrail: only allow calls under the configured API prefix
        raise ValueError(f"QuadS path must start with {settings.QUADS_API_PREFIX}")
    
    r = quads_session_mgr.call_with_refresh("GET", api_path, params=params)
    
    if "application/json" in r.headers.get("Content-Type", ""):
        return r.json()
    
    # Try to parse as JSON anyway
    try:
        return r.json()
    except:
        return r.text  # fallback if endpoint sends other content types

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
    """Fetch knit orders from eFab using the correct endpoint."""
    try:
        # Use the correct eFab endpoint for knit orders
        data = erp_get_json("/api/knitorder/list")
        if isinstance(data, list):
            logger.info(f"Fetched {len(data)} knit orders")
            return data
        elif isinstance(data, dict) and 'data' in data:
            logger.info(f"Fetched {len(data['data'])} knit orders")
            return data['data']
        else:
            logger.warning(f"Unexpected knit orders response type: {type(data)}")
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
                    # Map warehouse to correct endpoint
                    if wh == "F01":
                        data = erp_get_json("/api/finished/f01")
                    elif wh == "G00":
                        data = erp_get_json("/api/greige/g00")  
                    elif wh == "G02":
                        data = erp_get_json("/api/greige/g02")
                    elif wh == "I01":
                        data = erp_get_json("/api/finished/i01")
                    else:
                        continue
                    if isinstance(data, list):
                        for item in data:
                            item['warehouse'] = wh
                        all_inventory.extend(data)
                        logger.info(f"Fetched {len(data)} items from warehouse {wh}")
                except Exception as e:
                    logger.warning(f"Failed to fetch inventory for {wh}: {e}")
            
            return all_inventory
        else:
            # Map warehouse to correct endpoint
            if warehouse == "F01":
                data = erp_get_json("/api/finished/f01")
            elif warehouse == "G00":
                data = erp_get_json("/api/greige/g00")  
            elif warehouse == "G02":
                data = erp_get_json("/api/greige/g02")
            elif warehouse == "I01":
                data = erp_get_json("/api/finished/i01")
            else:
                return []
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

def get_greige_styles() -> List[Dict[str, Any]]:
    """Fetch greige styles from QuadS."""
    try:
        data = quads_get_json("/api/styles/greige/active")
        if isinstance(data, list):
            logger.info(f"Fetched {len(data)} greige styles")
            return data
        elif isinstance(data, dict) and 'data' in data:
            logger.info(f"Fetched {len(data['data'])} greige styles")
            return data['data']
        else:
            logger.warning(f"Unexpected greige styles response type: {type(data)}")
            return []
    except Exception as e:
        logger.error(f"Error fetching greige styles: {e}")
        raise

def get_finished_styles() -> List[Dict[str, Any]]:
    """Fetch finished styles from QuadS."""
    try:
        data = quads_get_json("/api/styles/finished/active")
        if isinstance(data, list):
            logger.info(f"Fetched {len(data)} finished styles")
            return data
        elif isinstance(data, dict) and 'data' in data:
            logger.info(f"Fetched {len(data['data'])} finished styles")
            return data['data']
        else:
            logger.warning(f"Unexpected finished styles response type: {type(data)}")
            return []
    except Exception as e:
        logger.error(f"Error fetching finished styles: {e}")
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

def test_quads_connection() -> bool:
    """Test if we can connect to the QuadS API."""
    try:
        # Try to fetch a simple endpoint
        data = quads_get_json("/api/styles/greige/active")
        return data is not None
    except Exception as e:
        logger.error(f"QuadS connection test failed: {e}")
        return False