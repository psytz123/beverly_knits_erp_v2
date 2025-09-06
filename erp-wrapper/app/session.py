import json
import time
import httpx
from bs4 import BeautifulSoup
from typing import Dict, Optional, Any
from .config import settings
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Maintains an authenticated httpx.Client with ERP session cookie.
    - Loads/stores cookie to disk
    - Re-logins on 401 automatically (caller triggers via call_with_refresh)
    """

    def __init__(self):
        self.client: Optional[httpx.Client] = None
        self.cookie_cache: Optional[Dict[str, str]] = None
        self._ensure_client()

    # ---------- Public API ----------

    def call_with_refresh(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Perform a request against the ERP, refreshing the session on 401."""
        url = self._erp_url(path)
        
        # Remove verify from kwargs if present, use settings
        kwargs.pop('verify', None)
        
        try:
            r = self.client.request(method, url, **kwargs)
            
            if r.status_code in (401, 403):
                # session likely expired -> relogin and retry once
                logger.info(f"Got {r.status_code}, refreshing session...")
                self.login(force=True)
                r = self.client.request(method, url, **kwargs)
            
            r.raise_for_status()
            return r
            
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text[:200]}")
            raise

    def login(self, force: bool = False) -> None:
        """Ensure we are logged in: tries cookie-from-disk, otherwise form-login."""
        if not force and self._load_cookie_from_disk():
            self._attach_cookie_to_client()
            # quick probe to test if cookie is still valid
            try:
                probe_path = f"{settings.ERP_API_PREFIX}/sales-order/plan/list"
                probe = self.client.get(self._erp_url(probe_path))
                if probe.status_code < 400:
                    logger.info("Using cached session cookie")
                    return
            except Exception as e:
                logger.debug(f"Cookie probe failed: {e}")
        
        logger.info("Performing fresh login...")
        self._form_login()
        self._save_cookie_to_disk()

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        return {
            "has_cookie": self.cookie_cache is not None,
            "cookie_name": settings.SESSION_COOKIE_NAME,
            "base_url": str(settings.ERP_BASE_URL),
            "client_active": self.client is not None
        }

    # ---------- Internals ----------

    def _ensure_client(self) -> None:
        if self.client is None:
            self.client = httpx.Client(
                base_url=str(settings.ERP_BASE_URL),
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

    def _erp_url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        # normalize duplicate slashes
        if not path.startswith("/"):
            path = "/" + path
        base = str(settings.ERP_BASE_URL).rstrip('/')
        return f"{base}{path}"

    def _load_cookie_from_disk(self) -> bool:
        try:
            with open(settings.SESSION_STATE_PATH, "r") as f:
                data = json.load(f)
            self.cookie_cache = data
            return settings.SESSION_COOKIE_NAME in data
        except FileNotFoundError:
            logger.debug("No saved session found")
            return False
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return False

    def _save_cookie_to_disk(self) -> None:
        if not self.cookie_cache:
            return
        try:
            with open(settings.SESSION_STATE_PATH, "w") as f:
                json.dump(self.cookie_cache, f)
            logger.debug("Session saved to disk")
        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def _attach_cookie_to_client(self) -> None:
        if not self.cookie_cache:
            return
        cookie_value = self.cookie_cache.get(settings.SESSION_COOKIE_NAME)
        if cookie_value:
            self.client.cookies.set(
                settings.SESSION_COOKIE_NAME,
                cookie_value,
                domain=self._extract_domain(str(settings.ERP_BASE_URL)),
                path="/",
            )
            logger.debug("Cookie attached to client")

    def _extract_domain(self, url: str) -> str:
        return httpx.URL(url).host

    def _form_login(self) -> None:
        """Perform form-based login to get session cookie."""
        # Step 1: GET login page to pick up CSRF and initial cookies
        try:
            r = self.client.get(str(settings.ERP_LOGIN_URL))
            r.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to load login page: {e}")
            raise
        
        csrf_value = None
        if settings.ERP_CSRF_QUERY_SELECTOR:
            try:
                soup = BeautifulSoup(r.text, "html.parser")
                el = soup.select_one(settings.ERP_CSRF_QUERY_SELECTOR)
                if el and el.has_attr("value"):
                    csrf_value = el["value"]
                    logger.debug(f"Found CSRF token: {csrf_value[:10]}...")
            except Exception as e:
                logger.debug(f"No CSRF token found: {e}")

        # Step 2: POST credentials
        form = {
            settings.ERP_USER_FIELD: settings.ERP_USERNAME,
            settings.ERP_PASS_FIELD: settings.ERP_PASSWORD,
        }
        if csrf_value:
            form[settings.ERP_CSRF_FIELD_NAME] = csrf_value

        logger.info(f"Logging in as {settings.ERP_USERNAME}...")
        
        try:
            r2 = self.client.post(
                str(settings.ERP_LOGIN_URL), 
                data=form, 
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            # Check if login was successful
            if r2.status_code >= 400:
                logger.error(f"Login failed with status {r2.status_code}")
                logger.debug(f"Response: {r2.text[:500]}")
                raise RuntimeError(f"Login failed: {r2.status_code}")
                
        except Exception as e:
            logger.error(f"Login request failed: {e}")
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
            logger.error("Login succeeded but session cookie not found.")
            logger.debug(f"Cookies: {self.client.cookies}")
            raise RuntimeError("Login succeeded but session cookie not found.")
        
        logger.info(f"Login successful, got session cookie: {cookie_val[:10]}...")
        self.cookie_cache = {settings.SESSION_COOKIE_NAME: cookie_val}