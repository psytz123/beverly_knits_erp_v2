from pydantic import BaseModel, AnyHttpUrl
from typing import Optional
import os

class Settings(BaseModel):
    # ERP endpoints
    ERP_BASE_URL: AnyHttpUrl = "https://efab.bkiapps.com"
    ERP_LOGIN_URL: AnyHttpUrl = "https://efab.bkiapps.com/login"   # adjust if different
    ERP_API_PREFIX: str = "/api"                                   # upstream API root
    
    # QuadS endpoints
    QUADS_BASE_URL: AnyHttpUrl = "https://quads.bkiapps.com"
    QUADS_LOGIN_URL: AnyHttpUrl = "https://quads.bkiapps.com/login"
    QUADS_API_PREFIX: str = "/api"

    # Login form field names (typical for Dancer/backends; change if needed)
    ERP_USER_FIELD: str = "username"
    ERP_PASS_FIELD: str = "password"
    ERP_CSRF_QUERY_SELECTOR: str = "input[name=csrf_token]"        # if not used, leave as-is
    ERP_CSRF_FIELD_NAME: str = "csrf_token"

    # Credentials (use a service account)
    ERP_USERNAME: str = "psytz"
    ERP_PASSWORD: str = "big$cat"

    # Cookie settings
    SESSION_COOKIE_NAME: str = "dancer.session"
    SESSION_STATE_PATH: str = "/tmp/erp_session.json"
    QUADS_SESSION_STATE_PATH: str = "/tmp/quads_session.json"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # SSL Verification (set to False if using self-signed certs)
    VERIFY_SSL: bool = True
    
    # Request timeout
    REQUEST_TIMEOUT: int = 60

    @classmethod
    def load_from_env(cls):
        """Load settings from environment variables and .env file"""
        from dotenv import load_dotenv
        load_dotenv()
        
        return cls(
            ERP_BASE_URL=os.getenv("ERP_BASE_URL", "https://efab.bkiapps.com"),
            ERP_LOGIN_URL=os.getenv("ERP_LOGIN_URL", "https://efab.bkiapps.com/login"),
            ERP_API_PREFIX=os.getenv("ERP_API_PREFIX", "/api"),
            QUADS_BASE_URL=os.getenv("QUADS_BASE_URL", "https://quads.bkiapps.com"),
            QUADS_LOGIN_URL=os.getenv("QUADS_LOGIN_URL", "https://quads.bkiapps.com/login"),
            QUADS_API_PREFIX=os.getenv("QUADS_API_PREFIX", "/api"),
            ERP_USER_FIELD=os.getenv("ERP_USER_FIELD", "username"),
            ERP_PASS_FIELD=os.getenv("ERP_PASS_FIELD", "password"),
            ERP_CSRF_QUERY_SELECTOR=os.getenv("ERP_CSRF_QUERY_SELECTOR", "input[name=csrf_token]"),
            ERP_CSRF_FIELD_NAME=os.getenv("ERP_CSRF_FIELD_NAME", "csrf_token"),
            ERP_USERNAME=os.getenv("ERP_USERNAME", "psytz"),
            ERP_PASSWORD=os.getenv("ERP_PASSWORD", "big$cat"),
            SESSION_COOKIE_NAME=os.getenv("SESSION_COOKIE_NAME", "dancer.session"),
            SESSION_STATE_PATH=os.getenv("SESSION_STATE_PATH", "/tmp/erp_session.json"),
            QUADS_SESSION_STATE_PATH=os.getenv("QUADS_SESSION_STATE_PATH", "/tmp/quads_session.json"),
            HOST=os.getenv("HOST", "0.0.0.0"),
            PORT=int(os.getenv("PORT", "8000")),
            VERIFY_SSL=os.getenv("VERIFY_SSL", "True").lower() == "true",
            REQUEST_TIMEOUT=int(os.getenv("REQUEST_TIMEOUT", "60"))
        )

settings = Settings.load_from_env()