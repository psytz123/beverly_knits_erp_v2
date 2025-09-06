from pydantic import BaseSettings, AnyHttpUrl
from typing import Optional

class Settings(BaseSettings):
    # ERP endpoints
    ERP_BASE_URL: AnyHttpUrl = "https://efab.bklapps.com"
    ERP_LOGIN_URL: AnyHttpUrl = "https://efab.bklapps.com/login"   # adjust if different
    ERP_API_PREFIX: str = "/api"                                   # upstream API root

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

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # SSL Verification (set to False if using self-signed certs)
    VERIFY_SSL: bool = True
    
    # Request timeout
    REQUEST_TIMEOUT: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()