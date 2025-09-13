from typing import Optional
import os

class Settings:
    def __init__(self):
        # ERP endpoints
        self.ERP_BASE_URL = "https://efab.bkiapps.com"
        self.ERP_LOGIN_URL = "https://efab.bkiapps.com/login"
        self.ERP_API_PREFIX = "/api"

        # QuadS endpoints
        self.QUADS_BASE_URL = "https://quads.bkiapps.com"
        self.QUADS_LOGIN_URL = "https://quads.bkiapps.com/login"
        self.QUADS_API_PREFIX = "/api"

        # Login form field names (typical for Dancer/backends; change if needed)
        self.ERP_USER_FIELD = "username"
        self.ERP_PASS_FIELD = "password"
        self.ERP_CSRF_QUERY_SELECTOR = "input[name=csrf_token]"
        self.ERP_CSRF_FIELD_NAME = "csrf_token"

        # Credentials (use a service account)
        self.ERP_USERNAME = "psytz"
        self.ERP_PASSWORD = "big$cat"

        # Cookie settings
        self.SESSION_COOKIE_NAME = "dancer.session"
        self.SESSION_STATE_PATH = "/tmp/erp_session.json"
        self.QUADS_SESSION_STATE_PATH = "/tmp/quads_session.json"

        # Server
        self.HOST = "0.0.0.0"
        self.PORT = 8000

        # SSL Verification (set to False if using self-signed certs)
        self.VERIFY_SSL = True

        # Request timeout
        self.REQUEST_TIMEOUT = 60

    @classmethod
    def load_from_env(cls):
        """Load settings from environment variables and .env file"""
        from dotenv import load_dotenv
        load_dotenv()

        instance = cls()

        # Load from environment variables, keep defaults if not set
        instance.ERP_BASE_URL = os.getenv("ERP_BASE_URL", instance.ERP_BASE_URL)
        instance.ERP_LOGIN_URL = os.getenv("ERP_LOGIN_URL", instance.ERP_LOGIN_URL)
        instance.ERP_API_PREFIX = os.getenv("ERP_API_PREFIX", instance.ERP_API_PREFIX)
        instance.QUADS_BASE_URL = os.getenv("QUADS_BASE_URL", instance.QUADS_BASE_URL)
        instance.QUADS_LOGIN_URL = os.getenv("QUADS_LOGIN_URL", instance.QUADS_LOGIN_URL)
        instance.QUADS_API_PREFIX = os.getenv("QUADS_API_PREFIX", instance.QUADS_API_PREFIX)
        instance.ERP_USER_FIELD = os.getenv("ERP_USER_FIELD", instance.ERP_USER_FIELD)
        instance.ERP_PASS_FIELD = os.getenv("ERP_PASS_FIELD", instance.ERP_PASS_FIELD)
        instance.ERP_CSRF_QUERY_SELECTOR = os.getenv("ERP_CSRF_QUERY_SELECTOR", instance.ERP_CSRF_QUERY_SELECTOR)
        instance.ERP_CSRF_FIELD_NAME = os.getenv("ERP_CSRF_FIELD_NAME", instance.ERP_CSRF_FIELD_NAME)
        instance.ERP_USERNAME = os.getenv("ERP_USERNAME", instance.ERP_USERNAME)
        instance.ERP_PASSWORD = os.getenv("ERP_PASSWORD", instance.ERP_PASSWORD)
        instance.SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", instance.SESSION_COOKIE_NAME)
        instance.SESSION_STATE_PATH = os.getenv("SESSION_STATE_PATH", instance.SESSION_STATE_PATH)
        instance.QUADS_SESSION_STATE_PATH = os.getenv("QUADS_SESSION_STATE_PATH", instance.QUADS_SESSION_STATE_PATH)
        instance.HOST = os.getenv("HOST", instance.HOST)
        instance.PORT = int(os.getenv("PORT", str(instance.PORT)))
        instance.VERIFY_SSL = os.getenv("VERIFY_SSL", "True").lower() == "true"
        instance.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", str(instance.REQUEST_TIMEOUT)))

        return instance

settings = Settings.load_from_env()