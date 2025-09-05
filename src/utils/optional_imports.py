"""
Optional Import Wrapper
Handles missing optional dependencies gracefully
"""

import warnings
import sys
from typing import Any, Optional


class OptionalDependency:
    """Placeholder for optional dependencies"""
    
    def __init__(self, name: str, install_cmd: str):
        self.name = name
        self.install_cmd = install_cmd
        self.warning_shown = False
    
    def __getattr__(self, item):
        if not self.warning_shown:
            warnings.warn(
                f"Optional dependency '{self.name}' not installed. "
                f"Install with: {self.install_cmd}",
                UserWarning,
                stacklevel=2
            )
            self.warning_shown = True
        return lambda *args, **kwargs: None
    
    def __call__(self, *args, **kwargs):
        return None


def import_optional(module_name: str, install_cmd: str = None) -> Any:
    """
    Import optional module with graceful fallback
    
    Args:
        module_name: Name of module to import
        install_cmd: Installation command if module not found
        
    Returns:
        Module or placeholder
    """
    try:
        return __import__(module_name)
    except ImportError:
        if not install_cmd:
            install_cmd = f"pip install {module_name}"
        return OptionalDependency(module_name, install_cmd)


# Common optional imports with proper fallbacks
try:
    import seaborn as sns
except ImportError:
    sns = OptionalDependency("seaborn", "pip install seaborn")
    
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = OptionalDependency("matplotlib", "pip install matplotlib")

try:
    from office365.sharepoint.client_context import ClientContext
except ImportError:
    ClientContext = OptionalDependency(
        "Office365-REST-Python-Client", 
        "pip install Office365-REST-Python-Client"
    )


# Export for use
__all__ = ['import_optional', 'sns', 'plt', 'ClientContext']
