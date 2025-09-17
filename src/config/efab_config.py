#!/usr/bin/env python3
"""
EFab Integration Configuration
Minimal configuration for Yarn Demand report scheduler
Following Operating Charter: Document everything, minimal approach
"""

import os
from typing import Dict, List, Any

# EFab API Configuration
EFAB_CONFIG: Dict[str, Any] = {
    # Authentication
    'session_cookie': os.environ.get('EFAB_SESSION', 'aMdcwNLa0ov0pcbWcQ_zb5wyPLSkYF_B'),

    # API Endpoints
    'base_url': 'https://efab.bkiapps.com',
    'report_queue_endpoint': '/api/report/report_queue',

    # Schedule Configuration
    'refresh_times': ['10:00', '12:00'],  # 10 AM and 12 PM
    'timezone': 'America/New_York',  # Adjust as needed

    # Report Settings
    'report_type': 'yarn_demand',
    'target_filename': 'Expected_Yarn_Report.xlsx',

    # Retry Configuration
    'max_retries': 3,
    'retry_delay_seconds': 30,

    # Feature Flags
    'enable_scheduler': os.environ.get('ENABLE_YARN_SCHEDULER', 'true').lower() == 'true',
    'enable_archiving': True,  # Archive old reports
    'archive_days': 30,  # Keep archives for 30 days
}

def get_efab_config(key: str, default: Any = None) -> Any:
    """
    Get configuration value by key

    Args:
        key: Configuration key
        default: Default value if key not found

    Returns:
        Configuration value
    """
    return EFAB_CONFIG.get(key, default)

def update_session_cookie(new_cookie: str) -> None:
    """
    Update session cookie dynamically

    Args:
        new_cookie: New dancer.session cookie value
    """
    EFAB_CONFIG['session_cookie'] = new_cookie
    os.environ['EFAB_SESSION'] = new_cookie

def is_scheduler_enabled() -> bool:
    """Check if scheduler is enabled"""
    return EFAB_CONFIG.get('enable_scheduler', False)

def get_refresh_schedule() -> List[str]:
    """Get scheduled refresh times"""
    return EFAB_CONFIG.get('refresh_times', ['10:00', '12:00'])


# Validation on module load
if __name__ == "__main__":
    print("EFab Configuration:")
    print(f"  Scheduler Enabled: {is_scheduler_enabled()}")
    print(f"  Refresh Times: {get_refresh_schedule()}")
    print(f"  Session Cookie: {'Set' if EFAB_CONFIG['session_cookie'] else 'Not Set'}")
    print(f"  Base URL: {EFAB_CONFIG['base_url']}")