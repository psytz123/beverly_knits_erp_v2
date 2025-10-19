#!/usr/bin/env python3
"""
Manufacturing ERP System - Industry-Agnostic Supply Chain AI
Full-featured supply chain optimization with ML forecasting, multi-level BOM explosion,
procurement optimization, and intelligent inventory management for any manufacturing industry
"""


# Day 0 Emergency Fixes - Added 2025-09-02
try:
    from scripts.day0_emergency_fixes import (
        DynamicPathResolver,
        ColumnAliasSystem,
        PriceStringParser,
        RealKPICalculator,
        MultiLevelBOMNetting,
        EmergencyFixManager,
    )

    DAY0_FIXES_AVAILABLE = True
    print("[DAY0] Emergency fixes loaded successfully")
except ImportError as e:
    print(f"[DAY0] Emergency fixes not available: {e}")
    DAY0_FIXES_AVAILABLE = False

import sys
import os

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import (
    Flask,
    jsonify,
    render_template_string,
    request,
    send_file,
    send_from_directory,
    Response,
    redirect,
    url_for,
)

try:
    from flask_cors import CORS

    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Flask-CORS not available, CORS support disabled")
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime, timedelta
import json
from collections import defaultdict
import warnings
import io
import base64
from functools import lru_cache, wraps
import logging
import traceback
import math
import sqlite3
import requests

warnings.filterwarnings("ignore")

# Import feature flags for API consolidation
try:
    from config.feature_flags import (
        FEATURE_FLAGS,
        get_feature_flag,
        should_redirect_deprecated,
        should_log_deprecated_usage,
        is_consolidation_enabled,
    )

    FEATURE_FLAGS_AVAILABLE = True
except ImportError:
    FEATURE_FLAGS_AVAILABLE = False
    print("Feature flags not available, API consolidation disabled")

# Import Column Standardizer for flexible column detection
try:
    from utils.column_standardization import ColumnStandardizer
except ImportError:
    try:
        from src.utils.column_standardization import ColumnStandardizer
    except ImportError:
        print("ColumnStandardizer not available, using fallback column detection")
        ColumnStandardizer = None

# Import Cache Manager for performance optimization
try:
    from utils.cache_manager import CacheManager

    CACHE_MANAGER_AVAILABLE = True
except ImportError:
    CACHE_MANAGER_AVAILABLE = False
