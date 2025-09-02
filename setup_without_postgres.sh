#!/bin/bash
# Setup script for Beverly Knits ERP without PostgreSQL
# This script configures the system to run in file-only mode

echo "=========================================="
echo "Beverly Knits ERP Setup (File-Based Mode)"
echo "=========================================="

# Update configuration to use file-based mode only
cat > src/config/unified_config.json << 'EOF'
{
  "data_source": {
    "primary": "files",
    "fallback": "files",
    "enable_dual_source": false,
    "database": {
      "enabled": false,
      "note": "PostgreSQL not installed - using file-based mode"
    },
    "files": {
      "primary_path": "/mnt/c/Users/psytz/sc data/ERP Data",
      "legacy_paths": [
        "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/prompts/5",
        "C:/finalee/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/sharepoint_sync"
      ],
      "batch_size": 1000
    }
  },
  "api": {
    "port": 5000,
    "debug": true,
    "cors_enabled": true,
    "api_versions": {
      "v1": "file_based",
      "v2": "file_based"
    }
  },
  "integration": {
    "enable_postgresql": false,
    "enable_file_loading": true,
    "sync_interval_minutes": 15,
    "data_validation": true
  }
}
EOF

echo "✓ Configuration updated for file-based mode"

# Check if data directory exists
if [ -d "/mnt/c/Users/psytz/sc data/ERP Data" ]; then
    echo "✓ Data directory found"
    ls -la "/mnt/c/Users/psytz/sc data/ERP Data" | head -5
else
    echo "⚠ Data directory not found at /mnt/c/Users/psytz/sc data/ERP Data"
    echo "  Please ensure your data files are in this location"
fi

# Install Python dependencies (excluding PostgreSQL)
echo ""
echo "Installing Python dependencies..."
pip3 install flask flask-cors pandas numpy openpyxl xlrd xlsxwriter 2>/dev/null || {
    echo "⚠ Some dependencies may need manual installation"
}

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "The system is now configured to run in FILE-BASED MODE"
echo "This means:"
echo "  ✓ No PostgreSQL required"
echo "  ✓ Data loaded directly from Excel/CSV files"
echo "  ✓ All features available except database queries"
echo ""
echo "To start the system:"
echo "  cd /mnt/c/finalee/beverly_knits_erp_v2"
echo "  python3 src/core/beverly_comprehensive_erp.py"
echo ""
echo "Data location: /mnt/c/Users/psytz/sc data/ERP Data"