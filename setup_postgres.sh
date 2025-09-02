#!/bin/bash
# PostgreSQL Setup Script for Beverly Knits ERP

echo "======================================"
echo "PostgreSQL Setup for Beverly Knits ERP"
echo "======================================"
echo ""

# Create database and user
echo "Creating database and setting up authentication..."

# Create the database if it doesn't exist
sudo -u postgres psql <<EOF
-- Create database
CREATE DATABASE beverly_knits_erp;

-- Create user with password
CREATE USER erp_user WITH PASSWORD 'erp_password';

-- Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE beverly_knits_erp TO erp_user;

-- Also allow postgres user (for development)
\c beverly_knits_erp
GRANT ALL ON SCHEMA public TO erp_user;
GRANT ALL ON SCHEMA public TO postgres;

\q
EOF

echo "✓ Database 'beverly_knits_erp' created"
echo "✓ User 'erp_user' created with password 'erp_password'"

# Update configuration files
echo ""
echo "Updating configuration files..."

# Update database config with new credentials
cat > /mnt/c/finalee/beverly_knits_erp_v2/src/database/database_config.json <<EOF
{
    "host": "localhost",
    "port": 5432,
    "database": "beverly_knits_erp",
    "user": "erp_user",
    "password": "erp_password",
    "data_path": "/mnt/c/Users/psytz/sc data/ERP Data",
    "api_port": 5007,
    "cache_enabled": true,
    "cache_ttl": 300,
    "batch_size": 1000,
    "connection_pool_size": 10
}
EOF

# Update unified config to enable PostgreSQL
cat > /mnt/c/finalee/beverly_knits_erp_v2/src/config/unified_config.json <<EOF
{
  "data_source": {
    "primary": "database",
    "fallback": "files",
    "enable_dual_source": true,
    "database": {
      "host": "localhost",
      "port": 5432,
      "database": "beverly_knits_erp",
      "user": "erp_user",
      "password": "erp_password",
      "connection_pool_size": 10,
      "api_port": 5007,
      "cache_enabled": true,
      "cache_ttl": 300
    },
    "files": {
      "primary_path": "/mnt/c/Users/psytz/sc data/ERP Data",
      "legacy_paths": [],
      "batch_size": 1000
    }
  },
  "api": {
    "port": 5000,
    "debug": true,
    "cors_enabled": true,
    "api_versions": {
      "v1": "file_based",
      "v2": "database_based"
    }
  },
  "etl": {
    "enabled": true,
    "schedule": "0 6 * * *",
    "source_directory": "/mnt/c/Users/psytz/sc data/ERP Data",
    "log_level": "INFO"
  },
  "integration": {
    "enable_postgresql": true,
    "enable_file_loading": true,
    "sync_interval_minutes": 15,
    "data_validation": true
  }
}
EOF

echo "✓ Configuration files updated"

# Run the database setup SQL
echo ""
echo "Setting up database schema..."

sudo -u postgres psql -d beverly_knits_erp < /mnt/c/finalee/beverly_knits_erp_v2/src/database/setup.sql 2>/dev/null || {
    echo "⚠ Database schema setup will be run during ETL"
}

echo ""
echo "======================================"
echo "PostgreSQL Setup Complete!"
echo "======================================"
echo ""
echo "Database Credentials:"
echo "  Database: beverly_knits_erp"
echo "  User:     erp_user"
echo "  Password: erp_password"
echo "  Host:     localhost"
echo "  Port:     5432"
echo ""
echo "Next steps:"
echo "1. Run ETL to load data:"
echo "   python3 src/data_sync/database_etl_pipeline.py"
echo ""
echo "2. Test the connection:"
echo "   python3 test_integration.py"
echo ""
echo "3. Start the API server:"
echo "   python3 src/api/database_api_server.py"