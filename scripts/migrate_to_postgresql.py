#!/usr/bin/env python3
"""
Beverly Knits ERP - PostgreSQL Migration Script
Phase 1 Day 3-4: Migrate from SQLite to PostgreSQL with connection pooling
CRITICAL: Must preserve ALL business logic
"""

import os
import sys
import sqlite3
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, List
import hashlib
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """PostgreSQL database manager with connection pooling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool = None
        self._init_pool()
        
    def _init_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.get('min_connections', 5),
                maxconn=self.config.get('max_connections', 20),
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 5432),
                database=self.config.get('database', 'beverly_erp'),
                user=self.config.get('user', 'beverly'),
                password=self.config.get('password', 'beverly_secure_pass_2025')
            )
            logger.info(f"Connection pool created: {self.config.get('min_connections', 5)}-{self.config.get('max_connections', 20)} connections")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic cleanup"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def close_all(self):
        """Close all connections in pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("All connections closed")

class PostgreSQLMigrator:
    """Migrate Beverly Knits ERP from SQLite to PostgreSQL"""
    
    def __init__(self, sqlite_path: str, pg_config: Dict[str, Any]):
        self.sqlite_path = sqlite_path
        self.pg_config = pg_config
        self.db_manager = None
        self.migration_log = []
        self.backup_path = None
        
    def create_backup(self):
        """Create backup of SQLite database before migration"""
        try:
            backup_dir = Path("backups/database")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_path = backup_dir / f"beverly_erp_backup_{timestamp}.db"
            
            # Copy SQLite database
            import shutil
            if Path(self.sqlite_path).exists():
                shutil.copy2(self.sqlite_path, self.backup_path)
                logger.info(f"Backup created: {self.backup_path}")
                
                # Calculate checksum for verification
                with open(self.backup_path, 'rb') as f:
                    checksum = hashlib.md5(f.read()).hexdigest()
                self.migration_log.append({
                    "action": "backup_created",
                    "path": str(self.backup_path),
                    "checksum": checksum,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                logger.warning(f"SQLite database not found at {self.sqlite_path}")
                
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def create_postgresql_schema(self):
        """Create PostgreSQL schema with optimized indexes"""
        
        schema_sql = """
        -- Create schema if not exists
        CREATE SCHEMA IF NOT EXISTS erp;
        
        -- Yarn Inventory table (critical for Planning Balance)
        CREATE TABLE IF NOT EXISTS erp.yarn_inventory (
            id SERIAL PRIMARY KEY,
            "Desc#" VARCHAR(100) NOT NULL,  -- Yarn ID
            description TEXT,
            planning_balance DECIMAL(12,2),  -- CRITICAL field
            theoretical_balance DECIMAL(12,2),
            allocated DECIMAL(12,2),  -- Already negative in source
            on_order DECIMAL(12,2),
            consumed DECIMAL(12,2),
            unit_price DECIMAL(10,4),
            last_updated TIMESTAMP DEFAULT NOW(),
            CONSTRAINT uk_yarn_desc UNIQUE ("Desc#")
        );
        
        -- Indexes for yarn inventory (critical for performance)
        CREATE INDEX IF NOT EXISTS idx_yarn_desc ON erp.yarn_inventory ("Desc#");
        CREATE INDEX IF NOT EXISTS idx_planning_balance ON erp.yarn_inventory (planning_balance);
        CREATE INDEX IF NOT EXISTS idx_allocated ON erp.yarn_inventory (allocated);
        
        -- BOM (Bill of Materials) table
        CREATE TABLE IF NOT EXISTS erp.bom (
            id SERIAL PRIMARY KEY,
            "Style#" VARCHAR(100) NOT NULL,  -- Style reference
            "fStyle#" VARCHAR(100),  -- Alternative style reference
            "Yarn_ID" VARCHAR(100) NOT NULL,  -- References Desc# in yarn_inventory
            quantity DECIMAL(12,4),
            unit VARCHAR(20),
            last_updated TIMESTAMP DEFAULT NOW()
        );
        
        -- Indexes for BOM
        CREATE INDEX IF NOT EXISTS idx_bom_style ON erp.bom ("Style#");
        CREATE INDEX IF NOT EXISTS idx_bom_fstyle ON erp.bom ("fStyle#");
        CREATE INDEX IF NOT EXISTS idx_bom_yarn ON erp.bom ("Yarn_ID");
        
        -- Sales Activity table
        CREATE TABLE IF NOT EXISTS erp.sales_activity (
            id SERIAL PRIMARY KEY,
            transaction_date DATE,
            style_number VARCHAR(100),
            customer_name VARCHAR(200),
            quantity DECIMAL(12,2),
            unit_price DECIMAL(10,4),
            line_price DECIMAL(12,2),
            consumed_date DATE,
            last_updated TIMESTAMP DEFAULT NOW()
        );
        
        -- Indexes for sales activity
        CREATE INDEX IF NOT EXISTS idx_sales_date ON erp.sales_activity (transaction_date);
        CREATE INDEX IF NOT EXISTS idx_sales_style ON erp.sales_activity (style_number);
        CREATE INDEX IF NOT EXISTS idx_sales_customer ON erp.sales_activity (customer_name);
        
        -- Production Orders table
        CREATE TABLE IF NOT EXISTS erp.production_orders (
            id SERIAL PRIMARY KEY,
            order_number VARCHAR(100) UNIQUE,
            style VARCHAR(100),
            quantity DECIMAL(12,2),
            machine_id VARCHAR(50),
            work_center VARCHAR(50),
            status VARCHAR(50),
            scheduled_date DATE,
            last_updated TIMESTAMP DEFAULT NOW()
        );
        
        -- Indexes for production orders
        CREATE INDEX IF NOT EXISTS idx_po_order ON erp.production_orders (order_number);
        CREATE INDEX IF NOT EXISTS idx_po_style ON erp.production_orders (style);
        CREATE INDEX IF NOT EXISTS idx_po_machine ON erp.production_orders (machine_id);
        CREATE INDEX IF NOT EXISTS idx_po_status ON erp.production_orders (status);
        
        -- Machine assignments table
        CREATE TABLE IF NOT EXISTS erp.machine_assignments (
            id SERIAL PRIMARY KEY,
            machine_id VARCHAR(50),
            work_center_pattern VARCHAR(50),  -- e.g., "9.38.20.F"
            machine_name VARCHAR(100),
            capacity_lbs_per_day DECIMAL(10,2),
            last_updated TIMESTAMP DEFAULT NOW()
        );
        
        -- Indexes for machine assignments
        CREATE INDEX IF NOT EXISTS idx_ma_machine ON erp.machine_assignments (machine_id);
        CREATE INDEX IF NOT EXISTS idx_ma_pattern ON erp.machine_assignments (work_center_pattern);
        
        -- ML Forecasts table (for tracking accuracy)
        CREATE TABLE IF NOT EXISTS erp.ml_forecasts (
            id SERIAL PRIMARY KEY,
            yarn_id VARCHAR(100),
            forecast_date DATE,
            forecast_horizon_weeks INT,
            predicted_demand DECIMAL(12,2),
            actual_demand DECIMAL(12,2),
            model_type VARCHAR(50),
            accuracy_score DECIMAL(5,4),
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Indexes for ML forecasts
        CREATE INDEX IF NOT EXISTS idx_forecast_yarn ON erp.ml_forecasts (yarn_id);
        CREATE INDEX IF NOT EXISTS idx_forecast_date ON erp.ml_forecasts (forecast_date);
        CREATE INDEX IF NOT EXISTS idx_forecast_model ON erp.ml_forecasts (model_type);
        
        -- Migration metadata table
        CREATE TABLE IF NOT EXISTS erp.migration_log (
            id SERIAL PRIMARY KEY,
            table_name VARCHAR(100),
            records_migrated INT,
            migration_status VARCHAR(50),
            error_message TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            checksum VARCHAR(32)
        );
        """
        
        try:
            self.db_manager = DatabaseManager(self.pg_config)
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(schema_sql)
                    conn.commit()
                    logger.info("PostgreSQL schema created successfully")
                    
                    # Verify tables were created
                    cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'erp'
                        ORDER BY table_name
                    """)
                    tables = cur.fetchall()
                    logger.info(f"Created {len(tables)} tables: {[t[0] for t in tables]}")
                    
                    self.migration_log.append({
                        "action": "schema_created",
                        "tables": [t[0] for t in tables],
                        "timestamp": datetime.now().isoformat()
                    })
                    
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL schema: {e}")
            raise
    
    def migrate_sqlite_data(self):
        """Migrate data from SQLite to PostgreSQL"""
        
        if not Path(self.sqlite_path).exists():
            logger.warning(f"SQLite database not found at {self.sqlite_path}")
            return
        
        try:
            # Connect to SQLite
            sqlite_conn = sqlite3.connect(self.sqlite_path)
            
            # Get list of tables
            cursor = sqlite_conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            logger.info(f"Found {len(tables)} tables in SQLite database")
            
            for table_name, in tables:
                self._migrate_table(sqlite_conn, table_name)
            
            sqlite_conn.close()
            
        except Exception as e:
            logger.error(f"Failed to migrate SQLite data: {e}")
            raise
    
    def _migrate_table(self, sqlite_conn: sqlite3.Connection, table_name: str):
        """Migrate a single table from SQLite to PostgreSQL"""
        
        try:
            logger.info(f"Migrating table: {table_name}")
            start_time = time.time()
            
            # Read data from SQLite
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)
            
            if df.empty:
                logger.warning(f"Table {table_name} is empty")
                return
            
            # Map table names if necessary
            pg_table_name = self._map_table_name(table_name)
            
            # Migrate data to PostgreSQL
            with self.db_manager.get_connection() as pg_conn:
                # Truncate table first (during migration only)
                with pg_conn.cursor() as cur:
                    cur.execute(f"TRUNCATE TABLE erp.{pg_table_name} CASCADE")
                
                # Insert data
                df.to_sql(
                    pg_table_name,
                    pg_conn,
                    schema='erp',
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                pg_conn.commit()
            
            elapsed = time.time() - start_time
            logger.info(f"Migrated {len(df)} records from {table_name} in {elapsed:.2f}s")
            
            self.migration_log.append({
                "action": "table_migrated",
                "table": table_name,
                "records": len(df),
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to migrate table {table_name}: {e}")
            self.migration_log.append({
                "action": "table_migration_failed",
                "table": table_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    def _map_table_name(self, sqlite_table: str) -> str:
        """Map SQLite table names to PostgreSQL table names"""
        
        mappings = {
            'yarn': 'yarn_inventory',
            'styles': 'bom',
            'orders': 'production_orders',
            'sales': 'sales_activity',
            'machines': 'machine_assignments',
            'forecasts': 'ml_forecasts'
        }
        
        # Check if mapping exists
        for key, value in mappings.items():
            if key in sqlite_table.lower():
                return value
        
        # Return original name if no mapping found
        return sqlite_table.lower().replace(' ', '_')
    
    def verify_migration(self):
        """Verify data integrity after migration"""
        
        logger.info("Verifying migration integrity...")
        
        verification_queries = [
            ("Yarn inventory count", "SELECT COUNT(*) FROM erp.yarn_inventory"),
            ("BOM count", "SELECT COUNT(*) FROM erp.bom"),
            ("Sales activity count", "SELECT COUNT(*) FROM erp.sales_activity"),
            ("Planning balance check", """
                SELECT COUNT(*) 
                FROM erp.yarn_inventory 
                WHERE planning_balance = theoretical_balance + allocated + on_order
            """),
            ("Negative allocated check", """
                SELECT COUNT(*) 
                FROM erp.yarn_inventory 
                WHERE allocated > 0
            """)
        ]
        
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                for check_name, query in verification_queries:
                    try:
                        cur.execute(query)
                        result = cur.fetchone()[0]
                        logger.info(f"{check_name}: {result}")
                        
                        self.migration_log.append({
                            "action": "verification",
                            "check": check_name,
                            "result": result,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        logger.error(f"Verification failed for {check_name}: {e}")
    
    def create_rollback_script(self):
        """Create rollback script in case of issues"""
        
        rollback_script = f"""#!/bin/bash
# Beverly Knits ERP PostgreSQL Migration Rollback Script
# Generated: {datetime.now().isoformat()}

echo "Rolling back PostgreSQL migration..."

# Stop the application
pkill -f "python.*beverly"

# Restore SQLite database from backup
cp {self.backup_path} {self.sqlite_path}

# Update configuration to use SQLite
sed -i 's/DATABASE_TYPE=postgresql/DATABASE_TYPE=sqlite/g' .env

# Restart application
python src/core/beverly_comprehensive_erp.py &

echo "Rollback complete. Application reverted to SQLite."
"""
        
        rollback_path = Path("scripts/rollback_postgresql.sh")
        rollback_path.write_text(rollback_script)
        rollback_path.chmod(0o755)
        
        logger.info(f"Rollback script created: {rollback_path}")
        
        self.migration_log.append({
            "action": "rollback_script_created",
            "path": str(rollback_path),
            "timestamp": datetime.now().isoformat()
        })
    
    def save_migration_log(self):
        """Save migration log for audit trail"""
        
        log_path = Path("docs/reports/postgresql_migration_log.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(self.migration_log, f, indent=2, default=str)
        
        logger.info(f"Migration log saved: {log_path}")
    
    def run_migration(self):
        """Execute complete migration process"""
        
        logger.info("Starting PostgreSQL migration...")
        logger.info("=" * 60)
        
        try:
            # Step 1: Create backup
            logger.info("Step 1: Creating backup...")
            self.create_backup()
            
            # Step 2: Create PostgreSQL schema
            logger.info("\nStep 2: Creating PostgreSQL schema...")
            self.create_postgresql_schema()
            
            # Step 3: Migrate data
            logger.info("\nStep 3: Migrating data...")
            self.migrate_sqlite_data()
            
            # Step 4: Verify migration
            logger.info("\nStep 4: Verifying migration...")
            self.verify_migration()
            
            # Step 5: Create rollback script
            logger.info("\nStep 5: Creating rollback script...")
            self.create_rollback_script()
            
            # Step 6: Save migration log
            logger.info("\nStep 6: Saving migration log...")
            self.save_migration_log()
            
            logger.info("\n" + "=" * 60)
            logger.info("MIGRATION COMPLETE")
            logger.info("=" * 60)
            
            logger.info("\nNext steps:")
            logger.info("1. Update application configuration to use PostgreSQL")
            logger.info("2. Test all critical endpoints")
            logger.info("3. Monitor performance and memory usage")
            logger.info("4. If issues arise, run: scripts/rollback_postgresql.sh")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            logger.info("Run rollback script if needed: scripts/rollback_postgresql.sh")
            raise
        
        finally:
            if self.db_manager:
                self.db_manager.close_all()

def create_config_file():
    """Create PostgreSQL configuration file"""
    
    config = {
        "development": {
            "host": "localhost",
            "port": 5432,
            "database": "beverly_erp_dev",
            "user": "beverly",
            "password": "beverly_dev_pass_2025",
            "min_connections": 5,
            "max_connections": 20
        },
        "production": {
            "host": "localhost",
            "port": 5432,
            "database": "beverly_erp",
            "user": "beverly",
            "password": "beverly_secure_pass_2025",
            "min_connections": 10,
            "max_connections": 50
        }
    }
    
    config_path = Path("config/database_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration file created: {config_path}")
    return config

def main():
    """Run the PostgreSQL migration"""
    
    # Create configuration
    config = create_config_file()
    
    # Get environment (default to development)
    env = os.getenv("ENVIRONMENT", "development")
    pg_config = config[env]
    
    # SQLite database path
    sqlite_path = "data/beverly_erp.db"
    
    # Check for alternative paths
    if not Path(sqlite_path).exists():
        alt_paths = [
            "beverly_erp.db",
            "src/database/beverly_erp.db",
            "db/beverly_erp.db"
        ]
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                sqlite_path = alt_path
                break
    
    # Run migration
    migrator = PostgreSQLMigrator(sqlite_path, pg_config)
    migrator.run_migration()

if __name__ == "__main__":
    main()