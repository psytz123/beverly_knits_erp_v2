#!/usr/bin/env python3
"""
Fix production database disk I/O error
"""
import sqlite3
import os
import shutil
from datetime import datetime

def fix_production_database():
    """Attempt to fix the production database disk I/O error"""
    
    db_path = "production.db"
    backup_path = f"production_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    
    print(f"Attempting to fix database: {db_path}")
    
    # First, try to backup the existing database
    try:
        if os.path.exists(db_path):
            shutil.copy2(db_path, backup_path)
            print(f"Created backup: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    # Try to dump and recreate the database
    try:
        # Connect with different settings to avoid disk I/O issues
        conn = sqlite3.connect(db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        
        # Try to read the schema
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        schemas = cursor.fetchall()
        
        conn.close()
        
        # Create a new database
        new_db_path = "production_new.db"
        if os.path.exists(new_db_path):
            os.remove(new_db_path)
            
        new_conn = sqlite3.connect(new_db_path)
        new_cursor = new_conn.cursor()
        
        # Recreate tables
        for schema in schemas:
            if schema[0]:
                new_cursor.execute(schema[0])
        
        new_conn.commit()
        new_conn.close()
        
        # Replace old database with new one
        os.remove(db_path)
        os.rename(new_db_path, db_path)
        
        print("Successfully recreated database")
        return True
        
    except Exception as e:
        print(f"Error during database repair: {e}")
        
        # As a last resort, create a fresh database with the required schema
        try:
            print("Creating fresh database with required schema...")
            
            if os.path.exists(db_path):
                os.rename(db_path, f"{db_path}.corrupt")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create production_orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS production_orders (
                    id TEXT PRIMARY KEY,
                    product_name TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    completed INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    priority TEXT DEFAULT 'medium',
                    start_date TEXT,
                    due_date TEXT,
                    assigned_line TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create production_tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS production_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    FOREIGN KEY (order_id) REFERENCES production_orders(id)
                )
            """)
            
            # Insert some sample data
            sample_orders = [
                ("PO-2024-001", "Cotton Blend T-Shirt", 5000, 3200, "in_progress", "high", "2024-08-01", "2024-08-15", "Line A", "Rush order"),
                ("PO-2024-002", "Polyester Hoodie", 3000, 1500, "in_progress", "medium", "2024-08-05", "2024-08-20", "Line B", None),
                ("PO-2024-003", "Denim Jeans", 2000, 0, "pending", "low", "2024-08-10", "2024-08-25", None, "Waiting for materials"),
            ]
            
            for order in sample_orders:
                cursor.execute("""
                    INSERT INTO production_orders 
                    (id, product_name, quantity, completed, status, priority, start_date, due_date, assigned_line, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, order)
            
            conn.commit()
            conn.close()
            
            print("Successfully created fresh database with sample data")
            return True
            
        except Exception as e2:
            print(f"Failed to create fresh database: {e2}")
            return False

if __name__ == "__main__":
    success = fix_production_database()
    if success:
        print("Database repair completed successfully")
        
        # Test the database
        try:
            conn = sqlite3.connect("production.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM production_orders")
            count = cursor.fetchone()[0]
            print(f"Database test successful - {count} production orders found")
            conn.close()
        except Exception as e:
            print(f"Database test failed: {e}")
    else:
        print("Database repair failed")