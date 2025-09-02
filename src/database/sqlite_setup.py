#!/usr/bin/env python3
"""
SQLite Database Setup for Beverly Knits ERP
Alternative to PostgreSQL - no password required, works locally
"""

import sqlite3
import os
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

class SQLiteERPDatabase:
    """SQLite-based database for ERP system"""
    
    def __init__(self, db_path='beverly_knits.db'):
        """Initialize SQLite database"""
        self.db_path = db_path
        self.conn = None
        self.setup_database()
    
    def connect(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def setup_database(self):
        """Create database tables"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS yarns (
                yarn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                desc_id TEXT UNIQUE,
                yarn_description TEXT,
                blend TEXT,
                yarn_type TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS yarn_inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                yarn_id INTEGER,
                theoretical_balance REAL,
                allocated REAL,
                on_order REAL,
                planning_balance REAL,
                weeks_of_supply REAL,
                cost_per_pound REAL,
                snapshot_date DATE,
                FOREIGN KEY (yarn_id) REFERENCES yarns(yarn_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                so_number TEXT,
                customer_id TEXT,
                customer_name TEXT,
                style_number TEXT,
                quantity_ordered REAL,
                quantity_shipped REAL,
                balance REAL,
                ship_date DATE,
                available_qty REAL,
                snapshot_date DATE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fabric_inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                style_number TEXT,
                fstyle_number TEXT,
                style_description TEXT,
                inventory_stage TEXT,
                location TEXT,
                quantity_lbs REAL,
                quantity_yards REAL,
                snapshot_date DATE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knit_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ko_number TEXT,
                style_number TEXT,
                qty_ordered_lbs REAL,
                g00_lbs REAL,
                shipped_lbs REAL,
                balance_lbs REAL,
                machine TEXT,
                start_date DATE,
                quoted_date DATE,
                snapshot_date DATE
            )
        ''')
        
        conn.commit()
        print(f"✓ SQLite database created at: {self.db_path}")
        return conn
    
    def import_from_excel(self, data_path='/mnt/c/Users/psytz/sc data/ERP Data'):
        """Import data from Excel files into SQLite"""
        print(f"\nImporting data from: {data_path}")
        
        # Find most recent data folder
        data_dir = Path(data_path)
        if not data_dir.exists():
            print(f"⚠ Data directory not found: {data_path}")
            return False
        
        # Look for latest date folder
        date_folders = ['8-28-2025', '8-26-2025', '8-24-2025', '8-22-2025']
        active_folder = None
        
        for folder in date_folders:
            test_path = data_dir / folder
            if test_path.exists():
                active_folder = test_path
                break
        
        if not active_folder:
            active_folder = data_dir
        
        print(f"Using data from: {active_folder}")
        
        conn = self.connect()
        snapshot_date = datetime.now().date()
        
        # Import yarn inventory
        yarn_file = active_folder / 'yarn_inventory.xlsx'
        if yarn_file.exists():
            try:
                df = pd.read_excel(yarn_file)
                # Import to yarns table first
                for _, row in df.iterrows():
                    if pd.notna(row.get('Desc#')):
                        conn.execute('''
                            INSERT OR IGNORE INTO yarns (desc_id, yarn_description)
                            VALUES (?, ?)
                        ''', (row.get('Desc#'), row.get('Yarn Description', '')))
                
                # Import inventory data
                for _, row in df.iterrows():
                    if pd.notna(row.get('Desc#')):
                        yarn_id = conn.execute(
                            'SELECT yarn_id FROM yarns WHERE desc_id = ?',
                            (row.get('Desc#'),)
                        ).fetchone()
                        
                        if yarn_id:
                            conn.execute('''
                                INSERT INTO yarn_inventory 
                                (yarn_id, theoretical_balance, allocated, on_order,
                                 planning_balance, weeks_of_supply, cost_per_pound, snapshot_date)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                yarn_id[0],
                                row.get('Theoretical Balance', 0),
                                row.get('Allocated', 0),
                                row.get('On Order', 0),
                                row.get('Planning Balance', 0),
                                row.get('Weeks of Supply', 0),
                                row.get('Cost/Pound', 0),
                                snapshot_date
                            ))
                
                conn.commit()
                print(f"✓ Imported {len(df)} yarn inventory records")
            except Exception as e:
                print(f"⚠ Error importing yarn inventory: {e}")
        
        # Import sales orders
        for file_pattern in ['eFab_SO_List*.xlsx', 'eFab_SO_List*.csv']:
            files = list(active_folder.glob(file_pattern))
            if files:
                try:
                    so_file = files[0]
                    if so_file.suffix == '.csv':
                        df = pd.read_csv(so_file)
                    else:
                        df = pd.read_excel(so_file)
                    
                    for _, row in df.iterrows():
                        conn.execute('''
                            INSERT INTO sales_orders
                            (so_number, customer_name, style_number, 
                             quantity_ordered, quantity_shipped, balance,
                             ship_date, snapshot_date)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            row.get('SO#', ''),
                            row.get('Sold To', ''),
                            row.get('fStyle#', ''),
                            row.get('Ordered', 0),
                            row.get('Picked/Shipped', 0),
                            row.get('Balance', 0),
                            row.get('Ship Date'),
                            snapshot_date
                        ))
                    
                    conn.commit()
                    print(f"✓ Imported {len(df)} sales order records")
                    break
                except Exception as e:
                    print(f"⚠ Error importing sales orders: {e}")
        
        print("\n✓ Data import complete!")
        return True
    
    def test_connection(self):
        """Test database connection and show sample data"""
        conn = self.connect()
        cursor = conn.cursor()
        
        print("\nDatabase Statistics:")
        print("-" * 40)
        
        tables = ['yarns', 'yarn_inventory', 'sales_orders', 'fabric_inventory', 'knit_orders']
        for table in tables:
            count = cursor.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            print(f"{table:20} {count:,} records")
        
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("SQLite Database Setup for Beverly Knits ERP")
    print("=" * 60)
    print("\nNo password required - database stored locally")
    
    # Create database
    db = SQLiteERPDatabase()
    
    # Import data
    db.import_from_excel()
    
    # Test connection
    db.test_connection()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nDatabase created: beverly_knits.db")
    print("No password required - works immediately!")
    print("\nTo use in your code:")
    print("  import sqlite3")
    print("  conn = sqlite3.connect('beverly_knits.db')")
    print("  # No password needed!")