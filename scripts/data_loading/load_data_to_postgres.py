#!/usr/bin/env python3
"""
Load Beverly Knits ERP data from files into PostgreSQL database
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import json
import sys
from pathlib import Path
from datetime import datetime

# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "beverly_knits_erp",
    "user": "erp_user",
    "password": "erp_password"
}

# Data paths
DATA_PATH = Path("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5")

def connect_db():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✓ Connected to PostgreSQL database")
        return conn
    except Exception as e:
        print(f"✗ Failed to connect to database: {e}")
        sys.exit(1)

def load_yarn_inventory(conn):
    """Load yarn inventory data into database"""
    try:
        # Read yarn inventory
        yarn_file = DATA_PATH / "yarn_inventory.xlsx"
        df = pd.read_excel(yarn_file)
        print(f"  Loading {len(df)} yarn inventory records...")
        
        cur = conn.cursor()
        
        # First, insert/update yarns master data
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO production.yarns (desc_id, yarn_description, color, cost_per_pound)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (desc_id) DO UPDATE
                SET yarn_description = EXCLUDED.yarn_description,
                    color = EXCLUDED.color,
                    cost_per_pound = EXCLUDED.cost_per_pound,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                int(row['Desc#']),
                row.get('Description', ''),
                row.get('Color', ''),
                float(row.get('Cost/Pound', 0))
            ))
        
        # Create time-series yarn inventory table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS production.yarn_inventory_ts (
                id SERIAL,
                yarn_id INTEGER REFERENCES production.yarns(yarn_id),
                theoretical_balance DECIMAL(10,2),
                allocated DECIMAL(10,2),
                on_order DECIMAL(10,2),
                planning_balance DECIMAL(10,2),
                cost_per_pound DECIMAL(10,2),
                snapshot_date DATE DEFAULT CURRENT_DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, snapshot_date)
            );
        """)
        
        # Get yarn_id mapping
        cur.execute("SELECT yarn_id, desc_id FROM production.yarns")
        yarn_mapping = {desc_id: yarn_id for yarn_id, desc_id in cur.fetchall()}
        
        # Insert inventory data
        for _, row in df.iterrows():
            yarn_id = yarn_mapping.get(int(row['Desc#']))
            if yarn_id:
                cur.execute("""
                    INSERT INTO production.yarn_inventory_ts 
                    (yarn_id, theoretical_balance, allocated, on_order, planning_balance, cost_per_pound)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    yarn_id,
                    float(row.get('Theoretical Balance', 0)),
                    float(row.get('Allocated', 0)),
                    float(row.get('On Order', 0)),
                    float(row.get('Planning Balance', 0)),
                    float(row.get('Cost/Pound', 0))
                ))
        
        conn.commit()
        print(f"  ✓ Loaded yarn inventory data")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to load yarn inventory: {e}")
        conn.rollback()
        return False

def load_bom_data(conn):
    """Load BOM data into database"""
    try:
        # Read BOM data
        bom_file = DATA_PATH / "BOM_updated.csv"
        df = pd.read_csv(bom_file)
        print(f"  Loading {len(df)} BOM records...")
        
        cur = conn.cursor()
        
        # First, insert styles if not exists
        unique_styles = df['Style#'].unique()
        for style in unique_styles:
            if pd.notna(style):
                cur.execute("""
                    INSERT INTO production.styles (style_number)
                    VALUES (%s)
                    ON CONFLICT (style_number) DO NOTHING
                """, (str(style),))
        
        # Get style and yarn mappings
        cur.execute("SELECT style_id, style_number FROM production.styles")
        style_mapping = {style_num: style_id for style_id, style_num in cur.fetchall()}
        
        cur.execute("SELECT yarn_id, desc_id FROM production.yarns")
        yarn_mapping = {desc_id: yarn_id for yarn_id, desc_id in cur.fetchall()}
        
        # Insert BOM data
        for _, row in df.iterrows():
            style_id = style_mapping.get(str(row['Style#']))
            yarn_id = yarn_mapping.get(int(row['Desc#'])) if pd.notna(row['Desc#']) else None
            
            if style_id and yarn_id and pd.notna(row.get('BOM_Percentage', 0)):
                cur.execute("""
                    INSERT INTO production.style_bom (style_id, yarn_id, bom_percent)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (style_id, yarn_id) DO UPDATE
                    SET bom_percent = EXCLUDED.bom_percent,
                        updated_at = CURRENT_TIMESTAMP
                """, (style_id, yarn_id, float(row.get('BOM_Percentage', 0))))
        
        conn.commit()
        print(f"  ✓ Loaded BOM data")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to load BOM: {e}")
        conn.rollback()
        return False

def load_sales_orders(conn):
    """Load sales order data into database"""
    try:
        # Read sales data
        sales_file = DATA_PATH / "Sales Activity Report.csv"
        df = pd.read_csv(sales_file)
        print(f"  Loading {len(df)} sales records...")
        
        cur = conn.cursor()
        
        # Create sales orders table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS production.sales_orders_ts (
                id SERIAL,
                so_number VARCHAR(50),
                customer_id INTEGER,
                style_number VARCHAR(100),
                quantity_ordered DECIMAL(10,2),
                quantity_shipped DECIMAL(10,2),
                balance DECIMAL(10,2),
                ship_date DATE,
                available_qty DECIMAL(10,2),
                snapshot_date DATE DEFAULT CURRENT_DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, snapshot_date)
            );
        """)
        
        # Insert sales data
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO production.sales_orders_ts 
                (so_number, style_number, quantity_ordered, ship_date)
                VALUES (%s, %s, %s, %s)
            """, (
                str(row.get('Document', '')),
                str(row.get('fStyle#', '')),
                float(row.get('Yds_ordered', 0)),
                pd.to_datetime(row.get('Invoice Date')).date() if pd.notna(row.get('Invoice Date')) else None
            ))
        
        conn.commit()
        print(f"  ✓ Loaded sales orders")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to load sales orders: {e}")
        conn.rollback()
        return False

def load_knit_orders(conn):
    """Load knit orders data into database"""
    try:
        # Read knit orders
        ko_file = DATA_PATH / "eFab_Knit_Orders.xlsx"
        df = pd.read_excel(ko_file)
        print(f"  Loading {len(df)} knit orders...")
        
        cur = conn.cursor()
        
        # Create knit orders table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS production.knit_orders (
                id SERIAL PRIMARY KEY,
                order_number VARCHAR(50),
                style_number VARCHAR(100),
                qty_ordered_lbs DECIMAL(10,2),
                balance_lbs DECIMAL(10,2),
                ship_date DATE,
                status VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Insert knit orders
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO production.knit_orders 
                (order_number, style_number, qty_ordered_lbs, balance_lbs, ship_date)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                str(row.get('Order #', '')),
                str(row.get('fStyle#', '')),
                float(row.get('Qty Ordered (lbs)', 0)),
                float(row.get('Balance (lbs)', 0)),
                pd.to_datetime(row.get('Quoted Date')).date() if pd.notna(row.get('Quoted Date')) else None
            ))
        
        conn.commit()
        print(f"  ✓ Loaded knit orders")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to load knit orders: {e}")
        conn.rollback()
        return False

def main():
    """Main function to load all data"""
    print("=" * 60)
    print("Beverly Knits ERP - PostgreSQL Data Loader")
    print("=" * 60)
    
    # Connect to database
    conn = connect_db()
    
    try:
        print("\nLoading data from files...")
        
        # Load data in order
        success = True
        success = load_yarn_inventory(conn) and success
        success = load_bom_data(conn) and success
        success = load_sales_orders(conn) and success
        success = load_knit_orders(conn) and success
        
        if success:
            print("\n✓ All data loaded successfully!")
            
            # Print summary
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM production.yarns")
            yarn_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM production.styles")
            style_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM production.style_bom")
            bom_count = cur.fetchone()[0]
            
            print(f"\nDatabase Summary:")
            print(f"  - Yarns: {yarn_count}")
            print(f"  - Styles: {style_count}")
            print(f"  - BOM entries: {bom_count}")
            
        else:
            print("\n⚠ Some data failed to load. Check errors above.")
            
    finally:
        conn.close()
        print("\n✓ Database connection closed")

if __name__ == "__main__":
    main()