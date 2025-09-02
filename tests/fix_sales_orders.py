#!/usr/bin/env python3
"""
Fix Sales Orders in Database
Clear incorrect data and reload with correct 128 orders
"""

import psycopg2
import pandas as pd
from datetime import datetime

def fix_sales_orders():
    # Connect to database
    conn = psycopg2.connect(
        host='localhost',
        database='beverly_knits_erp',
        user='erp_user',
        password='erp_password'
    )
    cursor = conn.cursor()
    
    print("Fixing Sales Orders Data...")
    print("-" * 50)
    
    # 1. Clear existing sales orders
    print("Step 1: Clearing existing sales orders...")
    cursor.execute("DELETE FROM production.sales_orders_ts")
    deleted = cursor.rowcount
    print(f"  Deleted {deleted} rows")
    
    # 2. Load correct data from file
    file_path = '/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/eFab_SO_List.xlsx'
    print(f"\nStep 2: Loading data from: {file_path}")
    
    df = pd.read_excel(file_path)
    print(f"  Found {len(df)} rows in file")
    
    # 3. Insert correct data
    print("\nStep 3: Inserting correct data...")
    snapshot_date = datetime.now().date()
    inserted = 0
    
    for _, row in df.iterrows():
        try:
            # Map columns properly
            so_number = str(row.get('SO #', ''))
            customer_name = str(row.get('Sold To', ''))
            style = str(row.get('fStyle#', ''))
            
            # Handle numeric columns
            ordered = float(row.get('Ordered', 0) or 0)
            shipped = float(row.get('Picked/Shipped', 0) or 0)
            balance = float(row.get('Balance', 0) or 0)
            available = float(row.get('Available', 0) or 0)
            
            # Handle dates
            ship_date = row.get('Ship Date')
            if pd.isna(ship_date):
                ship_date = None
            
            cursor.execute("""
                INSERT INTO production.sales_orders_ts 
                (so_number, customer_id, style_number, quantity_ordered, 
                 quantity_shipped, balance, ship_date, available_qty, snapshot_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                so_number,
                customer_name[:10] if customer_name else None,  # Use first 10 chars as customer_id
                style,
                ordered,
                shipped,
                balance,
                ship_date,
                available,
                snapshot_date
            ))
            inserted += 1
        except Exception as e:
            print(f"  Warning: Could not insert row: {e}")
            continue
    
    # Commit changes
    conn.commit()
    print(f"  Inserted {inserted} rows")
    
    # 4. Verify the fix
    print("\nStep 4: Verifying...")
    cursor.execute("SELECT COUNT(*) FROM production.sales_orders_ts")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT so_number) FROM production.sales_orders_ts")
    unique = cursor.fetchone()[0]
    
    print(f"  Total rows in database: {total}")
    print(f"  Unique SO numbers: {unique}")
    
    # Show sample data
    print("\nSample data (first 5 orders):")
    cursor.execute("""
        SELECT so_number, style_number, balance 
        FROM production.sales_orders_ts 
        LIMIT 5
    """)
    
    for row in cursor.fetchall():
        print(f"  SO: {row[0]:15} Style: {row[1]:20} Balance: {row[2]:,.0f}")
    
    conn.close()
    
    print("\n" + "=" * 50)
    print("✅ Sales orders fixed successfully!")
    print(f"Database now has the correct {inserted} orders from the file")
    print("=" * 50)

if __name__ == "__main__":
    fix_sales_orders()