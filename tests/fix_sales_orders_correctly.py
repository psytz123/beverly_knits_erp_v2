#!/usr/bin/env python3
"""
Fix Sales Orders in Database - Correctly load exactly 128 orders
"""

import psycopg2
import pandas as pd
from datetime import datetime
import hashlib

def fix_sales_orders():
    # Connect to database
    conn = psycopg2.connect(
        host='localhost',
        database='beverly_knits_erp',
        user='erp_user',
        password='erp_password'
    )
    cursor = conn.cursor()
    
    print("=" * 70)
    print("FIXING SALES ORDERS - Loading Correct 128 Orders")
    print("=" * 70)
    
    try:
        # 1. Clear ALL existing sales orders
        print("\nStep 1: Clearing ALL existing sales orders...")
        cursor.execute("DELETE FROM production.sales_orders_ts")
        deleted = cursor.rowcount
        conn.commit()
        print(f"  ✓ Deleted {deleted} rows")
        
        # 2. Load the correct file from 8-28-2025 folder (live data)
        file_path = '/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025/eFab_SO_List.xlsx'
        print(f"\nStep 2: Loading data from correct file...")
        print(f"  File: {file_path}")
        
        df = pd.read_excel(file_path)
        print(f"  ✓ Found {len(df)} rows in file")
        
        # 3. Create customer mapping (since customer_id needs to be integer)
        print("\nStep 3: Creating customer mappings...")
        unique_customers = df['Sold To'].dropna().unique()
        
        # First, ensure customers exist in the customers table
        customer_map = {}
        for idx, customer_name in enumerate(unique_customers, start=1):
            customer_map[customer_name] = idx
            
            # Insert or update customer
            cursor.execute("""
                INSERT INTO production.customers (customer_id, customer_name, customer_code)
                VALUES (%s, %s, %s)
                ON CONFLICT (customer_id) DO UPDATE
                SET customer_name = EXCLUDED.customer_name
            """, (idx, str(customer_name)[:100], f"CUST{idx:04d}"))
        
        conn.commit()
        print(f"  ✓ Created mappings for {len(customer_map)} customers")
        
        # 4. Insert sales orders with proper data types
        print("\nStep 4: Inserting sales orders...")
        snapshot_date = datetime.now().date()
        inserted = 0
        errors = []
        
        for idx, row in df.iterrows():
            try:
                # Get customer_id from mapping
                customer_name = row.get('Sold To')
                customer_id = customer_map.get(customer_name) if pd.notna(customer_name) else None
                
                # Extract data with proper handling
                so_number = str(row.get('SO #', ''))
                style = str(row.get('cFVersion', '')) if pd.notna(row.get('cFVersion')) else ''
                
                # Handle numeric fields
                ordered = float(row.get('Ordered', 0) or 0)
                shipped = float(row.get('Picked/Shipped', 0) or 0)
                balance = float(row.get('Balance', 0) or 0)
                available = float(row.get('Available', 0) or 0)
                
                # Handle date
                ship_date = row.get('Ship Date')
                if pd.isna(ship_date) or ship_date == '' or ship_date == 'nan':
                    ship_date = None
                    
                # Handle other fields
                po_number = str(row.get('PO #', '')) if pd.notna(row.get('PO #')) else None
                uom = str(row.get('UOM', 'yds')) if pd.notna(row.get('UOM')) else 'yds'
                status = str(row.get('Status', 'Open')) if pd.notna(row.get('Status')) else 'Open'
                
                cursor.execute("""
                    INSERT INTO production.sales_orders_ts 
                    (so_number, customer_id, style_number, fstyle_number,
                     order_status, uom, quantity_ordered, quantity_shipped, 
                     balance, available_qty, ship_date, po_number,
                     snapshot_date, data_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    so_number,
                    customer_id,
                    style,  # style_number
                    style,  # fstyle_number (using same value)
                    status,
                    uom,
                    ordered,
                    shipped,
                    balance,
                    available,
                    ship_date,
                    po_number,
                    snapshot_date,
                    'Excel Import'
                ))
                inserted += 1
                
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)[:50]}")
                conn.rollback()
                continue
        
        # Commit all successful inserts
        conn.commit()
        print(f"  ✓ Successfully inserted {inserted} rows")
        
        if errors and len(errors) <= 5:
            print(f"  ⚠ {len(errors)} rows had issues:")
            for err in errors[:5]:
                print(f"    - {err}")
        
        # 5. Final verification
        print("\nStep 5: Final Verification...")
        cursor.execute("SELECT COUNT(*) FROM production.sales_orders_ts")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT so_number) FROM production.sales_orders_ts")
        unique_orders = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT customer_id) FROM production.sales_orders_ts WHERE customer_id IS NOT NULL")
        unique_customers = cursor.fetchone()[0]
        
        print(f"  Total rows in database: {total}")
        print(f"  Unique SO numbers: {unique_orders}")
        print(f"  Unique customers: {unique_customers}")
        
        # Show sample data
        print("\nSample Orders (first 5):")
        cursor.execute("""
            SELECT 
                so.so_number, 
                c.customer_name,
                so.style_number, 
                so.balance,
                so.ship_date
            FROM production.sales_orders_ts so
            LEFT JOIN production.customers c ON so.customer_id = c.customer_id
            ORDER BY so.so_number
            LIMIT 5
        """)
        
        print(f"  {'SO#':12} {'Customer':25} {'Style':15} {'Balance':10} {'Ship Date'}")
        print("  " + "-" * 75)
        for row in cursor.fetchall():
            customer = (row[1] or 'Unknown')[:25]
            ship_date = str(row[4]) if row[4] else 'Not set'
            print(f"  {row[0]:12} {customer:25} {row[2]:15} {row[3]:10.0f} {ship_date}")
        
        # Show summary by customer
        print("\nTop Customers by Order Count:")
        cursor.execute("""
            SELECT 
                c.customer_name,
                COUNT(*) as order_count,
                SUM(so.balance) as total_balance
            FROM production.sales_orders_ts so
            JOIN production.customers c ON so.customer_id = c.customer_id
            GROUP BY c.customer_name
            ORDER BY order_count DESC
            LIMIT 5
        """)
        
        print(f"  {'Customer':30} {'Orders':10} {'Total Balance'}")
        print("  " + "-" * 55)
        for row in cursor.fetchall():
            print(f"  {row[0][:30]:30} {row[1]:10} {row[2]:15,.0f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        conn.rollback()
        
    finally:
        conn.close()
    
    print("\n" + "=" * 70)
    if 'inserted' in locals() and inserted == 128:
        print("✅ SUCCESS! Database now has exactly 128 sales orders from the file")
    elif 'inserted' in locals():
        print(f"⚠ Loaded {inserted} orders (expected 128)")
    else:
        print("⚠ Process interrupted before completion")
    print("=" * 70)

if __name__ == "__main__":
    fix_sales_orders()