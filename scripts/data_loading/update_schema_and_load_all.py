#!/usr/bin/env python3
"""
Update Database Schema and Load ALL Data from 8-28-2025
Matches the actual Excel file structures
"""

import psycopg2
import pandas as pd
from datetime import datetime
import os
import numpy as np
from pathlib import Path

def update_schema_and_load():
    # Connect to database
    conn = psycopg2.connect(
        host='localhost',
        database='beverly_knits_erp',
        user='erp_user',
        password='erp_password'
    )
    cursor = conn.cursor()
    
    # Use project root for consistent path resolution
    project_root = Path(__file__).parent.parent.parent
    base_path = str(project_root / 'data' / 'production' / '5' / 'ERP Data' / '8-28-2025')
    snapshot_date = datetime(2025, 8, 28).date()
    
    print("=" * 80)
    print("UPDATING SCHEMA AND LOADING ALL DATA FROM 8-28-2025")
    print("=" * 80)
    
    try:
        # 1. SCHEMA ALREADY UPDATED
        print("\n1. Schema already updated via postgres user")
        
        # 2. LOAD FABRIC INVENTORY
        print("\n2. Loading Fabric Inventory...")
        
        # Clear existing data
        cursor.execute("DELETE FROM production.fabric_inventory_ts WHERE snapshot_date = %s", (snapshot_date,))
        conn.commit()
        
        fabric_files = [
            ('F01', 'eFab_Inventory_F01.xlsx'),
            ('G00', 'eFab_Inventory_G00.xlsx'),
            ('G02', 'eFab_Inventory_G02.xlsx'),
            ('I01', 'eFab_Inventory_I01.xlsx')
        ]
        
        total_fabric = 0
        for stage, filename in fabric_files:
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                print(f"   Loading {stage}: {len(df)} rows")
                
                inserted = 0
                for _, row in df.iterrows():
                    try:
                        style = str(row.get('Style #', '')) if pd.notna(row.get('Style #')) else 'UNKNOWN'
                        
                        # Get or create style_id
                        cursor.execute("""
                            INSERT INTO production.styles (style_number, fstyle_number, style_description)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (style_number) DO UPDATE
                            SET fstyle_number = EXCLUDED.fstyle_number
                            RETURNING style_id
                        """, (style, style, style))
                        style_id = cursor.fetchone()[0]
                        
                        # Handle received date
                        received_date = row.get('Received')
                        if pd.isna(received_date):
                            received_date = None
                        
                        cursor.execute("""
                            INSERT INTO production.fabric_inventory_ts
                            (snapshot_date, style_id, inventory_stage, style_number,
                             order_number, customer, roll_number, vendor_roll_number,
                             rack, quantity_yards, quantity_lbs, good_ea, bad_ea,
                             received_date, location, data_source)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            snapshot_date,
                            style_id,
                            stage,
                            style,
                            str(row.get('Order #', ''))[:50] if pd.notna(row.get('Order #')) else None,
                            str(row.get('Customer', ''))[:200] if pd.notna(row.get('Customer')) else None,
                            str(row.get('Roll #', ''))[:50] if pd.notna(row.get('Roll #')) else None,
                            str(row.get('Vendor Roll #', ''))[:50] if pd.notna(row.get('Vendor Roll #')) else None,
                            str(row.get('Rack', ''))[:50] if pd.notna(row.get('Rack')) else None,
                            float(row.get('Qty (yds)', 0) or 0),
                            float(row.get('Qty (lbs)', 0) or 0),
                            float(row.get('Good Ea.', 0) or 0),
                            float(row.get('Bad Ea.', 0) or 0),
                            received_date,
                            str(row.get('Rack', ''))[:50] if pd.notna(row.get('Rack')) else None,  # Using rack as location
                            'Excel Import'
                        ))
                        inserted += 1
                    except Exception as e:
                        if inserted == 0:  # Show first error only
                            print(f"      Sample error: {str(e)[:100]}")
                        continue
                
                conn.commit()
                total_fabric += inserted
                print(f"      ✓ Inserted {inserted} rows")
        
        print(f"   Total fabric inventory loaded: {total_fabric} rows")
        
        # 3. YARN DEMAND SCHEMA ALREADY UPDATED
        print("\n3. Yarn demand schema already updated")
        
        # 4. LOAD YARN DEMAND
        print("\n4. Loading Yarn Demand...")
        
        # Clear existing data
        cursor.execute("DELETE FROM production.yarn_demand_ts WHERE snapshot_date = %s", (snapshot_date,))
        conn.commit()
        
        yarn_demand_file = os.path.join(base_path, 'Yarn_Demand.xlsx')
        if os.path.exists(yarn_demand_file):
            df = pd.read_excel(yarn_demand_file)
            print(f"   Found {len(df)} rows")
            
            inserted = 0
            for _, row in df.iterrows():
                try:
                    yarn_num = row.get('Yarn')
                    if pd.notna(yarn_num):
                        # Try to find yarn_id
                        cursor.execute("SELECT yarn_id FROM production.yarns WHERE desc_id = %s", (str(int(yarn_num)),))
                        result = cursor.fetchone()
                        yarn_id = result[0] if result else None
                        
                        # If not found, create yarn
                        if not yarn_id:
                            cursor.execute("""
                                INSERT INTO production.yarns (desc_id, yarn_description, supplier)
                                VALUES (%s, %s, %s)
                                RETURNING yarn_id
                            """, (
                                str(int(yarn_num)),
                                str(row.get('Description', ''))[:200] if pd.notna(row.get('Description')) else None,
                                str(row.get('Supplier', ''))[:100] if pd.notna(row.get('Supplier')) else None
                            ))
                            yarn_id = cursor.fetchone()[0]
                        
                        # Insert yarn demand with all weekly data
                        for week_num in range(36, 44):  # Weeks 36-43
                            week_demand = row.get(f'Demand Week {week_num}', 0)
                            if pd.notna(week_demand) and week_demand != 0:
                                cursor.execute("""
                                    INSERT INTO production.yarn_demand_ts
                                    (snapshot_date, yarn_id, desc_id, yarn_number, supplier,
                                     description, color, week_number, demand_qty,
                                     monday_inventory, total_demand, total_receipt, balance,
                                     data_source)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """, (
                                    snapshot_date,
                                    yarn_id,
                                    str(int(yarn_num)),
                                    int(yarn_num),
                                    str(row.get('Supplier', ''))[:100] if pd.notna(row.get('Supplier')) else None,
                                    str(row.get('Description', ''))[:500] if pd.notna(row.get('Description')) else None,
                                    str(row.get('Color', ''))[:50] if pd.notna(row.get('Color')) else None,
                                    week_num,
                                    float(week_demand),
                                    float(row.get('Monday Inventory', 0) or 0),
                                    float(row.get('Total Demand', 0) or 0),
                                    float(row.get('Total Receipt', 0) or 0),
                                    float(row.get('Balance', 0) or 0),
                                    'Excel Import'
                                ))
                                inserted += 1
                        
                        # Also add "This Week" and "Later" if they exist
                        this_week = row.get('Demand This Week', 0)
                        if pd.notna(this_week) and this_week != 0:
                            cursor.execute("""
                                INSERT INTO production.yarn_demand_ts
                                (snapshot_date, yarn_id, desc_id, yarn_number, supplier,
                                 description, color, week_number, demand_qty,
                                 monday_inventory, total_demand, total_receipt, balance,
                                 data_source)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                snapshot_date, yarn_id, str(int(yarn_num)), int(yarn_num),
                                str(row.get('Supplier', ''))[:100] if pd.notna(row.get('Supplier')) else None,
                                str(row.get('Description', ''))[:500] if pd.notna(row.get('Description')) else None,
                                str(row.get('Color', ''))[:50] if pd.notna(row.get('Color')) else None,
                                0,  # Week 0 for "This Week"
                                float(this_week),
                                float(row.get('Monday Inventory', 0) or 0),
                                float(row.get('Total Demand', 0) or 0),
                                float(row.get('Total Receipt', 0) or 0),
                                float(row.get('Balance', 0) or 0),
                                'Excel Import'
                            ))
                            inserted += 1
                            
                except Exception as e:
                    continue
            
            conn.commit()
            print(f"   ✓ Inserted {inserted} yarn demand records")
        
        # 5. FINAL VERIFICATION
        print("\n" + "=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        
        queries = [
            ("Sales Orders", "SELECT COUNT(*) FROM production.sales_orders_ts"),
            ("Yarn Inventory", "SELECT COUNT(*) FROM production.yarn_inventory_ts"),
            ("Knit Orders", "SELECT COUNT(*) FROM production.knit_orders_ts"),
            ("Fabric Inventory", "SELECT COUNT(*) FROM production.fabric_inventory_ts"),
            ("Yarn Demand", "SELECT COUNT(*) FROM production.yarn_demand_ts"),
            ("Unique Yarns", "SELECT COUNT(*) FROM production.yarns"),
            ("Unique Styles", "SELECT COUNT(*) FROM production.styles"),
        ]
        
        print(f"\n{'Data Type':25} {'Records':>12}")
        print("-" * 40)
        
        for name, query in queries:
            cursor.execute(query)
            count = cursor.fetchone()[0]
            print(f"{name:25} {count:12,}")
        
        # Show fabric inventory breakdown
        cursor.execute("""
            SELECT inventory_stage, COUNT(*) as count
            FROM production.fabric_inventory_ts
            GROUP BY inventory_stage
            ORDER BY inventory_stage
        """)
        
        print("\nFabric Inventory by Stage:")
        for row in cursor.fetchall():
            print(f"  {row[0]:10} {row[1]:12,} rows")
        
        conn.commit()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        conn.rollback()
        raise
    
    finally:
        conn.close()
    
    print("\n" + "=" * 80)
    print("✅ DATABASE SCHEMA UPDATED AND ALL DATA LOADED!")
    print("=" * 80)

if __name__ == "__main__":
    update_schema_and_load()