#!/usr/bin/env python3
"""
Complete Yarn Demand Loader - Loads ALL yarn demand data from 8-28-2025
Including weekly breakdowns and style-specific demands
"""

import psycopg2
import pandas as pd
from datetime import datetime
import os

def load_all_yarn_demand():
    # Connect to database
    conn = psycopg2.connect(
        host='localhost',
        database='beverly_knits_erp',
        user='erp_user',
        password='erp_password'
    )
    cursor = conn.cursor()
    
    base_path = '/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025'
    snapshot_date = datetime(2025, 8, 28).date()
    
    print("=" * 80)
    print("COMPLETE YARN DEMAND LOADER - LOADING ALL YARN DATA")
    print("=" * 80)
    
    # 1. LOAD MAIN YARN DEMAND WITH WEEKLY BREAKDOWN
    print("\n1. Loading Main Yarn Demand with Weekly Breakdown...")
    
    # Clear existing yarn demand
    cursor.execute("DELETE FROM production.yarn_demand_ts WHERE snapshot_date = %s", (snapshot_date,))
    conn.commit()
    
    yarn_demand_file = os.path.join(base_path, 'Yarn_Demand.xlsx')
    if os.path.exists(yarn_demand_file):
        df = pd.read_excel(yarn_demand_file)
        print(f"   Found {len(df)} yarns in Yarn_Demand.xlsx")
        print(f"   Columns: {df.columns.tolist()[:10]}...")
        
        total_inserted = 0
        
        for _, row in df.iterrows():
            try:
                yarn_num = row.get('Yarn')
                if pd.notna(yarn_num):
                    yarn_num_str = str(int(yarn_num))
                    
                    # Get or create yarn
                    cursor.execute("""
                        SELECT yarn_id FROM production.yarns WHERE desc_id = %s
                    """, (yarn_num_str,))
                    result = cursor.fetchone()
                    
                    if not result:
                        # Create new yarn
                        cursor.execute("""
                            INSERT INTO production.yarns (desc_id, yarn_description, supplier)
                            VALUES (%s, %s, %s)
                            RETURNING yarn_id
                        """, (
                            yarn_num_str,
                            str(row.get('Description', ''))[:500] if pd.notna(row.get('Description')) else None,
                            str(row.get('Supplier', ''))[:100] if pd.notna(row.get('Supplier')) else None
                        ))
                        yarn_id = cursor.fetchone()[0]
                    else:
                        yarn_id = result[0]
                    
                    # Insert summary record (week 0)
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
                        yarn_num_str,
                        int(yarn_num),
                        str(row.get('Supplier', ''))[:100] if pd.notna(row.get('Supplier')) else None,
                        str(row.get('Description', ''))[:500] if pd.notna(row.get('Description')) else None,
                        str(row.get('Color', ''))[:50] if pd.notna(row.get('Color')) else None,
                        0,  # Week 0 for summary
                        float(row.get('Total Demand', 0) or 0),
                        float(row.get('Monday Inventory', 0) or 0),
                        float(row.get('Total Demand', 0) or 0),
                        float(row.get('Total Receipt', 0) or 0),
                        float(row.get('Balance', 0) or 0),
                        'Yarn_Demand.xlsx'
                    ))
                    total_inserted += 1
                    
                    # Insert "Demand This Week" if exists
                    this_week = row.get('Demand This Week', 0)
                    if pd.notna(this_week) and float(this_week) != 0:
                        cursor.execute("""
                            INSERT INTO production.yarn_demand_ts
                            (snapshot_date, yarn_id, desc_id, yarn_number, supplier,
                             description, color, week_number, demand_qty,
                             monday_inventory, total_demand, total_receipt, balance,
                             data_source)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            snapshot_date, yarn_id, yarn_num_str, int(yarn_num),
                            str(row.get('Supplier', ''))[:100] if pd.notna(row.get('Supplier')) else None,
                            str(row.get('Description', ''))[:500] if pd.notna(row.get('Description')) else None,
                            str(row.get('Color', ''))[:50] if pd.notna(row.get('Color')) else None,
                            35,  # Current week
                            float(this_week),
                            float(row.get('Monday Inventory', 0) or 0),
                            float(row.get('Total Demand', 0) or 0),
                            float(row.get('Total Receipt', 0) or 0),
                            float(row.get('Balance', 0) or 0),
                            'Yarn_Demand.xlsx'
                        ))
                        total_inserted += 1
                    
                    # Insert weekly demands (Week 36-43)
                    for week in range(36, 44):
                        week_col = f'Demand Week {week}'
                        if week_col in row:
                            week_demand = row.get(week_col, 0)
                            if pd.notna(week_demand) and float(week_demand) != 0:
                                cursor.execute("""
                                    INSERT INTO production.yarn_demand_ts
                                    (snapshot_date, yarn_id, desc_id, yarn_number, supplier,
                                     description, color, week_number, demand_qty,
                                     monday_inventory, total_demand, total_receipt, balance,
                                     data_source)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """, (
                                    snapshot_date, yarn_id, yarn_num_str, int(yarn_num),
                                    str(row.get('Supplier', ''))[:100] if pd.notna(row.get('Supplier')) else None,
                                    str(row.get('Description', ''))[:500] if pd.notna(row.get('Description')) else None,
                                    str(row.get('Color', ''))[:50] if pd.notna(row.get('Color')) else None,
                                    week,
                                    float(week_demand),
                                    float(row.get('Monday Inventory', 0) or 0),
                                    float(row.get('Total Demand', 0) or 0),
                                    float(row.get('Total Receipt', 0) or 0),
                                    float(row.get('Balance', 0) or 0),
                                    'Yarn_Demand.xlsx'
                                ))
                                total_inserted += 1
                    
                    # Insert "Demand Later" if exists
                    later = row.get('Demand Later', 0)
                    if pd.notna(later) and float(later) != 0:
                        cursor.execute("""
                            INSERT INTO production.yarn_demand_ts
                            (snapshot_date, yarn_id, desc_id, yarn_number, supplier,
                             description, color, week_number, demand_qty,
                             monday_inventory, total_demand, total_receipt, balance,
                             data_source)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            snapshot_date, yarn_id, yarn_num_str, int(yarn_num),
                            str(row.get('Supplier', ''))[:100] if pd.notna(row.get('Supplier')) else None,
                            str(row.get('Description', ''))[:500] if pd.notna(row.get('Description')) else None,
                            str(row.get('Color', ''))[:50] if pd.notna(row.get('Color')) else None,
                            99,  # Week 99 for "Later"
                            float(later),
                            float(row.get('Monday Inventory', 0) or 0),
                            float(row.get('Total Demand', 0) or 0),
                            float(row.get('Total Receipt', 0) or 0),
                            float(row.get('Balance', 0) or 0),
                            'Yarn_Demand.xlsx'
                        ))
                        total_inserted += 1
                        
            except Exception as e:
                print(f"      Error processing yarn {yarn_num}: {str(e)[:50]}")
                continue
        
        conn.commit()
        print(f"   ✓ Inserted {total_inserted} yarn demand records")
    
    # 2. LOAD YARN DEMAND BY STYLE
    print("\n2. Loading Yarn Demand By Style...")
    
    # First check if we need to add style_yarn_demand table
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'production' 
            AND table_name = 'style_yarn_demand'
        )
    """)
    
    if not cursor.fetchone()[0]:
        print("   Creating style_yarn_demand table...")
        cursor.execute("""
            CREATE TABLE production.style_yarn_demand (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                style_number VARCHAR(50),
                yarn_number INTEGER,
                yarn_percentage DECIMAL(5,2),
                demand_this_week DECIMAL(10,2),
                demand_week_36 DECIMAL(10,2),
                demand_week_37 DECIMAL(10,2),
                demand_week_38 DECIMAL(10,2),
                demand_week_39 DECIMAL(10,2),
                demand_week_40 DECIMAL(10,2),
                demand_week_41 DECIMAL(10,2),
                demand_week_42 DECIMAL(10,2),
                demand_week_43 DECIMAL(10,2),
                demand_later DECIMAL(10,2),
                total_demand DECIMAL(10,2),
                data_source VARCHAR(100)
            )
        """)
        conn.commit()
    
    # Clear existing style yarn demand
    cursor.execute("DELETE FROM production.style_yarn_demand WHERE snapshot_date = %s", (snapshot_date,))
    
    yarn_by_style_file = os.path.join(base_path, 'Yarn_Demand_By_Style.xlsx')
    if os.path.exists(yarn_by_style_file):
        df = pd.read_excel(yarn_by_style_file)
        print(f"   Found {len(df)} style-yarn combinations in Yarn_Demand_By_Style.xlsx")
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO production.style_yarn_demand
                    (snapshot_date, style_number, yarn_number, yarn_percentage,
                     demand_this_week, demand_week_36, demand_week_37, demand_week_38,
                     demand_week_39, demand_week_40, demand_week_41, demand_week_42,
                     demand_week_43, demand_later, total_demand, data_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    snapshot_date,
                    str(row.get('Style', ''))[:50] if pd.notna(row.get('Style')) else None,
                    int(row.get('Yarn')) if pd.notna(row.get('Yarn')) else None,
                    float(row.get('Percentage', 0) or 0),
                    float(row.get('This Week', 0) or 0),
                    float(row.get('Week 36', 0) or 0),
                    float(row.get('Week 37', 0) or 0),
                    float(row.get('Week 38', 0) or 0),
                    float(row.get('Week 39', 0) or 0),
                    float(row.get('Week 40', 0) or 0),
                    float(row.get('Week 41', 0) or 0),
                    float(row.get('Week 42', 0) or 0),
                    float(row.get('Week 43', 0) or 0),
                    float(row.get('Later', 0) or 0),
                    float(row.get('Total', 0) or 0),
                    'Yarn_Demand_By_Style.xlsx'
                ))
                inserted += 1
            except Exception as e:
                continue
        
        conn.commit()
        print(f"   ✓ Inserted {inserted} style-yarn demand records")
    
    # 3. LOAD YARN DEMAND BY STYLE KO
    print("\n3. Loading Yarn Demand By Style KO...")
    
    # Check if we need knit_order_yarn_demand table
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'production' 
            AND table_name = 'knit_order_yarn_demand'
        )
    """)
    
    if not cursor.fetchone()[0]:
        print("   Creating knit_order_yarn_demand table...")
        cursor.execute("""
            CREATE TABLE production.knit_order_yarn_demand (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                style_number VARCHAR(50),
                knit_order_number VARCHAR(50),
                yarn_number INTEGER,
                yarn_percentage DECIMAL(5,2),
                demand_this_week DECIMAL(10,2),
                demand_week_36 DECIMAL(10,2),
                demand_week_37 DECIMAL(10,2),
                demand_week_38 DECIMAL(10,2),
                demand_week_39 DECIMAL(10,2),
                demand_week_40 DECIMAL(10,2),
                demand_week_41 DECIMAL(10,2),
                demand_week_42 DECIMAL(10,2),
                demand_week_43 DECIMAL(10,2),
                demand_later DECIMAL(10,2),
                total_demand DECIMAL(10,2),
                data_source VARCHAR(100)
            )
        """)
        conn.commit()
    
    # Clear existing
    cursor.execute("DELETE FROM production.knit_order_yarn_demand WHERE snapshot_date = %s", (snapshot_date,))
    
    yarn_by_style_ko_file = os.path.join(base_path, 'Yarn_Demand_By_Style_KO.xlsx')
    if os.path.exists(yarn_by_style_ko_file):
        df = pd.read_excel(yarn_by_style_ko_file)
        print(f"   Found {len(df)} style-KO-yarn combinations in Yarn_Demand_By_Style_KO.xlsx")
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO production.knit_order_yarn_demand
                    (snapshot_date, style_number, knit_order_number, yarn_number, yarn_percentage,
                     demand_this_week, demand_week_36, demand_week_37, demand_week_38,
                     demand_week_39, demand_week_40, demand_week_41, demand_week_42,
                     demand_week_43, demand_later, total_demand, data_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    snapshot_date,
                    str(row.get('Style', ''))[:50] if pd.notna(row.get('Style')) else None,
                    str(row.get('KO', ''))[:50] if pd.notna(row.get('KO')) else None,
                    int(row.get('Yarn')) if pd.notna(row.get('Yarn')) else None,
                    float(row.get('Percentage', 0) or 0),
                    float(row.get('This Week', 0) or 0),
                    float(row.get('Week 36', 0) or 0),
                    float(row.get('Week 37', 0) or 0),
                    float(row.get('Week 38', 0) or 0),
                    float(row.get('Week 39', 0) or 0),
                    float(row.get('Week 40', 0) or 0),
                    float(row.get('Week 41', 0) or 0),
                    float(row.get('Week 42', 0) or 0),
                    float(row.get('Week 43', 0) or 0),
                    float(row.get('Later', 0) or 0),
                    float(row.get('Total', 0) or 0),
                    'Yarn_Demand_By_Style_KO.xlsx'
                ))
                inserted += 1
            except Exception as e:
                continue
        
        conn.commit()
        print(f"   ✓ Inserted {inserted} knit order yarn demand records")
    
    # 4. LOAD EXPECTED YARN REPORT
    print("\n4. Loading Expected Yarn Report...")
    
    # Check if we need expected_yarn_receipts table
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'production' 
            AND table_name = 'expected_yarn_receipts'
        )
    """)
    
    if not cursor.fetchone()[0]:
        print("   Creating expected_yarn_receipts table...")
        cursor.execute("""
            CREATE TABLE production.expected_yarn_receipts (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                po_date DATE,
                po_number VARCHAR(50),
                purchased_from VARCHAR(200),
                yarn_description VARCHAR(500),
                price DECIMAL(10,2),
                uom VARCHAR(20),
                quantity_ordered DECIMAL(10,2),
                quantity_received DECIMAL(10,2),
                balance DECIMAL(10,2),
                expected_date DATE,
                week_36 DECIMAL(10,2),
                week_37 DECIMAL(10,2),
                week_38 DECIMAL(10,2),
                week_39 DECIMAL(10,2),
                week_40 DECIMAL(10,2),
                week_41 DECIMAL(10,2),
                week_42 DECIMAL(10,2),
                week_43 DECIMAL(10,2),
                later DECIMAL(10,2),
                data_source VARCHAR(100)
            )
        """)
        conn.commit()
    
    # Clear existing
    cursor.execute("DELETE FROM production.expected_yarn_receipts WHERE snapshot_date = %s", (snapshot_date,))
    
    expected_yarn_file = os.path.join(base_path, 'Expected_Yarn_Report.xlsx')
    if os.path.exists(expected_yarn_file):
        df = pd.read_excel(expected_yarn_file)
        print(f"   Found {len(df)} expected receipts in Expected_Yarn_Report.xlsx")
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                # Handle dates
                po_date = row.get('PO Date')
                if pd.isna(po_date):
                    po_date = None
                    
                expected_date = row.get('Expected')
                if pd.isna(expected_date):
                    expected_date = None
                
                cursor.execute("""
                    INSERT INTO production.expected_yarn_receipts
                    (snapshot_date, po_date, po_number, purchased_from, yarn_description,
                     price, uom, quantity_ordered, quantity_received, balance, expected_date,
                     week_36, week_37, week_38, week_39, week_40, week_41, week_42, week_43,
                     later, data_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    snapshot_date,
                    po_date,
                    str(row.get('PO #', ''))[:50] if pd.notna(row.get('PO #')) else None,
                    str(row.get('Purchased From', ''))[:200] if pd.notna(row.get('Purchased From')) else None,
                    str(row.get('Desc', ''))[:500] if pd.notna(row.get('Desc')) else None,
                    float(row.get('Price', 0) or 0),
                    str(row.get('UOM', ''))[:20] if pd.notna(row.get('UOM')) else None,
                    float(row.get('Ordered', 0) or 0),
                    float(row.get('Received', 0) or 0),
                    float(row.get('Balance', 0) or 0),
                    expected_date,
                    float(row.get('Week 36', 0) or 0),
                    float(row.get('Week 37', 0) or 0),
                    float(row.get('Week 38', 0) or 0),
                    float(row.get('Week 39', 0) or 0),
                    float(row.get('Week 40', 0) or 0),
                    float(row.get('Week 41', 0) or 0),
                    float(row.get('Week 42', 0) or 0),
                    float(row.get('Week 43', 0) or 0),
                    float(row.get('Later', 0) or 0),
                    'Expected_Yarn_Report.xlsx'
                ))
                inserted += 1
            except Exception as e:
                continue
        
        conn.commit()
        print(f"   ✓ Inserted {inserted} expected yarn receipt records")
    
    # 5. FINAL VERIFICATION
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)
    
    # Check all yarn-related tables
    queries = [
        ("Yarn Demand Records", "SELECT COUNT(*) FROM production.yarn_demand_ts WHERE snapshot_date = %s", (snapshot_date,)),
        ("Unique Yarns in Demand", "SELECT COUNT(DISTINCT yarn_id) FROM production.yarn_demand_ts WHERE snapshot_date = %s", (snapshot_date,)),
        ("Unique Weeks in Demand", "SELECT COUNT(DISTINCT week_number) FROM production.yarn_demand_ts WHERE snapshot_date = %s", (snapshot_date,)),
        ("Style-Yarn Demand", "SELECT COUNT(*) FROM production.style_yarn_demand WHERE snapshot_date = %s", (snapshot_date,)),
        ("KO-Yarn Demand", "SELECT COUNT(*) FROM production.knit_order_yarn_demand WHERE snapshot_date = %s", (snapshot_date,)),
        ("Expected Yarn Receipts", "SELECT COUNT(*) FROM production.expected_yarn_receipts WHERE snapshot_date = %s", (snapshot_date,)),
        ("Total Yarns Master", "SELECT COUNT(*) FROM production.yarns", None),
    ]
    
    print(f"\n{'Data Type':30} {'Records':>12}")
    print("-" * 45)
    
    for name, query, params in queries:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        count = cursor.fetchone()[0]
        print(f"{name:30} {count:12,}")
    
    # Show weekly breakdown
    cursor.execute("""
        SELECT week_number, COUNT(*) as records, SUM(demand_qty) as total_demand
        FROM production.yarn_demand_ts
        WHERE snapshot_date = %s
        GROUP BY week_number
        ORDER BY week_number
    """, (snapshot_date,))
    
    print("\nYarn Demand by Week:")
    print(f"  {'Week':10} {'Records':>10} {'Total Demand':>15}")
    print("  " + "-" * 40)
    for row in cursor.fetchall():
        week_label = f"Week {row[0]}" if row[0] > 0 else "Summary"
        if row[0] == 35:
            week_label = "This Week"
        elif row[0] == 99:
            week_label = "Later"
        total_demand = row[2] if row[2] else 0
        print(f"  {week_label:10} {row[1]:10,} {total_demand:15,.1f}")
    
    # File comparison
    print("\n" + "=" * 80)
    print("FILE VS DATABASE COMPARISON")
    print("=" * 80)
    
    file_counts = {
        'Yarn_Demand.xlsx': 177,
        'Yarn_Demand_By_Style.xlsx': 537,
        'Yarn_Demand_By_Style_KO.xlsx': 516,
        'Expected_Yarn_Report.xlsx': 81,
    }
    
    print(f"\n{'File':30} {'File Rows':>12} {'DB Records':>12} {'Status'}")
    print("-" * 70)
    
    # Get actual counts
    cursor.execute("SELECT COUNT(DISTINCT yarn_id) FROM production.yarn_demand_ts WHERE snapshot_date = %s AND week_number = 0", (snapshot_date,))
    yd_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM production.style_yarn_demand WHERE snapshot_date = %s", (snapshot_date,))
    syd_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM production.knit_order_yarn_demand WHERE snapshot_date = %s", (snapshot_date,))
    koyd_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM production.expected_yarn_receipts WHERE snapshot_date = %s", (snapshot_date,))
    eyr_count = cursor.fetchone()[0]
    
    db_counts = {
        'Yarn_Demand.xlsx': yd_count,
        'Yarn_Demand_By_Style.xlsx': syd_count,
        'Yarn_Demand_By_Style_KO.xlsx': koyd_count,
        'Expected_Yarn_Report.xlsx': eyr_count,
    }
    
    all_match = True
    for file_name, file_count in file_counts.items():
        db_count = db_counts.get(file_name, 0)
        if file_count == db_count:
            status = '✅ Match'
        else:
            status = f'⚠️  {db_count - file_count:+d}'
            all_match = False
        print(f"{file_name:30} {file_count:12,} {db_count:12,}  {status}")
    
    conn.close()
    
    print("\n" + "=" * 80)
    if all_match:
        print("✅ SUCCESS! ALL YARN DEMAND DATA IS FULLY LOADED!")
    else:
        print("✅ YARN DEMAND LOADING COMPLETE! Check any mismatches above.")
    print("=" * 80)

if __name__ == "__main__":
    load_all_yarn_demand()