#!/usr/bin/env python3
"""
Database Review Tool for Beverly Knits ERP
Quick way to explore your PostgreSQL database
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd

def review_database():
    """Review database contents"""
    
    # Connect to database
    conn = psycopg2.connect(
        host='localhost',
        database='beverly_knits_erp',
        user='erp_user',
        password='erp_password'
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("=" * 80)
    print("BEVERLY KNITS ERP DATABASE REVIEW")
    print("=" * 80)
    
    # 1. Database Overview
    print("\n📊 DATABASE OVERVIEW")
    print("-" * 40)
    
    # Get all tables with row counts
    cursor.execute("""
        SELECT 
            table_schema as schema,
            table_name as table
        FROM information_schema.tables
        WHERE table_schema IN ('production', 'api', 'analytics')
        AND table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name
    """)
    
    tables = cursor.fetchall()
    for table in tables:
        # Get row count for each table
        cursor.execute(f"SELECT COUNT(*) as count FROM {table['schema']}.{table['table']}")
        count = cursor.fetchone()['count']
        print(f"{table['schema']}.{table['table']:30} {count:,} rows")
    
    # 2. Sample Data from Key Tables
    print("\n📋 SAMPLE DATA")
    print("-" * 40)
    
    # Yarns
    print("\n🧶 YARNS (Top 5):")
    cursor.execute("""
        SELECT desc_id, yarn_description, blend, yarn_type
        FROM production.yarns
        LIMIT 5
    """)
    yarns = cursor.fetchall()
    for yarn in yarns:
        print(f"  {yarn['desc_id']:10} {yarn['yarn_description'][:40]:40} {yarn['blend'][:20] if yarn['blend'] else 'N/A'}")
    
    # Yarn Inventory
    print("\n📦 YARN INVENTORY STATUS:")
    cursor.execute("""
        SELECT 
            COUNT(*) as total_yarns,
            COUNT(CASE WHEN planning_balance < 0 THEN 1 END) as shortages,
            COUNT(CASE WHEN planning_balance > 1000 THEN 1 END) as overstocked,
            ROUND(AVG(weeks_of_supply)::numeric, 1) as avg_weeks_supply
        FROM production.yarn_inventory_ts
        WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM production.yarn_inventory_ts)
    """)
    inv_stats = cursor.fetchone()
    print(f"  Total Yarns: {inv_stats['total_yarns']}")
    print(f"  Shortages: {inv_stats['shortages']}")
    print(f"  Overstocked: {inv_stats['overstocked']}")
    print(f"  Avg Weeks Supply: {inv_stats['avg_weeks_supply']}")
    
    # Critical Yarns
    print("\n⚠️ CRITICAL YARNS (Negative Balance):")
    cursor.execute("""
        SELECT 
            y.desc_id,
            y.yarn_description,
            yi.planning_balance
        FROM production.yarns y
        JOIN production.yarn_inventory_ts yi ON y.yarn_id = yi.yarn_id
        WHERE yi.planning_balance < 0
        AND yi.snapshot_date = (SELECT MAX(snapshot_date) FROM production.yarn_inventory_ts)
        ORDER BY yi.planning_balance
        LIMIT 5
    """)
    critical = cursor.fetchall()
    if critical:
        for yarn in critical:
            print(f"  {yarn['desc_id']:10} {yarn['yarn_description'][:35]:35} Balance: {yarn['planning_balance']:,.0f} lbs")
    else:
        print("  None - All yarns have positive balance!")
    
    # Sales Orders
    print("\n📈 SALES ORDERS:")
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT so_number) as total_orders,
            COUNT(DISTINCT customer_id) as total_customers,
            SUM(balance) as total_backlog,
            MIN(ship_date) as earliest_ship,
            MAX(ship_date) as latest_ship
        FROM production.sales_orders_ts
        WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM production.sales_orders_ts)
    """)
    sales_stats = cursor.fetchone()
    print(f"  Total Orders: {sales_stats['total_orders']}")
    print(f"  Total Customers: {sales_stats['total_customers']}")
    print(f"  Total Backlog: {sales_stats['total_backlog']:,.0f} units")
    print(f"  Ship Dates: {sales_stats['earliest_ship']} to {sales_stats['latest_ship']}")
    
    # Top Customers
    print("\n👥 TOP CUSTOMERS BY ORDER VOLUME:")
    cursor.execute("""
        SELECT 
            customer_id,
            COUNT(DISTINCT so_number) as order_count,
            SUM(quantity_ordered) as total_ordered
        FROM production.sales_orders_ts
        WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM production.sales_orders_ts)
        GROUP BY customer_id
        ORDER BY total_ordered DESC
        LIMIT 5
    """)
    customers = cursor.fetchall()
    for cust in customers:
        print(f"  Customer {cust['customer_id']:15} Orders: {cust['order_count']:3}  Volume: {cust['total_ordered']:,.0f}")
    
    # 3. Data Quality Check
    print("\n✅ DATA QUALITY CHECK")
    print("-" * 40)
    
    # Check for nulls in critical fields
    cursor.execute("""
        SELECT 
            'Yarns without description' as check,
            COUNT(*) as count
        FROM production.yarns
        WHERE yarn_description IS NULL OR yarn_description = ''
        UNION ALL
        SELECT 
            'Orders without customer' as check,
            COUNT(*) as count
        FROM production.sales_orders_ts
        WHERE customer_id IS NULL
        UNION ALL
        SELECT 
            'Inventory without cost' as check,
            COUNT(*) as count
        FROM production.yarn_inventory_ts
        WHERE cost_per_pound IS NULL OR cost_per_pound = 0
    """)
    quality_checks = cursor.fetchall()
    for check in quality_checks:
        status = "✓" if check['count'] == 0 else "⚠"
        print(f"  {status} {check['check']:35} {check['count']}")
    
    # 4. Latest Data Timestamps
    print("\n🕐 DATA FRESHNESS")
    print("-" * 40)
    
    cursor.execute("""
        SELECT 
            'Yarn Inventory' as data_type,
            MAX(snapshot_date) as latest_date
        FROM production.yarn_inventory_ts
        UNION ALL
        SELECT 
            'Sales Orders' as data_type,
            MAX(snapshot_date) as latest_date
        FROM production.sales_orders_ts
        UNION ALL
        SELECT 
            'Fabric Inventory' as data_type,
            MAX(snapshot_date) as latest_date
        FROM production.fabric_inventory_ts
        UNION ALL
        SELECT 
            'Knit Orders' as data_type,
            MAX(snapshot_date) as latest_date
        FROM production.knit_orders_ts
    """)
    timestamps = cursor.fetchall()
    for ts in timestamps:
        date_str = str(ts['latest_date']) if ts['latest_date'] else 'No data'
        print(f"  {ts['data_type']:20} {date_str}")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("Review complete! Database is operational.")
    print("=" * 80)

if __name__ == "__main__":
    try:
        review_database()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure PostgreSQL is running and credentials are correct:")
        print("  Database: beverly_knits_erp")
        print("  User: erp_user")
        print("  Password: erp_password")