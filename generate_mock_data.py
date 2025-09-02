#!/usr/bin/env python3
"""
Generate realistic mock data for Beverly Knits ERP
Matches production data structure but with anonymized values
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import random
import string

# Create mock data directory
MOCK_DATA_DIR = "/mnt/c/finalee/beverly_knits_erp_v2/data/mock"
os.makedirs(MOCK_DATA_DIR, exist_ok=True)
os.makedirs(f"{MOCK_DATA_DIR}/5", exist_ok=True)
os.makedirs(f"{MOCK_DATA_DIR}/5/ERP Data", exist_ok=True)

print("🔧 Generating realistic mock data for Beverly Knits ERP...")

# Generate mock yarn inventory data
def generate_yarn_inventory():
    """Generate realistic yarn inventory data"""
    print("📦 Generating yarn inventory data...")
    
    yarn_types = ['Cotton', 'Polyester', 'Nylon', 'Wool', 'Acrylic', 'Blend', 'Spandex', 'Rayon']
    colors = ['Black', 'White', 'Navy', 'Grey', 'Red', 'Blue', 'Green', 'Natural', 'Beige', 'Brown']
    suppliers = ['SupplierA', 'SupplierB', 'SupplierC', 'SupplierD', 'SupplierE']
    
    data = []
    for i in range(1200):  # Match production count of ~1200 yarns
        yarn_id = f"YRN{i+1000:04d}"
        desc = f"{random.choice(yarn_types)}-{random.choice(colors)}-{random.randint(10,100)}"
        
        # Generate realistic balance values
        beginning = random.uniform(100, 5000)
        consumed = -random.uniform(0, beginning * 0.3) if random.random() > 0.3 else 0
        received = random.uniform(0, 2000) if random.random() > 0.5 else 0
        theoretical = beginning + consumed + received
        allocated = -random.uniform(0, 1000) if random.random() > 0.4 else 0
        on_order = random.uniform(0, 3000) if random.random() > 0.6 else 0
        planning_balance = theoretical + allocated + on_order
        
        data.append({
            'yarn_id': yarn_id,
            'Desc#': desc,
            'QS Cust': f"CUST{random.randint(1,10):02d}",
            'supplier': random.choice(suppliers),
            'description': f"{desc} - Mock Yarn",
            'color': random.choice(colors),
            'beginning_balance': round(beginning, 2),
            'received': round(received, 2),
            'consumed': round(consumed, 2),
            'adjustments': 0,
            'theoretical_balance': round(theoretical, 2),
            'Misc': None,
            'on_order': round(on_order, 2),
            'allocated': round(allocated, 2),
            'planning_balance': round(planning_balance, 2),
            'Planning_Balance': round(planning_balance, 2),  # Duplicate for compatibility
            'reconcile_date': datetime.now().strftime('%Y-%m-%d'),
            'cost_per_pound': round(random.uniform(2, 15), 2),
            'total_cost': round(theoretical * random.uniform(2, 15), 2)
        })
    
    df = pd.DataFrame(data)
    df.to_excel(f"{MOCK_DATA_DIR}/5/yarn_inventory.xlsx", index=False)
    df.to_excel(f"{MOCK_DATA_DIR}/5/ERP Data/yarn_inventory.xlsx", index=False)
    print(f"✅ Generated {len(data)} yarn inventory records")
    return df

# Generate mock BOM data
def generate_bom_data():
    """Generate realistic BOM data"""
    print("🔨 Generating BOM data...")
    
    styles = [f"STY{i:04d}" for i in range(1, 301)]  # 300 styles
    yarns = [f"YRN{i+1000:04d}" for i in range(1200)]
    
    data = []
    for style in styles:
        # Each style uses 2-8 different yarns
        num_yarns = random.randint(2, 8)
        selected_yarns = random.sample(yarns, num_yarns)
        
        for yarn in selected_yarns:
            data.append({
                'Style#': style,
                'fStyle#': f"F{style}",
                'Yarn_ID': yarn,
                'Desc#': yarn,
                'Usage_LBS': round(random.uniform(0.5, 15), 2),
                'Component': f"Component-{random.randint(1,5)}",
                'Percentage': round(random.uniform(10, 100), 1)
            })
    
    df = pd.DataFrame(data)
    df.to_csv(f"{MOCK_DATA_DIR}/5/BOM_updated.csv", index=False)
    df.to_csv(f"{MOCK_DATA_DIR}/5/Style_BOM.csv", index=False)
    print(f"✅ Generated {len(data)} BOM entries")
    return df

# Generate mock sales data
def generate_sales_data():
    """Generate realistic sales activity data"""
    print("💰 Generating sales data...")
    
    styles = [f"STY{i:04d}" for i in range(1, 301)]
    customers = [f"Customer{i:02d}" for i in range(1, 51)]
    
    data = []
    for i in range(10000):  # Generate 10000 sales records
        order_date = datetime.now() - timedelta(days=random.randint(0, 365))
        ship_date = order_date + timedelta(days=random.randint(1, 30))
        
        data.append({
            'Order#': f"ORD{i+10000:06d}",
            'Style#': random.choice(styles),
            'fStyle#': f"F{random.choice(styles)}",
            'Customer': random.choice(customers),
            'Ordered': random.randint(10, 1000),
            'Picked/Shipped': random.randint(5, 900),
            'Open': random.randint(0, 100),
            'Order Date': order_date.strftime('%Y-%m-%d'),
            'Quoted Date': ship_date.strftime('%Y-%m-%d'),
            'Ship Date': ship_date.strftime('%Y-%m-%d'),
            'Price': round(random.uniform(10, 200), 2),
            'Status': random.choice(['Open', 'Shipped', 'Partial', 'Complete'])
        })
    
    df = pd.DataFrame(data)
    df.to_csv(f"{MOCK_DATA_DIR}/5/Sales Activity Report.csv", index=False)
    print(f"✅ Generated {len(data)} sales records")
    return df

# Generate mock knit orders
def generate_knit_orders():
    """Generate realistic knit production orders"""
    print("🏭 Generating knit orders...")
    
    styles = [f"STY{i:04d}" for i in range(1, 301)]
    machines = [f"Machine{i:02d}" for i in range(1, 21)]
    
    data = []
    for i in range(1500):  # Generate 1500 knit orders
        start_date = datetime.now() + timedelta(days=random.randint(-30, 60))
        end_date = start_date + timedelta(days=random.randint(1, 14))
        
        data.append({
            'KO#': f"KO{i+5000:05d}",
            'Style#': random.choice(styles),
            'fStyle#': f"F{random.choice(styles)}",
            'Quantity': random.randint(100, 5000),
            'Machine': random.choice(machines),
            'Start_Date': start_date.strftime('%Y-%m-%d'),
            'End_Date': end_date.strftime('%Y-%m-%d'),
            'Status': random.choice(['Planned', 'In Progress', 'Complete', 'On Hold']),
            'Priority': random.randint(1, 5),
            'Efficiency': round(random.uniform(0.7, 1.0), 2)
        })
    
    df = pd.DataFrame(data)
    df.to_excel(f"{MOCK_DATA_DIR}/5/eFab_Knit_Orders.xlsx", index=False)
    df.to_excel(f"{MOCK_DATA_DIR}/5/ERP Data/eFab_Knit_Orders.xlsx", index=False)
    print(f"✅ Generated {len(data)} knit orders")
    return df

# Generate mock inventory stage data
def generate_inventory_stages():
    """Generate inventory data for different production stages"""
    print("📊 Generating stage inventory data...")
    
    styles = [f"STY{i:04d}" for i in range(1, 301)]
    stages = {
        'F01': 'Finished Goods',
        'G00': 'Greige Stage 1', 
        'G02': 'Greige Stage 2',
        'I01': 'QC Inspection'
    }
    
    for stage, desc in stages.items():
        data = []
        for style in random.sample(styles, min(200, len(styles))):
            data.append({
                'Style#': style,
                'fStyle#': f"F{style}",
                'Stage': stage,
                'Description': desc,
                'Quantity': random.randint(0, 5000),
                'Location': f"Warehouse-{random.randint(1,5)}",
                'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        df = pd.DataFrame(data)
        filename = f"eFab_Inventory_{stage}.xlsx"
        df.to_excel(f"{MOCK_DATA_DIR}/5/ERP Data/{filename}", index=False)
        print(f"✅ Generated {len(data)} records for {stage} ({desc})")

# Generate configuration file
def generate_config():
    """Generate configuration to use mock data"""
    print("⚙️ Generating configuration...")
    
    config = {
        "data_source": {
            "type": "mock",
            "files": {
                "primary_path": MOCK_DATA_DIR,
                "fallback_path": f"{MOCK_DATA_DIR}/5/ERP Data"
            }
        },
        "mock_settings": {
            "enabled": True,
            "realistic": True,
            "record_counts": {
                "yarns": 1200,
                "styles": 300,
                "sales": 10000,
                "knit_orders": 1500,
                "bom_entries": 28000
            }
        }
    }
    
    with open(f"{MOCK_DATA_DIR}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration saved")

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("BEVERLY KNITS ERP - MOCK DATA GENERATOR")
    print("="*60 + "\n")
    
    # Generate all mock data
    yarn_df = generate_yarn_inventory()
    bom_df = generate_bom_data()
    sales_df = generate_sales_data()
    knit_df = generate_knit_orders()
    generate_inventory_stages()
    generate_config()
    
    print("\n" + "="*60)
    print("✨ MOCK DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"\n📁 Mock data saved to: {MOCK_DATA_DIR}")
    print("\nSummary:")
    print(f"  • Yarn Inventory: {len(yarn_df)} records")
    print(f"  • BOM Entries: {len(bom_df)} records")
    print(f"  • Sales Records: {len(sales_df)} records")
    print(f"  • Knit Orders: {len(knit_df)} records")
    print(f"  • Stage Inventories: 4 files generated")
    
    print("\n🚀 To use mock data, set environment variable:")
    print("   export DATA_BASE_DIR=\"" + MOCK_DATA_DIR + "/5\"")
    print("\n   Or run:")
    print(f"   DATA_BASE_DIR=\"{MOCK_DATA_DIR}/5\" python3 src/core/beverly_comprehensive_erp.py")
    print("\n" + "="*60)