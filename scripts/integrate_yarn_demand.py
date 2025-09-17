#!/usr/bin/env python3
"""
Script to integrate Yarn Demand time-phased data with the ERP system
"""

import pandas as pd
import json
import sys
from pathlib import Path

def load_yarn_demand(file_path):
    """Load and parse the Yarn Demand CSV file"""
    try:
        # Read the CSV, skipping the first two rows (header info)
        df = pd.read_csv(file_path, skiprows=2, encoding='utf-8-sig')

        # Clean column names
        df.columns = df.columns.str.strip()

        # Rename columns for consistency
        column_mapping = {
            'Yarn': 'yarn_id',
            'Supplier': 'supplier',
            'Description': 'description',
            'Color': 'color',
            'Monday Inventory': 'monday_inventory',
            'Balance': 'balance',
            'Balance This Week': 'balance_week_37',
            'Balance Week 38': 'balance_week_38',
            'Balance Week 39': 'balance_week_39',
            'Balance Week 40': 'balance_week_40',
            'Balance Week 41': 'balance_week_41',
            'Balance Week 42': 'balance_week_42',
            'Balance Week 43': 'balance_week_43',
            'Balance Week 44': 'balance_week_44',
            'Balance Week 45': 'balance_week_45',
            'Balance Later': 'balance_later'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Convert yarn_id to string for consistency
        df['yarn_id'] = df['yarn_id'].astype(str)

        # Clean numeric columns (remove commas and parentheses for negatives)
        numeric_cols = ['monday_inventory', 'balance'] + [f'balance_week_{i}' for i in range(37, 46)] + ['balance_later']

        for col in numeric_cols:
            if col in df.columns:
                # Remove commas and handle parentheses for negative values
                df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = df[col].str.replace('(', '-').str.replace(')', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    except Exception as e:
        print(f"Error loading Yarn Demand file: {e}")
        return None

def compare_with_inventory(yarn_demand_df, inventory_path):
    """Compare Yarn Demand data with current inventory"""
    try:
        # Load inventory data
        inv_df = pd.read_csv(inventory_path)
        inv_df['yarn_id'] = inv_df['Desc#'].astype(str)

        # Merge datasets
        comparison = pd.merge(
            yarn_demand_df,
            inv_df[['yarn_id', 'Planning Balance', 'Allocated', 'On Order']],
            on='yarn_id',
            how='outer',
            suffixes=('_demand', '_inventory')
        )

        # Identify discrepancies
        comparison['has_shortage'] = comparison[[col for col in comparison.columns if 'balance_week' in col]].apply(
            lambda row: any(row < 0), axis=1
        )

        shortages = comparison[comparison['has_shortage'] == True]

        return comparison, shortages

    except Exception as e:
        print(f"Error comparing with inventory: {e}")
        return None, None

def generate_api_format(yarn_demand_df, inventory_df=None, filter_allocated_only=True):
    """Convert Yarn Demand data to API-compatible format

    Args:
        yarn_demand_df: DataFrame with yarn demand data
        inventory_df: Optional DataFrame with inventory data for allocated amounts
        filter_allocated_only: If True, only include yarns with allocated production
    """
    yarns_data = []

    # Load inventory data if provided to get allocated amounts
    allocated_map = {}
    if inventory_df is not None:
        inventory_df['yarn_id'] = inventory_df['Desc#'].astype(str)
        for _, inv_row in inventory_df.iterrows():
            yarn_id = str(inv_row['yarn_id'])
            allocated = float(inv_row.get('Allocated', 0))
            allocated_map[yarn_id] = allocated

    for _, row in yarn_demand_df.iterrows():
        yarn_id = str(row['yarn_id'])

        # Get allocated amount from inventory
        allocated = allocated_map.get(yarn_id, 0)

        # Skip if filtering for allocated only and this yarn has no allocation
        if filter_allocated_only and allocated >= 0:
            continue

        # Calculate weekly projections
        weekly_balances = {}
        for week in range(37, 46):
            col_name = f'balance_week_{week}'
            if col_name in row:
                weekly_balances[f'week_{week}'] = float(row[col_name])

        # Determine if yarn has shortage
        has_shortage = any(val < 0 for val in weekly_balances.values())

        yarn_data = {
            'yarn_id': yarn_id,
            'description': row['description'],
            'supplier': row['supplier'],
            'color': row['color'],
            'current_inventory': float(row['monday_inventory']),
            'planning_balance': float(row['balance']),
            'allocated': allocated,
            'weekly_balances': weekly_balances,
            'has_shortage': has_shortage,
            'risk_level': 'CRITICAL' if has_shortage else 'OK',
            'is_allocated': allocated < 0  # Negative means allocated to production
        }

        yarns_data.append(yarn_data)

    return {
        'status': 'success',
        'source': 'yarn_demand_integration',
        'filter': 'allocated_only' if filter_allocated_only else 'all',
        'total_yarns': len(yarns_data),
        'yarns_with_shortages': sum(1 for y in yarns_data if y['has_shortage']),
        'yarns_allocated': sum(1 for y in yarns_data if y['is_allocated']),
        'yarns': yarns_data
    }

def main():
    # File paths
    yarn_demand_path = "/mnt/c/Users/psytz/Downloads/Yarn_Demand_2025-09-14_0442.csv"
    inventory_path = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/yarn_inventory.csv"
    output_path = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/yarn_demand_integrated.json"

    print("Loading Yarn Demand data...")
    yarn_demand_df = load_yarn_demand(yarn_demand_path)

    if yarn_demand_df is not None:
        print(f"Loaded {len(yarn_demand_df)} yarns from Yarn Demand report")

        # Compare with inventory
        print("\nComparing with current inventory...")
        comparison, shortages = compare_with_inventory(yarn_demand_df, inventory_path)

        if shortages is not None:
            print(f"Found {len(shortages)} yarns with time-phased shortages")
            print("\nTop 10 yarns with shortages:")
            for _, yarn in shortages.head(10).iterrows():
                print(f"  - {yarn['yarn_id']}: {yarn['description']}")

        # Load inventory data for allocated amounts
        print("\nLoading inventory data for allocated amounts...")
        try:
            inventory_df = pd.read_csv(inventory_path)
            print(f"Loaded {len(inventory_df)} inventory records")
        except Exception as e:
            print(f"Warning: Could not load inventory data: {e}")
            inventory_df = None

        # Generate API format with allocated filter
        print("\nGenerating API-compatible format (allocated yarns only)...")
        api_data = generate_api_format(yarn_demand_df, inventory_df, filter_allocated_only=True)

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(api_data, f, indent=2)

        print(f"\nSaved integrated data to {output_path}")
        print(f"Filter: {api_data['filter']}")
        print(f"Total yarns (allocated to production): {api_data['total_yarns']}")
        print(f"Yarns with shortages: {api_data['yarns_with_shortages']}")
        print(f"Yarns allocated: {api_data['yarns_allocated']}")

        # Show sample yarn with shortage
        shortage_yarns = [y for y in api_data['yarns'] if y['has_shortage']]
        if shortage_yarns:
            sample = shortage_yarns[0]
            print(f"\nSample yarn with shortage:")
            print(f"  Yarn ID: {sample['yarn_id']}")
            print(f"  Description: {sample['description']}")
            print(f"  Current Inventory: {sample['current_inventory']}")
            print(f"  Planning Balance: {sample['planning_balance']}")
            print(f"  Allocated: {sample.get('allocated', 0)}")
            print(f"  Weekly Balances:")
            for week, balance in sample['weekly_balances'].items():
                status = "SHORTAGE" if balance < 0 else "OK"
                print(f"    {week}: {balance:,.0f} lbs ({status})")

    else:
        print("Failed to load Yarn Demand data")
        sys.exit(1)

if __name__ == "__main__":
    main()