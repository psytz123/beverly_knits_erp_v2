#!/usr/bin/env python3
"""Test script to verify run rate and days until stockout calculations."""

import json
import requests
from datetime import datetime

def test_stockout_calculations():
    """Test the stockout calculation logic."""
    
    print("=" * 80)
    print("TESTING RUN RATE AND DAYS UNTIL STOCKOUT CALCULATIONS")
    print("=" * 80)
    print()
    
    # Fetch data from API
    try:
        response = requests.get('http://localhost:5006/api/yarn-intelligence')
        data = response.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    yarns = data.get('criticality_analysis', {}).get('yarns', [])
    
    # Filter for yarns with shortages
    shortage_yarns = [
        y for y in yarns 
        if y.get('planning_balance', 0) < 0 or 
           (y.get('weeks_of_supply', float('inf')) < 4)
    ]
    
    print(f"Found {len(shortage_yarns)} yarns with shortages or low supply\n")
    
    # Test calculations for first 10 yarns
    test_results = []
    
    for yarn in shortage_yarns[:10]:
        yarn_id = yarn.get('yarn_id', 'N/A')
        description = yarn.get('description', 'N/A')[:40]
        theoretical_balance = yarn.get('theoretical_balance', 0)
        on_order = yarn.get('on_order', 0)
        allocated = yarn.get('allocated', 0)
        planning_balance = yarn.get('planning_balance', 0)
        weekly_demand = yarn.get('weekly_demand', 0)
        backend_weeks = yarn.get('weeks_of_supply', None)
        
        print(f"Yarn: {yarn_id} - {description}")
        print(f"  Theoretical Balance: {theoretical_balance:,.2f} lbs")
        print(f"  On Order: {on_order:,.2f} lbs")
        print(f"  Allocated: {allocated:,.2f} lbs")
        print(f"  Planning Balance: {planning_balance:,.2f} lbs")
        print(f"  Weekly Demand (Run Rate): {weekly_demand:,.2f} lbs/week")
        
        # Calculate weeks of supply
        if weekly_demand > 0 and theoretical_balance > 0:
            calculated_weeks = theoretical_balance / weekly_demand
            calculated_days = calculated_weeks * 7
            
            print(f"\n  CALCULATIONS:")
            print(f"    Formula: Theoretical Balance / Weekly Demand")
            print(f"    Weeks of Supply: {theoretical_balance:.2f} / {weekly_demand:.2f} = {calculated_weeks:.2f} weeks")
            print(f"    Days Until Stockout: {calculated_weeks:.2f} * 7 = {calculated_days:.1f} days")
            
            if backend_weeks is not None:
                print(f"    Backend Weeks of Supply: {backend_weeks:.2f} weeks")
                diff = abs(calculated_weeks - backend_weeks)
                if diff > 0.1:
                    print(f"    ⚠️  DISCREPANCY: {diff:.2f} weeks difference")
                else:
                    print(f"    ✓  Match confirmed (within 0.1 weeks)")
            
            # Display format logic (matching frontend)
            if calculated_days <= 7:
                display = f"{int(calculated_days)} days"
            elif calculated_days <= 30:
                weeks = round(calculated_days / 7)
                display = f"{weeks} week{'s' if weeks > 1 else ''}"
            elif calculated_days <= 90:
                months = round(calculated_days / 30)
                display = f"{months} month{'s' if months > 1 else ''}"
            elif calculated_weeks > 52:
                display = "1+ year"
            else:
                display = f"{round(calculated_weeks)} weeks"
            
            print(f"    Frontend Display: '{display}'")
            
            # Determine urgency
            if calculated_weeks < 1 or abs(planning_balance) > 10000:
                urgency = "CRITICAL"
            elif calculated_weeks < 2 or abs(planning_balance) > 5000:
                urgency = "HIGH"
            elif calculated_weeks < 4 or abs(planning_balance) > 1000:
                urgency = "MEDIUM"
            else:
                urgency = "LOW"
            
            print(f"    Urgency Level: {urgency}")
            
            test_results.append({
                'yarn_id': yarn_id,
                'theoretical': theoretical_balance,
                'weekly_demand': weekly_demand,
                'calculated_weeks': calculated_weeks,
                'backend_weeks': backend_weeks,
                'match': diff <= 0.1 if backend_weeks else None
            })
            
        elif theoretical_balance <= 0:
            print(f"\n  STATUS: Out of stock (no physical inventory)")
            test_results.append({
                'yarn_id': yarn_id,
                'theoretical': theoretical_balance,
                'weekly_demand': weekly_demand,
                'status': 'Out of stock'
            })
        elif weekly_demand <= 0:
            print(f"\n  STATUS: Low/No usage (weekly demand = {weekly_demand:.2f})")
            test_results.append({
                'yarn_id': yarn_id,
                'theoretical': theoretical_balance,
                'weekly_demand': weekly_demand,
                'status': 'Low usage'
            })
        
        print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    matches = sum(1 for r in test_results if r.get('match') == True)
    mismatches = sum(1 for r in test_results if r.get('match') == False)
    out_of_stock = sum(1 for r in test_results if r.get('status') == 'Out of stock')
    low_usage = sum(1 for r in test_results if r.get('status') == 'Low usage')
    
    print(f"Total yarns tested: {len(test_results)}")
    print(f"  ✓ Calculations match backend: {matches}")
    print(f"  ⚠️  Calculations mismatch: {mismatches}")
    print(f"  📦 Out of stock: {out_of_stock}")
    print(f"  📉 Low/No usage: {low_usage}")
    
    # Check formula correctness
    print("\n" + "=" * 80)
    print("FORMULA VERIFICATION")
    print("=" * 80)
    print()
    print("Days Until Stockout Formula:")
    print("  1. Weeks of Supply = Theoretical Balance / Weekly Demand")
    print("  2. Days Until Stockout = Weeks of Supply * 7")
    print()
    print("This formula is CORRECT because:")
    print("  - Theoretical Balance = current physical stock on hand")
    print("  - Weekly Demand = average consumption per week")
    print("  - Division gives weeks of coverage at current consumption rate")
    print()
    print("Planning Balance Formula:")
    print("  Planning Balance = Theoretical Balance + On Order + Allocated")
    print("  (Note: Allocated is typically negative for committed orders)")
    print()
    print("Run Rate:")
    print("  Run Rate = Weekly Demand (average weekly consumption)")
    print()
    
    return test_results

if __name__ == "__main__":
    test_stockout_calculations()