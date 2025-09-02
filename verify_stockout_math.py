#!/usr/bin/env python3
"""Verify the math for run rate and days until stockout calculations."""

def verify_stockout_math():
    """Test the stockout calculation with sample data."""
    
    print("=" * 80)
    print("STOCKOUT MATH VERIFICATION")
    print("=" * 80)
    print()
    
    # Test cases with known values
    test_cases = [
        {
            'name': 'Case 1: Normal usage',
            'theoretical_balance': 16643.30,
            'weekly_demand': 7410.40,
            'expected_weeks': 2.25,
            'expected_days': 15.7,
            'expected_display': '2 weeks'
        },
        {
            'name': 'Case 2: Low usage',
            'theoretical_balance': 16469.57,
            'weekly_demand': 251.12,
            'expected_weeks': 65.58,
            'expected_days': 459.1,
            'expected_display': '66 weeks'
        },
        {
            'name': 'Case 3: Critical - less than 1 week',
            'theoretical_balance': 500,
            'weekly_demand': 1000,
            'expected_weeks': 0.5,
            'expected_days': 3.5,
            'expected_display': '4 days'
        },
        {
            'name': 'Case 4: Out of stock',
            'theoretical_balance': 0,
            'weekly_demand': 1000,
            'expected_weeks': 0,
            'expected_days': 0,
            'expected_display': 'Out of stock'
        },
        {
            'name': 'Case 5: No demand',
            'theoretical_balance': 1000,
            'weekly_demand': 0,
            'expected_weeks': float('inf'),
            'expected_days': float('inf'),
            'expected_display': 'Low usage'
        },
        {
            'name': 'Case 6: One month supply',
            'theoretical_balance': 4000,
            'weekly_demand': 1000,
            'expected_weeks': 4,
            'expected_days': 28,
            'expected_display': '4 weeks'
        },
        {
            'name': 'Case 7: Three months supply',
            'theoretical_balance': 12000,
            'weekly_demand': 1000,
            'expected_weeks': 12,
            'expected_days': 84,
            'expected_display': '3 months'
        },
        {
            'name': 'Case 8: One year+ supply',
            'theoretical_balance': 60000,
            'weekly_demand': 1000,
            'expected_weeks': 60,
            'expected_days': 420,
            'expected_display': '60 weeks'
        }
    ]
    
    all_passed = True
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print("-" * 40)
        print(f"  Theoretical Balance: {test['theoretical_balance']:,.2f} lbs")
        print(f"  Weekly Demand: {test['weekly_demand']:,.2f} lbs/week")
        
        # Calculate
        if test['weekly_demand'] > 0 and test['theoretical_balance'] > 0:
            weeks = test['theoretical_balance'] / test['weekly_demand']
            days = weeks * 7
            
            # Determine display format (matching frontend logic)
            if days <= 7:
                display = f"{round(days)} days"
            elif days <= 30:
                weeks_rounded = round(days / 7)
                display = f"{weeks_rounded} week{'s' if weeks_rounded > 1 else ''}"
            elif days <= 90:
                months = round(days / 30)
                display = f"{months} month{'s' if months > 1 else ''}"
            elif weeks > 52:
                display = "1+ year"
            else:
                display = f"{round(weeks)} weeks"
                
        elif test['theoretical_balance'] <= 0:
            weeks = 0
            days = 0
            display = 'Out of stock'
        else:  # weekly_demand <= 0
            weeks = float('inf')
            days = float('inf')
            display = 'Low usage'
        
        print(f"\n  CALCULATED:")
        print(f"    Weeks of Supply: {weeks:.2f}")
        print(f"    Days Until Stockout: {days:.1f}")
        print(f"    Display Format: '{display}'")
        
        print(f"\n  EXPECTED:")
        print(f"    Weeks of Supply: {test['expected_weeks']:.2f}")
        print(f"    Days Until Stockout: {test['expected_days']:.1f}")
        print(f"    Display Format: '{test['expected_display']}'")
        
        # Verify
        weeks_match = abs(weeks - test['expected_weeks']) < 0.01 if weeks != float('inf') else weeks == test['expected_weeks']
        days_match = abs(days - test['expected_days']) < 0.1 if days != float('inf') else days == test['expected_days']
        display_match = display == test['expected_display']
        
        if weeks_match and days_match and display_match:
            print(f"\n  ✓ PASS: All calculations correct")
        else:
            print(f"\n  ✗ FAIL:")
            if not weeks_match:
                print(f"    - Weeks mismatch: {weeks:.2f} vs {test['expected_weeks']:.2f}")
            if not days_match:
                print(f"    - Days mismatch: {days:.1f} vs {test['expected_days']:.1f}")
            if not display_match:
                print(f"    - Display mismatch: '{display}' vs '{test['expected_display']}'")
            all_passed = False
    
    print("\n" + "=" * 80)
    print("FORMULA VERIFICATION SUMMARY")
    print("=" * 80)
    
    print("\n✓ CORRECT FORMULAS:")
    print("  1. Run Rate = Weekly Demand (direct from data)")
    print("  2. Weeks of Supply = Theoretical Balance ÷ Weekly Demand")
    print("  3. Days Until Stockout = Weeks of Supply × 7")
    
    print("\n✓ DISPLAY LOGIC:")
    print("  - ≤ 7 days: Show as 'X days'")
    print("  - ≤ 30 days: Show as 'X week(s)'")
    print("  - ≤ 90 days: Show as 'X month(s)'")
    print("  - > 52 weeks: Show as '1+ year'")
    print("  - Otherwise: Show as 'X weeks'")
    
    print("\n✓ SPECIAL CASES:")
    print("  - Theoretical Balance ≤ 0: 'Out of stock'")
    print("  - Weekly Demand ≤ 0: 'Low usage'")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Math is correct!")
    else:
        print("\n⚠️  Some tests failed - Review calculations")
    
    return all_passed

if __name__ == "__main__":
    verify_stockout_math()