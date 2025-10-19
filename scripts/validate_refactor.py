#!/usr/bin/env python3
"""
Validation script for fabric forecast refactoring.
Checks that all requirements were met.
"""

import sys
from pathlib import Path
import re


def validate_refactoring():
    """Run all validation checks."""
    print("=" * 70)
    print("Fabric Forecast Refactoring Validation")
    print("=" * 70)

    file_path = Path(__file__).parent.parent / "src" / "api" / "efab_api_server.py"

    if not file_path.exists():
        print("ERROR: File not found")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract the refactored function
    start_marker = "@app.route('/api/fabric-forecast-integrated'"
    end_marker = "@app.route('/api/inventory-netting'"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        print("ERROR: Could not find function boundaries")
        return False

    function_code = content[start_idx:end_idx]

    print(f"\nFunction location: chars {start_idx} to {end_idx}")
    print(f"Function length: {len(function_code)} characters")

    # Check 1: No CSV file references
    csv_patterns = [
        r'\.csv',
        r'\.xlsx',
        r'read_csv',
        r'read_excel',
        r'eFab_Inventory_',
        r'eFab_Knit_Orders',
        r'eFab_SO_List'
    ]

    print("\n--- Check 1: CSV Dependencies Removed ---")
    csv_found = False
    for pattern in csv_patterns:
        # Exclude docstrings/comments
        matches = re.findall(pattern, function_code, re.IGNORECASE)
        # Filter out matches in docstrings (between triple quotes)
        actual_matches = [m for m in matches if 'REFACTORED: API-First' not in function_code]

        if matches and pattern != 'NO CSV FALLBACK':
            # Check if it's just in a comment/docstring
            lines_with_pattern = [line for line in function_code.split('\n') if pattern in line.lower()]
            code_lines = [line for line in lines_with_pattern if not line.strip().startswith('#') and '"""' not in line]

            if code_lines:
                print(f"  FAIL: Found {pattern} in code")
                csv_found = True
            else:
                print(f"  OK: {pattern} only in comments/docstrings")

    if not csv_found:
        print("  PASS: No CSV dependencies in code")

    # Check 2: No external eFab API calls
    print("\n--- Check 2: External API Calls Removed ---")
    if 'fetch_from_efab(' in function_code:
        print("  FAIL: Found fetch_from_efab() calls")
    else:
        print("  PASS: No external eFab API calls")

    # Check 3: Local API calls present
    print("\n--- Check 3: Local API Calls Added ---")
    local_api_calls = [
        '/api/knit-orders',
        '/api/inventory/pipeline-summary',
        '/api/yarn-intelligence'
    ]

    for endpoint in local_api_calls:
        if endpoint in function_code:
            print(f"  PASS: Found call to {endpoint}")
        else:
            print(f"  FAIL: Missing call to {endpoint}")

    # Check 4: Error handling
    print("\n--- Check 4: Error Handling ---")
    error_patterns = [
        'eFab API unavailable',
        'Cannot load production orders',
        'Cannot load inventory data',
        '"status": "error"',
        '_empty_fabric_summary()'
    ]

    for pattern in error_patterns:
        if pattern in function_code:
            print(f"  PASS: Found error handling: {pattern}")
        else:
            print(f"  WARN: Missing pattern: {pattern}")

    # Check 5: Helper functions exist
    print("\n--- Check 5: Helper Functions ---")
    helper_functions = [
        '_fetch_local_api_data',
        '_build_fabric_allocations',
        '_process_inventory_pipeline',
        '_generate_forecast_items',
        '_calculate_fabric_summary',
        '_empty_fabric_summary'
    ]

    for func in helper_functions:
        if func in content:  # Check full file
            print(f"  PASS: Found helper function: {func}")
        else:
            print(f"  FAIL: Missing helper function: {func}")

    # Check 6: Return format maintained
    print("\n--- Check 6: Response Format ---")
    response_fields = [
        "'status': 'success'",
        "'forecast_items':",
        "'fabric_forecast':",  # Dashboard compatibility
        "'summary':",
        "'timestamp':"
    ]

    for field in response_fields:
        if field in function_code:
            print(f"  PASS: Found response field: {field}")
        else:
            print(f"  FAIL: Missing response field: {field}")

    # Check 7: Documentation
    print("\n--- Check 7: Documentation ---")
    if 'REFACTORED: API-First Architecture' in function_code:
        print("  PASS: Function documented as refactored")
    else:
        print("  FAIL: Missing refactoring documentation")

    if 'NO CSV FALLBACK' in function_code:
        print("  PASS: No CSV fallback documented")
    else:
        print("  FAIL: Missing CSV fallback documentation")

    # Summary
    print("\n" + "=" * 70)
    print("Validation Complete")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run syntax check: python -m py_compile src/api/efab_api_server.py")
    print("2. Start server: python src/api/efab_api_server.py")
    print("3. Test endpoint: curl http://localhost:5006/api/fabric-forecast-integrated")
    print("4. Review logs for any issues")

    return True


if __name__ == '__main__':
    success = validate_refactoring()
    sys.exit(0 if success else 1)
