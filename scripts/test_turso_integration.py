#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Turso Integration for Forecast System
Verifies all Turso tables and data flows work correctly
"""

import os
import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

def test_turso_connection():
    """Test basic Turso connection"""
    print("\n" + "="*80)
    print("TEST 1: Turso Database Connection")
    print("="*80)

    try:
        from src.database.turso_client import get_turso_client

        client = get_turso_client()
        result = client.execute("SELECT 1 as test")

        if result:
            print("✅ Connected to Turso successfully")
            return True
        else:
            print("❌ Connection failed")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_style_mappings():
    """Test style mappings table and StyleMapper"""
    print("\n" + "="*80)
    print("TEST 2: Style Mappings")
    print("="*80)

    try:
        from src.utils.style_mapper import get_style_mapper

        mapper = get_style_mapper()
        stats = mapper.get_mapping_stats()

        print(f"Total fStyles: {stats['total_fstyles']}")
        print(f"Unique gBases: {stats['unique_gbases']}")
        print(f"Source: {stats['source']}")

        if stats['total_fstyles'] > 0:
            print("✅ Style mappings loaded from Turso")

            # Test a mapping
            test_fstyle = list(mapper.fstyle_to_gbase.keys())[0]
            gbase = mapper.fstyle_to_gbase[test_fstyle]
            print(f"✓ Sample mapping: {test_fstyle} → {gbase}")
            return True
        else:
            print("⚠️  No style mappings found")
            print("   Run: python scripts/import_style_mappings_to_turso.py")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bom_table():
    """Test BOM table access"""
    print("\n" + "="*80)
    print("TEST 3: BOM Table")
    print("="*80)

    try:
        from src.database.turso_client import get_turso_client

        client = get_turso_client()

        # Count BOM entries
        result = client.execute("SELECT COUNT(*) as cnt FROM bom")
        if result and len(result) > 0:
            count = result[0].get('cnt', 0)
            print(f"Total BOM entries: {count}")

        # Get sample
        result = client.execute("SELECT * FROM bom LIMIT 3")
        if result:
            print("\nSample BOM entries:")
            for row in result:
                print(f"  Style: {row.get('style')} → Yarn: {row.get('yarn_id')} ({row.get('percentage')*100:.1f}%)")
            print("✅ BOM table accessible")
            return True
        else:
            print("⚠️  No BOM data found")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_fabric_specs():
    """Test fabric_specs table"""
    print("\n" + "="*80)
    print("TEST 4: Fabric Specs Table")
    print("="*80)

    try:
        from src.database.turso_client import get_turso_client

        client = get_turso_client()

        result = client.execute("SELECT COUNT(*) as cnt FROM fabric_specs")
        if result and len(result) > 0:
            count = result[0].get('cnt', 0)
            print(f"Total fabric specs: {count}")

        result = client.execute("SELECT * FROM fabric_specs LIMIT 3")
        if result:
            print("\nSample fabric specs:")
            for row in result:
                print(f"  Style: {row.get('style')} → {row.get('yds_per_lb')} yds/lb")
            print("✅ Fabric specs table accessible")
            return True
        else:
            print("⚠️  No fabric specs found")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_historical_sales():
    """Test historical_sales table"""
    print("\n" + "="*80)
    print("TEST 5: Historical Sales Table")
    print("="*80)

    try:
        from src.database.turso_client import get_turso_client

        client = get_turso_client()

        result = client.execute("SELECT COUNT(*) as cnt FROM historical_sales")
        if result and len(result) > 0:
            count = result[0].get('cnt', 0)
            print(f"Total sales records: {count}")

        result = client.execute("SELECT * FROM historical_sales LIMIT 3")
        if result:
            print("\nSample sales records:")
            for row in result:
                print(f"  {row.get('date')}: {row.get('style')} - {row.get('quantity')} {row.get('units')}")
            print("✅ Historical sales table accessible")
            return True
        else:
            print("⚠️  No historical sales data found")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_forecast_tables():
    """Test new forecast tables"""
    print("\n" + "="*80)
    print("TEST 6: Forecast Tables")
    print("="*80)

    try:
        from src.database.turso_client import get_turso_client

        client = get_turso_client()

        tables = [
            'external_forecasts',
            'forecast_accuracy',
            'forecast_blend_weights'
        ]

        all_ok = True
        for table in tables:
            try:
                result = client.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                if result and len(result) > 0:
                    count = result[0].get('cnt', 0)
                    print(f"✓ {table}: {count} records")
                else:
                    print(f"✓ {table}: exists (empty)")
            except Exception as e:
                print(f"❌ {table}: {e}")
                all_ok = False

        if all_ok:
            print("✅ All forecast tables accessible")
            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "🔍 "*20)
    print("TURSO INTEGRATION TEST SUITE")
    print("🔍 "*20)

    results = []

    results.append(("Turso Connection", test_turso_connection()))
    results.append(("Style Mappings", test_style_mappings()))
    results.append(("BOM Table", test_bom_table()))
    results.append(("Fabric Specs", test_fabric_specs()))
    results.append(("Historical Sales", test_historical_sales()))
    results.append(("Forecast Tables", test_forecast_tables()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Turso integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
        print("\nCommon fixes:")
        print("  - Run: python scripts/import_bom_and_specs_to_turso.py")
        print("  - Run: python scripts/import_style_mappings_to_turso.py --create-table")
        print("  - Run: python scripts/import_sales_to_turso.py")


if __name__ == "__main__":
    main()
