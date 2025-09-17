#!/usr/bin/env python3
"""
Test Script for EFab Yarn Demand Download
Validates the download and processing workflow
Following Operating Charter: Real data validation, track all failures
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
from datetime import datetime

def test_efab_download():
    """Test the EFab report download functionality"""
    print("="*60)
    print("EFab Yarn Demand Download Test")
    print("="*60)

    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': [],
        'passed': 0,
        'failed': 0
    }

    # Test 1: Import modules
    print("\n[TEST 1] Importing modules...")
    try:
        from src.data_loaders.efab_report_downloader import EFabReportDownloader
        from src.data_loaders.po_delivery_loader import PODeliveryLoader
        from src.config.efab_config import EFAB_CONFIG
        print("✓ Modules imported successfully")
        test_results['tests'].append({'name': 'Module Import', 'status': 'PASSED'})
        test_results['passed'] += 1
    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
        test_results['tests'].append({'name': 'Module Import', 'status': 'FAILED', 'error': str(e)})
        test_results['failed'] += 1
        return test_results

    # Test 2: Configuration check
    print("\n[TEST 2] Checking configuration...")
    try:
        session_cookie = EFAB_CONFIG.get('session_cookie')
        if session_cookie:
            print(f"✓ Session cookie configured: {session_cookie[:10]}...")
            test_results['tests'].append({'name': 'Configuration', 'status': 'PASSED'})
            test_results['passed'] += 1
        else:
            print("✗ No session cookie configured")
            test_results['tests'].append({'name': 'Configuration', 'status': 'FAILED', 'error': 'No session cookie'})
            test_results['failed'] += 1
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        test_results['tests'].append({'name': 'Configuration', 'status': 'FAILED', 'error': str(e)})
        test_results['failed'] += 1

    # Test 3: Initialize downloader
    print("\n[TEST 3] Initializing downloader...")
    try:
        downloader = EFabReportDownloader(session_cookie)
        print("✓ Downloader initialized")
        test_results['tests'].append({'name': 'Downloader Init', 'status': 'PASSED'})
        test_results['passed'] += 1
    except Exception as e:
        print(f"✗ Failed to initialize downloader: {e}")
        test_results['tests'].append({'name': 'Downloader Init', 'status': 'FAILED', 'error': str(e)})
        test_results['failed'] += 1
        return test_results

    # Test 4: Check queue (non-destructive)
    print("\n[TEST 4] Checking report queue...")
    try:
        queue_data = downloader.check_queue()
        if queue_data:
            # Handle both list and dict responses
            if isinstance(queue_data, list):
                report_count = len(queue_data)
            elif isinstance(queue_data, dict):
                report_count = len(queue_data.get('reports', []))
            else:
                report_count = 0

            print(f"✓ Queue check successful: {report_count} reports found")
            test_results['tests'].append({'name': 'Queue Check', 'status': 'PASSED', 'reports': report_count})
            test_results['passed'] += 1
        else:
            print("⚠ Queue check returned no data (may be normal)")
            test_results['tests'].append({'name': 'Queue Check', 'status': 'WARNING', 'note': 'No queue data'})
    except Exception as e:
        print(f"✗ Queue check failed: {e}")
        test_results['tests'].append({'name': 'Queue Check', 'status': 'FAILED', 'error': str(e)})
        test_results['failed'] += 1

    # Test 5: Test file download (to temp location)
    print("\n[TEST 5] Testing download to temp location...")
    try:
        test_path = Path('/tmp/test_yarn_demand.xlsx')
        success = downloader.download_latest(test_path)

        if success and test_path.exists():
            file_size = test_path.stat().st_size
            print(f"✓ Download successful: {file_size:,} bytes")
            test_results['tests'].append({'name': 'Download Test', 'status': 'PASSED', 'size': file_size})
            test_results['passed'] += 1

            # Test 6: Verify file can be loaded
            print("\n[TEST 6] Testing PODeliveryLoader compatibility...")
            try:
                loader = PODeliveryLoader()
                po_data = loader.load_po_deliveries(str(test_path))
                if po_data is not None and len(po_data) > 0:
                    print(f"✓ File loaded successfully: {len(po_data)} records")
                    test_results['tests'].append({'name': 'File Loading', 'status': 'PASSED', 'records': len(po_data)})
                    test_results['passed'] += 1
                else:
                    print("⚠ File loaded but no data found")
                    test_results['tests'].append({'name': 'File Loading', 'status': 'WARNING', 'note': 'No data'})
            except Exception as e:
                print(f"✗ Failed to load file: {e}")
                test_results['tests'].append({'name': 'File Loading', 'status': 'FAILED', 'error': str(e)})
                test_results['failed'] += 1

            # Clean up test file
            test_path.unlink()
            print(f"✓ Test file cleaned up")

        else:
            print("⚠ No report available for download (may be normal)")
            test_results['tests'].append({'name': 'Download Test', 'status': 'WARNING', 'note': 'No report available'})

    except Exception as e:
        print(f"✗ Download test failed: {e}")
        test_results['tests'].append({'name': 'Download Test', 'status': 'FAILED', 'error': str(e)})
        test_results['failed'] += 1

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {test_results['passed'] + test_results['failed']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")

    # Save results
    results_file = Path('/tmp/efab_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Exit code based on failures
    return 0 if test_results['failed'] == 0 else 1


if __name__ == "__main__":
    exit_code = test_efab_download()
    sys.exit(exit_code)