#!/usr/bin/env python3
"""
Test script for modularized components
Verifies that the extracted services work correctly
"""
import sys
import os
sys.path.insert(0, '/mnt/c/finalee/beverly_knits_erp_v2/src')

from services.inventory_analyzer_core import InventoryAnalyzer
from services.inventory_pipeline_core import InventoryManagementPipeline
import pandas as pd


def test_inventory_analyzer():
    """Test the extracted InventoryAnalyzer class"""
    print("Testing InventoryAnalyzer...")
    
    analyzer = InventoryAnalyzer()
    
    # Test with sample data
    test_inventory = [
        {'id': 'YARN001', 'quantity': 50},
        {'id': 'YARN002', 'quantity': 5},
        {'id': 'YARN003', 'quantity': 200}
    ]
    
    test_forecast = {
        'YARN001': 100,
        'YARN002': 50,
        'YARN003': 150
    }
    
    # Test analyze_inventory_levels
    analysis = analyzer.analyze_inventory_levels(test_inventory, test_forecast)
    print(f"  ✓ analyze_inventory_levels returned {len(analysis)} items")
    
    # Test risk calculation
    risk = analyzer.calculate_risk(current=50, required=100, days_supply=5)
    assert risk == 'CRITICAL', f"Expected CRITICAL, got {risk}"
    print(f"  ✓ calculate_risk working correctly")
    
    # Test analyze_inventory
    sample_df = pd.DataFrame({
        'Desc#': ['YARN001', 'YARN002', 'YARN003'],
        'Planning Balance': [-50, 75, 150]
    })
    
    inventory_analysis = analyzer.analyze_inventory(sample_df)
    assert 'total_items' in inventory_analysis
    assert inventory_analysis['total_items'] == 3
    print(f"  ✓ analyze_inventory processed {inventory_analysis['total_items']} items")
    print(f"  ✓ Found {inventory_analysis['summary']['critical_count']} critical items")
    
    print("✅ InventoryAnalyzer tests passed!\n")
    return True


def test_inventory_pipeline():
    """Test the extracted InventoryManagementPipeline class"""
    print("Testing InventoryManagementPipeline...")
    
    pipeline = InventoryManagementPipeline()
    
    # Create test data
    inventory_df = pd.DataFrame({
        'Description': ['YARN001', 'YARN002', 'YARN003'],
        'Planning Balance': [50, -20, 150]
    })
    
    sales_df = pd.DataFrame({
        'Description': ['YARN001', 'YARN002', 'YARN003'],
        'Consumed': [100, 50, 75]
    })
    
    # Run complete analysis
    results = pipeline.run_complete_analysis(
        sales_data=sales_df,
        inventory_data=inventory_df,
        yarn_data=inventory_df
    )
    
    assert 'sales_forecast' in results
    print(f"  ✓ Generated sales forecast for {len(results['sales_forecast'])} items")
    
    assert 'inventory_analysis' in results
    print(f"  ✓ Analyzed {len(results.get('inventory_analysis', []))} inventory items")
    
    assert 'recommendations' in results
    print(f"  ✓ Generated {len(results['recommendations'])} recommendations")
    
    # Test production plan generation
    test_analysis = [
        {'product_id': 'YARN001', 'reorder_needed': True, 'reorder_quantity': 100, 'shortage_risk': 'HIGH'},
        {'product_id': 'YARN002', 'reorder_needed': False, 'reorder_quantity': 0, 'shortage_risk': 'LOW'}
    ]
    
    production_plan = pipeline.generate_production_plan(test_analysis, {'YARN001': 50})
    assert 'YARN001' in production_plan
    assert production_plan['YARN001']['priority'] == 'HIGH'
    print(f"  ✓ Generated production plan for {len(production_plan)} items")
    
    print("✅ InventoryManagementPipeline tests passed!\n")
    return True


def test_integration():
    """Test that the services work together"""
    print("Testing Integration...")
    
    # Create analyzer and pipeline
    analyzer = InventoryAnalyzer()
    pipeline = InventoryManagementPipeline()
    
    # Verify pipeline uses analyzer
    assert hasattr(pipeline, 'inventory_analyzer')
    assert isinstance(pipeline.inventory_analyzer, InventoryAnalyzer)
    print("  ✓ Pipeline correctly integrates InventoryAnalyzer")
    
    # Test data flow
    test_data = pd.DataFrame({
        'Description': ['YARN001'],
        'Planning Balance': [-100]
    })
    
    # Direct analyzer test
    direct_analysis = analyzer.analyze_inventory(test_data)
    
    # Pipeline test  
    pipeline_results = pipeline.run_complete_analysis(inventory_data=test_data)
    
    print("  ✓ Both direct and pipeline analysis completed successfully")
    
    print("✅ Integration tests passed!\n")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("MODULARIZATION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        # Test individual components
        test_inventory_analyzer()
        test_inventory_pipeline()
        test_integration()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("The modularized components are working correctly.")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())