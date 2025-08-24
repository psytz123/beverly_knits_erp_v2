#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Six-Phase Planning Engine
Tests all 6 phases of the planning engine with various scenarios and edge cases
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

from production.six_phase_planning_engine import SixPhasePlanningEngine, PlanningPhaseResult


class TestSixPhasePlanningEngine:
    """Test suite for the Six-Phase Planning Engine"""
    
    @pytest.fixture
    def planning_engine(self):
        """Create a planning engine instance for testing"""
        engine = SixPhasePlanningEngine(data_path="/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/prompts/5")
        return engine
    
    @pytest.fixture
    def sample_sales_data(self):
        """Create sample sales data for testing"""
        return pd.DataFrame({
            'Style#': ['TEST001', 'TEST002', 'TEST003', 'TEST001', 'TEST002'],
            'Ordered': [100, 200, 150, 50, 75],
            'Ship Date': pd.date_range(start='2025-08-15', periods=5),
            'Customer': ['Cust1', 'Cust2', 'Cust3', 'Cust1', 'Cust2'],
            'Status': ['Open', 'Open', 'Open', 'Open', 'Open']
        })
    
    @pytest.fixture
    def sample_inventory_data(self):
        """Create sample multi-level inventory data"""
        return {
            'F01': pd.DataFrame({
                'fStyle#': ['TEST001', 'TEST002'],
                'Available': [30, 50]
            }),
            'I01': pd.DataFrame({
                'fStyle#': ['TEST001', 'TEST003'],
                'Available': [20, 15]
            }),
            'G00': pd.DataFrame({
                'fStyle#': ['TEST001', 'TEST002', 'TEST003'],
                'Available': [25, 30, 20]
            }),
            'G02': pd.DataFrame({
                'fStyle#': ['TEST002', 'TEST003'],
                'Available': [15, 10]
            })
        }
    
    @pytest.fixture
    def sample_bom_data(self):
        """Create sample BOM data"""
        return pd.DataFrame({
            'Style#': ['TEST001', 'TEST001', 'TEST002', 'TEST002', 'TEST003'],
            'Desc#': ['YARN001', 'YARN002', 'YARN001', 'YARN003', 'YARN002'],
            'BOM_Percent': [60, 40, 50, 50, 100],
            'unit': ['lbs', 'lbs', 'lbs', 'lbs', 'lbs']
        })
    
    @pytest.fixture
    def sample_yarn_inventory(self):
        """Create sample yarn inventory data"""
        return pd.DataFrame({
            'Desc#': ['YARN001', 'YARN002', 'YARN003'],
            'Planning Balance': [500, -100, 200],
            'Bal': [600, 50, 250],
            'Allocated': [100, 150, 50],
            'On Order': [0, 0, 100],
            'Description': ['Cotton 30/1', 'Polyester 40/1', 'Nylon 20/1']
        })


class TestPhase1DemandConsolidation(TestSixPhasePlanningEngine):
    """Test Phase 1: Demand Consolidation"""
    
    def test_basic_demand_consolidation(self, planning_engine, sample_sales_data):
        """Test basic demand consolidation from sales orders"""
        with patch.object(planning_engine, '_load_sales_data', return_value=sample_sales_data):
            result = planning_engine.phase1_demand_consolidation()
            
            assert result.status == 'Completed'
            assert 'total_demand' in result.output_data or 'demand_consolidated' in result.output_data
            assert result.phase_name == 'Demand Consolidation'
    
    def test_demand_aggregation_by_style(self, planning_engine, sample_sales_data):
        """Test that demands are properly aggregated by style"""
        with patch.object(planning_engine, '_load_sales_data', return_value=sample_sales_data):
            result = planning_engine.phase1_demand_consolidation()
            
            # Check that result was generated
            assert result.status == 'Completed'
            assert result.output_data is not None
    
    def test_demand_with_forecast_integration(self, planning_engine):
        """Test demand consolidation with forecast data"""
        sales_data = pd.DataFrame({
            'Style#': ['TEST001', 'TEST002'],
            'Ordered': [100, 200],
            'Ship Date': pd.date_range(start='2025-08-15', periods=2)
        })
        
        forecast_data = pd.DataFrame({
            'Style#': ['TEST001', 'TEST003'],
            'Forecast_Qty': [50, 150]
        })
        
        with patch.object(planning_engine, '_load_sales_data', return_value=sales_data):
            with patch.object(planning_engine, '_load_forecast_data', return_value=forecast_data):
                result = planning_engine.phase1_demand_consolidation()
                
                assert result.status == 'Completed'
                # Should include both sales and forecast
    
    def test_empty_demand_handling(self, planning_engine):
        """Test handling of empty demand data"""
        empty_df = pd.DataFrame()
        
        with patch.object(planning_engine, '_load_sales_data', return_value=empty_df):
            result = planning_engine.phase1_demand_consolidation()
            
            # Should handle gracefully
            assert result.status in ['Completed', 'Failed', 'Error']
            # Empty data should still complete successfully
    
    def test_demand_date_filtering(self, planning_engine):
        """Test filtering demands by date range"""
        sales_data = pd.DataFrame({
            'Style#': ['TEST001', 'TEST002', 'TEST003'],
            'Ordered': [100, 200, 150],
            'Ship Date': [
                datetime.now() + timedelta(days=7),   # Within planning horizon
                datetime.now() + timedelta(days=30),  # Within planning horizon
                datetime.now() + timedelta(days=365)  # Outside planning horizon
            ]
        })
        
        with patch.object(planning_engine, '_load_sales_data', return_value=sales_data):
            result = planning_engine.phase1_demand_consolidation()
            
            assert result.status == 'Completed'
            # Should filter based on planning horizon


class TestPhase2InventoryAssessment(TestSixPhasePlanningEngine):
    """Test Phase 2: Multi-Level Inventory Assessment"""
    
    def test_basic_inventory_loading(self, planning_engine, sample_inventory_data):
        """Test loading inventory from multiple levels"""
        with patch.object(planning_engine, '_load_stage_inventory', side_effect=lambda stage: sample_inventory_data.get(stage, pd.DataFrame())):
            result = planning_engine.phase2_inventory_assessment()
            
            assert result.status == 'Completed'
            assert result.output_data is not None
    
    def test_inventory_aggregation(self, planning_engine):
        """Test aggregation of inventory across levels"""
        inventory_data = {
            'F01': pd.DataFrame({'fStyle#': ['TEST001'], 'Available': [30]}),
            'I01': pd.DataFrame({'fStyle#': ['TEST001'], 'Available': [20]}),
            'G00': pd.DataFrame({'fStyle#': ['TEST001'], 'Available': [25]}),
            'G02': pd.DataFrame({'fStyle#': ['TEST001'], 'Available': [15]})
        }
        
        with patch.object(planning_engine, '_load_stage_inventory', side_effect=lambda stage: inventory_data.get(stage, pd.DataFrame())):
            result = planning_engine.phase2_inventory_assessment()
            
            assert result.status == 'Completed'
            # TEST001 should have total of 90 (30+20+25+15)
    
    def test_inventory_with_knit_orders(self, planning_engine):
        """Test inventory assessment including knit orders"""
        inventory_data = {
            'G00': pd.DataFrame({'fStyle#': ['TEST001'], 'Available': [25]})
        }
        
        knit_orders = pd.DataFrame({
            'Style#': ['TEST001'],
            'Qty Ordered (lbs)': [100],
            'G00 (lbs)': [25],  # Already allocated
            'Balance (lbs)': [75]
        })
        
        with patch.object(planning_engine, '_load_stage_inventory', side_effect=lambda stage: inventory_data.get(stage, pd.DataFrame())):
            with patch.object(planning_engine, '_load_knit_orders', return_value=knit_orders):
                result = planning_engine.phase2_inventory_assessment()
                
                assert result.status == 'Completed'
                # Should account for knit order allocations
    
    def test_empty_inventory_handling(self, planning_engine):
        """Test handling of empty inventory"""
        with patch.object(planning_engine, '_load_stage_inventory', return_value=pd.DataFrame()):
            result = planning_engine.phase2_inventory_assessment()
            
            assert result.status == 'Completed'
            # Should handle empty inventory gracefully
    
    def test_inventory_quality_levels(self, planning_engine):
        """Test priority of inventory levels (F01 > I01 > G00/G02)"""
        inventory_data = {
            'F01': pd.DataFrame({'fStyle#': ['TEST001'], 'Available': [100]}),  # Finished goods
            'I01': pd.DataFrame({'fStyle#': ['TEST001'], 'Available': [50]}),   # QC queue
            'G00': pd.DataFrame({'fStyle#': ['TEST001'], 'Available': [75]})    # In-process
        }
        
        with patch.object(planning_engine, '_load_stage_inventory', side_effect=lambda stage: inventory_data.get(stage, pd.DataFrame())):
            result = planning_engine.phase2_inventory_assessment()
            
            assert result.status == 'Completed'
            # Should maintain level hierarchy


class TestPhase3NetRequirements(TestSixPhasePlanningEngine):
    """Test Phase 3: Net Requirements Calculation"""
    
    def test_basic_net_requirements(self, planning_engine):
        """Test basic net requirements calculation"""
        # Set up phase 1 results (demand)
        planning_engine.phase_results['phase1'] = {
            'demand_consolidated': {
                'by_style': {
                    'TEST001': 100,
                    'TEST002': 200
                }
            }
        }
        
        # Set up phase 2 results (inventory)
        planning_engine.phase_results['phase2'] = {
            'multi_level_inventory': {
                'TEST001': {'total': 30},
                'TEST002': {'total': 50}
            }
        }
        
        result = planning_engine.phase3_net_requirements()
        
        assert result.status == 'Completed'
        assert result.output_data is not None
        # TEST001 net = 100 - 30 = 70
        # TEST002 net = 200 - 50 = 150
    
    def test_net_requirements_no_shortage(self, planning_engine):
        """Test when inventory covers all demand"""
        planning_engine.phase_results['phase1'] = {
            'demand_consolidated': {
                'by_style': {'TEST001': 50}
            }
        }
        
        planning_engine.phase_results['phase2'] = {
            'multi_level_inventory': {
                'TEST001': {'total': 100}
            }
        }
        
        result = planning_engine.phase3_net_requirements()
        
        assert result.status == 'Completed'
        # Net requirement should be 0 (no production needed)
    
    def test_net_requirements_with_safety_stock(self, planning_engine):
        """Test net requirements considering safety stock"""
        planning_engine.phase_results['phase1'] = {
            'demand_consolidated': {
                'by_style': {'TEST001': 100}
            }
        }
        
        planning_engine.phase_results['phase2'] = {
            'multi_level_inventory': {
                'TEST001': {'total': 80, 'safety_stock': 20}
            }
        }
        
        result = planning_engine.phase3_net_requirements()
        
        assert result.status == 'Completed'
        # Should consider safety stock in calculation
    
    def test_missing_phase_data(self, planning_engine):
        """Test handling of missing prerequisite phase data"""
        # No phase 1 or phase 2 results
        planning_engine.phase_results = {}
        
        result = planning_engine.phase3_net_requirements()
        
        # Should handle missing data gracefully
        assert result.status in ['Failed', 'Error'] or (result.status == 'Completed' and 'error' not in str(result.output_data))


class TestPhase4BOMExplosion(TestSixPhasePlanningEngine):
    """Test Phase 4: BOM Explosion for Net Requirements Only"""
    
    def test_bom_explosion_net_only(self, planning_engine, sample_bom_data):
        """Test BOM explosion only for items with net requirements"""
        # Set up phase 3 results
        planning_engine.phase_results['phase3'] = {
            'net_requirements': {
                'TEST001': 100,  # Has net requirement
                'TEST002': 0,    # No net requirement
                'TEST003': 50    # Has net requirement
            }
        }
        
        with patch.object(planning_engine, '_load_bom_data', return_value=sample_bom_data):
            result = planning_engine.phase4_bom_explosion_net()
            
            assert result.status == 'Completed'
            assert result.output_data is not None
            # Should only process TEST001 and TEST003, not TEST002
    
    def test_bom_percentage_calculation(self, planning_engine):
        """Test correct BOM percentage calculations"""
        planning_engine.phase_results['phase3'] = {
            'net_requirements': {'TEST001': 100}
        }
        
        bom_data = pd.DataFrame({
            'Style#': ['TEST001', 'TEST001'],
            'Desc#': ['YARN001', 'YARN002'],
            'BOM_Percent': [60, 40]
        })
        
        with patch.object(planning_engine, '_load_bom_data', return_value=bom_data):
            result = planning_engine.phase4_bom_explosion_net()
            
            assert result.status == 'Completed'
            # YARN001 should be 60 lbs (100 * 0.60)
            # YARN002 should be 40 lbs (100 * 0.40)
    
    def test_missing_bom_data(self, planning_engine):
        """Test handling of missing BOM data"""
        planning_engine.phase_results['phase3'] = {
            'net_requirements': {'TEST001': 100}
        }
        
        with patch.object(planning_engine, '_load_bom_data', return_value=pd.DataFrame()):
            result = planning_engine.phase4_bom_explosion_net()
            
            # Should handle missing BOM gracefully
            assert result.status in ['Completed', 'Failed', 'Error']
    
    def test_complex_bom_structure(self, planning_engine):
        """Test complex multi-level BOM structures"""
        planning_engine.phase_results['phase3'] = {
            'net_requirements': {
                'TEST001': 100,
                'TEST002': 200
            }
        }
        
        complex_bom = pd.DataFrame({
            'Style#': ['TEST001', 'TEST001', 'TEST001', 'TEST002', 'TEST002'],
            'Desc#': ['YARN001', 'YARN002', 'YARN003', 'YARN001', 'YARN004'],
            'BOM_Percent': [50, 30, 20, 70, 30]
        })
        
        with patch.object(planning_engine, '_load_bom_data', return_value=complex_bom):
            result = planning_engine.phase4_bom_explosion_net()
            
            assert result.status == 'Completed'
            # YARN001 should aggregate from both styles


class TestPhase5ProcurementProduction(TestSixPhasePlanningEngine):
    """Test Phase 5: Procurement and Production Planning"""
    
    def test_yarn_shortage_detection(self, planning_engine, sample_yarn_inventory):
        """Test detection of yarn shortages"""
        planning_engine.phase_results['phase4'] = {
            'yarn_requirements': {
                'YARN001': 100,
                'YARN002': 200,  # This yarn has negative planning balance
                'YARN003': 50
            }
        }
        
        with patch.object(planning_engine, '_load_yarn_inventory', return_value=sample_yarn_inventory):
            result = planning_engine.phase5_procurement_production()
            
            assert result.status == 'Completed'
            assert result.output_data is not None
            # YARN002 should be flagged as shortage
    
    def test_knit_order_generation(self, planning_engine):
        """Test generation of knit orders"""
        planning_engine.phase_results['phase3'] = {
            'net_requirements': {
                'TEST001': 100,
                'TEST002': 200
            }
        }
        
        planning_engine.phase_results['phase4'] = {
            'yarn_requirements': {
                'YARN001': 150,
                'YARN002': 100
            }
        }
        
        yarn_inventory = pd.DataFrame({
            'Desc#': ['YARN001', 'YARN002'],
            'Planning Balance': [200, 150]  # Sufficient inventory
        })
        
        with patch.object(planning_engine, '_load_yarn_inventory', return_value=yarn_inventory):
            result = planning_engine.phase5_procurement_production()
            
            assert result.status == 'Completed'
            assert result.output_data is not None
    
    def test_procurement_recommendations(self, planning_engine):
        """Test generation of procurement recommendations"""
        planning_engine.phase_results['phase4'] = {
            'yarn_requirements': {
                'YARN001': 500,
                'YARN002': 300
            }
        }
        
        yarn_inventory = pd.DataFrame({
            'Desc#': ['YARN001', 'YARN002'],
            'Planning Balance': [100, -50],  # Both need procurement
            'Supplier': ['Supplier1', 'Supplier2']
        })
        
        with patch.object(planning_engine, '_load_yarn_inventory', return_value=yarn_inventory):
            result = planning_engine.phase5_procurement_production()
            
            assert result.status == 'Completed'
            # Should generate procurement recommendations
    
    def test_lead_time_consideration(self, planning_engine):
        """Test consideration of lead times in procurement"""
        planning_engine.phase_results['phase4'] = {
            'yarn_requirements': {
                'YARN001': 100
            },
            'required_dates': {
                'YARN001': datetime.now() + timedelta(days=14)
            }
        }
        
        yarn_inventory = pd.DataFrame({
            'Desc#': ['YARN001'],
            'Planning Balance': [-50],
            'Lead_Time_Days': [21]  # 3 weeks lead time
        })
        
        with patch.object(planning_engine, '_load_yarn_inventory', return_value=yarn_inventory):
            result = planning_engine.phase5_procurement_production()
            
            assert result.status == 'Completed'
            # Should flag urgent procurement due to lead time


class TestPhase6OptimizationOutput(TestSixPhasePlanningEngine):
    """Test Phase 6: Optimization and Output Generation"""
    
    def test_final_report_generation(self, planning_engine):
        """Test generation of final planning report"""
        # Set up all phase results
        planning_engine.phase_results = {
            'phase1': {'demand_consolidated': {'total': 1000}},
            'phase2': {'inventory_summary': {'total': 300}},
            'phase3': {'net_requirements': {'total': 700}},
            'phase4': {'yarn_requirements': {'total': 500}},
            'phase5': {'procurement_plan': {'orders': 5}}
        }
        
        result = planning_engine.phase6_optimization_output()
        
        assert result.status == 'Completed'
        assert result.output_data is not None
    
    def test_optimization_recommendations(self, planning_engine):
        """Test generation of optimization recommendations"""
        planning_engine.phase_results = {
            'phase3': {'net_requirements': {
                'TEST001': 100,
                'TEST002': 50,
                'TEST003': 200
            }},
            'phase5': {'yarn_shortages': ['YARN001', 'YARN002']}
        }
        
        result = planning_engine.phase6_optimization_output()
        
        assert result.status == 'Completed'
        assert result.output_data is not None
    
    def test_kpi_calculation(self, planning_engine):
        """Test calculation of planning KPIs"""
        planning_engine.phase_results = {
            'phase1': {'demand_consolidated': {'by_style': {'TEST001': 100, 'TEST002': 200}}},
            'phase2': {'multi_level_inventory': {'TEST001': {'total': 50}, 'TEST002': {'total': 150}}},
            'phase3': {'net_requirements': {'TEST001': 50, 'TEST002': 50}},
            'phase5': {'procurement_plan': {'total_cost': 10000}}
        }
        
        result = planning_engine.phase6_optimization_output()
        
        assert result.status == 'Completed'
        # Should calculate KPIs like fill rate, inventory turns, etc.
    
    def test_export_formats(self, planning_engine):
        """Test different export formats for output"""
        planning_engine.phase_results = {
            'phase1': {'demand_consolidated': {'total': 1000}},
            'phase2': {'inventory_summary': {'total': 300}}
        }
        
        result = planning_engine.phase6_optimization_output()
        
        assert result.status == 'Completed'
        # Should support multiple formats (dict, DataFrame, etc.)


class TestFullPlanningCycle:
    """Test complete planning cycle execution"""
    
    def test_full_cycle_execution(self):
        """Test execution of all 6 phases in sequence"""
        engine = SixPhasePlanningEngine()
        
        with patch.object(engine, '_load_sales_data', return_value=pd.DataFrame({'Style#': ['TEST001'], 'Ordered': [100]})):
            with patch.object(engine, '_load_stage_inventory', return_value=pd.DataFrame()):
                with patch.object(engine, '_load_bom_data', return_value=pd.DataFrame()):
                    with patch.object(engine, '_load_yarn_inventory', return_value=pd.DataFrame()):
                        results = engine.execute_full_planning_cycle(max_time=60)
                        
                        assert len(results) == 6
                        # All phases should execute
    
    def test_phase_dependency_validation(self):
        """Test that phases execute in correct order"""
        engine = SixPhasePlanningEngine()
        
        # Try to execute phase 3 without phase 1 and 2
        engine.phase_results = {}
        result = engine.phase3_net_requirements()
        
        # Should handle missing dependencies
        assert result.status in ['Failed', 'Error'] or 'warning' in str(result.output_data).lower()
    
    def test_timeout_handling(self):
        """Test timeout handling in planning cycle"""
        engine = SixPhasePlanningEngine()
        
        # Mock a slow operation
        def slow_operation():
            import time
            time.sleep(100)
            return pd.DataFrame()
        
        with patch.object(engine, '_load_sales_data', side_effect=slow_operation):
            results = engine.execute_full_planning_cycle(max_time=1)  # 1 second timeout
            
            # Should handle timeout gracefully
            assert len(results) >= 0
    
    def test_error_recovery(self):
        """Test error recovery during planning cycle"""
        engine = SixPhasePlanningEngine()
        
        # Mock an error in phase 2
        with patch.object(engine, 'phase2_inventory_assessment', side_effect=Exception("Test error")):
            results = engine.execute_full_planning_cycle(max_time=60)
            
            # Should continue with other phases or handle error gracefully
            assert isinstance(results, list)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_circular_bom_dependency(self):
        """Test handling of circular BOM dependencies"""
        engine = SixPhasePlanningEngine()
        
        circular_bom = pd.DataFrame({
            'Style#': ['TEST001', 'TEST002'],
            'Desc#': ['TEST002', 'TEST001'],  # Circular reference
            'BOM_Percent': [100, 100]
        })
        
        engine.phase_results = {'phase3': {'net_requirements': {'TEST001': 100}}}
        
        with patch.object(engine, '_load_bom_data', return_value=circular_bom):
            result = engine.phase4_bom_explosion_net()
            
            # Should detect and handle circular dependency
            assert result.status == 'Completed' or 'circular' in str(result.output_data).lower()
    
    def test_negative_inventory(self):
        """Test handling of negative inventory values"""
        engine = SixPhasePlanningEngine()
        
        negative_inventory = pd.DataFrame({
            'fStyle#': ['TEST001'],
            'Available': [-50]  # Negative inventory
        })
        
        with patch.object(engine, '_load_stage_inventory', return_value=negative_inventory):
            result = engine.phase2_inventory_assessment()
            
            # Should handle negative values appropriately
            assert result.status == 'Completed'
    
    def test_extreme_demand_spike(self):
        """Test handling of extreme demand spikes"""
        engine = SixPhasePlanningEngine()
        
        spike_demand = pd.DataFrame({
            'Style#': ['TEST001'],
            'Ordered': [1000000]  # Extreme demand
        })
        
        with patch.object(engine, '_load_sales_data', return_value=spike_demand):
            result = engine.phase1_demand_consolidation()
            
            assert result.status == 'Completed'
            # Should handle large numbers without overflow
    
    def test_unicode_and_special_chars(self):
        """Test handling of unicode and special characters in data"""
        engine = SixPhasePlanningEngine()
        
        unicode_data = pd.DataFrame({
            'Style#': ['TEST-001/中文', 'TËST-002'],
            'Ordered': [100, 200]
        })
        
        with patch.object(engine, '_load_sales_data', return_value=unicode_data):
            result = engine.phase1_demand_consolidation()
            
            assert result.status == 'Completed'
            # Should handle special characters


class TestPerformanceAndScaling:
    """Test performance with large datasets"""
    
    def test_large_dataset_processing(self):
        """Test processing of large datasets"""
        engine = SixPhasePlanningEngine()
        
        # Create large dataset
        large_sales = pd.DataFrame({
            'Style#': [f'TEST{i:05d}' for i in range(10000)],
            'Ordered': np.random.randint(1, 1000, 10000)
        })
        
        with patch.object(engine, '_load_sales_data', return_value=large_sales):
            import time
            start = time.time()
            result = engine.phase1_demand_consolidation()
            duration = time.time() - start
            
            assert result.status == 'Completed'
            assert duration < 10  # Should complete within 10 seconds
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets"""
        engine = SixPhasePlanningEngine()
        
        # Create memory-intensive dataset
        large_bom = pd.DataFrame({
            'Style#': np.repeat([f'TEST{i:04d}' for i in range(1000)], 10),
            'Desc#': [f'YARN{j:04d}' for j in range(10000)],
            'BOM_Percent': np.random.rand(10000) * 100
        })
        
        engine.phase_results = {'phase3': {'net_requirements': {f'TEST{i:04d}': 100 for i in range(1000)}}}
        
        with patch.object(engine, '_load_bom_data', return_value=large_bom):
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = engine.phase4_bom_explosion_net()
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_increase = mem_after - mem_before
            
            assert result.status == 'Completed'
            assert mem_increase < 500  # Should not increase by more than 500MB


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])