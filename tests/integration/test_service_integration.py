#!/usr/bin/env python3
"""
Integration tests for extracted services
Verifies that all modularized services work together correctly
"""

import sys
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# No need to modify sys.path with proper module structure

from services.service_manager import ServiceManager
from services.inventory_analyzer_service import InventoryAnalyzerService, InventoryConfig
from services.sales_forecasting_service import SalesForecastingService, ForecastConfig
from services.capacity_planning_service import CapacityPlanningService, CapacityConfig


class TestServiceIntegration:
    """Test integration between all extracted services"""
    
    @pytest.fixture
    def service_manager(self):
        """Create a service manager with test configuration"""
        config = {
            'inventory': {
                'safety_stock_multiplier': 1.5,
                'lead_time_days': 30
            },
            'forecast': {
                'horizon_days': 90,
                'target_accuracy': 0.85
            },
            'capacity': {
                'bottleneck_threshold': 0.85,
                'critical_threshold': 0.95
            }
        }
        return ServiceManager(config)
    
    @pytest.fixture
    def sample_inventory_data(self):
        """Create sample inventory data"""
        return [
            {'product_id': 'YARN001', 'quantity': 100, 'location': 'F01'},
            {'product_id': 'YARN002', 'quantity': 50, 'location': 'F01'},
            {'product_id': 'YARN003', 'quantity': 200, 'location': 'G00'},
            {'product_id': 'FABRIC001', 'quantity': 500, 'location': 'F01'},
            {'product_id': 'FABRIC002', 'quantity': 300, 'location': 'I01'}
        ]
    
    @pytest.fixture
    def sample_sales_history(self):
        """Create sample sales history data"""
        # Generate realistic sales data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        return {
            'YARN001': pd.DataFrame({
                'date': dates,
                'quantity': np.random.normal(100, 10, 30)
            }),
            'YARN002': pd.DataFrame({
                'date': dates,
                'quantity': np.random.normal(50, 5, 30)
            }),
            'YARN003': pd.DataFrame({
                'date': dates,
                'quantity': np.random.normal(200, 20, 30)
            }),
            'FABRIC001': pd.DataFrame({
                'date': dates,
                'quantity': np.random.normal(500, 50, 30)
            })
        }
    
    @pytest.fixture
    def sample_production_plan(self):
        """Create sample production plan"""
        return {
            'FABRIC_LIGHT_001': 600,
            'FABRIC_MEDIUM_002': 900,
            'FABRIC_HEAVY_003': 400
        }
    
    def test_all_services_initialized(self, service_manager):
        """Test that all services are properly initialized"""
        # Check service manager is initialized
        assert service_manager.initialized
        
        # Check all three services are available
        assert service_manager.get_inventory_analyzer() is not None
        assert service_manager.get_sales_forecaster() is not None
        assert service_manager.get_capacity_planner() is not None
        
        # Check service count (now includes InventoryManagementPipelineService)
        status = service_manager.get_system_status()
        assert status['total_services'] == 4
    
    def test_inventory_analyzer_service(self, service_manager, sample_inventory_data):
        """Test inventory analyzer service functionality"""
        inventory_service = service_manager.get_inventory_analyzer()
        assert inventory_service is not None
        
        # Create forecast data
        forecast = {
            'YARN001': 150,
            'YARN002': 100,
            'YARN003': 180
        }
        
        # Analyze inventory
        analysis = inventory_service.analyze_inventory_levels(
            sample_inventory_data[:3], 
            forecast
        )
        
        assert len(analysis) == 3
        assert all('shortage_risk' in item for item in analysis)
        assert all('reorder_quantity' in item for item in analysis)
    
    def test_sales_forecasting_service(self, service_manager, sample_sales_history):
        """Test sales forecasting service functionality"""
        forecast_service = service_manager.get_sales_forecaster()
        assert forecast_service is not None
        
        # Test ML availability
        assert forecast_service.ML_AVAILABLE
        
        # Generate forecast for one product
        forecast_result = forecast_service.forecast_with_consistency(
            sample_sales_history['YARN001'],
            horizon_days=30
        )
        
        assert 'forecast' in forecast_result
        assert 'confidence' in forecast_result
        assert 'method' in forecast_result
        assert forecast_result['confidence'] > 0
    
    def test_capacity_planning_service(self, service_manager, sample_production_plan):
        """Test capacity planning service functionality"""
        capacity_service = service_manager.get_capacity_planner()
        assert capacity_service is not None
        
        # Calculate capacity requirements
        requirements = capacity_service.calculate_finite_capacity_requirements(
            sample_production_plan,
            time_horizon_days=30
        )
        
        assert len(requirements) == 3
        assert all('machine_hours' in req for req in requirements.values())
        assert all('total_days' in req for req in requirements.values())
        
        # Test bottleneck identification
        utilization = {
            'Machine_45': 0.75,
            'Machine_88': 0.92,
            'Machine_127': 0.88
        }
        
        bottlenecks = capacity_service.identify_capacity_bottlenecks(utilization)
        assert len(bottlenecks) == 2  # Two machines above 0.85 threshold
        assert bottlenecks[0]['resource'] == 'Machine_88'  # Highest utilization
    
    def test_integrated_analysis(self, service_manager, sample_inventory_data, sample_sales_history):
        """Test integrated analysis using all services together"""
        # Perform integrated analysis
        results = service_manager.perform_integrated_analysis(
            sample_inventory_data,
            sample_sales_history
        )
        
        # Check structure
        assert 'inventory_analysis' in results
        assert 'forecasts' in results
        assert 'recommendations' in results
        
        # Check forecasts were generated
        if results['forecasts']:
            assert 'forecasts' in results['forecasts']
            assert 'average_confidence' in results['forecasts']
        
        # Check inventory was analyzed
        if results['inventory_analysis']:
            assert len(results['inventory_analysis']) > 0
            assert all('shortage_risk' in item for item in results['inventory_analysis'])
    
    def test_service_to_service_communication(self, service_manager, sample_sales_history, sample_production_plan):
        """Test that services can work together in a workflow"""
        # Step 1: Generate forecasts
        forecast_service = service_manager.get_sales_forecaster()
        forecasts = {}
        
        for product_id, history in sample_sales_history.items():
            result = forecast_service.forecast_with_consistency(history, 30)
            forecasts[product_id] = result['forecast']
        
        # Step 2: Use forecasts for inventory analysis
        inventory_service = service_manager.get_inventory_analyzer()
        inventory_data = [
            {'product_id': pid, 'quantity': np.random.randint(50, 500)}
            for pid in forecasts.keys()
        ]
        
        analysis = inventory_service.analyze_inventory_levels(inventory_data, forecasts)
        assert len(analysis) == len(forecasts)
        
        # Step 3: Plan capacity based on requirements
        capacity_service = service_manager.get_capacity_planner()
        
        # Convert inventory needs to production plan
        production_needs = {
            f"Product_{i}": max(0, forecasts[pid] - item['quantity'])
            for i, (pid, item) in enumerate(zip(forecasts.keys(), inventory_data))
        }
        
        allocation = capacity_service.optimize_capacity_allocation(production_needs)
        assert '_summary' in allocation
        assert allocation['_summary']['overall_fulfillment_rate'] >= 0
    
    def test_service_manager_convenience_methods(self, service_manager, sample_inventory_data):
        """Test service manager convenience methods"""
        # Test analyze_inventory convenience method
        forecast = {'YARN001': 150, 'YARN002': 100}
        analysis = service_manager.analyze_inventory(
            sample_inventory_data[:2],
            forecast
        )
        assert len(analysis) == 2
        
        # Test generate_forecast convenience method
        sample_data = pd.DataFrame({'quantity': [100, 110, 105, 95, 100]})
        forecast_result = service_manager.generate_forecast(sample_data, 30)
        assert 'forecast' in forecast_result
        assert 'confidence' in forecast_result
    
    def test_service_configuration(self, service_manager):
        """Test that services respect configuration"""
        # Check inventory service config
        inventory_service = service_manager.get_inventory_analyzer()
        assert inventory_service.safety_stock_multiplier == 1.5
        assert inventory_service.lead_time_days == 30
        
        # Check forecast service config
        forecast_service = service_manager.get_sales_forecaster()
        assert forecast_service.forecast_horizon == 90
        assert forecast_service.target_accuracy == 0.85
        
        # Check capacity service config
        capacity_service = service_manager.get_capacity_planner()
        assert capacity_service.config.bottleneck_threshold == 0.85
        assert capacity_service.config.critical_threshold == 0.95
    
    def test_service_manager_shutdown(self, service_manager):
        """Test graceful shutdown of service manager"""
        # Verify services are available (now includes InventoryManagementPipelineService)
        assert len(service_manager.services) == 4
        assert service_manager.initialized
        
        # Shutdown
        service_manager.shutdown()
        
        # Verify shutdown
        assert len(service_manager.services) == 0
        assert not service_manager.initialized


def test_service_extraction_completeness():
    """Verify that extracted services maintain all original functionality"""
    # This test verifies the extraction was complete
    
    # Test InventoryAnalyzerService has all methods
    inventory_service = InventoryAnalyzerService()
    assert hasattr(inventory_service, 'analyze_inventory_levels')
    assert hasattr(inventory_service, 'calculate_risk')
    assert hasattr(inventory_service, 'get_critical_items')
    assert hasattr(inventory_service, 'generate_reorder_report')
    
    # Test SalesForecastingService has all methods
    forecast_service = SalesForecastingService()
    assert hasattr(forecast_service, 'forecast_with_consistency')
    assert hasattr(forecast_service, 'calculate_consistency_score')
    assert hasattr(forecast_service, 'get_forecast_summary')
    assert hasattr(forecast_service, 'ml_engines')
    
    # Test CapacityPlanningService has all methods
    capacity_service = CapacityPlanningService()
    assert hasattr(capacity_service, 'calculate_finite_capacity_requirements')
    assert hasattr(capacity_service, 'identify_capacity_bottlenecks')
    assert hasattr(capacity_service, 'optimize_capacity_allocation')
    assert hasattr(capacity_service, 'get_capacity_metrics')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])