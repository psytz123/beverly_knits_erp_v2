"""
ERP Service Manager
Central coordinator for all modularized services
Created during Phase 2 modularization
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

# Import extracted services
from .inventory_analyzer_service import InventoryAnalyzer, InventoryManagementPipeline
from .sales_forecasting_service import SalesForecastingEngine
from .capacity_planning_service import CapacityPlanningEngine
from .business_rules import BusinessRules, ValidationRules

# Configure logging
logger = logging.getLogger(__name__)


class ERPServiceManager:
    """
    Central service manager that coordinates all ERP services
    Replaces embedded classes in beverly_comprehensive_erp.py
    """
    
    def __init__(self, data_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize all ERP services
        
        Args:
            data_path: Path to data directory
            config: Configuration dictionary
        """
        self.data_path = Path(data_path) if data_path else None
        self.config = config or {}
        
        # Service status tracking (must be before initialization)
        self.service_status = {
            'inventory': False,
            'forecasting': False,
            'capacity': False,
            'business_rules': False
        }
        
        # Initialize services
        self._initialize_services()
        
        # Validate initialization
        self._validate_services()
        
        logger.info("ERP Service Manager initialized successfully")
    
    def _initialize_services(self):
        """Initialize all individual services"""
        try:
            # Inventory services
            self.inventory_analyzer = InventoryAnalyzer(data_path=self.data_path)
            self.inventory_pipeline = InventoryManagementPipeline()
            self.service_status['inventory'] = True
            logger.info("Inventory services initialized")
        except Exception as e:
            logger.error(f"Failed to initialize inventory services: {e}")
            self.inventory_analyzer = None
            self.inventory_pipeline = None
        
        try:
            # Forecasting service
            self.forecasting_engine = SalesForecastingEngine()
            self.service_status['forecasting'] = True
            logger.info("Forecasting service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize forecasting service: {e}")
            self.forecasting_engine = None
        
        try:
            # Capacity planning service
            self.capacity_engine = CapacityPlanningEngine()
            self.service_status['capacity'] = True
            logger.info("Capacity planning service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize capacity planning service: {e}")
            self.capacity_engine = None
        
        try:
            # Business rules service
            self.business_rules = BusinessRules()
            self.validation_rules = ValidationRules()
            self.service_status['business_rules'] = True
            logger.info("Business rules service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize business rules service: {e}")
            self.business_rules = None
            self.validation_rules = None
    
    def _validate_services(self):
        """Validate that critical services are available"""
        critical_services = ['inventory', 'business_rules']
        missing_services = [s for s in critical_services if not self.service_status[s]]
        
        if missing_services:
            logger.warning(f"Critical services not available: {missing_services}")
        
        # Report overall status
        active_services = sum(1 for status in self.service_status.values() if status)
        total_services = len(self.service_status)
        logger.info(f"Service initialization: {active_services}/{total_services} services active")
    
    # ========== INVENTORY SERVICES ==========
    
    def analyze_inventory(self, inventory_data=None) -> Dict:
        """
        Analyze inventory and return insights
        
        Args:
            inventory_data: Inventory DataFrame or data
            
        Returns:
            Inventory analysis results
        """
        if not self.inventory_analyzer:
            return {'error': 'Inventory analyzer not available'}
        
        try:
            return self.inventory_analyzer.analyze_inventory(inventory_data)
        except Exception as e:
            logger.error(f"Inventory analysis failed: {e}")
            return {'error': str(e)}
    
    def analyze_inventory_levels(self, current_inventory, forecast) -> list:
        """
        Compare current inventory against forecasted demand
        
        Args:
            current_inventory: Current inventory data
            forecast: Demand forecast
            
        Returns:
            Inventory level analysis
        """
        if not self.inventory_analyzer:
            return []
        
        try:
            return self.inventory_analyzer.analyze_inventory_levels(
                current_inventory, forecast
            )
        except Exception as e:
            logger.error(f"Inventory level analysis failed: {e}")
            return []
    
    def run_inventory_pipeline(self, sales_data=None, inventory_data=None, yarn_data=None) -> Dict:
        """
        Execute complete inventory analysis pipeline
        
        Args:
            sales_data: Sales data
            inventory_data: Inventory data
            yarn_data: Yarn inventory data
            
        Returns:
            Complete pipeline analysis results
        """
        if not self.inventory_pipeline:
            return {'error': 'Inventory pipeline not available'}
        
        try:
            return self.inventory_pipeline.run_complete_analysis(
                sales_data=sales_data,
                inventory_data=inventory_data,
                yarn_data=yarn_data
            )
        except Exception as e:
            logger.error(f"Inventory pipeline failed: {e}")
            return {'error': str(e)}
    
    # ========== FORECASTING SERVICES ==========
    
    def generate_forecast(self, sales_data, horizon_days=90) -> Dict:
        """
        Generate sales forecast
        
        Args:
            sales_data: Historical sales data
            horizon_days: Forecast horizon in days
            
        Returns:
            Forecast results
        """
        if not self.forecasting_engine:
            return {'error': 'Forecasting engine not available'}
        
        try:
            return self.forecasting_engine.generate_forecast_output(sales_data)
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            return {'error': str(e)}
    
    def calculate_consistency_score(self, style_history) -> Dict:
        """
        Calculate consistency score for a style's historical sales
        
        Args:
            style_history: Historical sales data for a style
            
        Returns:
            Consistency analysis
        """
        if not self.forecasting_engine:
            return {'consistency_score': 0, 'cv': 1.0, 'recommendation': 'service_unavailable'}
        
        try:
            return self.forecasting_engine.calculate_consistency_score(style_history)
        except Exception as e:
            logger.error(f"Consistency score calculation failed: {e}")
            return {'consistency_score': 0, 'cv': 1.0, 'error': str(e)}
    
    def analyze_portfolio_consistency(self, sales_data):
        """
        Analyze consistency across entire product portfolio
        
        Args:
            sales_data: Sales data for all products
            
        Returns:
            Portfolio consistency analysis DataFrame
        """
        if not self.forecasting_engine:
            return None
        
        try:
            return self.forecasting_engine.analyze_portfolio_consistency(sales_data)
        except Exception as e:
            logger.error(f"Portfolio consistency analysis failed: {e}")
            return None
    
    # ========== CAPACITY PLANNING SERVICES ==========
    
    def calculate_capacity_requirements(self, production_plan, time_horizon_days=30) -> Dict:
        """
        Calculate capacity requirements for production plan
        
        Args:
            production_plan: Production plan dictionary
            time_horizon_days: Planning horizon
            
        Returns:
            Capacity requirements
        """
        if not self.capacity_engine:
            return {'error': 'Capacity engine not available'}
        
        try:
            return self.capacity_engine.calculate_finite_capacity_requirements(
                production_plan, time_horizon_days
            )
        except Exception as e:
            logger.error(f"Capacity calculation failed: {e}")
            return {'error': str(e)}
    
    def identify_bottlenecks(self, capacity_utilization) -> list:
        """
        Identify production bottlenecks
        
        Args:
            capacity_utilization: Resource utilization data
            
        Returns:
            List of bottlenecks
        """
        if not self.capacity_engine:
            return []
        
        try:
            return self.capacity_engine.identify_capacity_bottlenecks(capacity_utilization)
        except Exception as e:
            logger.error(f"Bottleneck identification failed: {e}")
            return []
    
    def optimize_capacity_allocation(self, demand_forecast, capacity_constraints) -> Dict:
        """
        Optimize capacity allocation across production lines
        
        Args:
            demand_forecast: Demand forecast
            capacity_constraints: Capacity constraints
            
        Returns:
            Optimized allocation plan
        """
        if not self.capacity_engine:
            return {'error': 'Capacity engine not available'}
        
        try:
            return self.capacity_engine.optimize_capacity_allocation(
                demand_forecast, capacity_constraints
            )
        except Exception as e:
            logger.error(f"Capacity optimization failed: {e}")
            return {'error': str(e)}
    
    # ========== BUSINESS RULES SERVICES ==========
    
    def calculate_planning_balance(self, theoretical_balance: float, 
                                  allocated: float, on_order: float) -> float:
        """
        Calculate planning balance using business rules
        
        CRITICAL: Allocated is already negative in source data
        
        Args:
            theoretical_balance: Theoretical balance
            allocated: Allocated quantity (already negative)
            on_order: On order quantity
            
        Returns:
            Planning balance
        """
        if not self.business_rules:
            logger.error("Business rules not available - using direct calculation")
            return theoretical_balance + allocated + on_order
        
        return self.business_rules.calculate_planning_balance(
            theoretical_balance, allocated, on_order
        )
    
    def calculate_weekly_demand(self, consumed_data=None, 
                               allocated_qty=None, 
                               monthly_consumed=None) -> float:
        """
        Calculate weekly demand using business rules
        
        Args:
            consumed_data: Historical consumption
            allocated_qty: Allocated quantity
            monthly_consumed: Monthly consumption
            
        Returns:
            Weekly demand
        """
        if not self.business_rules:
            logger.error("Business rules not available - using default")
            return 10
        
        return self.business_rules.calculate_weekly_demand(
            consumed_data, allocated_qty, monthly_consumed
        )
    
    def validate_inventory_data(self, df) -> list:
        """
        Validate inventory data integrity
        
        Args:
            df: Inventory DataFrame
            
        Returns:
            List of validation errors
        """
        if not self.validation_rules:
            return []
        
        try:
            return self.validation_rules.validate_inventory_data(df)
        except Exception as e:
            logger.error(f"Inventory validation failed: {e}")
            return [str(e)]
    
    # ========== INTEGRATED OPERATIONS ==========
    
    def run_integrated_analysis(self, inventory_data=None, 
                               sales_data=None, 
                               production_orders=None) -> Dict:
        """
        Run complete integrated analysis across all services
        
        Args:
            inventory_data: Current inventory
            sales_data: Historical sales
            production_orders: Current production orders
            
        Returns:
            Comprehensive analysis results
        """
        results = {
            'status': 'success',
            'timestamp': str(Path.cwd()),
            'services_available': self.service_status,
            'analysis': {}
        }
        
        # Run inventory analysis
        if self.service_status['inventory'] and inventory_data is not None:
            results['analysis']['inventory'] = self.analyze_inventory(inventory_data)
        
        # Run forecasting
        if self.service_status['forecasting'] and sales_data is not None:
            results['analysis']['forecast'] = self.generate_forecast(sales_data)
        
        # Run capacity planning
        if self.service_status['capacity'] and production_orders is not None:
            capacity_util = self._calculate_utilization(production_orders)
            results['analysis']['bottlenecks'] = self.identify_bottlenecks(capacity_util)
        
        # Validate data integrity
        if self.service_status['business_rules']:
            errors = []
            if inventory_data is not None:
                errors.extend(self.validate_inventory_data(inventory_data))
            results['validation'] = {
                'errors': errors,
                'valid': len(errors) == 0
            }
        
        return results
    
    def _calculate_utilization(self, production_orders) -> Dict:
        """
        Calculate resource utilization from production orders
        
        Args:
            production_orders: Production order data
            
        Returns:
            Utilization by resource
        """
        # Simplified utilization calculation
        utilization = {}
        
        if hasattr(production_orders, 'iterrows'):
            for _, order in production_orders.iterrows():
                machine = order.get('Machine', 'unassigned')
                qty = order.get('Qty', 0)
                
                if machine not in utilization:
                    utilization[machine] = 0
                
                # Simple utilization based on quantity
                # Real calculation would consider machine capacity
                utilization[machine] += qty / 1000  # Normalize
        
        return utilization
    
    def get_service_status(self) -> Dict:
        """
        Get status of all services
        
        Returns:
            Service status dictionary
        """
        return {
            'services': self.service_status,
            'summary': {
                'active': sum(1 for s in self.service_status.values() if s),
                'total': len(self.service_status),
                'health': 'healthy' if all(self.service_status.values()) else 'degraded'
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown all services"""
        logger.info("Shutting down ERP Service Manager")
        # Add cleanup code if needed
        pass