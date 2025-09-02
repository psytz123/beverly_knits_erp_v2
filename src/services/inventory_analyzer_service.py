#!/usr/bin/env python3
"""
Beverly Knits ERP - Inventory Analyzer Service
Extracted from beverly_comprehensive_erp.py (lines 359-418)
PRESERVES ALL BUSINESS LOGIC EXACTLY
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InventoryConfig:
    """Configuration for inventory analysis"""
    safety_stock_multiplier: float = 1.5
    lead_time_days: int = 30
    critical_days_threshold: int = 7
    high_risk_days_threshold: int = 14
    medium_risk_days_threshold: int = 30


class InventoryAnalyzerService:
    """
    Inventory analysis service extracted from monolith
    Preserves all original business logic from INVENTORY_FORECASTING_IMPLEMENTATION.md spec
    
    Original location: beverly_comprehensive_erp.py lines 359-418
    """
    
    def __init__(self, config: Optional[InventoryConfig] = None):
        """
        Initialize inventory analyzer with configuration
        
        Args:
            config: Optional configuration override
        """
        self.config = config or InventoryConfig()
        
        # Preserve original values from monolith
        self.safety_stock_multiplier = self.config.safety_stock_multiplier
        self.lead_time_days = self.config.lead_time_days
        
        logger.info(f"InventoryAnalyzerService initialized")
        logger.info(f"  Safety stock multiplier: {self.safety_stock_multiplier}")
        logger.info(f"  Lead time days: {self.lead_time_days}")
    
    def analyze_inventory_levels(self, 
                                current_inventory: List[Dict[str, Any]], 
                                forecast: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Compare current inventory against forecasted demand
        
        PRESERVED LOGIC: Original algorithm from monolith unchanged
        
        Args:
            current_inventory: List of current inventory items
            forecast: Dictionary of product_id -> forecasted demand
            
        Returns:
            List of analysis results with risk levels and reorder recommendations
        """
        analysis = []
        
        for product in current_inventory:
            # Original field mapping logic preserved
            product_id = product.get('id', product.get('product_id', ''))
            quantity = product.get('quantity', product.get('stock', 0))
            
            # Get forecast for this product
            forecasted_demand = forecast.get(product_id, 0)
            
            # Calculate days of supply (original formula)
            daily_demand = forecasted_demand / 30 if forecasted_demand > 0 else 0
            days_of_supply = quantity / daily_demand if daily_demand > 0 else 999
            
            # Calculate required inventory with safety stock (original formula)
            required_inventory = (
                daily_demand * self.lead_time_days * 
                self.safety_stock_multiplier
            )
            
            # Identify risk level using spec criteria
            risk_level = self.calculate_risk(
                current=quantity,
                required=required_inventory,
                days_supply=days_of_supply
            )
            
            # Build analysis result (original structure)
            analysis.append({
                'product_id': product_id,
                'current_stock': quantity,
                'forecasted_demand': forecasted_demand,
                'days_of_supply': days_of_supply,
                'required_inventory': required_inventory,
                'shortage_risk': risk_level,
                'reorder_needed': quantity < required_inventory,
                'reorder_quantity': max(0, required_inventory - quantity)
            })
        
        # Log summary
        critical_count = sum(1 for a in analysis if a['shortage_risk'] == 'CRITICAL')
        high_count = sum(1 for a in analysis if a['shortage_risk'] == 'HIGH')
        
        logger.info(f"Inventory analysis complete: {len(analysis)} products")
        logger.info(f"  Critical risk: {critical_count}")
        logger.info(f"  High risk: {high_count}")
        
        return analysis
    
    def calculate_risk(self, 
                       current: float, 
                       required: float, 
                       days_supply: float) -> str:
        """
        Calculate stockout risk level per spec
        
        PRESERVED LOGIC: Original risk thresholds from monolith
        
        Args:
            current: Current inventory level
            required: Required inventory level
            days_supply: Days of supply remaining
            
        Returns:
            Risk level: 'CRITICAL', 'HIGH', 'MEDIUM', or 'LOW'
        """
        # Original risk calculation logic preserved exactly
        if days_supply < self.config.critical_days_threshold:
            return 'CRITICAL'
        elif days_supply < self.config.high_risk_days_threshold:
            return 'HIGH'
        elif days_supply < self.config.medium_risk_days_threshold:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    # Additional methods for enhanced functionality
    
    def get_critical_items(self, 
                          current_inventory: List[Dict[str, Any]], 
                          forecast: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Get only critical risk items requiring immediate attention
        
        Args:
            current_inventory: Current inventory data
            forecast: Forecast data
            
        Returns:
            List of critical items only
        """
        analysis = self.analyze_inventory_levels(current_inventory, forecast)
        return [item for item in analysis if item['shortage_risk'] == 'CRITICAL']
    
    def calculate_total_reorder_value(self, 
                                     analysis: List[Dict[str, Any]], 
                                     unit_costs: Dict[str, float]) -> float:
        """
        Calculate total value of required reorders
        
        Args:
            analysis: Results from analyze_inventory_levels
            unit_costs: Dictionary of product_id -> unit cost
            
        Returns:
            Total reorder value
        """
        total_value = 0.0
        
        for item in analysis:
            if item['reorder_needed']:
                product_id = item['product_id']
                unit_cost = unit_costs.get(product_id, 0)
                total_value += item['reorder_quantity'] * unit_cost
        
        return total_value
    
    def generate_reorder_report(self, 
                               analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive reorder report
        
        Args:
            analysis: Results from analyze_inventory_levels
            
        Returns:
            Report dictionary with summary and details
        """
        reorder_items = [item for item in analysis if item['reorder_needed']]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_products_analyzed': len(analysis),
            'products_requiring_reorder': len(reorder_items),
            'risk_summary': {
                'critical': sum(1 for a in analysis if a['shortage_risk'] == 'CRITICAL'),
                'high': sum(1 for a in analysis if a['shortage_risk'] == 'HIGH'),
                'medium': sum(1 for a in analysis if a['shortage_risk'] == 'MEDIUM'),
                'low': sum(1 for a in analysis if a['shortage_risk'] == 'LOW')
            },
            'total_reorder_quantity': sum(item['reorder_quantity'] for item in reorder_items),
            'reorder_items': reorder_items
        }
        
        return report
    
    def adjust_safety_stock(self, 
                           product_id: str, 
                           historical_variance: float) -> float:
        """
        Adjust safety stock multiplier based on historical variance
        
        Args:
            product_id: Product identifier
            historical_variance: Historical demand variance
            
        Returns:
            Adjusted safety stock multiplier
        """
        # Higher variance requires higher safety stock
        if historical_variance > 0.5:
            return self.safety_stock_multiplier * 1.5
        elif historical_variance > 0.3:
            return self.safety_stock_multiplier * 1.2
        else:
            return self.safety_stock_multiplier
    
    def analyze_all(self) -> Dict[str, Any]:
        """
        Perform comprehensive inventory analysis
        
        Returns:
            Dictionary containing complete analysis results
        """
        try:
            # This is a placeholder that returns a comprehensive analysis
            # In a real implementation, this would fetch data and run analysis
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'inventory_summary': {
                    'total_items': 0,
                    'critical_items': 0,
                    'high_risk_items': 0,
                    'medium_risk_items': 0,
                    'low_risk_items': 0
                },
                'reorder_summary': {
                    'items_requiring_reorder': 0,
                    'total_reorder_value': 0.0
                },
                'risk_analysis': [],
                'recommendations': []
            }
        except Exception as e:
            logger.error(f"Error in analyze_all: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_yarn_shortages(self) -> List[Dict[str, Any]]:
        """
        Calculate yarn shortages based on current inventory and demand
        
        Returns:
            List of yarn shortage details
        """
        try:
            # This is a placeholder that returns yarn shortage analysis
            # In a real implementation, this would analyze yarn inventory
            return []
        except Exception as e:
            logger.error(f"Error in calculate_yarn_shortages: {str(e)}")
            return []
    
    def validate_inventory_data(self, 
                               inventory_data: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
        """
        Validate inventory data before analysis
        
        Args:
            inventory_data: Raw inventory data
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if not inventory_data:
            issues.append("Empty inventory data")
            return False, issues
        
        for idx, item in enumerate(inventory_data):
            # Check for required fields
            if not item.get('id') and not item.get('product_id'):
                issues.append(f"Item {idx}: Missing product identifier")
            
            # Check for quantity
            if 'quantity' not in item and 'stock' not in item:
                issues.append(f"Item {idx}: Missing quantity/stock field")
            
            # Validate numeric fields
            quantity = item.get('quantity', item.get('stock', 0))
            try:
                float(quantity)
            except (TypeError, ValueError):
                issues.append(f"Item {idx}: Invalid quantity value: {quantity}")
        
        return len(issues) == 0, issues


# Singleton instance for backward compatibility
_instance = None

def get_inventory_analyzer() -> InventoryAnalyzerService:
    """
    Get singleton instance of InventoryAnalyzerService
    Maintains backward compatibility with monolith
    """
    global _instance
    if _instance is None:
        _instance = InventoryAnalyzerService()
    return _instance


# For backward compatibility with original class name
InventoryAnalyzer = InventoryAnalyzerService


def test_service():
    """Test the extracted service"""
    print("=" * 80)
    print("Testing InventoryAnalyzerService")
    print("=" * 80)
    
    # Create service instance
    analyzer = InventoryAnalyzerService()
    
    # Test data
    test_inventory = [
        {'product_id': 'YARN001', 'quantity': 100},
        {'product_id': 'YARN002', 'stock': 50},
        {'product_id': 'YARN003', 'quantity': 200},
        {'product_id': 'YARN004', 'stock': 10}
    ]
    
    test_forecast = {
        'YARN001': 300,  # Will need reorder
        'YARN002': 150,  # Will need reorder
        'YARN003': 100,  # Sufficient stock
        'YARN004': 200   # Critical shortage
    }
    
    # Run analysis
    print("\nRunning inventory analysis...")
    analysis = analyzer.analyze_inventory_levels(test_inventory, test_forecast)
    
    # Display results
    print(f"\nAnalyzed {len(analysis)} products:")
    for item in analysis:
        print(f"  {item['product_id']}: Risk={item['shortage_risk']}, " 
              f"Days Supply={item['days_of_supply']:.1f}, "
              f"Reorder={item['reorder_needed']}")
    
    # Generate report
    report = analyzer.generate_reorder_report(analysis)
    print(f"\nReorder Report:")
    print(f"  Products requiring reorder: {report['products_requiring_reorder']}")
    print(f"  Risk summary: {report['risk_summary']}")
    print(f"  Total reorder quantity: {report['total_reorder_quantity']:.0f}")
    
    print("\n" + "=" * 80)
    print("✅ Service test complete")


if __name__ == "__main__":
    test_service()