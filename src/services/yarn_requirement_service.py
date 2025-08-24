#!/usr/bin/env python3
"""
Beverly Knits ERP - Yarn Requirement Calculator Service
Extracted from beverly_comprehensive_erp.py (lines 1793-1907)
Processes BOM entries to calculate yarn requirements
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class YarnRequirementConfig:
    """Configuration for yarn requirement service"""
    critical_threshold: float = 1000.0  # Threshold for critical yarn identification
    high_priority_factor: float = 0.5   # Factor for high priority classification
    default_data_path: str = "/mnt/c/finalee/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5"


class YarnRequirementCalculatorService:
    """
    Processes BOM entries to calculate yarn requirements
    Handles 55,160+ BOM entries for comprehensive requirement analysis
    Extracted from monolith for modular architecture
    """
    
    def __init__(self, config: Optional[YarnRequirementConfig] = None):
        """
        Initialize yarn requirement calculator service
        
        Args:
            config: Optional configuration for yarn requirement calculation
        """
        self.config = config or YarnRequirementConfig()
        self.data_path = Path(self.config.default_data_path)
        
        # Data storage
        self.bom_data = None
        self.yarn_requirements = {}
        self.unique_yarns = set()
        self.column_standardizer = None  # Optional column standardizer
        
        logger.info(f"YarnRequirementCalculatorService initialized with data path: {self.data_path}")
    
    def set_data_path(self, data_path: str):
        """
        Set the data path for loading BOM and inventory files
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        logger.info(f"Data path updated to: {self.data_path}")
    
    def set_column_standardizer(self, standardizer):
        """
        Set optional column standardizer for data cleaning
        
        Args:
            standardizer: Column standardization utility
        """
        self.column_standardizer = standardizer
        logger.debug("Column standardizer configured")
    
    def load_bom_data(self, bom_file_name: str = "BOM_2(Sheet1).csv") -> bool:
        """
        Load and process BOM entries
        
        Args:
            bom_file_name: Name of BOM file to load
            
        Returns:
            True if successful, False otherwise
        """
        bom_file = self.data_path / bom_file_name
        
        # Try alternate names if primary not found
        alternate_names = ["Style_BOM.csv", "BOM_updated.csv", "BOM_Master_Sheet1.csv"]
        
        if not bom_file.exists():
            for alt_name in alternate_names:
                alt_file = self.data_path / alt_name
                if alt_file.exists():
                    bom_file = alt_file
                    logger.info(f"Using alternate BOM file: {alt_name}")
                    break
        
        if bom_file.exists():
            try:
                # Load CSV or Excel based on extension
                if bom_file.suffix == '.csv':
                    self.bom_data = pd.read_csv(bom_file)
                else:
                    self.bom_data = pd.read_excel(bom_file)
                
                # Standardize columns if standardizer available
                if self.column_standardizer:
                    self.bom_data = self.column_standardizer.standardize_columns(self.bom_data)
                    logger.debug("Columns standardized")
                
                logger.info(f"Loaded {len(self.bom_data)} BOM entries from {bom_file.name}")
                return True
                
            except Exception as e:
                logger.error(f"Error loading BOM data: {e}")
                return False
        else:
            logger.warning(f"BOM file not found: {bom_file}")
            return False
    
    def process_yarn_requirements(self) -> Dict[str, Any]:
        """
        Calculate total yarn requirements from BOM explosion
        
        Returns:
            Dictionary of yarn requirements by yarn ID
        """
        # Load BOM data if not already loaded
        if self.bom_data is None:
            if not self.load_bom_data():
                logger.error("Failed to load BOM data")
                return {}
        
        # Reset requirements
        self.yarn_requirements = {}
        self.unique_yarns = set()
        
        # Process BOM entries
        if self.bom_data is not None:
            # Identify yarn ID column
            yarn_col = self._find_yarn_column()
            quantity_col = self._find_quantity_column()
            product_col = self._find_product_column()
            
            if not yarn_col:
                logger.warning("No yarn ID column found in BOM")
                return {}
            
            # Group by yarn type and calculate total requirements
            for idx, row in self.bom_data.iterrows():
                yarn_id = str(row.get(yarn_col, ''))
                quantity = float(row.get(quantity_col, 0))
                product = str(row.get(product_col, f'Product_{idx}'))
                
                if yarn_id and yarn_id != 'nan':
                    self.unique_yarns.add(yarn_id)
                    
                    if yarn_id not in self.yarn_requirements:
                        self.yarn_requirements[yarn_id] = {
                            'total_required': 0,
                            'products_using': [],
                            'average_usage': 0,
                            'min_usage': float('inf'),
                            'max_usage': 0
                        }
                    
                    self.yarn_requirements[yarn_id]['total_required'] += quantity
                    
                    if product and product not in self.yarn_requirements[yarn_id]['products_using']:
                        self.yarn_requirements[yarn_id]['products_using'].append(product)
                    
                    # Track min/max usage
                    if quantity > 0:
                        self.yarn_requirements[yarn_id]['min_usage'] = min(
                            self.yarn_requirements[yarn_id]['min_usage'], quantity
                        )
                        self.yarn_requirements[yarn_id]['max_usage'] = max(
                            self.yarn_requirements[yarn_id]['max_usage'], quantity
                        )
            
            # Calculate averages and statistics
            for yarn_id in self.yarn_requirements:
                req = self.yarn_requirements[yarn_id]
                count = len(req['products_using'])
                
                if count > 0:
                    req['average_usage'] = req['total_required'] / count
                
                # Fix inf values
                if req['min_usage'] == float('inf'):
                    req['min_usage'] = 0
            
            logger.info(f"Processed {len(self.unique_yarns)} unique yarns from BOM")
            logger.debug(f"Total yarn requirements calculated: {len(self.yarn_requirements)}")
            
        return self.yarn_requirements
    
    def get_critical_yarns(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Identify yarns with high requirements
        
        Args:
            threshold: Optional threshold override (default from config)
            
        Returns:
            List of critical yarns sorted by requirement
        """
        if not self.yarn_requirements:
            self.process_yarn_requirements()
        
        threshold = threshold or self.config.critical_threshold
        critical = []
        
        for yarn_id, req in self.yarn_requirements.items():
            if req['total_required'] > threshold:
                critical.append({
                    'yarn_id': yarn_id,
                    'total_required': req['total_required'],
                    'products_count': len(req['products_using']),
                    'average_usage': req['average_usage'],
                    'min_usage': req['min_usage'],
                    'max_usage': req['max_usage'],
                    'criticality_score': self._calculate_criticality_score(req)
                })
        
        # Sort by criticality score then total required
        return sorted(critical, key=lambda x: (x['criticality_score'], x['total_required']), reverse=True)
    
    def calculate_procurement_needs(self, inventory_data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Calculate procurement needs based on BOM requirements vs inventory
        
        Args:
            inventory_data: Optional inventory DataFrame (will load if not provided)
            
        Returns:
            List of procurement needs sorted by shortage
        """
        if not self.yarn_requirements:
            self.process_yarn_requirements()
        
        procurement_list = []
        
        # Load current inventory if not provided
        if inventory_data is None:
            inventory_data = self._load_inventory_data()
        
        if inventory_data is not None:
            # Create inventory lookup
            inventory_dict = self._create_inventory_lookup(inventory_data)
            
            # Calculate procurement needs
            for yarn_id, req in self.yarn_requirements.items():
                current_stock = inventory_dict.get(yarn_id, 0)
                required = req['total_required']
                shortage = required - current_stock
                
                if shortage > 0:
                    priority = self._calculate_priority(current_stock, required, shortage)
                    
                    procurement_list.append({
                        'yarn_id': yarn_id,
                        'required': required,
                        'current_stock': current_stock,
                        'shortage': shortage,
                        'products_affected': len(req['products_using']),
                        'priority': priority,
                        'coverage_days': int(current_stock / (required / 30)) if required > 0 else 0,
                        'recommended_order': shortage * 1.2  # Add 20% buffer
                    })
        else:
            logger.warning("No inventory data available for procurement calculation")
        
        # Sort by priority then shortage
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        return sorted(procurement_list, 
                     key=lambda x: (priority_order.get(x['priority'], 3), -x['shortage']))
    
    def get_yarn_usage_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of yarn usage
        
        Returns:
            Summary statistics of yarn requirements
        """
        if not self.yarn_requirements:
            self.process_yarn_requirements()
        
        total_yarns = len(self.unique_yarns)
        total_requirement = sum(req['total_required'] for req in self.yarn_requirements.values())
        
        # Calculate distribution
        requirement_values = [req['total_required'] for req in self.yarn_requirements.values()]
        
        summary = {
            'total_unique_yarns': total_yarns,
            'total_requirement_quantity': total_requirement,
            'average_requirement_per_yarn': total_requirement / total_yarns if total_yarns > 0 else 0,
            'yarns_above_critical_threshold': sum(
                1 for req in self.yarn_requirements.values() 
                if req['total_required'] > self.config.critical_threshold
            )
        }
        
        if requirement_values:
            summary['min_requirement'] = min(requirement_values)
            summary['max_requirement'] = max(requirement_values)
            summary['median_requirement'] = sorted(requirement_values)[len(requirement_values) // 2]
        
        # Product coverage
        all_products = set()
        for req in self.yarn_requirements.values():
            all_products.update(req['products_using'])
        summary['total_products_covered'] = len(all_products)
        
        return summary
    
    def _find_yarn_column(self) -> Optional[str]:
        """Find the yarn ID column in BOM data"""
        possible_names = ['Yarn_ID', 'Yarn ID', 'Component_ID', 'Component ID', 
                         'Desc#', 'Desc', 'Yarn', 'Material']
        
        if self.bom_data is not None:
            for col in self.bom_data.columns:
                if col in possible_names:
                    return col
        return None
    
    def _find_quantity_column(self) -> Optional[str]:
        """Find the quantity column in BOM data"""
        possible_names = ['Quantity', 'Usage', 'Amount', 'Qty', 'Required']
        
        if self.bom_data is not None:
            for col in self.bom_data.columns:
                if col in possible_names:
                    return col
        return 'Quantity'  # Default
    
    def _find_product_column(self) -> Optional[str]:
        """Find the product ID column in BOM data"""
        possible_names = ['Product_ID', 'Product ID', 'Style', 'Style#', 'Item']
        
        if self.bom_data is not None:
            for col in self.bom_data.columns:
                if col in possible_names:
                    return col
        return 'Product_ID'  # Default
    
    def _load_inventory_data(self) -> Optional[pd.DataFrame]:
        """Load inventory data from file"""
        inventory_files = [
            "yarn_inventory (4).xlsx",
            "yarn_inventory (4).csv",
            "yarn_inventory (3).xlsx",
            "yarn_inventory (1).xlsx"
        ]
        
        for file_name in inventory_files:
            inv_file = self.data_path / file_name
            if inv_file.exists():
                try:
                    if inv_file.suffix == '.csv':
                        return pd.read_csv(inv_file)
                    else:
                        return pd.read_excel(inv_file)
                except Exception as e:
                    logger.error(f"Error loading inventory file {file_name}: {e}")
        
        return None
    
    def _create_inventory_lookup(self, inventory_data: pd.DataFrame) -> Dict[str, float]:
        """Create inventory lookup dictionary"""
        inventory_dict = {}
        
        # Find relevant columns
        yarn_col = None
        balance_col = None
        
        for col in inventory_data.columns:
            if 'ID' in col or 'Desc' in col:
                yarn_col = col
            if 'Balance' in col or 'Planning' in col or 'Quantity' in col:
                balance_col = col
        
        if yarn_col and balance_col:
            for _, row in inventory_data.iterrows():
                yarn_id = str(row.get(yarn_col, ''))
                balance = float(row.get(balance_col, 0))
                if yarn_id and yarn_id != 'nan':
                    inventory_dict[yarn_id] = balance
        
        return inventory_dict
    
    def _calculate_priority(self, current_stock: float, required: float, shortage: float) -> str:
        """Calculate procurement priority"""
        if current_stock <= 0:
            return 'CRITICAL'
        elif shortage > required * self.config.high_priority_factor:
            return 'HIGH'
        elif shortage > 0:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_criticality_score(self, requirement: Dict) -> float:
        """Calculate criticality score for a yarn"""
        # Factors: total requirement, number of products, average usage
        score = (
            requirement['total_required'] * 0.5 +
            len(requirement['products_using']) * 100 +
            requirement['average_usage'] * 0.3
        )
        return score


def test_yarn_requirement_service():
    """Test the yarn requirement calculator service"""
    print("=" * 80)
    print("Testing YarnRequirementCalculatorService")
    print("=" * 80)
    
    # Create service with custom config
    config = YarnRequirementConfig(
        critical_threshold=500,
        high_priority_factor=0.6
    )
    service = YarnRequirementCalculatorService(config)
    
    # Test 1: Load BOM data
    print("\n1. Testing BOM Data Loading:")
    if service.load_bom_data():
        print(f"  ✓ BOM data loaded: {len(service.bom_data)} entries")
    else:
        print("  ✗ Failed to load BOM data (file may not exist in test environment)")
        # Create sample data for testing
        service.bom_data = pd.DataFrame({
            'Yarn_ID': ['YARN001', 'YARN002', 'YARN001', 'YARN003'],
            'Quantity': [100, 200, 150, 300],
            'Product_ID': ['PROD_A', 'PROD_A', 'PROD_B', 'PROD_C']
        })
        print("  Using sample BOM data for testing")
    
    # Test 2: Process yarn requirements
    print("\n2. Testing Yarn Requirement Processing:")
    requirements = service.process_yarn_requirements()
    print(f"  ✓ Processed {len(requirements)} unique yarns")
    
    if requirements:
        # Show sample requirement
        sample_yarn = list(requirements.keys())[0]
        sample_req = requirements[sample_yarn]
        print(f"  Sample requirement for {sample_yarn}:")
        print(f"    Total required: {sample_req['total_required']}")
        print(f"    Products using: {len(sample_req['products_using'])}")
        print(f"    Average usage: {sample_req['average_usage']:.2f}")
    
    # Test 3: Get critical yarns
    print("\n3. Testing Critical Yarn Identification:")
    critical_yarns = service.get_critical_yarns(threshold=200)
    print(f"  ✓ Identified {len(critical_yarns)} critical yarns")
    
    for yarn in critical_yarns[:3]:
        print(f"    {yarn['yarn_id']}: {yarn['total_required']} units "
              f"({yarn['products_count']} products)")
    
    # Test 4: Calculate procurement needs
    print("\n4. Testing Procurement Calculation:")
    # Create sample inventory
    sample_inventory = pd.DataFrame({
        'Yarn ID': ['YARN001', 'YARN002', 'YARN003'],
        'Balance': [100, 50, 200]
    })
    
    procurement_needs = service.calculate_procurement_needs(sample_inventory)
    print(f"  ✓ Calculated procurement for {len(procurement_needs)} items")
    
    for item in procurement_needs[:3]:
        print(f"    {item['yarn_id']}: Shortage {item['shortage']} "
              f"({item['priority']} priority)")
    
    # Test 5: Get usage summary
    print("\n5. Yarn Usage Summary:")
    summary = service.get_yarn_usage_summary()
    print(f"  Total unique yarns: {summary['total_unique_yarns']}")
    print(f"  Total requirement: {summary['total_requirement_quantity']}")
    print(f"  Average per yarn: {summary['average_requirement_per_yarn']:.2f}")
    print(f"  Critical yarns: {summary['yarns_above_critical_threshold']}")
    
    print("\n" + "=" * 80)
    print("✅ YarnRequirementCalculatorService test complete")


if __name__ == "__main__":
    test_yarn_requirement_service()