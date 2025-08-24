"""
Global pytest configuration and fixtures for Beverly Knits ERP tests

Provides shared fixtures and test data for all test modules
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


# ==================== Sample Data Fixtures ====================

@pytest.fixture(scope="session")
def sample_yarn_inventory():
    """Sample yarn inventory data for testing"""
    return pd.DataFrame({
        'Item': ['YARN001', 'YARN002', 'YARN003', 'YARN004', 'YARN005'],
        'Desc#': ['Cotton 30/1 White', 'Poly 40/1 Black', 'Cotton 30/1 Combed', 
                  'Nylon 20/1 Red', 'Cotton/Poly Blend'],
        'Material': ['Cotton', 'Polyester', 'Cotton', 'Nylon', 'Blend'],
        'Size': ['30/1', '40/1', '30/1', '20/1', '30/1'],
        'Color': ['White', 'Black', 'White', 'Red', 'Gray'],
        'Theoretical Balance': [1000.0, 500.0, -200.0, 800.0, 600.0],
        'Allocated': [200.0, 100.0, 50.0, 150.0, 100.0],
        'On Order': [0.0, 300.0, 500.0, 0.0, 200.0],
        'Planning Balance': [800.0, 700.0, 250.0, 650.0, 700.0],
        'UOM': ['LBS', 'LBS', 'LBS', 'LBS', 'LBS'],
        'Unit Cost': [5.50, 4.25, 6.00, 7.50, 5.00],
        'Lead Time Days': [7, 14, 7, 10, 14],
        'Safety Stock': [100, 80, 120, 90, 100]
    })


@pytest.fixture(scope="session")
def sample_yarn_demand():
    """Sample yarn demand data for testing"""
    return pd.DataFrame({
        'Yarn_Code': ['YARN001', 'YARN002', 'YARN003', 'YARN004', 'YARN005'],
        'Yarn_Description': ['Cotton 30/1 White', 'Poly 40/1 Black', 'Cotton 30/1 Combed',
                            'Nylon 20/1 Red', 'Cotton/Poly Blend'],
        'Total_Demand': [1500.0, 300.0, 800.0, 400.0, 600.0],
        'Week_1': [300.0, 60.0, 160.0, 80.0, 120.0],
        'Week_2': [300.0, 60.0, 160.0, 80.0, 120.0],
        'Week_3': [300.0, 60.0, 160.0, 80.0, 120.0],
        'Week_4': [300.0, 60.0, 160.0, 80.0, 120.0],
        'Week_5': [300.0, 60.0, 160.0, 80.0, 120.0]
    })


@pytest.fixture(scope="session")
def sample_sales_orders():
    """Sample sales orders data for testing"""
    return pd.DataFrame({
        'Order_ID': ['SO001', 'SO002', 'SO003', 'SO004', 'SO005'],
        'Customer': ['Customer A', 'Customer B', 'Customer C', 'Customer A', 'Customer D'],
        'Product': ['Jersey Fabric', 'Rib Fabric', 'Interlock', 'Jersey Fabric', 'Fleece'],
        'Quantity': [1000, 500, 750, 1200, 600],
        'Unit': ['Yards', 'Yards', 'Yards', 'Yards', 'Yards'],
        'Due_Date': [
            datetime.now() + timedelta(days=14),
            datetime.now() + timedelta(days=21),
            datetime.now() + timedelta(days=10),
            datetime.now() + timedelta(days=28),
            datetime.now() + timedelta(days=18)
        ],
        'Status': ['In Production', 'Planned', 'In Production', 'Planned', 'In Production']
    })


@pytest.fixture(scope="session")
def sample_bom_data():
    """Sample Bill of Materials data for testing"""
    return pd.DataFrame({
        'Product': ['Jersey Fabric', 'Jersey Fabric', 'Rib Fabric', 'Rib Fabric', 'Interlock'],
        'Component': ['YARN001', 'YARN002', 'YARN001', 'YARN003', 'YARN004'],
        'Quantity_Per_Unit': [0.8, 0.2, 0.6, 0.4, 1.0],
        'Unit': ['LBS/Yard', 'LBS/Yard', 'LBS/Yard', 'LBS/Yard', 'LBS/Yard']
    })


@pytest.fixture(scope="session")
def sample_supplier_data():
    """Sample supplier data for testing"""
    return pd.DataFrame({
        'Supplier_ID': ['SUP001', 'SUP002', 'SUP003', 'SUP004'],
        'Supplier_Name': ['Cotton Mills Inc', 'Poly Suppliers Ltd', 'Global Yarns', 'Express Yarns'],
        'Materials': [['Cotton'], ['Polyester', 'Nylon'], ['Cotton', 'Polyester'], ['All']],
        'Lead_Time_Days': [7, 10, 14, 3],
        'Reliability_Score': [0.95, 0.90, 0.85, 0.98],
        'Cost_Factor': [1.0, 0.95, 0.90, 1.5],
        'Min_Order_Qty': [500, 1000, 750, 100],
        'Payment_Terms': ['Net 30', 'Net 45', 'Net 30', 'COD']
    })


@pytest.fixture(scope="session")
def sample_production_capacity():
    """Sample production capacity data for testing"""
    return {
        'knitting': {
            'machines': 20,
            'capacity_per_machine': 100,  # yards/day
            'efficiency': 0.85,
            'shifts': 2
        },
        'dyeing': {
            'machines': 5,
            'capacity_per_machine': 500,  # yards/day
            'efficiency': 0.90,
            'shifts': 1
        },
        'finishing': {
            'machines': 10,
            'capacity_per_machine': 200,  # yards/day
            'efficiency': 0.95,
            'shifts': 2
        }
    }


# ==================== Test Data Files ====================

@pytest.fixture(scope="session")
def temp_data_directory():
    """Create temporary directory with test data files"""
    temp_dir = tempfile.mkdtemp()
    
    # Create test Excel files
    yarn_inventory = pd.DataFrame({
        'Item': ['TEST001', 'TEST002'],
        'Planning Balance': [100, -50]
    })
    
    yarn_demand = pd.DataFrame({
        'Yarn_Code': ['TEST001', 'TEST002'],
        'Total_Demand': [200, 150]
    })
    
    # Save to temp directory
    yarn_inventory.to_excel(f"{temp_dir}/yarn_inventory.xlsx", index=False)
    yarn_demand.to_excel(f"{temp_dir}/yarn_demand.xlsx", index=False)
    
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


# ==================== Mock Objects ====================

@pytest.fixture
def mock_inventory_analyzer():
    """Mock InventoryAnalyzer for testing"""
    from unittest.mock import Mock
    import beverly_comprehensive_erp as erp
    
    analyzer = Mock(spec=erp.InventoryAnalyzer)
    analyzer.yarn_inventory = pd.DataFrame({
        'Item': ['MOCK001', 'MOCK002'],
        'Planning Balance': [100, -50],
        'Material': ['Cotton', 'Polyester']
    })
    analyzer.yarn_demand = pd.DataFrame({
        'Yarn_Code': ['MOCK001', 'MOCK002'],
        'Total_Demand': [150, 100]
    })
    
    return analyzer


@pytest.fixture
def mock_flask_app():
    """Mock Flask application for testing with proper context handling"""
    import beverly_comprehensive_erp as erp
    
    erp.app.config['TESTING'] = True
    erp.app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
    
    # Create application context
    ctx = erp.app.app_context()
    ctx.push()
    
    yield erp.app
    
    # Clean up context
    ctx.pop()


@pytest.fixture
def client(mock_flask_app):
    """Test client for Flask app with proper thread safety"""
    with mock_flask_app.test_client() as client:
        with mock_flask_app.app_context():
            yield client


@pytest.fixture
def mock_inventory_data():
    """Mock inventory data for API testing"""
    return pd.DataFrame({
        'Item': ['YARN001', 'YARN002', 'YARN003'],
        'Desc#': ['Cotton 30/1', 'Polyester 40/1', 'Cotton Blend'],
        'Planning Balance': [1000, -500, 200],
        'Theoretical Balance': [1200, -300, 400],
        'Allocated': [200, 100, 150],
        'On Order': [0, 300, 0],
        'Material': ['Cotton', 'Polyester', 'Blend'],
        'Color': ['White', 'Black', 'Gray']
    })


# ==================== Helper Functions ====================

@pytest.fixture
def calculate_planning_balance():
    """Helper function to calculate planning balance"""
    def _calculate(theoretical, allocated, on_order):
        return theoretical - allocated + on_order
    return _calculate


@pytest.fixture
def calculate_days_of_supply():
    """Helper function to calculate days of supply"""
    def _calculate(current_stock, daily_consumption):
        if daily_consumption <= 0:
            return float('inf')
        return current_stock / daily_consumption
    return _calculate


@pytest.fixture
def calculate_safety_stock():
    """Helper function to calculate safety stock"""
    def _calculate(avg_demand, std_dev, lead_time, service_level=0.95):
        z_scores = {0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
        z = z_scores.get(service_level, 1.645)
        return z * std_dev * np.sqrt(lead_time)
    return _calculate


# ==================== Test Configuration ====================

@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings"""
    return {
        'api_timeout': 10,
        'max_retries': 3,
        'test_data_dir': '/tmp/test_data',
        'coverage_threshold': 80,
        'performance_benchmarks': {
            'api_response_time': 0.2,  # seconds
            'planning_execution': 120,  # seconds for 1000 items
            'forecast_generation': 5    # seconds
        }
    }


# ==================== Database Fixtures ====================

@pytest.fixture
def in_memory_database():
    """Create in-memory SQLite database for testing"""
    import sqlite3
    
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create test tables
    cursor.execute('''
        CREATE TABLE inventory (
            item_id TEXT PRIMARY KEY,
            quantity REAL,
            last_updated TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            item_id TEXT,
            quantity REAL,
            order_date TIMESTAMP,
            FOREIGN KEY (item_id) REFERENCES inventory (item_id)
        )
    ''')
    
    conn.commit()
    
    yield conn
    
    conn.close()


# ==================== Performance Testing ====================

@pytest.fixture
def performance_timer():
    """Timer for performance testing"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            if self.start_time:
                self.elapsed = time.time() - self.start_time
                return self.elapsed
            return None
        
        def assert_under(self, seconds):
            assert self.elapsed is not None, "Timer not stopped"
            assert self.elapsed < seconds, f"Operation took {self.elapsed}s, expected under {seconds}s"
    
    return Timer()


# ==================== Pytest Hooks ====================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "ml: Machine learning tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add slow marker for specific tests
        if "test_planning_engine_large_dataset" in item.name:
            item.add_marker(pytest.mark.slow)
        if "test_ml_" in item.name:
            item.add_marker(pytest.mark.ml)