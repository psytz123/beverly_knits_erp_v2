"""Unit tests for Knit Order Processor."""

import pytest
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np

from src.rules.knit_order_processor import (
    KnitOrderProcessor,
    KnitOrderAssignment
)
from src.rules.efab_integration import eFabIntegration
from src.rules.cache_manager import CacheManager


class TestKnitOrderProcessor:
    """Test suite for KnitOrderProcessor."""

    @pytest.fixture
    def mock_efab(self) -> MagicMock:
        """Mock eFab integration."""
        mock = MagicMock(spec=eFabIntegration)
        return mock

    @pytest.fixture
    def mock_cache(self) -> MagicMock:
        """Mock cache manager."""
        mock = MagicMock(spec=CacheManager)
        mock.cache_key.return_value = "test_key"
        mock.get.return_value = None  # No cached data by default
        return mock

    @pytest.fixture
    def sample_orders(self) -> pd.DataFrame:
        """Sample knit orders data."""
        return pd.DataFrame([
            {
                'Order #': 'ORD001',
                'Style #': 'ST123',
                'Machine': 161,
                'Balance (lbs)': '1,500.5',
                'G00 (lbs)': 500.0,
                'Shipped (lbs)': 0.0,
                'Status': 'Active'
            },
            {
                'Order #': 'ORD002',
                'Style #': 'ST123',
                'Machine': np.nan,  # Unassigned
                'Balance (lbs)': '800',
                'G00 (lbs)': 0.0,
                'Shipped (lbs)': 0.0,
                'Status': 'Active'
            },
            {
                'Order #': 'ORD003',
                'Style #': 'ST456',
                'Machine': 225,
                'Balance (lbs)': '2,000',
                'G00 (lbs)': 1000.0,
                'Shipped (lbs)': 500.0,
                'Status': 'In Production'
            },
            {
                'Order #': 'ORD004',
                'Style #': 'ST456',
                'Machine': 177,
                'Balance (lbs)': '1,200',
                'G00 (lbs)': 0.0,
                'Shipped (lbs)': 300.0,
                'Status': 'In Production'
            },
            {
                'Order #': 'ORD005',
                'Style #': 'ST789',
                'Machine': 161,
                'Balance (lbs)': '0',  # Completed
                'G00 (lbs)': 1000.0,
                'Shipped (lbs)': 1000.0,
                'Status': 'Completed'
            },
            {
                'Order #': 'ORD006',
                'Style #': 'ST789',
                'Machine': 161,
                'Balance (lbs)': '-100',  # Negative balance
                'G00 (lbs)': 0.0,
                'Shipped (lbs)': 0.0,
                'Status': 'Completed'
            }
        ])

    @pytest.fixture
    def processor(self, mock_efab: MagicMock, mock_cache: MagicMock) -> KnitOrderProcessor:
        """Create processor with mocks."""
        return KnitOrderProcessor(
            efab_integration=mock_efab,
            cache_manager=mock_cache,
            use_efab_api=False  # Use local API for testing
        )

    def test_initialization(self, processor: KnitOrderProcessor) -> None:
        """Test processor initialization."""
        assert processor is not None
        assert processor.machine_workloads == {}
        assert processor.machine_assignments == {}
        assert processor.suggested_workloads == {}
        assert processor.processing_stats['total_orders'] == 0

    def test_categorize_order_status(
        self,
        processor: KnitOrderProcessor,
        sample_orders: pd.DataFrame
    ) -> None:
        """Test order status categorization."""
        # Categorize orders
        active_orders = processor.categorize_order_status(sample_orders)

        # Should have 4 non-complete orders (ORD001-004)
        assert len(active_orders) == 4
        assert processor.processing_stats['completed_filtered'] == 2
        assert processor.processing_stats['total_orders'] == 6
        # ORD001 has G00=500 -> In Production
        # ORD003 has G00=1000, Shipped=500 -> In Production
        # ORD004 has Shipped=300 -> In Production
        assert processor.processing_stats['in_production_count'] == 3  # ORD001, ORD003, ORD004
        assert processor.processing_stats['active_count'] == 1  # ORD002 only

        # Check that completed orders are not in result
        order_ids = active_orders['Order #'].tolist()
        assert 'ORD005' not in order_ids  # Balance = 0
        assert 'ORD006' not in order_ids  # Balance < 0

        # Check status column was added
        assert 'order_status' in active_orders.columns

    def test_clean_balance_value(self, processor: KnitOrderProcessor) -> None:
        """Test balance value cleaning."""
        assert processor._clean_balance_value('1,500.5') == 1500.5
        assert processor._clean_balance_value('800') == 800.0
        assert processor._clean_balance_value(1200) == 1200.0
        assert processor._clean_balance_value(np.nan) == 0.0
        assert processor._clean_balance_value(None) == 0.0

    def test_assign_machines_intelligently_inheritance(
        self,
        processor: KnitOrderProcessor,
        sample_orders: pd.DataFrame
    ) -> None:
        """Test machine inheritance for same style."""
        # Categorize and filter to active orders
        active_orders = processor.categorize_order_status(sample_orders)

        # Assign machines
        assignments = processor.assign_machines_intelligently(active_orders)

        # Check ST123: ORD001 has machine 161, ORD002 should inherit it
        machine_161_assignments = assignments.get('161', [])

        # Should have assignments for both orders
        st123_assignments = [a for a in machine_161_assignments if a.style == 'ST123']
        assert len(st123_assignments) > 0

        # Check that unassigned order got suggested assignment
        suggested = [a for a in processor.order_assignments
                    if a.style == 'ST123' and a.is_suggested]
        assert len(suggested) > 0

    def test_assign_machines_intelligently_distribution(
        self,
        processor: KnitOrderProcessor
    ) -> None:
        """Test equal distribution of unassigned orders."""
        # Create orders with multiple machines for one style
        orders = pd.DataFrame([
            {
                'Order #': 'ORD001',
                'Style #': 'ST999',
                'Machine': 100,
                'Balance (lbs)': '1000',
                'G00 (lbs)': 0.0,
                'Shipped (lbs)': 0.0,
                'order_status': 'Active'
            },
            {
                'Order #': 'ORD002',
                'Style #': 'ST999',
                'Machine': 200,
                'Balance (lbs)': '1000',
                'G00 (lbs)': 0.0,
                'Shipped (lbs)': 0.0,
                'order_status': 'Active'
            },
            {
                'Order #': 'ORD003',
                'Style #': 'ST999',
                'Machine': np.nan,  # Unassigned
                'Balance (lbs)': '2000',
                'G00 (lbs)': 0.0,
                'Shipped (lbs)': 0.0,
                'order_status': 'Active'
            }
        ])

        assignments = processor.assign_machines_intelligently(orders)

        # Check that unassigned workload was distributed
        assert '100' in assignments
        assert '200' in assignments

        # Each machine should get half of the unassigned workload
        machine_100_suggested = processor.suggested_workloads.get('100', 0)
        machine_200_suggested = processor.suggested_workloads.get('200', 0)

        assert machine_100_suggested > 0
        assert machine_200_suggested > 0
        assert abs(machine_100_suggested - machine_200_suggested) < 1  # Equal distribution

    def test_calculate_machine_workloads(self, processor: KnitOrderProcessor) -> None:
        """Test machine workload calculation."""
        # Set up test workloads
        processor.machine_workloads = {
            '161': 3080.0,  # 5 days work (100% utilization)
            '225': 1540.0,  # 2.5 days work (50% utilization)
            '177': 308.0    # 0.5 days work (10% utilization)
        }
        processor.suggested_workloads = {
            '161': 500.0
        }
        processor.machine_assignments = {
            '161': 'ST123',
            '225': 'ST456',
            '177': 'ST456'
        }

        workloads = processor.calculate_machine_workloads()

        # Check machine 161
        assert workloads['161']['total_lbs'] == 3080.0
        assert workloads['161']['assigned_lbs'] == 2580.0
        assert workloads['161']['suggested_lbs'] == 500.0
        assert workloads['161']['utilization_percent'] == 100.0
        assert workloads['161']['is_overloaded'] is True

        # Check machine 225
        assert workloads['225']['total_lbs'] == 1540.0
        assert workloads['225']['utilization_percent'] == 50.0
        assert workloads['225']['is_overloaded'] is False

    def test_suggest_machine_assignments(self, processor: KnitOrderProcessor) -> None:
        """Test machine assignment suggestions."""
        # Set up existing assignments
        processor.machine_workloads = {
            '161': 1540.0,  # 50% utilization
            '225': 2464.0,  # 80% utilization
            '177': 308.0    # 10% utilization
        }
        processor.machine_assignments = {
            '161': 'ST123',
            '225': 'ST456',
            '177': 'ST789'
        }

        # Test 1: Suggest for existing style
        suggestions = processor.suggest_machine_assignments('ST123', 500)
        assert len(suggestions) > 0
        assert suggestions[0]['machine_id'] == '161'  # Should prefer same style
        assert suggestions[0]['reason'] == 'Already assigned to this style'

        # Test 2: Suggest for new style
        suggestions = processor.suggest_machine_assignments('ST999', 500)
        assert len(suggestions) > 0
        # Should prefer machine with lowest utilization
        assert suggestions[0]['machine_id'] == '177'

    @patch('requests.get')
    def test_fetch_knit_orders_from_api(
        self,
        mock_get: MagicMock,
        processor: KnitOrderProcessor,
        sample_orders: pd.DataFrame
    ) -> None:
        """Test API fetching."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'status': 'success',
            'data': sample_orders.to_dict('records')
        }
        mock_get.return_value = mock_response

        # Fetch orders
        orders = processor.fetch_knit_orders_from_api()

        # Check results
        assert len(orders) == 6
        assert 'Order #' in orders.columns
        assert 'Style #' in orders.columns
        assert 'Machine' in orders.columns

        # Check cache was set
        processor.cache.set.assert_called_once()

    def test_process_knit_orders_full_flow(
        self,
        processor: KnitOrderProcessor,
        sample_orders: pd.DataFrame
    ) -> None:
        """Test full processing flow."""
        # Mock API fetch
        with patch.object(processor, 'fetch_knit_orders_from_api',
                         return_value=sample_orders):
            results = processor.process_knit_orders()

        # Check results
        assert results['status'] == 'success'
        assert 'Processed 4 active orders' in results['message']
        assert results['stats']['total_orders'] == 6
        assert results['stats']['completed_filtered'] == 2
        assert results['total_machines'] > 0
        assert results['total_workload_lbs'] > 0

    def test_process_knit_orders_no_data(
        self,
        processor: KnitOrderProcessor
    ) -> None:
        """Test processing with no data."""
        # Mock empty API response
        with patch.object(processor, 'fetch_knit_orders_from_api',
                         return_value=pd.DataFrame()):
            results = processor.process_knit_orders()

        assert results['status'] == 'no_data'
        assert 'No knit orders available' in results['message']

    def test_process_knit_orders_all_completed(
        self,
        processor: KnitOrderProcessor
    ) -> None:
        """Test processing when all orders are completed."""
        # Create all completed orders
        completed_orders = pd.DataFrame([
            {
                'Order #': 'ORD001',
                'Style #': 'ST123',
                'Machine': 161,
                'Balance (lbs)': '0'
            },
            {
                'Order #': 'ORD002',
                'Style #': 'ST456',
                'Machine': 225,
                'Balance (lbs)': '-100'
            }
        ])

        with patch.object(processor, 'fetch_knit_orders_from_api',
                         return_value=completed_orders):
            results = processor.process_knit_orders()

        assert results['status'] == 'all_completed'
        assert 'All knit orders are completed' in results['message']

    def test_get_processing_summary(self, processor: KnitOrderProcessor) -> None:
        """Test processing summary generation."""
        from src.rules.knit_order_processor import KnitOrderAssignment

        # Set up some state
        processor.processing_stats = {
            'total_orders': 10,
            'completed_filtered': 2,
            'in_production_count': 3,
            'active_count': 5,
            'assigned_orders': 6,
            'unassigned_orders': 2,
            'suggested_assignments': 2
        }
        processor.machine_workloads = {
            '161': 3080.0,
            '225': 1540.0
        }
        processor.suggested_workloads = {
            '161': 500.0
        }
        processor.machine_assignments = {
            '161': 'ST123',
            '225': 'ST456'
        }
        # Add some order assignments for status breakdown
        processor.order_assignments = [
            KnitOrderAssignment(
                order_id='ORD001', style='ST123', machine_id='161',
                balance_lbs=1000.0, g00_lbs=500.0, shipped_lbs=0.0,
                order_status='In Production', is_assigned=True,
                is_suggested=False, assignment_reason='Original'
            ),
            KnitOrderAssignment(
                order_id='ORD002', style='ST456', machine_id='225',
                balance_lbs=500.0, g00_lbs=0.0, shipped_lbs=0.0,
                order_status='Active', is_assigned=True,
                is_suggested=False, assignment_reason='Original'
            )
        ]

        summary = processor.get_processing_summary()

        assert summary['processing_stats']['total_orders'] == 10
        assert summary['machine_count'] == 2
        assert summary['total_workload_lbs'] == 4620.0
        assert summary['assigned_workload_lbs'] == 4120.0
        assert summary['suggested_workload_lbs'] == 500.0
        assert summary['styles_processed'] == 2
        assert summary['overloaded_machines'] == 1  # Machine 161 is overloaded
        assert 'order_status_breakdown' in summary
        assert summary['order_status_breakdown']['completed'] == 2
        assert summary['order_status_breakdown']['in_production'] == 3
        assert summary['order_status_breakdown']['active'] == 5

    def test_api_fallback_on_error(
        self,
        processor: KnitOrderProcessor,
        sample_orders: pd.DataFrame
    ) -> None:
        """Test fallback to cached data on API error."""
        # Set up cached data
        processor.cache.get.side_effect = [
            None,  # First call (no recent cache)
            sample_orders.to_dict('records')  # Fallback call
        ]

        # Mock API error
        with patch('requests.get', side_effect=Exception("API Error")):
            orders = processor.fetch_knit_orders_from_api()

        # Should return cached data
        assert len(orders) == 6
        # Cache.get should be called twice (once for fresh, once for fallback)
        assert processor.cache.get.call_count == 2

    def test_order_status_with_production_fields(self, processor: KnitOrderProcessor) -> None:
        """Test order status determination with G00 and Shipped fields."""
        orders = pd.DataFrame([
            {  # Active order (no production)
                'Order #': 'ORD001',
                'Style #': 'ST123',
                'Machine': 161,
                'Balance (lbs)': 1000,
                'G00 (lbs)': 0,
                'Shipped (lbs)': 0
            },
            {  # In Production (G00 > 0, Balance > 0)
                'Order #': 'ORD002',
                'Style #': 'ST456',
                'Machine': 225,
                'Balance (lbs)': 800,
                'G00 (lbs)': 500,
                'Shipped (lbs)': 0
            },
            {  # In Production (Shipped > 0, Balance > 0)
                'Order #': 'ORD003',
                'Style #': 'ST789',
                'Machine': 177,
                'Balance (lbs)': 600,
                'G00 (lbs)': 0,
                'Shipped (lbs)': 400
            },
            {  # Complete (Balance = 0, even though G00+Shipped > 0)
                'Order #': 'ORD004',
                'Style #': 'ST999',
                'Machine': 161,
                'Balance (lbs)': 0,
                'G00 (lbs)': 1000,
                'Shipped (lbs)': 1000
            },
            {  # Complete (Balance < 0)
                'Order #': 'ORD005',
                'Style #': 'ST888',
                'Machine': 225,
                'Balance (lbs)': -50,
                'G00 (lbs)': 500,  # Has production but still complete
                'Shipped (lbs)': 500
            }
        ])

        active_orders = processor.categorize_order_status(orders)

        # Check correct categorization
        assert len(active_orders) == 3  # Only non-complete orders
        assert processor.processing_stats['completed_filtered'] == 2
        assert processor.processing_stats['in_production_count'] == 2
        assert processor.processing_stats['active_count'] == 1

        # Verify status assignment
        statuses = active_orders['order_status'].tolist()
        assert 'Active' in statuses
        assert statuses.count('In Production') == 2

    def test_order_status_priority(self, processor: KnitOrderProcessor) -> None:
        """Test that Complete status takes priority over In Production."""
        orders = pd.DataFrame([
            {  # Should be Complete even with production
                'Order #': 'TEST001',
                'Style #': 'ST100',
                'Machine': 100,
                'Balance (lbs)': 0,
                'G00 (lbs)': 1000,
                'Shipped (lbs)': 500
            },
            {  # Should be Complete even with negative balance and production
                'Order #': 'TEST002',
                'Style #': 'ST200',
                'Machine': 200,
                'Balance (lbs)': -100,
                'G00 (lbs)': 2000,
                'Shipped (lbs)': 2100
            },
            {  # Should be In Production (balance > 0, has production)
                'Order #': 'TEST003',
                'Style #': 'ST300',
                'Machine': 300,
                'Balance (lbs)': 500,
                'G00 (lbs)': 100,
                'Shipped (lbs)': 0
            }
        ])

        active_orders = processor.categorize_order_status(orders)

        # All orders should have status assigned
        assert 'order_status' in orders.columns

        # Check each order's status
        test001_status = orders[orders['Order #'] == 'TEST001']['order_status'].iloc[0]
        test002_status = orders[orders['Order #'] == 'TEST002']['order_status'].iloc[0]
        test003_status = orders[orders['Order #'] == 'TEST003']['order_status'].iloc[0]

        # Verify priority: Complete takes precedence over production
        assert test001_status == 'Complete', "Balance=0 should be Complete even with G00+Shipped>0"
        assert test002_status == 'Complete', "Balance<0 should be Complete even with G00+Shipped>0"
        assert test003_status == 'In Production', "Balance>0 with G00+Shipped>0 should be In Production"

        # Only TEST003 should be in active_orders (non-complete)
        assert len(active_orders) == 1
        assert active_orders['Order #'].iloc[0] == 'TEST003'

    def test_efab_api_mode(self, mock_efab: MagicMock, mock_cache: MagicMock) -> None:
        """Test using eFab API mode."""
        # Create processor in eFab API mode
        processor = KnitOrderProcessor(
            efab_integration=mock_efab,
            cache_manager=mock_cache,
            use_efab_api=True  # Use eFab API
        )

        # Mock eFab API response (direct list format)
        mock_efab.fetch_with_retry.return_value = [
            {
                'order_number': 'K2509044',
                'style_number': 'GG6020-GR/0',
                'machine_id': 161,
                'balance': 2000.0,
                'g00': 500.0,
                'shipped': 0.0
            },
            {
                'order_number': 'K2509043',
                'style_number': '1942/0',
                'machine_id': None,
                'balance': 2784.0,
                'g00': 0.0,
                'shipped': 0.0
            }
        ]

        # Fetch orders
        orders = processor.fetch_knit_orders_from_api()

        # Check that eFab API was called
        mock_efab.fetch_with_retry.assert_called_once_with('/api/knitorder/list')

        # Check column mapping worked
        assert 'Order #' in orders.columns
        assert 'Style #' in orders.columns
        assert 'Balance (lbs)' in orders.columns
        assert len(orders) == 2

    @pytest.mark.parametrize("balance_col,machine_col,style_col", [
        ('Balance (lbs)', 'Machine', 'Style #'),
        ('Balance_lbs', 'machine', 'Style'),
        ('balance_lbs', 'Machine_ID', 'style')
    ])
    def test_column_name_flexibility(
        self,
        processor: KnitOrderProcessor,
        balance_col: str,
        machine_col: str,
        style_col: str
    ) -> None:
        """Test handling of different column names."""
        # Create orders with different column names
        orders = pd.DataFrame([
            {
                'Order #': 'ORD001',
                style_col: 'ST123',
                machine_col: 161,
                balance_col: '1000',
                'order_status': 'Active'  # Add status for assignment logic
            },
            {
                'Order #': 'ORD002',
                style_col: 'ST123',
                machine_col: np.nan,
                balance_col: '500',
                'order_status': 'Active'
            }
        ])

        # Should handle different column names
        assignments = processor.assign_machines_intelligently(orders)

        # Should have assignments
        assert len(processor.order_assignments) > 0
        assert len(processor.machine_workloads) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])