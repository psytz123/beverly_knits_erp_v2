"""Integration tests for Phase 2 data layer consolidation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.infrastructure.data.unified_data_loader import UnifiedDataLoader, FileDataSource
from src.infrastructure.data.column_mapper import ColumnMapper
from src.infrastructure.data.data_validator import DataValidator, ValidationResult
from src.infrastructure.cache.multi_tier_cache import MultiTierCache
from src.infrastructure.data.cache_warmer import EnhancedCacheWarmer
from src.infrastructure.data.exceptions import DataLoadException, DataValidationException


class TestUnifiedDataLoader:
    """Test unified data loader functionality."""
    
    @pytest.fixture
    async def data_loader(self):
        """Create data loader instance for testing."""
        config = {
            'file_path': '/test/data/path',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'api_url': None,
            'db_connection': None
        }
        loader = UnifiedDataLoader(config)
        yield loader
        loader.close()
    
    @pytest.mark.asyncio
    async def test_load_with_fallback(self, data_loader):
        """Test data loading with fallback strategy."""
        # Create mock data
        mock_df = pd.DataFrame({
            'yarn_id': ['Y001', 'Y002', 'Y003'],
            'description': ['Yarn 1', 'Yarn 2', 'Yarn 3'],
            'theoretical_balance': [100, 200, 300],
            'allocated': [-50, -75, -100],
            'on_order': [25, 50, 75]
        })
        
        # Mock the file source
        with patch.object(FileDataSource, 'load', return_value=mock_df):
            result = await data_loader.load_yarn_inventory()
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert 'planning_balance' in result.columns
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, data_loader):
        """Test that cache is used when available."""
        mock_df = pd.DataFrame({'test': [1, 2, 3]})
        
        # Pre-populate cache
        await data_loader.cache.set('data:yarn_inventory', mock_df, 'yarn_inventory')
        
        # Load should hit cache
        result = await data_loader.load_yarn_inventory()
        
        assert result.equals(mock_df)
        
        # Verify cache was hit
        stats = data_loader.get_cache_stats()
        assert stats['l1']['hits'] > 0 or stats['l2']['hits'] > 0
    
    @pytest.mark.asyncio
    async def test_data_type_conversion(self, data_loader):
        """Test automatic data type conversion."""
        # Create data with string numbers
        mock_df = pd.DataFrame({
            'yarn_id': ['Y001'],
            'theoretical_balance': ['1,000.50'],
            'allocated': ['$-500.25'],
            'on_order': ['250']
        })
        
        result = data_loader._convert_data_types(mock_df, 'yarn_inventory')
        
        assert pd.api.types.is_numeric_dtype(result['theoretical_balance'])
        assert result['theoretical_balance'].iloc[0] == 1000.50
        assert result['allocated'].iloc[0] == -500.25
        assert result['on_order'].iloc[0] == 250.0
    
    @pytest.mark.asyncio
    async def test_parallel_loading(self, data_loader):
        """Test parallel loading of multiple data sources."""
        # Mock multiple data sources
        mock_data = {
            'yarn_inventory': pd.DataFrame({'yarn': [1, 2, 3]}),
            'bom_data': pd.DataFrame({'bom': [4, 5, 6]}),
            'production_orders': pd.DataFrame({'orders': [7, 8, 9]})
        }
        
        with patch.object(data_loader, '_load_with_fallback', side_effect=lambda dt: mock_data[dt]):
            results = await data_loader.load_all_data_sources()
            
            assert 'yarn_inventory' in results
            assert 'bom_data' in results
            assert 'production_orders' in results
            assert len(results) >= 3


class TestColumnMapper:
    """Test column standardization functionality."""
    
    @pytest.fixture
    def mapper(self):
        """Create column mapper instance."""
        return ColumnMapper()
    
    def test_standardize_columns(self, mapper):
        """Test column name standardization."""
        # Create DataFrame with various column variations
        df = pd.DataFrame({
            'Desc#': ['Y001', 'Y002'],
            'Planning Balance': [100, 200],
            'Planning_Balance': [100, 200],  # Duplicate variation
            'on_order': [50, 75],
            'Unknown Column': ['A', 'B']
        })
        
        result = mapper.standardize(df)
        
        # Check standardized names
        assert 'yarn_id' in result.columns
        assert 'planning_balance' in result.columns
        assert 'on_order' in result.columns
        assert 'Unknown Column' in result.columns  # Unmapped kept as-is
    
    def test_validate_required_columns(self, mapper):
        """Test validation of required columns."""
        df = pd.DataFrame({
            'yarn_id': ['Y001'],
            'description': ['Test'],
            'Missing_Column': ['Value']
        })
        
        required = ['yarn_id', 'description', 'theoretical_balance']
        
        validation = mapper.validate_required_columns(df, required)
        
        assert not validation.is_valid
        assert len(validation.errors) == 1
        assert 'theoretical_balance' in validation.errors[0]
    
    def test_detect_column_type(self, mapper):
        """Test automatic column type detection."""
        df = pd.DataFrame({
            'percentage': [10, 20, 30, 40],
            'balance': [-100, 50, -200, 300],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
            'id': ['ID001', 'ID002', 'ID003', 'ID004'],
            'category': ['A', 'A', 'B', 'B']
        })
        
        assert mapper.detect_column_type(df, 'percentage') == 'percentage'
        assert mapper.detect_column_type(df, 'balance') == 'balance_or_adjustment'
        assert mapper.detect_column_type(df, 'id') == 'identifier'
        assert mapper.detect_column_type(df, 'category') == 'category'
    
    def test_suggest_standard_name(self, mapper):
        """Test standard name suggestion for unmapped columns."""
        df = pd.DataFrame({
            'Material_Quantity': [100, 200],
            'Order_Status': ['pending', 'complete'],
            'Total_Price': [1000, 2000]
        })
        
        assert mapper.suggest_standard_name('Material_Quantity', df) == 'quantity'
        assert mapper.suggest_standard_name('Order_Status', df) == 'status'
        assert mapper.suggest_standard_name('Total_Price', df) == 'cost'


class TestDataValidator:
    """Test data validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create data validator instance."""
        return DataValidator()
    
    def test_validate_yarn_inventory(self, validator):
        """Test yarn inventory validation."""
        df = pd.DataFrame({
            'yarn_id': ['Y001', 'Y002', 'Y002'],  # Duplicate
            'description': ['Yarn 1', 'Yarn 2', None],  # Null
            'theoretical_balance': [100, -50, 200],  # Negative
            'allocated': [-50, -75, -100],
            'on_order': [25, 50, 75],
            'planning_balance': [75, -75, 175]
        })
        
        result = validator.validate(df, 'yarn_inventory')
        
        # Check for expected issues
        assert not result.is_valid  # Due to duplicates
        assert any('duplicate' in e for e in result.errors)
        assert any('null' in w for w in result.warnings)
    
    def test_validate_bom_data(self, validator):
        """Test BOM data validation."""
        df = pd.DataFrame({
            'style_id': ['S001', 'S001', 'S002'],
            'yarn_id': ['Y001', 'Y002', 'Y001'],
            'quantity': [10, 20, 0],  # Zero quantity
            'bom_percent': [40, 60, 100]
        })
        
        result = validator.validate(df, 'bom_data')
        
        # Check for BOM percentage validation
        assert result.has_warnings
        assert any('zero quantity' in w for w in result.warnings)
    
    def test_validate_production_orders(self, validator):
        """Test production order validation."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002'],
            'style_id': ['S001', 'S002'],
            'qty_ordered': [100, 200],
            'qty_produced': [150, 180],  # Over-production
            'due_date': ['2024-01-01', '2024-12-31'],
            'status': ['in_progress', 'in_progress']
        })
        
        result = validator.validate(df, 'production_orders')
        
        assert result.has_warnings
        assert any('produced more than ordered' in w for w in result.warnings)
    
    def test_business_rule_validation(self, validator):
        """Test business rule validations."""
        df = pd.DataFrame({
            'yarn_id': ['Y001', 'Y002'],
            'theoretical_balance': [1000, 2000],
            'allocated': [-500, -3000],
            'on_order': [200, 500],
            'planning_balance': [700, -500]  # Critical shortage
        })
        
        result = validator.validate(df, 'yarn_inventory')
        
        assert result.has_warnings
        # Should detect critical shortage
        assert any('critical' in w.lower() for w in result.warnings)


class TestMultiTierCache:
    """Test multi-tier caching functionality."""
    
    @pytest.fixture
    async def cache(self):
        """Create cache instance."""
        cache = MultiTierCache(enable_disk=False)
        yield cache
        cache.close()
    
    @pytest.mark.asyncio
    async def test_cache_hierarchy(self, cache):
        """Test L1 and L2 cache hierarchy."""
        test_data = {'key': 'value', 'number': 42}
        
        # Set in cache
        await cache.set('test_key', test_data, 'default', ttl=60)
        
        # Get from L1 (memory)
        result = await cache.get('test_key')
        assert result == test_data
        assert cache.l1_hits > 0
        
        # Clear L1 to force L2 hit
        cache.clear_l1()
        
        # Get from L2 (Redis) if available
        if cache.l2_enabled:
            result = await cache.get('test_key')
            assert result == test_data
            assert cache.l2_hits > 0
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self, cache):
        """Test cache TTL expiry."""
        # Set with short TTL
        await cache.set('expiry_test', 'data', ttl=1)
        
        # Should be available immediately
        result = await cache.get('expiry_test')
        assert result == 'data'
        
        # Wait for expiry
        await asyncio.sleep(1.5)
        
        # Should be expired
        result = await cache.get('expiry_test')
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache):
        """Test cache statistics tracking."""
        # Perform some operations
        await cache.set('stat_test1', 'data1')
        await cache.get('stat_test1')  # Hit
        await cache.get('nonexistent')  # Miss
        await cache.delete('stat_test1')
        
        stats = cache.get_stats()
        
        assert stats['total']['requests'] > 0
        assert stats['total']['sets'] > 0
        assert stats['total']['deletes'] > 0
        assert stats['l1']['hit_rate'] >= 0


class TestCacheWarmer:
    """Test cache warming functionality."""
    
    @pytest.fixture
    async def warmer(self):
        """Create cache warmer instance."""
        cache = MultiTierCache()
        loader = Mock(spec=UnifiedDataLoader)
        
        # Mock data loader methods
        loader.load_yarn_inventory = AsyncMock(
            return_value=pd.DataFrame({'test': [1, 2, 3]})
        )
        loader.load_bom_data = AsyncMock(
            return_value=pd.DataFrame({'test': [4, 5, 6]})
        )
        loader.load_production_orders = AsyncMock(
            return_value=pd.DataFrame({'test': [7, 8, 9]})
        )
        
        warmer = EnhancedCacheWarmer(cache, loader)
        yield warmer
        warmer.close()
    
    @pytest.mark.asyncio
    async def test_startup_warming(self, warmer):
        """Test cache warming on startup."""
        results = await warmer.warm_on_startup()
        
        assert 'critical' in results
        assert 'important' in results
        
        # Check that critical data was warmed
        critical_results = results.get('critical', {})
        for data_type in ['yarn_inventory', 'bom_data', 'production_orders']:
            if data_type in critical_results:
                assert critical_results[data_type]['success']
    
    @pytest.mark.asyncio
    async def test_manual_warming(self, warmer):
        """Test manual cache warming."""
        results = await warmer.warm_specific(['yarn_inventory', 'bom_data'])
        
        assert 'yarn_inventory' in results
        assert 'bom_data' in results
        
        # Check statistics
        stats = warmer.get_statistics()
        assert stats['overall']['total_warmings'] > 0
    
    @pytest.mark.asyncio
    async def test_warming_statistics(self, warmer):
        """Test warming statistics collection."""
        # Warm some data
        await warmer.warm_specific(['yarn_inventory'])
        
        stats = warmer.get_statistics()
        
        assert 'overall' in stats
        assert 'by_data_type' in stats
        assert 'strategies' in stats
        
        # Check yarn_inventory stats
        if 'yarn_inventory' in stats['by_data_type']:
            yarn_stats = stats['by_data_type']['yarn_inventory']
            assert yarn_stats['warm_count'] > 0
            assert yarn_stats['avg_time_ms'] > 0