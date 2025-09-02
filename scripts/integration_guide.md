# Day 0 Emergency Fixes Integration Guide

## Overview

The `day0_emergency_fixes.py` script provides production-ready fixes for the Beverly Knits ERP system. This guide shows how to integrate these fixes into the main ERP application.

## Integration Steps

### 1. Import the Emergency Fixes

Add this to the top of your main ERP file (`src/core/beverly_comprehensive_erp.py`):

```python
# Import Day 0 Emergency Fixes
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

try:
    from day0_emergency_fixes import (
        DynamicPathResolver, 
        ColumnAliasSystem, 
        PriceStringParser, 
        RealKPICalculator,
        MultiLevelBOMNetting,
        initialize_emergency_fixes
    )
    EMERGENCY_FIXES_AVAILABLE = True
    print("Day 0 Emergency Fixes loaded successfully")
except ImportError as e:
    EMERGENCY_FIXES_AVAILABLE = False
    print(f"Day 0 Emergency Fixes not available: {e}")
```

### 2. Initialize Emergency Fixes in Your ERP Class

In your main ERP class constructor:

```python
class BeverlyKnitsERP:
    def __init__(self):
        # Initialize emergency fixes if available
        if EMERGENCY_FIXES_AVAILABLE:
            self.emergency_components = initialize_emergency_fixes()
            self.path_resolver = self.emergency_components['path_resolver']
            self.column_system = self.emergency_components['column_system']
            self.price_parser = self.emergency_components['price_parser']
            self.kpi_calculator = self.emergency_components['kpi_calculator']
            self.bom_netting = self.emergency_components['bom_netting']
            print("Emergency fixes initialized in ERP system")
        else:
            # Fallback to original methods
            self.path_resolver = None
            self.column_system = None
            # ... etc
```

### 3. Replace File Loading Logic

Replace hardcoded paths with dynamic resolution:

```python
# OLD WAY (hardcoded):
def load_yarn_inventory(self):
    file_path = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/yarn_inventory.xlsx"
    return pd.read_excel(file_path)

# NEW WAY (dynamic):
def load_yarn_inventory(self):
    if self.path_resolver:
        file_path = self.path_resolver.resolve_data_file('yarn_inventory')
        if file_path:
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            return self.column_system.standardize_columns(df)
    
    # Fallback to original method
    return self._load_yarn_inventory_fallback()
```

### 4. Replace Column Access Logic

Replace hardcoded column names with alias system:

```python
# OLD WAY (hardcoded):
def get_planning_balance(self, yarn_df, yarn_id):
    try:
        row = yarn_df[yarn_df['Desc#'] == yarn_id].iloc[0]
        return float(row['Planning Balance'])
    except:
        return 0.0

# NEW WAY (dynamic):
def get_planning_balance(self, yarn_df, yarn_id):
    if self.column_system:
        yarn_col = self.column_system.find_column(yarn_df, 'yarn_id')
        balance_col = self.column_system.find_column(yarn_df, 'planning_balance')
        
        if yarn_col and balance_col:
            try:
                row = yarn_df[yarn_df[yarn_col] == yarn_id].iloc[0]
                return float(row[balance_col])
            except:
                pass
    
    # Fallback to original method
    return self._get_planning_balance_fallback(yarn_df, yarn_id)
```

### 5. Replace Price Parsing

Replace manual price parsing with robust parser:

```python
# OLD WAY (basic):
def parse_price(self, price_str):
    try:
        return float(str(price_str).replace('$', '').replace(',', ''))
    except:
        return 0.0

# NEW WAY (robust):
def parse_price(self, price_str):
    if self.price_parser:
        result = self.price_parser.parse_price(price_str)
        return result['value'] if result['is_valid'] else 0.0
    
    # Fallback to original method
    return self._parse_price_fallback(price_str)
```

### 6. Replace KPI Calculation

Replace hardcoded zeros with real calculations:

```python
# OLD WAY (returns zeros):
def calculate_comprehensive_kpis(self):
    return {
        'total_yarn_items': 0,
        'total_planning_balance': 0.0,
        # ... more zeros
    }

# NEW WAY (real calculations):
def calculate_comprehensive_kpis(self):
    if self.kpi_calculator:
        return self.kpi_calculator.calculate_comprehensive_kpis()
    
    # Fallback to original method (zeros)
    return self._calculate_kpis_fallback()
```

### 7. Add BOM Netting Endpoint

Add a new API endpoint for multi-level BOM netting:

```python
@app.route('/api/bom-netting', methods=['POST'])
def api_bom_netting():
    """Calculate multi-level BOM net requirements"""
    try:
        data = request.get_json()
        style_demands = data.get('style_demands', {})
        
        if hasattr(self, 'bom_netting') and self.bom_netting:
            result = self.bom_netting.calculate_net_requirements(style_demands)
            return jsonify(result)
        else:
            return jsonify({
                'error': 'BOM netting not available',
                'status': 'emergency_fixes_not_loaded'
            }), 501
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

## API Usage Examples

### Test Dynamic Path Resolution

```bash
curl -X GET "http://localhost:5006/api/data-files" 
```

### Test Price Parsing

```bash
curl -X POST "http://localhost:5006/api/parse-prices" \
  -H "Content-Type: application/json" \
  -d '{"prices": ["$4.07", "$14.95 (kg)", "€100.50"]}'
```

### Calculate Real KPIs

```bash
curl -X GET "http://localhost:5006/api/comprehensive-kpis?real=true"
```

### Multi-level BOM Netting

```bash
curl -X POST "http://localhost:5006/api/bom-netting" \
  -H "Content-Type: application/json" \
  -d '{"style_demands": {"STYLE001": 100.0, "STYLE002": 50.0}}'
```

## Configuration

### Environment Variables

Set these to customize behavior:

```bash
export BKI_ENABLE_EMERGENCY_FIXES=true
export BKI_LOG_LEVEL=INFO
export BKI_CACHE_ENABLED=true
export BKI_PATH_CACHE_TTL=3600
```

### Feature Flags

Add to your configuration:

```python
EMERGENCY_FIXES_CONFIG = {
    'path_resolution_enabled': True,
    'column_standardization_enabled': True,
    'price_parsing_enhanced': True,
    'real_kpi_calculation': True,
    'bom_netting_enabled': True,
    'cache_ttl_seconds': 3600
}
```

## Testing

### Unit Tests

Create tests for emergency fixes integration:

```python
import unittest
from scripts.day0_emergency_fixes import initialize_emergency_fixes

class TestEmergencyFixesIntegration(unittest.TestCase):
    def setUp(self):
        self.components = initialize_emergency_fixes()
    
    def test_path_resolution(self):
        files = self.components['path_resolver'].resolve_all_files()
        self.assertGreater(len(files), 5)  # Should find at least 5 files
    
    def test_price_parsing(self):
        result = self.components['price_parser'].parse_price('$4.07')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['value'], 4.07)
    
    def test_kpi_calculation(self):
        kpis = self.components['kpi_calculator'].calculate_comprehensive_kpis()
        self.assertEqual(kpis['calculation_status'], 'success')
```

### Health Monitoring

Add health check endpoint:

```python
@app.route('/api/health/emergency-fixes')
def health_emergency_fixes():
    """Health check for emergency fixes"""
    from scripts.day0_emergency_fixes import run_comprehensive_health_check
    
    health_results = run_comprehensive_health_check()
    
    status_code = 200 if health_results['overall_status'] == 'healthy' else 503
    return jsonify(health_results), status_code
```

## Performance Considerations

1. **Caching**: The emergency fixes include comprehensive caching for data files and calculations
2. **Lazy Loading**: Components are initialized only when needed
3. **Fallback Logic**: All fixes include fallback to original methods if fixes fail
4. **Error Handling**: Comprehensive error handling prevents system crashes

## Monitoring and Logging

The emergency fixes provide detailed logging:

```python
# Enable debug logging
import logging
logging.getLogger('day0_emergency_fixes').setLevel(logging.DEBUG)
```

Monitor key metrics:
- Path resolution success rate
- Price parsing accuracy
- KPI calculation performance
- BOM netting execution time

## Rollback Plan

If issues occur, disable emergency fixes:

```python
# In your configuration
EMERGENCY_FIXES_ENABLED = False

# Or set environment variable
export BKI_ENABLE_EMERGENCY_FIXES=false
```

The system will automatically fall back to original methods.

## Production Deployment

1. Deploy the `day0_emergency_fixes.py` script to the scripts directory
2. Update the main ERP file with integration code
3. Test thoroughly in staging environment
4. Enable feature flags gradually in production
5. Monitor health checks and performance metrics

## Support

If you encounter issues:

1. Check the health check endpoint: `/api/health/emergency-fixes`
2. Review logs for error messages
3. Test individual components using the script's CLI
4. Fall back to original methods if needed

For debugging:

```bash
# Test specific components
python3 scripts/day0_emergency_fixes.py --test-paths
python3 scripts/day0_emergency_fixes.py --test-prices
python3 scripts/day0_emergency_fixes.py --calculate-kpis
python3 scripts/day0_emergency_fixes.py --health-check
```