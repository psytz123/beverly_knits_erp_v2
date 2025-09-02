# Planning Balance Formula - Official Documentation

## Formula Definition

```
Planning Balance = Theoretical Balance + Allocated + On Order
```

## Critical Business Logic

### Understanding Allocated Values

The `Allocated` field represents yarn that has been **consumed or committed** to production orders. These values are **NEGATIVE** by design:

- **Negative Allocated** = Yarn consumed/committed (reduces available balance)
- **Positive Allocated** = Rare, typically corrections or returns

### Formula Breakdown

1. **Theoretical Balance**: Current physical inventory based on counts
2. **Allocated**: Amount consumed (negative) or returned (positive)
3. **On Order**: Incoming purchase orders not yet received
4. **Planning Balance**: Available inventory for planning purposes

### Mathematical Example

```
Given:
- Theoretical Balance = 224.10 lbs
- Allocated = -185.14 lbs (consumed)
- On Order = 0.00 lbs

Calculation:
Planning Balance = 224.10 + (-185.14) + 0.00
Planning Balance = 38.96 lbs
```

## Validation Results

- **Test Date**: 2025-09-02
- **Rows Tested**: 982
- **Accuracy**: 97.56%
- **Failures**: 24 (due to NaN values in source data)

## Implementation Locations

### Primary Calculation
- **File**: `src/core/beverly_comprehensive_erp.py`
- **Lines**: 11886-11896
- **Function**: `inventory-intelligence-enhanced` endpoint

### Fallback Calculations
- **File**: `src/services/inventory_analyzer_service.py`
- **Method**: `calculate_planning_balance()`

## Common Misunderstandings

1. **"Allocated should be positive"** - INCORRECT
   - Allocated is negative when yarn is consumed
   - The formula ADDS the negative value (subtracting consumption)

2. **"Formula should subtract Allocated"** - INCORRECT
   - Formula correctly adds Allocated
   - Allocated is already negative, so adding it subtracts

## Testing the Formula

```python
def validate_planning_balance(row):
    """Validate Planning Balance calculation for a data row"""
    tb = float(row['Theoretical Balance'])
    allocated = float(row['Allocated'])  # Usually negative
    on_order = float(row['On Order'])
    expected = float(row['Planning Balance'])
    
    calculated = tb + allocated + on_order
    
    # Allow for floating point precision
    assert abs(calculated - expected) < 0.01, \
        f"Mismatch: {calculated} != {expected}"
    
    return True
```

## SQL Equivalent

```sql
SELECT 
    "Desc#",
    "Theoretical Balance",
    "Allocated",
    "On Order",
    "Theoretical Balance" + "Allocated" + "On Order" AS "Calculated_Planning_Balance",
    "Planning Balance" AS "Stored_Planning_Balance"
FROM yarn_inventory
WHERE ABS(("Theoretical Balance" + "Allocated" + "On Order") - "Planning Balance") > 0.01;
```

## Change History

- 2025-09-02: Formula documented and validated (Phase 1 implementation)
- 2025-08-29: API consolidation maintained formula
- 2025-08-23: Formula verified during optimization

## Verification Steps

To verify the formula is working correctly:

1. **Check API Response**:
```bash
curl -s http://localhost:5006/api/inventory-intelligence-enhanced | python3 -m json.tool | head -100
```

2. **Run Validation Script**:
```python
python3 scripts/validate_planning_balance.py
```

3. **Check Database**:
Run the SQL query above to find any mismatches

## Business Impact

The Planning Balance is critical for:
- **Production Planning**: Determines what can be manufactured
- **Yarn Shortage Detection**: Identifies when to reorder
- **Capacity Planning**: Affects production scheduling
- **Financial Reporting**: Impacts inventory valuation

## Support

For questions about this formula:
- Review test cases in `tests/unit/test_inventory_fixes.py`
- Check implementation in `beverly_comprehensive_erp.py:11886`
- Contact system administrator if discrepancies found