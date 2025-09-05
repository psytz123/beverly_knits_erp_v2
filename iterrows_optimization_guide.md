# DataFrame.iterrows() Optimization Guide

================================================================================
DataFrame.iterrows() Performance Optimization Report
================================================================================

Total iterrows() instances found: 94
Files affected: 6

Files with most instances:
  - beverly_comprehensive_erp.py: 56 instances
  - six_phase_planning_engine.py: 10 instances
  - database_etl_pipeline.py: 9 instances
  - time_phased_mrp_service.py: 7 instances
  - yarn_repository.py: 6 instances

Pattern Analysis:
  - unknown: 64 instances
  - dictionary_build: 28 instances
  - conditional: 2 instances

Estimated Performance Improvement:
  - Current: ~10-15 seconds for all iterrows operations
  - Optimized: ~0.1-0.5 seconds (10-100x faster)
  - Time saved per run: ~10+ seconds

Next Steps:
  1. Review the suggested optimizations
  2. Apply vectorized replacements carefully
  3. Test each optimization thoroughly
  4. Benchmark before and after changes

## Specific Fixes

### Planning Balance
```python

# BEFORE (with iterrows):
for idx, row in df.iterrows():
    df.at[idx, 'planning_balance'] = row['theoretical_balance'] + row['allocated'] + row['on_order']

# AFTER (vectorized):
df['planning_balance'] = df['theoretical_balance'] + df['allocated'] + df['on_order']
```

### Shortage Detection
```python

# BEFORE (with iterrows):
shortages = []
for idx, row in df.iterrows():
    if row['planning_balance'] < 0:
        shortages.append(row)

# AFTER (vectorized):
shortages = df[df['planning_balance'] < 0].copy()
```

### Conditional Update
```python

# BEFORE (with iterrows):
for idx, row in df.iterrows():
    if row['quantity'] > threshold:
        df.at[idx, 'status'] = 'HIGH'

# AFTER (vectorized):
df.loc[df['quantity'] > threshold, 'status'] = 'HIGH'
```

### Accumulation
```python

# BEFORE (with iterrows):
total = 0
for idx, row in df.iterrows():
    total += row['quantity'] * row['price']

# AFTER (vectorized):
total = (df['quantity'] * df['price']).sum()
```

### Groupby Aggregation
```python

# BEFORE (with iterrows):
results = {}
for idx, row in df.iterrows():
    key = row['category']
    if key not in results:
        results[key] = 0
    results[key] += row['value']

# AFTER (vectorized):
results = df.groupby('category')['value'].sum().to_dict()
```

