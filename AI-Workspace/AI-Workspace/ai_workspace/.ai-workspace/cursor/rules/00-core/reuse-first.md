---
description: Reuse-First Workflow - Search before creating, wrapper before implementing
applies_to: all
priority: high
---

# Reuse-First Workflow

**Principle:** Every line of code is a liability. Reuse existing solutions before creating new ones.

---

## 🔍 Mandatory Search Workflow

**BEFORE writing ANY code, complete this checklist:**

### Step 1: Search Internal Codebase
```bash
# Search for similar functionality
grep -r "function_name" src/
grep -r "class_name" src/

# Use semantic search if available
@search-specialist find similar implementation for [feature]
```

### Step 2: Check Shared Libraries
- **Python:** `shared/`, `common/`, `utils/`
- **TypeScript:** `lib/`, `utils/`, `shared/`
- **Look for:**
  - Similar data transformations
  - Comparable validation logic
  - Analogous API patterns

### Step 3: Search Dependencies
```bash
# Python
pip search [functionality]
# Check existing deps for capability

# JavaScript
npm search [functionality]
# Review package.json dependencies
```

### Step 4: Calculate Reuse Percentage
```
Reuse % = (Lines reusable / Total lines needed) × 100

If Reuse % ≥ 70%:
  → Create wrapper/adapter
  → Document reuse in code comments
  → Add tests for wrapper

If Reuse % < 70%:
  → Implement new code
  → Document why reuse failed in ADR
  → Add to shared library if generalizable
```

---

## ✅ Decision Matrix

| Reuse % | Action | Documentation Required |
|---------|--------|------------------------|
| **90-100%** | Use directly, no wrapper needed | Inline comment with source |
| **70-89%** | Create thin wrapper/adapter | Function docstring + ADR |
| **50-69%** | Consider wrapper vs. new | ADR with trade-off analysis |
| **<50%** | Implement new code | ADR explaining why reuse failed |

---

## 🎯 Wrapper Patterns

### Example: Reusing 80% of existing code

```python
# existing_service.py (in shared/)
def process_data(data: dict, config: dict) -> dict:
    """Generic data processor."""
    # 100 lines of complex logic
    ...

# your_service.py (new code)
from shared.existing_service import process_data

def process_order_data(order: Order) -> ProcessedOrder:
    """Process order data using existing processor.

    Reuses: shared.existing_service.process_data (80% match)
    Adaptation: Order → dict conversion, dict → ProcessedOrder
    """
    # Convert Order to dict (new code, 10 LOC)
    data_dict = {
        "id": order.id,
        "items": [item.to_dict() for item in order.items],
        "total": order.total,
    }

    # Reuse existing processor (0 new LOC)
    result = process_data(data_dict, config={"mode": "order"})

    # Convert result to ProcessedOrder (new code, 5 LOC)
    return ProcessedOrder(**result)

# Result: 15 new LOC instead of 115 LOC (87% reuse)
```

---

## 🚫 Anti-Patterns

### ❌ BAD: Copy-Paste Programming
```python
# service_a.py
def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# service_b.py (COPIED CODE!)
def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
```

### ✅ GOOD: Shared Utility
```python
# shared/validators.py
def validate_email(email: str) -> bool:
    """Validate email format per RFC 5322."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# service_a.py
from shared.validators import validate_email

# service_b.py
from shared.validators import validate_email
```

---

## 📋 Reuse Report Template

When creating new code, document your reuse analysis:

```markdown
## Reuse Analysis for [Feature Name]

**Date:** YYYY-MM-DD
**Author:** [Your Name / Agent]

### Search Results
- **Internal Search:** Found 3 similar implementations
  - `shared/utils/data_processor.py` - 60% match
  - `service_x/helpers.py` - 40% match
  - `service_y/transforms.py` - 30% match

- **Dependency Search:** Checked existing deps
  - `pandas` - Has transform capability but overkill (adds 50MB)
  - `pydantic` - Already using, no matching feature

### Best Match Analysis
- **Source:** `shared/utils/data_processor.py`
- **Match Percentage:** 60%
- **Gaps:**
  1. No support for nested objects (20% gap)
  2. Different error handling (10% gap)
  3. Missing async support (10% gap)

### Decision
- **Reuse %:** 60% (below 70% threshold)
- **Action:** Implement new code
- **Rationale:**
  - Wrapper would require 40% custom code
  - Async support is critical (not in original)
  - Better to implement clean async version
- **Future:** Consider refactoring shared version to support async

### Impact
- **LOC Added:** 75 lines
- **LOC Saved:** 45 lines (reused patterns)
- **Net New LOC:** 30 lines
- **Added to:** `shared/async_processor.py` for future reuse
```

---

## 🎓 Best Practices

### 1. Always Check Standard Library First
```python
# ❌ BAD: Reinventing the wheel
def deep_merge_dicts(dict1, dict2):
    # 30 lines of complex merge logic
    ...

# ✅ GOOD: Use standard library
from collections import ChainMap
merged = ChainMap(dict1, dict2)

# Or for Python 3.9+
merged = dict1 | dict2
```

### 2. Favor Composition Over Duplication
```python
# ❌ BAD: Duplicate logic
class UserService:
    def validate_and_save(self, user_data):
        # validation logic (20 LOC)
        # save logic (15 LOC)

class ProductService:
    def validate_and_save(self, product_data):
        # same validation logic (20 LOC)
        # same save logic (15 LOC)

# ✅ GOOD: Compose from shared components
from shared.validators import validate_data
from shared.repository import save_entity

class UserService:
    def validate_and_save(self, user_data):
        validate_data(user_data, UserSchema)
        return save_entity(user_data, User)

class ProductService:
    def validate_and_save(self, product_data):
        validate_data(product_data, ProductSchema)
        return save_entity(product_data, Product)
```

### 3. Extract Common Patterns
```python
# After writing similar code 3 times, extract pattern:

# Service A, B, C all do:
def process_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

# Extract to shared/retry.py
# Now all services reuse instead of duplicating
```

---

## 🔧 Tools & Automation

### Semantic Code Search
```bash
# Use grep with context
grep -r -A 5 -B 5 "pattern" src/

# Use specialized tools
rg --type py "function_pattern"

# AI-powered search
@search-specialist find code that does [X]
```

### Dependency Audit
```bash
# Python
pip list | grep [package]
pip show [package]

# JavaScript
npm list [package]
npm info [package]
```

### Duplication Detection
```bash
# Python
pylint --disable=all --enable=duplicate-code src/

# JavaScript
npx jscpd src/

# Universal
npx jscpd --threshold 3 .
```

---

## 📊 Reuse Metrics

Track reuse effectiveness:

```yaml
# .agent-workspace/metrics/reuse-stats.yml
month: 2025-10
total_features: 25
reuse_analysis_completed: 25
breakdown:
  reused_90_plus: 8  # Used existing code directly
  wrapped_70_89: 7   # Created wrappers
  custom_50_69: 5    # Partial reuse
  net_new_below_50: 5  # Implemented from scratch

avg_reuse_percentage: 71.2%
loc_saved: 1,847
loc_written: 982
efficiency_ratio: 1.88  # 1.88x more efficient than greenfield
```

---

**Remember:** The best code is no code. The second-best code is someone else's well-tested code.
