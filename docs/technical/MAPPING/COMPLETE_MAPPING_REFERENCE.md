# Complete Data Mapping Reference - Beverly Knits ERP

## Overview

This reference documents all column mappings for both fabric/style identifiers and yarn identifiers across different data sources.

---

## FABRIC/STYLE MAPPING

### Mapping Table


| <br />Data Source                                                | Style Column              | Maps To                   | Notes                                   |
| ---------------------------------------------------------------- | ------------------------- | ------------------------- | --------------------------------------- |
| api/sales-order/plan/list                                        | `cFVersion` + `fBase` | `cFVersion` + `fBase` | Has BOTH columns                        |
| `GET /api/knitorder/list`                                      | `Style #`               | `Style#`                | Production orders                       |
| `GET /api/finished/i01`                                        | `Style #`               | `Style#`                | QC inspection                           |
| `GET /api/greige/g00`                                          | `Style #`               | `Style#`                | Greige stage 1                          |
| `GET /api/finished/f01`                                        | `Style #`               | `Style#`                | Finished goods                          |
| `GET /api/greige/g02`                                          | `fStyle`                | `fStyle#`               | Greige stage 2 (only file using fStyle) |
| `GET /api/styles/greige/active` & /api/styles/finished/active | `Style#`                | `Style#`                | QuadS fabric list                       |
| **BOM_updated**                                            | `Style#`                | `Style#`                | Bill of materials                       |
| **Sales Activity Report**                                  | `Style`                 | `cFVersion`             | ⚠️ Special: Style = cFVersion         |

- `GET /api/styles/greige/active` - Greige styles from QuadS
- `GET /api/styles/finished/active` - Finished styles from QuadS

### Standardized Style Columns

- **`Style#`** - Primary style identifier (most files)
- **`fStyle#`** - Fabric style (only G02 inventory)
- **`fBase`** - Fabric base (SO List)
- **`cFVersion`** - Fabric version (SO List & Sales)

---

## YARN MAPPING

### Mapping Table

| Data Source                        | Yarn Column | Maps To   | Notes              |
| ---------------------------------- | ----------- | --------- | ------------------ |
| **BOM_updated**              | `Desc#`   | `Desc#` | Already standard   |
| `GET /api/yarn/active`           | `Desc#`   | `Desc#` | Already standard   |
| `GET /api/yarn/active`           | `Desc#`   | `Desc#` | Already standard   |
| `GET /api/yarn/active`           | `Desc#`   | `Desc#` | Already standard   |
| `/api/report/yarn_expected`      | `Desc`    | `Desc#` | Maps Desc → Desc# |
| `GET /api/report/yarn_demand`    | `Yarn`    | `Desc#` | Maps Yarn → Desc# |
| `GET /api/report/yarn_demand`    | `Yarn`    | `Desc#` | Maps Yarn → Desc# |
| `GET /api/report/yarn_demand_ko` | `Yarn`    | `Desc#` | Maps Yarn → Desc# |

### Standardized Yarn Column

- **`Desc#`** - Universal yarn identifier (ALL files)

### Yarn Attribute Columns

These columns are also standardized when found:

- **Color** → `Yarn_Color`
- **Type** → `Yarn_Type`
- **Count** → `Yarn_Count`
- **Supplier** → `Yarn_Supplier`

---

## PROCESSING ORDER

The data parser applies mappings in this sequence:

1. **Fabric Mapping** - Style/fabric columns based on file type
2. **Yarn Mapping** - Yarn identifiers to Desc#
3. **General Column Cleaning** - Fix typos and variations
4. **Data Type Validation** - Ensure correct data types
5. **Value Normalization** - Clean and standardize values

---

## COMMON VARIATIONS HANDLED

### Style Variations

- `Style #` → `Style#`
- `Style Number` → `Style#`
- `fStyle` → `fStyle#`
- `Fabric Style` → `fStyle#`

### Yarn Variations

- `Yarn_ID` → `Desc#`
- `YarnID` → `Desc#`
- `Yarn ID` → `Desc#`
- `Description` → `Desc#`
- `Yarn` → `Desc#` (in demand reports)
- `Desc` → `Desc#` (in expected reports)

---

## FILE IDENTIFICATION

Files are identified by these patterns in their names:

### Fabric/Style Files

- Contains `eFab_`
- Contains `QuadS_`
- Contains `BOM`
- Contains `Sales Activity`

### Yarn Files

- Contains `yarn_inventory`
- Contains `Yarn_ID`
- Contains `Yarn_Demand`
- Contains `Expected_Yarn`

---

## VALIDATION RULES

### Style/Fabric Values

- Remove "Style " prefix
- Standardize spacing
- Preserve alphanumeric codes

### Yarn Values

- Remove leading zeros from numeric codes
- Remove "YARN-" or "Y-" prefixes
- Convert to uppercase for demand reports
- Must not be empty after cleaning

---

## CRITICAL RELATIONSHIPS

### Style Flow

```
Sales (cFVersion) → SO List (cFVersion + fBase) → Production (Style#) → Inventory (Style# or fStyle#)
```

### Yarn Flow

```
BOM (Desc#) → Inventory (Desc#) → Demand Reports (Yarn → Desc#)
```

---

## TROUBLESHOOTING

### If columns aren't mapping:

1. Check filename matches expected pattern
2. Verify column exists in source file
3. Look for typos in column names
4. Check cleaning report for details

### Common Issues:

- **Multiple style columns**: Parser preserves all, maps each appropriately
- **Missing columns**: Logged as warnings, processing continues
- **Invalid values**: Cleaned but flagged in report

---

## DATA QUALITY BENEFITS

1. **Consistency**: Same identifiers across pipeline
2. **Traceability**: Track items from sales to inventory
3. **Integration**: Easy joins between data sources
4. **Automation**: No manual column renaming
5. **Validation**: Catches data issues early

The mapping system ensures all data files use consistent column names regardless of source variations!
