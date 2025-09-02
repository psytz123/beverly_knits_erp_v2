# Complete Data Mapping Reference - Beverly Knits ERP

## Overview
This reference documents all column mappings for both fabric/style identifiers and yarn identifiers across different data sources.

---

## FABRIC/STYLE MAPPING

### Mapping Table

| Data Source | Style Column | Maps To | Notes |
|------------|--------------|---------|-------|
| **eFab_SO_List_** | `cFVersion` + `fBase` | `cFVersion` + `fBase` | Has BOTH columns |
| **eFab_Knit_Orders_** | `Style #` | `Style#` | Production orders |
| **eFab_Inventory_I01_** | `Style #` | `Style#` | QC inspection |
| **eFab_Inventory_G00_** | `Style #` | `Style#` | Greige stage 1 |
| **eFab_Inventory_F01_** | `Style #` | `Style#` | Finished goods |
| **eFab_Inventory_G02_** | `fStyle` | `fStyle#` | Greige stage 2 (only file using fStyle) |
| **QuadS_finishedFabric** | `Style#` | `Style#` | QuadS fabric list |
| **BOM_updated** | `Style#` | `Style#` | Bill of materials |
| **Sales Activity Report** | `Style` | `cFVersion` | ⚠️ Special: Style = cFVersion |

### Standardized Style Columns
- **`Style#`** - Primary style identifier (most files)
- **`fStyle#`** - Fabric style (only G02 inventory)
- **`fBase`** - Fabric base (SO List)
- **`cFVersion`** - Fabric version (SO List & Sales)

---

## YARN MAPPING

### Mapping Table

| Data Source | Yarn Column | Maps To | Notes |
|------------|-------------|---------|-------|
| **BOM_updated** | `Desc#` | `Desc#` | Already standard |
| **Yarn_ID** | `Desc#` | `Desc#` | Already standard |
| **Yarn_ID_Master** | `Desc#` | `Desc#` | Already standard |
| **yarn_inventory** | `Desc#` | `Desc#` | Already standard |
| **Expected_Yarn_Report** | `Desc` | `Desc#` | Maps Desc → Desc# |
| **Yarn_Demand_Report** | `Yarn` | `Desc#` | Maps Yarn → Desc# |
| **Yarn_Demand_By_Style** | `Yarn` | `Desc#` | Maps Yarn → Desc# |
| **Yarn_Demand_By_Style_KO** | `Yarn` | `Desc#` | Maps Yarn → Desc# |

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