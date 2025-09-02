# Fabric Mapping Reference Guide

## Overview
Different data sources use different column names for style/fabric identifiers. This reference shows the exact mapping for each source.

## Mapping Table

| Data Source | Style Column Name | Maps To | Description |
|------------|-------------------|---------|-------------|
| **eFab_SO_List_** | `cFVersion` + `fBase` | `cFVersion` + `fBase` | Has both fabric version and base |
| **eFab_Knit_Orders_** | `Style #` | `Style#` | Style number in knit orders |
| **eFab_Inventory_I01_** | `Style #` | `Style#` | Style in QC inspection inventory |
| **eFab_Inventory_G00_** | `Style #` | `Style#` | Style in greige stage 1 |
| **eFab_Inventory_G02_** | `fStyle` | `fStyle#` | Fabric style in greige stage 2 |
| **eFab_Inventory_F01_** | `Style #` | `Style#` | Style in finished goods |
| **QuadS_finishedFabricList_** | `Style#` | `Style#` | Style in QuadS fabric list |
| **BOM_updated** | `Style#` | `Style#` | Style in bill of materials |
| **Sales Activity Report** | `Style` | `cFVersion` | Style in sales = cFVersion |

## Standardized Output Columns

After processing, all style-related columns are mapped to these standard names:

- **`Style#`** - Primary style identifier (used in most files)
- **`fStyle#`** - Fabric-specific style (used in F01 and G02 inventory)
- **`fBase`** - Fabric base (used in SO List)
- **`cFVersion`** - Fabric version (used in eFab Styles)
- **`Desc#`** - Yarn/component identifier (standardized across all files)

## File-Specific Rules

### Files using `fStyle#`:
- eFab_Inventory_G02_*.xlsx (Only G02 uses fStyle)

### Files using `Style#`:
- eFab_Knit_Orders_*.xlsx
- eFab_Inventory_I01_*.xlsx
- eFab_Inventory_G00_*.xlsx
- eFab_Inventory_F01_*.xlsx
- QuadS_finishedFabricList_*.csv
- BOM_updated.csv

### Special Cases:
- **eFab_SO_List_**: Contains BOTH `cFVersion` and `fBase` columns
- **Sales Activity Report**: `Style` column maps to `cFVersion`

## Data Flow Example

```
Sales Order (eFab_SO_List)
    fBase: "ABC123"
    ↓
Style Mapping (eFab_Styles)
    cFVersion: "V1.2"
    ↓
Production Order (eFab_Knit_Orders)
    Style #: "ABC123-001"
    ↓
Inventory Stages:
    G00: Style # = "ABC123-001"
    G02: fStyle = "ABC123-F"
    I01: Style # = "ABC123-001"
    F01: fStyle = "ABC123-F"
```

## Column Name Variations

The parser handles these common variations automatically:

| Original | Cleaned To |
|----------|------------|
| `Style #` | `Style#` |
| `Style Number` | `Style#` |
| `StyleNumber` | `Style#` |
| `fStyle` | `fStyle#` |
| `fStyle #` | `fStyle#` |
| `Fabric Style` | `fStyle#` |

## Processing Order

1. **Fabric Mapping** - Apply source-specific mappings first
2. **Column Standardization** - Fix common naming variations
3. **Data Type Validation** - Ensure correct data types
4. **Value Normalization** - Clean and standardize values

## Benefits

1. **Consistency** - Same column names across all files
2. **Traceability** - Track styles through production stages
3. **Integration** - Easy joins between different data sources
4. **Automation** - No manual column renaming needed

## Troubleshooting

If style columns are not mapping correctly:

1. Check the filename matches the expected pattern
2. Verify the source column exists in the file
3. Look for typos in column names
4. Check the cleaning report for warnings

The system will log all mappings applied during data cleaning.