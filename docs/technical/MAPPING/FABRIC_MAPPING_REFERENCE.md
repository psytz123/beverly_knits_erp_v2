# Fabric Mapping Reference Guide

```

```

## Overview

Different data sources use different column names for style/fabric identifiers. This reference shows the exact mapping for each source.

## Mapping Table

| Data Source                                                          | Style Column Name         | Maps To                   | Description                      |
| -------------------------------------------------------------------- | ------------------------- | ------------------------- | -------------------------------- |
| /api/sales-order/plan/list                                           | `cFVersion` + `fBase` | `cFVersion` + `fBase` | Has both fabric version and base |
| `GET /api/knitorder/list`                                          | `Style #`               | `Style#`                | Style number in knit orders      |
| `GET /api/finished/i01`                                            | `Style #`               | `Style#`                | Style in QC inspection inventory |
| `GET /api/greige/g00`                                              | `Style #`               | `Style#`                | Style in greige stage 1          |
| `GET /api/greige/g02`                                              | `fStyle`                | `fStyle#`               | Fabric style in greige stage 2   |
| `GET /api/finished/f01`                                            | `Style #`               | `Style#`                | Style in finished goods          |
| `GET /api/styles/greige/active`<br />/api/styles/finished/active  | `Style#`                | `Style#`                | Style in QuadS fabric list       |
| **BOM_updated**                                                | `Style#`                | `Style#`                | Style in bill of materials       |
| `GET /api/styles`                                                 | `Style`                 | `cFVersion`             | Style in sales = cFVersion       |

## Standardized Output Columns

After processing, all style-related columns are mapped to these standard names:

- **`Style#`** - Primary style identifier (used in most files)
- **`fStyle#`** - Fabric-specific style (used in F01 and G02 inventory)
- **`fBase`** - Fabric base (used in SO List)
- **`cFVersion`** - Fabric version (used in eFab Styles)
- **`Desc#`** - Yarn/component identifier (standardized across all files)
- 

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

| Original         | Cleaned To  |
| ---------------- | ----------- |
| `Style #`      | `Style#`  |
| `Style Number` | `Style#`  |
| `StyleNumber`  | `Style#`  |
| `fStyle`       | `fStyle#` |
| `fStyle #`     | `fStyle#` |
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
