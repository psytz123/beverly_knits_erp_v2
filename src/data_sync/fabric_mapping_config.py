#!/usr/bin/env python3
"""
Fabric Mapping Configuration for Beverly Knits ERP
Defines the specific naming conventions for style/fabric references across different data sources
"""

from typing import Dict, List, Optional

class FabricMappingConfig:
    """Central configuration for fabric/style naming across all data sources"""
    
    # Define the fabric mapping based on data source
    FABRIC_MAPPINGS = {
        # Data Source Pattern -> Column Name for Style/Fabric Reference
        "eFab_SO_List_": {
            "style_columns": ["cFVersion", "fBase"],  # Has both columns
            "primary_column": "cFVersion",
            "alternate_names": ["cFVersion", "fBase", "Fabric Version", "Fabric Base"],
            "description": "Sales order list with fabric version and base"
        },
        
        "eFab_Knit_Orders_": {
            "style_column": "Style #",
            "alternate_names": ["Style #", "Style#", "Style Number"],
            "description": "Style number in knit orders"
        },
        
        "eFab_Inventory_I01_": {
            "style_column": "Style #",
            "alternate_names": ["Style #", "Style#", "Style Number"],
            "description": "Style number in I01 inventory"
        },
        
        "eFab_Inventory_G00_": {
            "style_column": "Style #",
            "alternate_names": ["Style #", "Style#", "Style Number"],
            "description": "Style number in G00 inventory"
        },
        
        "eFab_Inventory_G02_": {
            "style_column": "fStyle",
            "alternate_names": ["fStyle", "fStyle#", "Fabric Style"],
            "description": "Fabric style in G02 inventory"
        },
        
        "eFab_Inventory_F01_": {
            "style_column": "Style #",
            "alternate_names": ["Style #", "Style#", "Style Number"],
            "description": "Style number in F01 finished goods inventory"
        },
        
        "QuadS_finishedFabricList_": {
            "style_column": "Style#",
            "alternate_names": ["Style#", "Style #", "StyleNumber"],
            "description": "Style number in QuadS finished fabric"
        },
        
        "BOM_updated": {
            "style_column": "Style#",
            "alternate_names": ["Style#", "Style #", "StyleNumber"],
            "description": "Style number in BOM"
        },
        
        "Sales Activity Report": {
            "style_column": "Style",
            "maps_to": "cFVersion",  # Style in sales = cFVersion
            "alternate_names": ["Style", "Style #", "Style#", "StyleNumber"],
            "description": "Style in sales activity maps to cFVersion"
        }
    }
    
    # Define standardized output column names
    STANDARD_COLUMNS = {
        "style_primary": "Style#",          # Primary style identifier
        "style_fabric": "fStyle#",          # Fabric-specific style
        "style_base": "fBase",              # Fabric base
        "style_version": "cFVersion",       # Fabric version
        "yarn_id": "Desc#",                 # Yarn/component identifier
    }
    
    # Mapping rules for consolidation
    CONSOLIDATION_RULES = {
        # When to use fStyle# vs Style#
        "use_fstyle": ["eFab_Inventory_G02_"],  # Only G02 uses fStyle
        "use_style": ["eFab_Knit_Orders_", "eFab_Inventory_I01_", "eFab_Inventory_G00_", 
                     "eFab_Inventory_F01_", "QuadS_finishedFabricList_", "BOM_updated"],
        "use_cfversion": ["Sales Activity Report"],  # Sales Style maps to cFVersion
        "use_both": ["eFab_SO_List_"]  # Has both cFVersion and fBase
    }
    
    @classmethod
    def get_style_column_for_file(cls, filename: str) -> Optional[Dict[str, any]]:
        """Get the correct style column name based on filename"""
        for pattern, config in cls.FABRIC_MAPPINGS.items():
            if pattern in filename:
                return config
        return None
    
    @classmethod
    def get_standard_mapping(cls, filename: str, df_columns: List[str]) -> Dict[str, str]:
        """Get column mapping to standardize a dataframe based on filename"""
        mapping = {}
        file_config = cls.get_style_column_for_file(filename)
        
        if file_config:
            # Special handling for eFab_SO_List which has multiple columns
            if "style_columns" in file_config:
                for style_col in file_config["style_columns"]:
                    if style_col in df_columns:
                        if style_col == "cFVersion":
                            mapping[style_col] = cls.STANDARD_COLUMNS["style_version"]
                        elif style_col == "fBase":
                            mapping[style_col] = cls.STANDARD_COLUMNS["style_base"]
            
            # Special handling for Sales Activity Report
            elif "maps_to" in file_config:
                style_col = file_config.get("style_column", "Style")
                if style_col in df_columns:
                    mapping[style_col] = cls.STANDARD_COLUMNS["style_version"]  # Maps to cFVersion
                # Also check alternates
                for alt_name in file_config["alternate_names"]:
                    if alt_name in df_columns and alt_name not in mapping:
                        mapping[alt_name] = cls.STANDARD_COLUMNS["style_version"]
                        break
            
            # Standard handling for other files
            elif "style_column" in file_config:
                style_col = file_config["style_column"]
                
                # Look for the style column in the dataframe
                for col in df_columns:
                    # Check primary name
                    if col == style_col:
                        # Determine which standard column to map to
                        if filename.startswith(tuple(cls.CONSOLIDATION_RULES["use_fstyle"])):
                            mapping[col] = cls.STANDARD_COLUMNS["style_fabric"]
                        else:
                            mapping[col] = cls.STANDARD_COLUMNS["style_primary"]
                        break
                    
                    # Check alternate names
                    for alt_name in file_config["alternate_names"]:
                        if col.lower() == alt_name.lower():
                            if filename.startswith(tuple(cls.CONSOLIDATION_RULES["use_fstyle"])):
                                mapping[col] = cls.STANDARD_COLUMNS["style_fabric"]
                            else:
                                mapping[col] = cls.STANDARD_COLUMNS["style_primary"]
                            break
        
        return mapping
    
    @classmethod
    def normalize_style_value(cls, value: any, source_file: str = None) -> str:
        """Normalize a style value to consistent format"""
        if pd.isna(value) or value is None:
            return ""
        
        # Convert to string and clean
        value = str(value).strip()
        
        # Remove common prefixes/suffixes
        value = value.replace("Style ", "").replace("Style#", "").replace("Style# ", "")
        
        # Handle special cases based on source
        if source_file:
            if "eFab_SO_List_" in source_file:
                # fBase values might have special formatting
                value = value.upper()
            elif "eFab_Styles_" in source_file:
                # cFVersion might have version numbers
                value = value.replace("v", "V")
        
        return value
    
    @classmethod
    def get_mapping_report(cls) -> str:
        """Generate a human-readable mapping report"""
        report = "FABRIC MAPPING CONFIGURATION\n"
        report += "="*50 + "\n\n"
        
        for source, config in cls.FABRIC_MAPPINGS.items():
            report += f"Source: {source}\n"
            report += f"  Primary Column: {config['style_column']}\n"
            report += f"  Alternates: {', '.join(config['alternate_names'])}\n"
            report += f"  Description: {config['description']}\n"
            report += "\n"
        
        report += "\nSTANDARDIZED OUTPUT COLUMNS:\n"
        report += "-"*30 + "\n"
        for key, value in cls.STANDARD_COLUMNS.items():
            report += f"  {key}: {value}\n"
        
        return report


# Integration with data parser
def apply_fabric_mapping(df, filename: str) -> tuple:
    """Apply fabric mapping to a dataframe based on filename"""
    import pandas as pd
    
    corrections = []
    
    # Get mapping for this file
    mapping = FabricMappingConfig.get_standard_mapping(filename, df.columns.tolist())
    
    if mapping:
        # Apply column renaming
        for old_col, new_col in mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
                corrections.append(f"Mapped '{old_col}' to standard '{new_col}'")
                
                # Normalize values in the column
                if new_col in df.columns:
                    df[new_col] = df[new_col].apply(
                        lambda x: FabricMappingConfig.normalize_style_value(x, filename)
                    )
                    corrections.append(f"Normalized values in '{new_col}'")
    
    # Additional style column detection
    style_patterns = ['style', 'fstyle', 'fabric']
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in style_patterns):
            if col not in mapping:
                # This might be a style column we missed
                corrections.append(f"Note: Found potential style column '{col}' not in mapping")
    
    return df, corrections


if __name__ == "__main__":
    # Test the mapping configuration
    print(FabricMappingConfig.get_mapping_report())
    
    # Test specific file mappings
    test_files = [
        "eFab_Styles_20250814.xlsx",
        "eFab_SO_List_202508160148.csv",
        "eFab_Knit_Orders_20250816.xlsx",
        "eFab_Inventory_F01_20250814.xlsx",
        "Sales Activity Report (6).csv"
    ]
    
    print("\n\nTEST FILE MAPPINGS:")
    print("-"*50)
    for filename in test_files:
        config = FabricMappingConfig.get_style_column_for_file(filename)
        if config:
            print(f"\n{filename}:")
            print(f"  Style Column: {config['style_column']}")
            
            # Determine standard output
            if filename.startswith(tuple(FabricMappingConfig.CONSOLIDATION_RULES["use_fstyle"])):
                print(f"  Maps to: fStyle#")
            elif filename.startswith(tuple(FabricMappingConfig.CONSOLIDATION_RULES["use_style"])):
                print(f"  Maps to: Style#")
            elif filename.startswith(tuple(FabricMappingConfig.CONSOLIDATION_RULES["use_fbase"])):
                print(f"  Maps to: fBase")
            elif filename.startswith(tuple(FabricMappingConfig.CONSOLIDATION_RULES["use_cfversion"])):
                print(f"  Maps to: cFVersion")