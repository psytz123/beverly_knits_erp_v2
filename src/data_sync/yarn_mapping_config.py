#!/usr/bin/env python3
"""
Yarn Mapping Configuration for Beverly Knits ERP
Defines the specific naming conventions for yarn identifiers across different data sources
All yarn columns should map to the standard Desc# format
"""

from typing import Dict, List, Optional
import pandas as pd

class YarnMappingConfig:
    """Central configuration for yarn naming across all data sources"""
    
    # Define the yarn mapping based on data source
    YARN_MAPPINGS = {
        # Data Source Pattern -> Column Name for Yarn Reference
        "BOM_updated": {
            "yarn_column": "Desc#",
            "alternate_names": ["Desc#", "Desc #", "Description#"],
            "description": "Yarn descriptor in BOM"
        },
        
        "Yarn_ID": {
            "yarn_column": "Desc#",
            "alternate_names": ["Desc#", "Yarn_ID", "YarnID", "Yarn ID"],
            "description": "Yarn ID file"
        },
        
        "Yarn_ID_Master": {
            "yarn_column": "Desc#",
            "alternate_names": ["Desc#", "Yarn_ID", "YarnID"],
            "description": "Master yarn ID list"
        },
        
        "yarn_inventory": {
            "yarn_column": "Desc#",
            "alternate_names": ["Desc#", "Desc #", "Description#", "Yarn_ID", "YarnID"],
            "description": "Yarn inventory file"
        },
        
        "Expected_Yarn_Report": {
            "yarn_column": "Desc",
            "maps_to": "Desc#",
            "alternate_names": ["Desc", "Description", "Yarn Description"],
            "description": "Expected yarn report - Desc maps to Desc#"
        },
        
        "Yarn_Demand_Report": {
            "yarn_column": "Yarn",
            "maps_to": "Desc#",
            "alternate_names": ["Yarn", "Yarn Description", "Yarn_Desc"],
            "description": "Yarn demand report - Yarn maps to Desc#"
        },
        
        "Yarn_Demand_By_Style": {
            "yarn_column": "Yarn",
            "maps_to": "Desc#",
            "alternate_names": ["Yarn", "Yarn Description", "Yarn_Desc"],
            "description": "Yarn demand by style - Yarn maps to Desc#"
        },
        
        "Yarn_Demand_By_Style_KO": {
            "yarn_column": "Yarn",
            "maps_to": "Desc#",
            "alternate_names": ["Yarn", "Yarn Description", "Yarn_Desc"],
            "description": "Yarn demand by style KO - Yarn maps to Desc#"
        }
    }
    
    # Standard output column name for all yarn references
    STANDARD_YARN_COLUMN = "Desc#"
    
    # Additional yarn-related columns that might appear
    YARN_ATTRIBUTE_COLUMNS = {
        "yarn_color": ["Color", "Yarn_Color", "Yarn Color", "YarnColor"],
        "yarn_type": ["Type", "Yarn_Type", "Yarn Type", "YarnType"],
        "yarn_count": ["Count", "Yarn_Count", "Yarn Count", "YarnCount"],
        "yarn_supplier": ["Supplier", "Yarn_Supplier", "Yarn Supplier", "Vendor"]
    }
    
    @classmethod
    def get_yarn_column_for_file(cls, filename: str) -> Optional[Dict[str, any]]:
        """Get the correct yarn column configuration based on filename"""
        filename_lower = filename.lower()
        
        for pattern, config in cls.YARN_MAPPINGS.items():
            if pattern.lower() in filename_lower:
                return config
        return None
    
    @classmethod
    def get_yarn_mapping(cls, filename: str, df_columns: List[str]) -> Dict[str, str]:
        """Get column mapping to standardize yarn columns in a dataframe"""
        mapping = {}
        file_config = cls.get_yarn_column_for_file(filename)
        
        if file_config:
            # Get the primary yarn column name
            yarn_col = file_config["yarn_column"]
            target_col = file_config.get("maps_to", cls.STANDARD_YARN_COLUMN)
            
            # Look for the yarn column in the dataframe
            for col in df_columns:
                # Check primary name
                if col == yarn_col and col != target_col:
                    mapping[col] = target_col
                    break
                
                # Check alternate names
                for alt_name in file_config["alternate_names"]:
                    if col.lower() == alt_name.lower() and col != target_col:
                        mapping[col] = target_col
                        break
        
        # Also check for common yarn column patterns not in specific files
        yarn_patterns = ['yarn_id', 'yarnid', 'yarn id', 'desc#', 'desc #', 'description#']
        for col in df_columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in yarn_patterns):
                if col not in mapping and col != cls.STANDARD_YARN_COLUMN:
                    # Only map if it's clearly a yarn ID column
                    if 'yarn' in col_lower or 'desc' in col_lower:
                        mapping[col] = cls.STANDARD_YARN_COLUMN
        
        # Map yarn attribute columns
        for attr_type, attr_names in cls.YARN_ATTRIBUTE_COLUMNS.items():
            for col in df_columns:
                for attr_name in attr_names:
                    if col.lower() == attr_name.lower():
                        # Standardize attribute column names
                        standard_name = attr_type.replace('_', ' ').title().replace(' ', '_')
                        if col != standard_name:
                            mapping[col] = standard_name
                        break
        
        return mapping
    
    @classmethod
    def normalize_yarn_value(cls, value: any, source_file: str = None) -> str:
        """Normalize a yarn value to consistent format"""
        if pd.isna(value) or value is None:
            return ""
        
        # Convert to string and clean
        value = str(value).strip()
        
        # Remove leading zeros if it's a numeric yarn code
        if value.isdigit():
            value = str(int(value))
        
        # Standardize common prefixes
        value = value.replace("YARN-", "").replace("yarn-", "")
        value = value.replace("Y-", "").replace("y-", "")
        
        # Handle special cases based on source
        if source_file:
            if "demand" in source_file.lower():
                # Demand reports might have additional formatting
                value = value.upper()
        
        return value
    
    @classmethod
    def validate_yarn_id(cls, yarn_id: str) -> bool:
        """Validate if a yarn ID follows expected format"""
        if not yarn_id or pd.isna(yarn_id):
            return False
        
        # Yarn IDs should not be empty after normalization
        normalized = cls.normalize_yarn_value(yarn_id)
        return len(normalized) > 0
    
    @classmethod
    def get_mapping_report(cls) -> str:
        """Generate a human-readable yarn mapping report"""
        report = "YARN MAPPING CONFIGURATION\n"
        report += "="*50 + "\n\n"
        
        report += f"Standard Yarn Column: {cls.STANDARD_YARN_COLUMN}\n\n"
        
        for source, config in cls.YARN_MAPPINGS.items():
            report += f"Source: {source}\n"
            report += f"  Yarn Column: {config['yarn_column']}\n"
            if "maps_to" in config:
                report += f"  Maps To: {config['maps_to']}\n"
            report += f"  Alternates: {', '.join(config['alternate_names'])}\n"
            report += f"  Description: {config['description']}\n"
            report += "\n"
        
        report += "\nYARN ATTRIBUTE COLUMNS:\n"
        report += "-"*30 + "\n"
        for attr_type, names in cls.YARN_ATTRIBUTE_COLUMNS.items():
            report += f"  {attr_type}: {', '.join(names)}\n"
        
        return report


# Integration with data parser
def apply_yarn_mapping(df, filename: str) -> tuple:
    """Apply yarn mapping to a dataframe based on filename"""
    corrections = []
    
    # Get mapping for this file
    mapping = YarnMappingConfig.get_yarn_mapping(filename, df.columns.tolist())
    
    if mapping:
        # Apply column renaming
        for old_col, new_col in mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
                corrections.append(f"Mapped yarn column '{old_col}' to standard '{new_col}'")
                
                # Normalize values in the yarn column
                if new_col == YarnMappingConfig.STANDARD_YARN_COLUMN:
                    df[new_col] = df[new_col].apply(
                        lambda x: YarnMappingConfig.normalize_yarn_value(x, filename)
                    )
                    corrections.append(f"Normalized yarn values in '{new_col}'")
                    
                    # Validate yarn IDs
                    invalid_count = (~df[new_col].apply(YarnMappingConfig.validate_yarn_id)).sum()
                    if invalid_count > 0:
                        corrections.append(f"Warning: Found {invalid_count} invalid yarn IDs")
    
    # Check for duplicate yarn columns after mapping
    yarn_cols = [col for col in df.columns if 'desc#' in col.lower() or 'yarn' in col.lower()]
    if len(yarn_cols) > 1:
        corrections.append(f"Note: Multiple yarn-related columns found: {yarn_cols}")
    
    return df, corrections


if __name__ == "__main__":
    # Test the yarn mapping configuration
    print(YarnMappingConfig.get_mapping_report())
    
    # Test specific file mappings
    test_files = [
        "BOM_updated.csv",
        "Yarn_ID.xlsx",
        "yarn_inventory (4).xlsx",
        "Expected_Yarn_Report.xlsx",
        "Yarn_Demand_Report.csv",
        "Yarn_Demand_By_Style.xlsx"
    ]
    
    print("\n\nTEST FILE MAPPINGS:")
    print("-"*50)
    for filename in test_files:
        config = YarnMappingConfig.get_yarn_column_for_file(filename)
        if config:
            print(f"\n{filename}:")
            print(f"  Yarn Column: {config['yarn_column']}")
            if "maps_to" in config:
                print(f"  Maps to: {config['maps_to']}")
            else:
                print(f"  Already standard: Desc#")