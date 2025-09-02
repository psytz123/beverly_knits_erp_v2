"""
Style Mapping Module for Beverly Knits ERP
Maps between fStyle# (sales) and Style# (BOM) using eFab_Styles mapping file
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)

class StyleMapper:
    """Maps between different style naming conventions across the ERP system"""
    
    def __init__(self, mapping_file_path: Optional[str] = None):
        """Initialize the style mapper with optional mapping file"""
        self.mapping_df = None
        self.fstyle_to_gbase = {}
        self.gbase_to_bom_styles = {}
        self.direct_mappings = {}
        
        if mapping_file_path:
            self.load_mapping_file(mapping_file_path)
    
    def load_mapping_file(self, file_path: str) -> bool:
        """Load the eFab_Styles mapping file"""
        try:
            if Path(file_path).exists():
                self.mapping_df = pd.read_excel(file_path)
                self._build_mappings()
                logger.info(f"Loaded style mappings from {file_path}")
                return True
            else:
                logger.warning(f"Mapping file not found: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading mapping file: {e}")
            return False
    
    def _build_mappings(self):
        """Build internal mapping dictionaries for fast lookup"""
        if self.mapping_df is None:
            return
        
        # Build fStyle to gBase mapping
        for _, row in self.mapping_df.iterrows():
            fstyle = str(row.get('fStyle', ''))
            gbase = str(row.get('gBase', ''))
            
            if fstyle and gbase:
                self.fstyle_to_gbase[fstyle] = gbase
                # Also store direct mapping if style column exists
                if 'style' in row:
                    self.direct_mappings[fstyle] = str(row['style'])
    
    def set_bom_styles(self, bom_styles: Set[str]):
        """Set the available BOM styles for matching"""
        self.gbase_to_bom_styles.clear()
        
        for bom_style in bom_styles:
            if isinstance(bom_style, str):
                # Extract base from BOM style (e.g., "C1B4014/1A" -> "C1B4014")
                base = bom_style.split('/')[0] if '/' in bom_style else bom_style
                base = base.split('-')[0] if '-' in base and '/' not in base else base
                
                if base not in self.gbase_to_bom_styles:
                    self.gbase_to_bom_styles[base] = []
                self.gbase_to_bom_styles[base].append(bom_style)
    
    def map_sales_to_bom(self, sales_style: str) -> List[str]:
        """
        Map a sales style (fStyle#) to potential BOM styles
        Returns a list of matching BOM styles
        """
        if not sales_style:
            return []
        
        sales_style = str(sales_style).strip()
        
        # First, get the gBase from the mapping
        gbase = self.fstyle_to_gbase.get(sales_style)
        
        if not gbase:
            # Try case-insensitive match
            for fstyle, base in self.fstyle_to_gbase.items():
                if fstyle.lower() == sales_style.lower():
                    gbase = base
                    break
        
        if not gbase:
            return []
        
        # Now find all BOM styles that match this gBase
        matching_bom_styles = []
        
        # Direct match
        if gbase in self.gbase_to_bom_styles:
            matching_bom_styles.extend(self.gbase_to_bom_styles[gbase])
        
        # Case-insensitive match
        gbase_lower = gbase.lower()
        for base, styles in self.gbase_to_bom_styles.items():
            if base.lower() == gbase_lower and base != gbase:
                matching_bom_styles.extend(styles)
        
        return list(set(matching_bom_styles))  # Remove duplicates
    
    def map_bom_to_sales(self, bom_style: str) -> List[str]:
        """
        Map a BOM style to potential sales styles (fStyle#)
        Returns a list of matching sales styles
        """
        if not bom_style:
            return []
        
        bom_style = str(bom_style).strip()
        
        # Extract base from BOM style
        base = bom_style.split('/')[0] if '/' in bom_style else bom_style
        
        # Find all fStyles that map to this base
        matching_sales_styles = []
        for fstyle, gbase in self.fstyle_to_gbase.items():
            if gbase == base or gbase.lower() == base.lower():
                matching_sales_styles.append(fstyle)
        
        return matching_sales_styles
    
    def get_all_mappings(self) -> Dict[str, List[str]]:
        """Get all sales to BOM mappings"""
        mappings = {}
        for fstyle in self.fstyle_to_gbase.keys():
            bom_styles = self.map_sales_to_bom(fstyle)
            if bom_styles:
                mappings[fstyle] = bom_styles
        return mappings
    
    def get_mapping_stats(self) -> Dict:
        """Get statistics about the loaded mappings"""
        return {
            'total_fstyles': len(self.fstyle_to_gbase),
            'unique_gbases': len(set(self.fstyle_to_gbase.values())),
            'bom_bases_mapped': len(self.gbase_to_bom_styles),
            'mapping_file_loaded': self.mapping_df is not None
        }


# Global instance
_style_mapper = None

def get_style_mapper(mapping_file_path: Optional[str] = None) -> StyleMapper:
    """Get or create the global style mapper instance"""
    global _style_mapper
    
    if _style_mapper is None:
        # Default path
        if mapping_file_path is None:
            mapping_file_path = "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/eFab_Styles_20250902.xlsx"
        
        _style_mapper = StyleMapper(mapping_file_path)
    
    return _style_mapper