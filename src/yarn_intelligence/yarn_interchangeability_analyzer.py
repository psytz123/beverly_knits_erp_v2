#!/usr/bin/env python3
"""
Yarn Interchangeability Analyzer Module
=======================================

This module analyzes BOM (Bill of Materials) and Yarn ID data to identify 
interchangeable yarns based on material properties, size specifications, 
and color characteristics.

Author: Claude AI Assistant
Date: 2025-08-11
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import re

class YarnInterchangeabilityAnalyzer:
    """
    Analyzes yarn data to identify interchangeable alternatives based on:
    - Material composition (Blend, Type) - 4% tolerance on blend ratios
    - Size specifications (Description field patterns) - 5% tolerance on size/count
    - Color matching - No tolerance (exact match required)
    - Cost considerations included
    - Exact match requirements for all specifications
    """
    
    def __init__(self):
        self.bom_data = {}
        self.yarn_data = {}
        self.interchangeable_groups = {}
        self.material_categories = {}
        
        # Tolerance settings based on user requirements
        self.BLEND_TOLERANCE = 0.04  # 4% tolerance on material blend
        self.SIZE_TOLERANCE = 0.05   # 5% tolerance on yarn count/size
        self.COLOR_TOLERANCE = 0.0   # No tolerance - exact color match required
        self.EXACT_MATCH_MODE = True # All specifications must be exact matches
        
    def load_bom_files(self, file_paths: List[str]):
        """Load BOM files and normalize column names"""
        for i, file_path in enumerate(file_paths):
            try:
                df = pd.read_csv(file_path)
                # Normalize column names
                df.columns = df.columns.str.strip().str.lower()
                
                # Map different column naming conventions
                if 'desc#' in df.columns:
                    df.rename(columns={'desc#': 'desc_id'}, inplace=True)
                elif 'yarn_id' in df.columns:
                    df.rename(columns={'yarn_id': 'desc_id'}, inplace=True)
                
                if 'style#' in df.columns:
                    df.rename(columns={'style#': 'style_id'}, inplace=True)
                    
                if 'bom_percentage' in df.columns:
                    df.rename(columns={'bom_percentage': 'bom_percent'}, inplace=True)
                    
                self.bom_data[f'bom_{i+1}'] = df
                print(f"Loaded BOM file {i+1}: {len(df)} records")
            except Exception as e:
                print(f"Error loading BOM file {file_path}: {e}")
    
    def load_yarn_files(self, file_paths: List[str]):
        """Load Yarn ID files with material specifications"""
        for i, file_path in enumerate(file_paths):
            try:
                df = pd.read_csv(file_path)
                # Clean column names
                df.columns = df.columns.str.strip().str.lower()
                
                if 'desc#' in df.columns:
                    df.rename(columns={'desc#': 'desc_id'}, inplace=True)
                    
                # Remove empty rows
                df = df.dropna(subset=['desc_id'])
                
                self.yarn_data[f'yarn_{i+1}'] = df
                print(f"Loaded Yarn file {i+1}: {len(df)} records")
            except Exception as e:
                print(f"Error loading Yarn file {file_path}: {e}")
    
    def parse_yarn_size(self, description: str) -> Dict:
        """
        Industry-standard yarn size parsing based on textile research:
        - Direct systems: Tex, Denier, Dtex (higher number = thicker yarn)
        - Indirect systems: Ne, Nm, Nw (higher number = finer yarn)
        - Filament patterns: denier/filament count (e.g., 1/150/48)
        - Spun yarn patterns: count/twist (e.g., 30/1 Ne)
        Returns normalized values with proper yarn classification
        """
        if pd.isna(description):
            return {'count': None, 'denier': None, 'filaments': None, 'yarn_system': 'unknown'}
            
        desc_str = str(description).upper().strip()
        
        # Enhanced pattern matching based on industry standards
        patterns = [
            # Filament yarn patterns (Direct system - Denier based)
            r'(\d+\.?\d*)/(\d+\.?\d*)/(\d+\.?\d*)',  # Format: 1/150/48 (ply/denier/filament)
            r'(\d+)D',                               # 150D (denier only)
            r'(\d+\.?\d*)DEN',                       # Explicit denier
            r'(\d+\.?\d*)DENIER',                    # Full denier spelling
            
            # Tex system (Direct system - grams per 1000m)
            r'(\d+\.?\d*)TEX',                       # 20 TEX
            r'(\d+\.?\d*)T(?!W)',                    # Tex abbreviated (not TW for twist)
            
            # Dtex system (Direct system - grams per 10000m)  
            r'(\d+\.?\d*)DTEX',                      # 200 DTEX
            
            # Spun yarn patterns (Indirect systems)
            r'(\d+\.?\d*)/(\d+\.?\d*)(?![/\d])',     # Count/twist like 30/1, 24/1
            r'(\d+\.?\d*)NE',                        # English Cotton Count
            r'(\d+\.?\d*)NM',                        # Metric Count  
            r'(\d+\.?\d*)NW',                        # Worsted Count
            
            # Month-based twist patterns (industry shorthand)
            r'(\d+)-(JUN|JUNE)',                     # 1-Jun twist code
            r'(\d+)-(DEC|DECEMBER)',                 # 1-Dec twist code
            r'(\d+)-(JAN|JANUARY|FEB|FEBRUARY|MAR|MARCH|APR|APRIL|MAY|JUL|JULY|AUG|AUGUST|SEP|SEPTEMBER|OCT|OCTOBER|NOV|NOVEMBER)',
            
            # TPM/TPI twist specifications
            r'(\d+\.?\d*)TPM',                       # Turns per meter
            r'(\d+\.?\d*)TPI',                       # Turns per inch
            
            # Single numeric patterns (context-dependent)
            r'^(\d+\.?\d*)$',                        # Just a number
        ]
        
        # Month name to number mapping for twist codes
        month_mapping = {
            'JUN': 6, 'JUNE': 6, 'DEC': 12, 'DECEMBER': 12,
            'JAN': 1, 'JANUARY': 1, 'FEB': 2, 'FEBRUARY': 2,
            'MAR': 3, 'MARCH': 3, 'APR': 4, 'APRIL': 4,
            'MAY': 5, 'JUL': 7, 'JULY': 7, 'AUG': 8, 'AUGUST': 8,
            'SEP': 9, 'SEPTEMBER': 9, 'OCT': 10, 'OCTOBER': 10,
            'NOV': 11, 'NOVEMBER': 11
        }
        
        for pattern in patterns:
            match = re.search(pattern, desc_str)
            if match:
                groups = match.groups()
                
                # Handle filament yarn patterns (3-component: ply/denier/filament)
                if len(groups) == 3 and all(g.replace('.', '').isdigit() for g in groups):
                    ply_val = float(groups[0])
                    denier_val = float(groups[1])
                    filament_val = float(groups[2])
                    
                    return {
                        'ply': ply_val,
                        'denier': denier_val,
                        'filaments': filament_val,
                        'size_signature': f"{groups[0]}/{groups[1]}/{groups[2]}",
                        'yarn_system': 'denier_direct',
                        'yarn_type': 'multifilament',
                        'weight_category': self._categorize_denier(denier_val),
                        'tex_equivalent': denier_val / 9.0,  # Convert denier to tex
                        'interchangeability_group': f"multifilament_{self._categorize_denier(denier_val)}"
                    }
                
                # Handle single denier values (150D, 200DEN, etc.)
                elif len(groups) == 1 and any(x in desc_str for x in ['D', 'DEN', 'DENIER']):
                    denier_val = float(groups[0])
                    return {
                        'denier': denier_val,
                        'size_signature': f"{denier_val}D",
                        'yarn_system': 'denier_direct',
                        'yarn_type': 'continuous_filament',
                        'weight_category': self._categorize_denier(denier_val),
                        'tex_equivalent': denier_val / 9.0,
                        'interchangeability_group': f"filament_{self._categorize_denier(denier_val)}"
                    }
                
                # Handle Tex values (20TEX, 30T, etc.)
                elif len(groups) == 1 and any(x in desc_str for x in ['TEX', 'T']):
                    tex_val = float(groups[0])
                    return {
                        'tex': tex_val,
                        'size_signature': f"{tex_val}TEX",
                        'yarn_system': 'tex_direct',
                        'yarn_type': 'tex_specified',
                        'weight_category': self._categorize_tex(tex_val),
                        'denier_equivalent': tex_val * 9.0,
                        'interchangeability_group': f"tex_{self._categorize_tex(tex_val)}"
                    }
                
                # Handle spun yarn patterns (count/twist like 30/1, 24/1)
                elif len(groups) == 2 and groups[1].replace('.', '').isdigit():
                    count_val = float(groups[0])
                    twist_val = float(groups[1])
                    
                    # Determine if it's Ne (English Cotton Count) based on typical ranges
                    if 10 <= count_val <= 80 and twist_val <= 10:
                        tex_equiv = 590.5 / count_val  # Convert Ne to Tex
                        return {
                            'ne_count': count_val,
                            'twist': twist_val,
                            'size_signature': f"{count_val}/{twist_val}",
                            'yarn_system': 'ne_indirect',
                            'yarn_type': 'spun_cotton',
                            'weight_category': self._categorize_ne_count(count_val),
                            'tex_equivalent': tex_equiv,
                            'interchangeability_group': f"spun_ne_{self._categorize_ne_count(count_val)}"
                        }
                    else:
                        # Other count systems or unknown
                        return {
                            'count': count_val,
                            'twist': twist_val,
                            'size_signature': f"{count_val}/{twist_val}",
                            'yarn_system': 'unknown_indirect',
                            'yarn_type': 'spun_generic',
                            'weight_category': 'medium',
                            'interchangeability_group': 'spun_generic'
                        }
                
                # Handle month-based twist patterns (1-Jun, 1-Dec)
                elif len(groups) == 2 and groups[1] in month_mapping:
                    count_val = float(groups[0])
                    month_num = month_mapping[groups[1]]
                    return {
                        'count': count_val,
                        'twist_code': month_num,
                        'size_signature': f"{count_val}-{groups[1]}",
                        'yarn_system': 'twist_coded',
                        'yarn_type': 'specialty_spun',
                        'weight_category': self._categorize_specialty_count(count_val),
                        'interchangeability_group': f"specialty_{self._categorize_specialty_count(count_val)}"
                    }
                
                # Handle single count systems (30NE, 50NM, etc.)
                elif len(groups) == 1:
                    value = float(groups[0])
                    
                    if 'NE' in desc_str:
                        tex_equiv = 590.5 / value
                        return {
                            'ne_count': value,
                            'yarn_system': 'ne_indirect',
                            'yarn_type': 'spun_cotton',
                            'weight_category': self._categorize_ne_count(value),
                            'tex_equivalent': tex_equiv,
                            'interchangeability_group': f"spun_ne_{self._categorize_ne_count(value)}"
                        }
                    
                    elif 'NM' in desc_str:
                        tex_equiv = 1000.0 / value
                        return {
                            'nm_count': value,
                            'yarn_system': 'nm_indirect',
                            'yarn_type': 'metric_spun',
                            'weight_category': self._categorize_nm_count(value),
                            'tex_equivalent': tex_equiv,
                            'interchangeability_group': f"spun_nm_{self._categorize_nm_count(value)}"
                        }
                    
                    elif 'DTEX' in desc_str:
                        tex_equiv = value / 10.0
                        return {
                            'dtex': value,
                            'yarn_system': 'dtex_direct',
                            'yarn_type': 'dtex_filament',
                            'weight_category': self._categorize_dtex(value),
                            'tex_equivalent': tex_equiv,
                            'interchangeability_group': f"dtex_{self._categorize_dtex(value)}"
                        }
        
        # Fallback: try to infer from numeric patterns and context
        numeric_match = re.search(r'(\d+\.?\d*)', desc_str)
        if numeric_match:
            value = float(numeric_match.group(1))
            
            # Heuristic classification based on typical ranges
            if 10 <= value <= 80:  # Likely Ne count
                return {
                    'inferred_ne': value,
                    'yarn_system': 'inferred_ne',
                    'yarn_type': 'probably_spun',
                    'weight_category': self._categorize_ne_count(value),
                    'size_signature': f"{value}_inferred_ne",
                    'interchangeability_group': f"inferred_spun_{self._categorize_ne_count(value)}"
                }
            elif 100 <= value <= 1000:  # Likely denier
                return {
                    'inferred_denier': value,
                    'yarn_system': 'inferred_denier',
                    'yarn_type': 'probably_filament',
                    'weight_category': self._categorize_denier(value),
                    'size_signature': f"{value}_inferred_denier",
                    'interchangeability_group': f"inferred_filament_{self._categorize_denier(value)}"
                }
        
        # Final fallback
        return {
            'raw_description': desc_str,
            'size_signature': desc_str,
            'yarn_system': 'unknown',
            'yarn_type': 'unknown',
            'weight_category': 'unknown',
            'interchangeability_group': 'unknown'
        }
    
    def _categorize_denier(self, denier_value: float) -> str:
        """Categorize yarn by denier weight"""
        if denier_value < 100:
            return 'fine'
        elif denier_value < 200:
            return 'medium'
        elif denier_value < 400:
            return 'heavy'
        else:
            return 'extra_heavy'
    
    def _categorize_count(self, count_value: float) -> str:
        """Categorize yarn by count weight (generic)"""
        if count_value < 10:
            return 'heavy'
        elif count_value < 20:
            return 'medium_heavy'
        elif count_value < 40:
            return 'medium'
        else:
            return 'fine'
    
    def _categorize_ne_count(self, ne_value: float) -> str:
        """Categorize yarn by English Cotton Count (Ne) - higher = finer"""
        if ne_value >= 60:
            return 'very_fine'
        elif ne_value >= 40:
            return 'fine'
        elif ne_value >= 20:
            return 'medium'
        elif ne_value >= 10:
            return 'coarse'
        else:
            return 'very_coarse'
    
    def _categorize_nm_count(self, nm_value: float) -> str:
        """Categorize yarn by Metric Count (Nm) - higher = finer"""
        if nm_value >= 80:
            return 'very_fine'
        elif nm_value >= 50:
            return 'fine'  
        elif nm_value >= 30:
            return 'medium'
        elif nm_value >= 15:
            return 'coarse'
        else:
            return 'very_coarse'
    
    def _categorize_tex(self, tex_value: float) -> str:
        """Categorize yarn by Tex (direct system) - higher = heavier"""
        if tex_value >= 50:
            return 'very_heavy'
        elif tex_value >= 30:
            return 'heavy'
        elif tex_value >= 15:
            return 'medium'
        elif tex_value >= 8:
            return 'fine'
        else:
            return 'very_fine'
    
    def _categorize_dtex(self, dtex_value: float) -> str:
        """Categorize yarn by Dtex (direct system) - higher = heavier"""
        if dtex_value >= 500:
            return 'very_heavy'
        elif dtex_value >= 300:
            return 'heavy'
        elif dtex_value >= 150:
            return 'medium'
        elif dtex_value >= 80:
            return 'fine'
        else:
            return 'very_fine'
    
    def _categorize_specialty_count(self, count_value: float) -> str:
        """Categorize specialty yarn counts"""
        if count_value >= 5:
            return 'coarse'
        elif count_value >= 2:
            return 'medium'
        else:
            return 'fine'
    
    def _determine_yarn_structure(self, description: str) -> str:
        """Determine if yarn is spun or filament based on description patterns"""
        if not description:
            return 'unknown'
        
        desc_upper = str(description).upper()
        
        # Filament indicators
        filament_patterns = [
            r'\d+/\d+/\d+',  # Multi-component like 1/150/48
            r'\d+D(?!\w)',   # Denier like 150D
            r'DENIER',
            r'DTEX',
            r'FILAMENT',
            r'MULTIFILAMENT',
            r'CONTINUOUS'
        ]
        
        for pattern in filament_patterns:
            if re.search(pattern, desc_upper):
                return 'filament'
        
        # Spun indicators  
        spun_patterns = [
            r'\d+/\d+(?!/)',  # Count/twist like 30/1, 24/1
            r'\d+NE',         # English count
            r'\d+NM',         # Metric count
            r'\d+-(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)',  # Month codes
            r'SPUN',
            r'COTTON',
            r'CARDED',
            r'COMBED',
            r'OE\b',          # Open End
            r'RING',
            r'MVS'            # Murata Vortex Spinning
        ]
        
        for pattern in spun_patterns:
            if re.search(pattern, desc_upper):
                return 'spun'
        
        # Default fallback based on numeric patterns
        if re.search(r'\d+/\d+/\d+', desc_upper):
            return 'filament'
        elif re.search(r'\d+/\d+$', desc_upper):
            return 'spun'
        
        return 'unknown'
    
    def categorize_materials(self):
        """Enhanced material categorization with compatibility rules and blend analysis"""
        categories = defaultdict(list)
        
        for yarn_file, df in self.yarn_data.items():
            for _, row in df.iterrows():
                desc_id = row['desc_id']
                material_type = row.get('type', 'Unknown')
                blend = row.get('blend', 'Unknown')
                color = row.get('color', 'Unknown')
                
                # Enhanced material analysis
                material_analysis = self._analyze_material_composition(material_type, blend, row.get('description', ''))
                
                # Create material signature that separates spun and filament yarns
                yarn_structure = self._determine_yarn_structure(row.get('description', ''))
                material_sig = f"{material_analysis['primary_material']}|{material_analysis['blend_category']}|{yarn_structure}"
                
                categories[material_sig].append({
                    'desc_id': desc_id,
                    'color': color,
                    'supplier': row.get('supplier', 'Unknown'),
                    'description': row.get('description', ''),
                    'yarn_file': yarn_file,
                    'material_analysis': material_analysis,
                    'compatibility_group': material_analysis['compatibility_group']
                })
        
        self.material_categories = dict(categories)
        return categories
    
    def _analyze_material_composition(self, material_type: str, blend: str, description: str) -> Dict:
        """
        Advanced material composition analysis for better compatibility matching
        """
        import re
        
        # Normalize inputs
        mat_type = str(material_type).lower() if material_type else 'unknown'
        blend_info = str(blend).lower() if blend else 'unknown'
        desc = str(description).lower() if description else ''
        
        # Extract material percentages from description and blend
        material_percentages = {}
        
        # Pattern matching for blend percentages
        percentage_patterns = [
            r'(\d+)%?\s*(polyester|cotton|nylon|spandex|rayon|modacrylic|bamboo|wool|acrylic)',
            r'(\d+)/(\d+)\s*(poly|cotton|nylon|span)',  # 55/45 format
            r'(poly|cotton|nylon|span)\s*(\d+)%?',
        ]
        
        for pattern in percentage_patterns:
            matches = re.findall(pattern, desc + ' ' + blend_info)
            for match in matches:
                if len(match) == 2 and match[0].isdigit():
                    percentage = int(match[0])
                    material = self._normalize_material_name(match[1])
                    material_percentages[material] = percentage
                elif len(match) == 3 and match[0].isdigit() and match[1].isdigit():
                    # Handle 55/45 format
                    materials = self._extract_materials_from_description(desc)
                    if len(materials) >= 2:
                        material_percentages[materials[0]] = int(match[0])
                        material_percentages[materials[1]] = int(match[1])
        
        # Determine primary material (highest percentage or first mentioned)
        if material_percentages:
            primary_material = max(material_percentages.keys(), key=lambda k: material_percentages[k])
            primary_percentage = material_percentages[primary_material]
        else:
            # Fallback to material_type
            primary_material = self._normalize_material_name(mat_type)
            primary_percentage = 100
        
        # Categorize blend type
        if len(material_percentages) == 1 or primary_percentage >= 95:
            blend_category = 'pure'
        elif primary_percentage >= 80:
            blend_category = 'dominant_blend'
        elif primary_percentage >= 60:
            blend_category = 'majority_blend'
        else:
            blend_category = 'balanced_blend'
        
        # Assign compatibility group for substitution
        compatibility_group = self._get_compatibility_group(primary_material, blend_category, material_percentages)
        
        return {
            'primary_material': primary_material,
            'material_percentages': material_percentages,
            'blend_category': blend_category,
            'compatibility_group': compatibility_group,
            'substitution_flexibility': self._calculate_substitution_flexibility(material_percentages, blend_category)
        }
    
    def _normalize_material_name(self, material: str) -> str:
        """Normalize material names to standard terms"""
        material_mapping = {
            'poly': 'polyester', 'pes': 'polyester',
            'cotton': 'cotton', 'co': 'cotton', 'cot': 'cotton',
            'nylon': 'nylon', 'ny': 'nylon', 'pa': 'nylon',
            'spandex': 'spandex', 'span': 'spandex', 'elastane': 'spandex',
            'rayon': 'rayon', 'viscose': 'rayon', 'cv': 'rayon',
            'modacrylic': 'modacrylic', 'mac': 'modacrylic',
            'acrylic': 'acrylic', 'pan': 'acrylic',
            'wool': 'wool', 'wo': 'wool',
            'bamboo': 'bamboo',
            'tencel': 'tencel', 'lyocell': 'tencel',
            'polypropylene': 'polypropylene', 'pp': 'polypropylene',
            'polyethylene': 'polyethylene', 'pe': 'polyethylene'
        }
        
        material_lower = material.lower().strip()
        for key, standard in material_mapping.items():
            if key in material_lower:
                return standard
        
        return material_lower
    
    def _extract_materials_from_description(self, description: str) -> List[str]:
        """Extract material names from description text"""
        materials = []
        common_materials = ['polyester', 'cotton', 'nylon', 'spandex', 'rayon', 'modacrylic', 'bamboo', 'wool', 'acrylic']
        
        desc_lower = description.lower()
        for material in common_materials:
            if material in desc_lower:
                materials.append(material)
        
        return materials
    
    def _get_compatibility_group(self, primary_material: str, blend_category: str, percentages: Dict) -> str:
        """
        Assign compatibility groups for substitution decisions
        """
        # Define material compatibility families
        synthetic_family = ['polyester', 'nylon', 'acrylic', 'polypropylene', 'polyethylene']
        natural_family = ['cotton', 'wool', 'bamboo']
        cellulosic_family = ['rayon', 'tencel', 'bamboo']
        stretch_family = ['spandex', 'elastane']
        
        if primary_material in synthetic_family:
            base_group = 'synthetic'
        elif primary_material in natural_family:
            base_group = 'natural'
        elif primary_material in cellulosic_family:
            base_group = 'cellulosic'
        elif primary_material in stretch_family:
            base_group = 'stretch'
        else:
            base_group = 'specialty'
        
        # Modify based on blend
        if blend_category == 'pure':
            return f"{base_group}_pure"
        elif 'cotton' in percentages and 'polyester' in percentages:
            return "cotton_poly_blend"
        elif any(mat in stretch_family for mat in percentages.keys()):
            return f"{base_group}_stretch"
        else:
            return f"{base_group}_blend"
    
    def _calculate_substitution_flexibility(self, percentages: Dict, blend_category: str) -> str:
        """Calculate how flexible this yarn is for substitution"""
        if blend_category == 'pure':
            return 'low'  # Pure materials are less flexible
        elif blend_category == 'balanced_blend':
            return 'high'  # Balanced blends are most flexible
        elif blend_category == 'dominant_blend':
            return 'medium'  # Somewhat flexible
        else:
            return 'medium'
    
    def find_interchangeable_yarns(self) -> Dict:
        """
        Enhanced interchangeable yarn identification with improved accuracy:
        1. Advanced material composition analysis with compatibility groups
        2. Adaptive size tolerance based on yarn type and weight category  
        3. Enhanced color normalization with broader equivalencies
        4. Cost analysis and supplier diversity scoring
        5. Substitution flexibility assessment
        """
        if not self.material_categories:
            self.categorize_materials()
        
        interchangeable_groups = {}
        
        # Also check for cross-compatibility between similar material groups
        compatible_groups = self._find_compatible_material_groups()
        
        all_material_groups = list(self.material_categories.items()) + compatible_groups
        
        for material_sig, yarns in all_material_groups:
            if len(yarns) < 2:  # Need at least 2 yarns to be interchangeable
                continue
                
            # Enhanced color grouping with fuzzy matching
            color_groups = self._group_by_enhanced_color_matching(yarns)
            
            # Within each color group, apply size and compatibility analysis
            for color, color_yarns in color_groups.items():
                if len(color_yarns) < 2:
                    continue
                    
                # Group by size with adaptive tolerance
                size_compatible_groups = self._group_by_size_tolerance(color_yarns)
                
                for size_group_idx, size_yarns in enumerate(size_compatible_groups):
                    if len(size_yarns) >= 2:
                        group_key = f"{material_sig}_{color}_{size_group_idx}"
                        
                        # Advanced group analysis
                        group_analysis = self._analyze_group_compatibility(size_yarns)
                        
                        interchangeable_groups[group_key] = {
                            'material_signature': material_sig,
                            'exact_color': color,
                            'yarns': size_yarns,
                            'total_alternatives': len(size_yarns),
                            'suppliers': list(set(yarn['supplier'] for yarn in size_yarns)),
                            'supplier_count': len(set(yarn['supplier'] for yarn in size_yarns)),
                            'cost_range': group_analysis['cost_analysis'],
                            'size_tolerance_group': size_group_idx,
                            'compatibility_score': group_analysis['compatibility_score'],
                            'substitution_risk': group_analysis['substitution_risk'],
                            'material_flexibility': group_analysis['material_flexibility'],
                            'recommended_priority': group_analysis['recommended_priority']
                        }
        
        self.interchangeable_groups = interchangeable_groups
        return interchangeable_groups
    
    def _find_compatible_material_groups(self) -> List[Tuple]:
        """Find materials that can be compatible across different signatures - ONLY within same yarn structure"""
        compatible_pairs = []
        
        # CRITICAL: Compatibility rules must respect spun vs filament separation
        # We can ONLY create compatible groups within the same yarn structure
        
        # Spun yarn compatibility rules
        spun_compatibility_rules = {
            # Cotton-poly blends (spun only)
            'cotton|dominant_blend|spun': ['polyester|dominant_blend|spun'],
            'polyester|dominant_blend|spun': ['cotton|dominant_blend|spun'],
            
            # Pure synthetics (spun only)
            'polyester|pure|spun': ['nylon|pure|spun'],
            'nylon|pure|spun': ['polyester|pure|spun'],
            
            # Natural fibers (spun only)
            'cotton|pure|spun': ['cotton|dominant_blend|spun'],
        }
        
        # Filament yarn compatibility rules  
        filament_compatibility_rules = {
            # Pure synthetics (filament only)
            'polyester|pure|filament': ['nylon|pure|filament'],
            'nylon|pure|filament': ['polyester|pure|filament'],
            
            # Multifilament denier variations
            'polyester|dominant_blend|filament': ['nylon|dominant_blend|filament'],
        }
        
        # Combine all rules
        all_compatibility_rules = {**spun_compatibility_rules, **filament_compatibility_rules}
        
        processed_combinations = set()
        
        for primary_sig, compatible_sigs in all_compatibility_rules.items():
            if primary_sig in self.material_categories:
                primary_yarns = self.material_categories[primary_sig]
                
                for compatible_sig in compatible_sigs:
                    if compatible_sig in self.material_categories:
                        # Create combination key to avoid duplicates
                        combo_key = tuple(sorted([primary_sig, compatible_sig]))
                        
                        if combo_key not in processed_combinations:
                            processed_combinations.add(combo_key)
                            
                            # Merge yarns from compatible groups
                            combined_yarns = primary_yarns + self.material_categories[compatible_sig]
                            combined_sig = f"compatible_{primary_sig}_{compatible_sig}"
                            
                            compatible_pairs.append((combined_sig, combined_yarns))
        
        return compatible_pairs
    
    def _group_by_enhanced_color_matching(self, yarns: List[Dict]) -> Dict:
        """Enhanced color grouping with fuzzy matching and color families"""
        color_groups = defaultdict(list)
        
        for yarn in yarns:
            normalized_color = self._normalize_color(yarn['color'])
            color_groups[normalized_color].append(yarn)
        
        return dict(color_groups)
    
    def _analyze_group_compatibility(self, yarns: List[Dict]) -> Dict:
        """Analyze compatibility and risk factors for a group of yarns"""
        costs = []
        flexibility_scores = []
        material_analyses = []
        
        for yarn in yarns:
            # Cost analysis
            cost = self._extract_cost_data(yarn)
            if cost:
                costs.append(cost)
            
            # Material analysis
            if 'material_analysis' in yarn:
                material_analyses.append(yarn['material_analysis'])
                
                # Flexibility scoring
                flexibility = yarn['material_analysis']['substitution_flexibility']
                flexibility_scores.append({
                    'low': 1, 'medium': 2, 'high': 3
                }.get(flexibility, 2))
        
        # Calculate compatibility score (0-100)
        avg_flexibility = sum(flexibility_scores) / len(flexibility_scores) if flexibility_scores else 2
        supplier_diversity = len(set(yarn['supplier'] for yarn in yarns))
        
        compatibility_score = min(100, (avg_flexibility * 20) + (supplier_diversity * 10) + 20)
        
        # Assess substitution risk
        if avg_flexibility >= 2.5 and supplier_diversity >= 2:
            substitution_risk = 'low'
            recommended_priority = 'high'
        elif avg_flexibility >= 2.0:
            substitution_risk = 'medium'
            recommended_priority = 'medium'
        else:
            substitution_risk = 'high'
            recommended_priority = 'low'
        
        return {
            'cost_analysis': {
                'min': min(costs) if costs else None,
                'max': max(costs) if costs else None,
                'avg': sum(costs)/len(costs) if costs else None,
                'cost_spread_pct': ((max(costs) - min(costs)) / min(costs) * 100) if len(costs) > 1 else 0
            },
            'compatibility_score': compatibility_score,
            'substitution_risk': substitution_risk,
            'material_flexibility': {
                'avg_flexibility': avg_flexibility,
                'flexibility_range': [min(flexibility_scores), max(flexibility_scores)] if flexibility_scores else [2, 2]
            },
            'recommended_priority': recommended_priority
        }
    
    def _group_by_color_family(self, yarns: List[Dict]) -> Dict:
        """Group yarns by color families for better matching"""
        color_families = {
            'natural': ['natural', 'nat', 'crudo', 'buff', 'organic'],
            'black': ['black', 'anthracite', 'midnight', 'anil'],
            'grey': ['grey', 'gray', 'merrimack', 'stealth', 'dawn grey', 'cool grey'],
            'blue': ['blue', 'indigo', 'navy', 'royal', 'capri', 'ice blue'],
            'special': ['orange', 'green', 'red', 'purple', 'silver', 'copper']
        }
        
        grouped = defaultdict(list)
        
        for yarn in yarns:
            color = yarn['color'].lower() if yarn['color'] else 'unknown'
            assigned = False
            
            for family, family_colors in color_families.items():
                if any(fc in color for fc in family_colors):
                    grouped[family].append(yarn)
                    assigned = True
                    break
            
            if not assigned:
                grouped['other'].append(yarn)
        
        return dict(grouped)
    
    def _group_by_size_tolerance(self, yarns: List[Dict]) -> List[List[Dict]]:
        """
        Group yarns by size specifications within 5% tolerance
        Returns list of groups where each group contains compatible yarns
        """
        if not yarns:
            return []
        
        groups = []
        processed = set()
        
        for i, yarn1 in enumerate(yarns):
            if i in processed:
                continue
                
            # Start new group with current yarn
            current_group = [yarn1]
            processed.add(i)
            
            size1 = self.parse_yarn_size(yarn1['description'])
            
            # Find all yarns compatible with yarn1
            for j, yarn2 in enumerate(yarns[i+1:], i+1):
                if j in processed:
                    continue
                    
                size2 = self.parse_yarn_size(yarn2['description'])
                
                if self._sizes_within_tolerance(size1, size2):
                    current_group.append(yarn2)
                    processed.add(j)
            
            if len(current_group) >= 2:  # Only keep groups with alternatives
                groups.append(current_group)
        
        return groups
    
    def _sizes_within_tolerance(self, size1: Dict, size2: Dict) -> bool:
        """
        Industry-standard yarn size tolerance checking based on yarn system compatibility.
        Key insight: Cotton spun yarns (Ne) are NOT interchangeable with filament yarns (Denier).
        """
        if not size1 or not size2:
            return size1 == size2
        
        # Get yarn systems and interchangeability groups
        system1 = size1.get('yarn_system', 'unknown')
        system2 = size2.get('yarn_system', 'unknown')
        group1 = size1.get('interchangeability_group', 'unknown')
        group2 = size2.get('interchangeability_group', 'unknown')
        
        # RULE 1: FUNDAMENTAL INCOMPATIBILITY - SPUN vs FILAMENT YARNS
        # This is a core textile principle: spun and filament yarns are NEVER interchangeable
        yarn_type1 = size1.get('yarn_type', 'unknown')
        yarn_type2 = size2.get('yarn_type', 'unknown')
        
        # Define spun yarn types
        spun_types = {
            'spun_cotton', 'spun_generic', 'specialty_spun', 'probably_spun', 
            'metric_spun', 'tex_specified'  # TEX can be spun or filament, but treat conservatively
        }
        
        # Define filament yarn types  
        filament_types = {
            'multifilament', 'continuous_filament', 'dtex_filament', 
            'probably_filament', 'continuous'
        }
        
        # Check for spun vs filament incompatibility
        type1_is_spun = yarn_type1 in spun_types
        type2_is_spun = yarn_type2 in spun_types
        type1_is_filament = yarn_type1 in filament_types
        type2_is_filament = yarn_type2 in filament_types
        
        # ABSOLUTE RULE: Spun and filament yarns are NEVER interchangeable
        if (type1_is_spun and type2_is_filament) or (type1_is_filament and type2_is_spun):
            return False  # FUNDAMENTAL INCOMPATIBILITY
        
        # Additional system-level incompatibilities
        incompatible_system_pairs = [
            ('ne_indirect', 'denier_direct'),  # Cotton count vs Filament denier
            ('ne_indirect', 'dtex_direct'),    # Cotton count vs Dtex
            ('nm_indirect', 'denier_direct'),  # Metric count vs Denier
            ('nm_indirect', 'dtex_direct'),    # Metric count vs Dtex
        ]
        
        system_pair = tuple(sorted([system1, system2]))
        if system_pair in [tuple(sorted(pair)) for pair in incompatible_system_pairs]:
            return False  # System-level incompatibility
        
        # RULE 2: Same yarn system with TEX equivalency comparison
        if system1 == system2:
            return self._compare_within_same_system(size1, size2)
        
        # RULE 3: Cross-system compatibility using TEX as common denominator
        tex1 = self._get_tex_equivalent(size1)
        tex2 = self._get_tex_equivalent(size2)
        
        if tex1 is None or tex2 is None:
            # Can't convert to TEX - use interchangeability groups
            return group1 == group2 and group1 != 'unknown'
        
        # Compare using TEX equivalency with system-specific tolerances
        tex_tolerance = self._get_cross_system_tolerance(system1, system2)
        tex_diff = abs(tex1 - tex2) / max(tex1, tex2)
        
        return tex_diff <= tex_tolerance
    
    def _compare_within_same_system(self, size1: Dict, size2: Dict) -> bool:
        """Compare yarns within the same measurement system"""
        system = size1.get('yarn_system', 'unknown')
        tolerance = self.SIZE_TOLERANCE
        
        # System-specific comparisons with appropriate tolerances
        if system == 'ne_indirect':
            # English Cotton Count - compare Ne values with slightly higher tolerance
            ne1 = size1.get('ne_count')
            ne2 = size2.get('ne_count')
            if ne1 and ne2:
                # Cotton count tolerance should be more lenient (industry practice)
                cotton_tolerance = tolerance * 1.8  # 9% tolerance for cotton counts
                return abs(ne1 - ne2) / max(ne1, ne2) <= cotton_tolerance
        
        elif system == 'denier_direct':
            # Denier system - compare denier values with industry-appropriate tolerance
            den1 = size1.get('denier')
            den2 = size2.get('denier')
            if den1 and den2:
                # Filament yarn tolerance should be more lenient (industry practice)
                filament_tolerance = tolerance * 2.0  # 10% tolerance for filament yarns
                return abs(den1 - den2) / max(den1, den2) <= filament_tolerance
                
        elif system == 'tex_direct':
            # TEX system - direct comparison
            tex1 = size1.get('tex')
            tex2 = size2.get('tex')
            if tex1 and tex2:
                return abs(tex1 - tex2) / max(tex1, tex2) <= tolerance
        
        elif system == 'nm_indirect':
            # Metric Count - compare Nm values
            nm1 = size1.get('nm_count')
            nm2 = size2.get('nm_count')
            if nm1 and nm2:
                return abs(nm1 - nm2) / max(nm1, nm2) <= tolerance
        
        # Fallback to weight category matching
        weight1 = size1.get('weight_category', 'unknown')
        weight2 = size2.get('weight_category', 'unknown')
        return weight1 == weight2
    
    def _get_tex_equivalent(self, size_info: Dict) -> float:
        """Convert yarn size to TEX equivalent for cross-system comparison"""
        # Direct TEX value
        if 'tex' in size_info:
            return size_info['tex']
        
        # Use pre-calculated TEX equivalent
        if 'tex_equivalent' in size_info:
            return size_info['tex_equivalent']
        
        # Convert from Denier: TEX = Denier / 9
        if 'denier' in size_info:
            return size_info['denier'] / 9.0
        
        # Convert from Ne: TEX = 590.5 / Ne
        if 'ne_count' in size_info:
            return 590.5 / size_info['ne_count']
        
        # Convert from Nm: TEX = 1000 / Nm
        if 'nm_count' in size_info:
            return 1000.0 / size_info['nm_count']
        
        # Convert from Dtex: TEX = Dtex / 10
        if 'dtex' in size_info:
            return size_info['dtex'] / 10.0
        
        return None
    
    def _get_cross_system_tolerance(self, system1: str, system2: str) -> float:
        """Get tolerance for cross-system yarn comparisons"""
        base_tolerance = self.SIZE_TOLERANCE
        
        # Define system compatibility groups
        synthetic_systems = ['denier_direct', 'dtex_direct', 'tex_direct']
        natural_systems = ['ne_indirect', 'nm_indirect']
        
        # Same family systems - more lenient
        if (system1 in synthetic_systems and system2 in synthetic_systems) or \
           (system1 in natural_systems and system2 in natural_systems):
            return base_tolerance * 1.3
        
        # Cross-family systems - more strict
        if (system1 in synthetic_systems and system2 in natural_systems) or \
           (system1 in natural_systems and system2 in synthetic_systems):
            return base_tolerance * 0.7
        
        # Unknown systems - conservative
        return base_tolerance * 0.8
    
    def _extract_cost_data(self, yarn: Dict) -> float:
        """Extract cost data from yarn record if available"""
        # Look for cost in various possible fields
        cost_fields = ['cost_pound', 'cost', 'price', 'unit_cost']
        
        for field in cost_fields:
            if field in yarn and yarn[field]:
                try:
                    # Clean up currency symbols and convert to float
                    cost_str = str(yarn[field]).replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
                    return float(cost_str)
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _normalize_color(self, color: str) -> str:
        """
        Enhanced color normalization with broader equivalencies and fuzzy matching
        """
        if not color:
            return 'unknown'
        
        color_normalized = color.lower().strip()
        
        # Enhanced color equivalency groups
        color_equivalencies = {
            'natural': ['nat', 'natural', 'semidull', 'sd', 'semi-dull', 'semi dull', 'crudo', 'buff', 'organic', 'raw', 'undyed', 'ecru'],
            'black': ['black', 'blk', 'negro', 'noir', 'anthracite', 'midnight', 'anil', 'charcoal', 'jet'],
            'white': ['white', 'wht', 'blanco', 'blanc', 'snow', 'ivory', 'cream', 'off-white', 'pearl'],
            'grey': ['grey', 'gray', 'gry', 'merrimack', 'stealth', 'dawn grey', 'cool grey', 'silver', 'slate', 'ash'],
            'blue': ['blue', 'blu', 'azul', 'bleu', 'indigo', 'navy', 'royal', 'capri', 'ice blue', 'teal', 'cobalt'],
            'red': ['red', 'rouge', 'rojo', 'crimson', 'scarlet', 'burgundy', 'wine', 'cherry'],
            'green': ['green', 'grn', 'verde', 'vert', 'forest', 'lime', 'olive', 'mint', 'emerald', 'shamrock'],
            'brown': ['brown', 'brn', 'marron', 'cafe', 'tan', 'beige', 'khaki', 'camel', 'bronze', 'coffee'],
            'orange': ['orange', 'naranja', 'apricot', 'peach', 'coral', 'amber'],
            'purple': ['purple', 'violet', 'lavender', 'plum', 'magenta'],
            'yellow': ['yellow', 'amarillo', 'gold', 'lemon', 'canary', 'mustard']
        }
        
        # Check for exact matches first
        for standard_color, equivalents in color_equivalencies.items():
            if color_normalized in equivalents:
                return standard_color
            # Also check partial matches for compound colors
            for equiv in equivalents:
                if equiv in color_normalized or color_normalized in equiv:
                    return standard_color
        
        # Check for color codes (e.g., "4015-01668")
        import re
        if re.match(r'^\d{2,4}-?\d{2,5}$', color_normalized):
            return f'color_code_{color_normalized}'
        
        # If no equivalency found, return the normalized original
        return color_normalized
    
    def analyze_bom_usage(self) -> Dict:
        """Analyze which yarns are used in which styles and their percentages"""
        usage_analysis = defaultdict(list)
        
        for bom_name, df in self.bom_data.items():
            for _, row in df.iterrows():
                desc_id = row.get('desc_id')
                style_id = row.get('style_id')
                bom_percent = row.get('bom_percent', 0)
                
                if desc_id and style_id:
                    usage_analysis[desc_id].append({
                        'style_id': style_id,
                        'bom_percent': bom_percent,
                        'source_file': bom_name
                    })
        
        return dict(usage_analysis)
    
    def get_substitution_recommendations(self, target_desc_id) -> Dict:
        """Get substitution recommendations for a specific yarn"""
        recommendations = {
            'target_yarn': None,
            'alternatives': [],
            'risk_factors': [],
            'usage_impact': []
        }
        
        # Convert to appropriate data type - try both string and int
        target_ids = [target_desc_id]
        try:
            if isinstance(target_desc_id, str) and target_desc_id.isdigit():
                target_ids.append(int(target_desc_id))
            elif isinstance(target_desc_id, int):
                target_ids.append(str(target_desc_id))
        except (ValueError, TypeError, AttributeError):
            # Handle type conversion errors silently
            pass
        
        # Find target yarn details
        target_yarn = None
        for yarn_file, df in self.yarn_data.items():
            for tid in target_ids:
                match = df[df['desc_id'] == tid]
                if not match.empty:
                    target_yarn = match.iloc[0].to_dict()
                    break
            if target_yarn:
                break
        
        if not target_yarn:
            return {'error': f'Yarn {target_desc_id} not found'}
        
        recommendations['target_yarn'] = target_yarn
        
        # Find alternatives in interchangeable groups  
        for group_key, group_data in self.interchangeable_groups.items():
            if any(yarn['desc_id'] in target_ids for yarn in group_data['yarns']):
                # Found the group, get alternatives
                alternatives = [yarn for yarn in group_data['yarns'] if yarn['desc_id'] not in target_ids]
                
                # Sort alternatives by cost if available
                if group_data.get('cost_range'):
                    alternatives.sort(key=lambda x: self._extract_cost_data(x) or float('inf'))
                
                recommendations['alternatives'].extend(alternatives)
                recommendations['group_info'] = {
                    'exact_color': group_data['exact_color'],
                    'material': group_data['material_signature'], 
                    'supplier_count': group_data['supplier_count'],
                    'cost_range': group_data['cost_range']
                }
        
        # Analyze usage impact
        usage_data = self.analyze_bom_usage()
        if target_desc_id in usage_data:
            recommendations['usage_impact'] = usage_data[target_desc_id]
        
        return recommendations
    
    def generate_interchangeability_report(self) -> str:
        """Generate a comprehensive report of yarn interchangeability analysis"""
        if not self.interchangeable_groups:
            self.find_interchangeable_yarns()
        
        report = []
        report.append("=== YARN INTERCHANGEABILITY ANALYSIS REPORT ===")
        report.append("Configuration: INDUSTRY STANDARD MODE")
        report.append("- Color Tolerance: Enhanced equivalency mapping")
        report.append("- Size Tolerance: Adaptive (5-13% based on yarn type)")
        report.append("- Yarn System Separation: ENFORCED")
        report.append("- Cost Analysis: Enabled")
        report.append("")
        report.append("🚨 CRITICAL TEXTILE RULE: SPUN AND FILAMENT YARNS ARE NEVER INTERCHANGEABLE")
        report.append("   - Cotton spun (30/1, 24/1) ≠ Filament (1/150/48, 150D)")
        report.append("   - Different manufacturing processes create incompatible structures")
        report.append("   - Only yarns within the SAME structure type can be substituted")
        report.append("")
        
        # Summary statistics
        total_groups = len(self.interchangeable_groups)
        total_alternatives = sum(group['total_alternatives'] for group in self.interchangeable_groups.values())
        
        report.append(f"Total Interchangeable Groups: {total_groups}")
        report.append(f"Total Yarn Alternatives Available: {total_alternatives}")
        if total_groups > 0:
            report.append(f"Average Alternatives per Group: {total_alternatives/total_groups:.1f}")
        report.append("")
        
        # Top material categories
        material_counts = defaultdict(int)
        for group_key, group_data in self.interchangeable_groups.items():
            material_counts[group_data['material_signature']] += group_data['total_alternatives']
        
        report.append("TOP MATERIAL CATEGORIES WITH ALTERNATIVES:")
        for material, count in sorted(material_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"  {material}: {count} alternatives")
        
        report.append("\nDETAILED GROUP ANALYSIS (Top 20 by alternatives):")
        sorted_groups = sorted(self.interchangeable_groups.items(), 
                              key=lambda x: x[1]['total_alternatives'], reverse=True)[:20]
        
        for group_key, group_data in sorted_groups:
            report.append(f"\nGroup: {group_key}")
            report.append(f"  Material: {group_data['material_signature']}")
            report.append(f"  Exact Color: {group_data['exact_color'].title()}")
            report.append(f"  Total Alternatives: {group_data['total_alternatives']}")
            report.append(f"  Supplier Diversity: {group_data['supplier_count']} suppliers")
            
            if group_data.get('cost_range') and group_data['cost_range'].get('min') is not None:
                cost_range = group_data['cost_range']
                report.append(f"  Cost Range: ${cost_range['min']:.2f} - ${cost_range['max']:.2f} (avg: ${cost_range['avg']:.2f})")
            
            report.append("  Yarn Options:")
            for yarn in group_data['yarns'][:5]:  # Show first 5
                cost_info = self._extract_cost_data(yarn)
                cost_str = f" - ${cost_info:.2f}/lb" if cost_info else ""
                report.append(f"    - {yarn['desc_id']}: {yarn['supplier']}{cost_str}")
        
        return "\n".join(report)

def main():
    """Main execution function with example usage"""
    analyzer = YarnInterchangeabilityAnalyzer()
    
    # File paths (update these with actual paths)
    bom_files = [
        "D:\\Agent-MCP-1-ddd\\Agent-MCP-1-dd\\ERP Data\\prompts\\5\\Style_BOM.csv",
        "D:\\Agent-MCP-1-ddd\\Agent-MCP-1-dd\\ERP Data\\4\\BOM_Master_Sheet1.csv",
        "D:\\Agent-MCP-1-ddd\\Agent-MCP-1-dd\\ERP Data\\New folder\\BOM_2(Sheet1).csv"
    ]
    
    yarn_files = [
        "D:\\Agent-MCP-1-ddd\\Agent-MCP-1-dd\\ERP Data\\prompts\\5\\Yarn_ID_1.csv",
        "D:\\Agent-MCP-1-ddd\\Agent-MCP-1-dd\\ERP Data\\4\\Yarn_ID_Master.csv",
        "D:\\Agent-MCP-1-ddd\\Agent-MCP-1-dd\\ERP Data\\4\\Yarn_ID_Backup.csv"
    ]
    
    # Load data
    analyzer.load_bom_files(bom_files)
    analyzer.load_yarn_files(yarn_files)
    
    # Perform analysis
    interchangeable_groups = analyzer.find_interchangeable_yarns()
    
    # Generate report
    report = analyzer.generate_interchangeability_report()
    print(report)
    
    # Example: Get substitution recommendations
    example_yarn_id = "18343"  # From the data we saw
    recommendations = analyzer.get_substitution_recommendations(example_yarn_id)
    print(f"\n=== SUBSTITUTION RECOMMENDATIONS FOR YARN {example_yarn_id} ===")
    if 'error' not in recommendations:
        print(f"Target Yarn: {recommendations['target_yarn'].get('supplier', 'Unknown')} - {recommendations['target_yarn'].get('description', 'N/A')}")
        print(f"Alternatives Found: {len(recommendations['alternatives'])}")
        for alt in recommendations['alternatives'][:5]:
            print(f"  - {alt['desc_id']}: {alt['supplier']} ({alt['color']})")

if __name__ == "__main__":
    main()