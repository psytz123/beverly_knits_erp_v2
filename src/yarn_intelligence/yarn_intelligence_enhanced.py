#!/usr/bin/env python3
"""
Enhanced Yarn Intelligence Module for Beverly Knits ERP
Provides deep analysis of yarn inventory, demand, and procurement needs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import platform
import math

# Import column standardizer for flexible column detection
try:
    from src.utils.column_standardization import ColumnStandardizer
except ImportError:
    try:
        from utils.column_standardization import ColumnStandardizer
    except ImportError:
        ColumnStandardizer = None

# Helper function for flexible column detection
def find_column(df, variations):
    """Find first matching column from list of variations"""
    if ColumnStandardizer:
        return ColumnStandardizer.find_column(df, variations)
    else:
        if hasattr(df, 'columns'):
            for col in variations:
                if col in df.columns:
                    return col
    return None

# Common column variations
YARN_ID_VARIATIONS = ['Desc#', 'desc#', 'Yarn', 'yarn', 'Yarn_ID', 'YarnID', 'yarn_id']
PLANNING_BALANCE_VARIATIONS = ['Planning Balance', 'Planning_Balance', 'Planning_Ballance', 'planning_balance']
ALLOCATED_VARIATIONS = ['Allocated', 'allocated']
ON_ORDER_VARIATIONS = ['On Order', 'On_Order', 'on_order']
THEORETICAL_BALANCE_VARIATIONS = ['Theoretical Balance', 'Theoretical_Balance', 'theoretical_balance']

def clean_for_json(obj):
    """Clean NaN and Infinity values for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
        return int(obj)
    return obj

class YarnIntelligenceEngine:
    def __init__(self, data_path=None):
        if data_path is None:
            if platform.system() == "Windows":
                data_path = "D:/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/prompts/5"
            else:
                data_path = "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/prompts/5"
        self.data_path = Path(data_path)
        self.yarn_demand = None
        self.yarn_by_style = None
        self.expected_yarn = None
        self.yarn_inventory = None
        self.style_bom = None
        
    def load_all_yarn_data(self):
        """Load all yarn-related data files"""
        try:
            # Load yarn demand with weekly breakdown (look for latest file)
            demand_files = list(self.data_path.glob("Yarn_Demand_*.xlsx"))
            if demand_files:
                # Filter out the by-style file
                demand_files = [f for f in demand_files if 'By_Style' not in f.name]
                if demand_files:
                    # The yarn demand file has headers in row 1, so skip first row
                    demand_file = max(demand_files, key=lambda f: f.stat().st_mtime)
                    self.yarn_demand = pd.read_excel(demand_file, skiprows=1)
                    # Rename 'Yarn' column to 'Desc#' if needed
                    if 'Yarn' in self.yarn_demand.columns and 'Desc#' not in self.yarn_demand.columns:
                        self.yarn_demand.rename(columns={'Yarn': 'Desc#'}, inplace=True)
                    print(f"Loaded yarn demand: {len(self.yarn_demand)} yarns from {demand_file.name}")
                else:
                    print("Warning: No standard yarn demand file found")
                    self.yarn_demand = pd.DataFrame()
            else:
                print("Warning: No yarn demand file found")
                self.yarn_demand = pd.DataFrame()
            
            # Load yarn demand by style
            yarn_by_style_file = self.data_path / "Yarn_Demand_By_Style (1).xlsx"
            if not yarn_by_style_file.exists():
                yarn_by_style_file = self.data_path / "Yarn_Demand_By_Style.xlsx"
            if yarn_by_style_file.exists():
                self.yarn_by_style = pd.read_excel(yarn_by_style_file)
            print(f"Loaded demand by style: {len(self.yarn_by_style)} style-yarn combinations")
            
            # Load expected yarn (on order) - may not exist
            expected_file = self.data_path / "Expected_Yarn_Report.xlsx"
            if expected_file.exists():
                self.expected_yarn = pd.read_excel(expected_file)
                print(f"Loaded expected yarn: {len(self.expected_yarn)} POs")
            else:
                print("Warning: Expected Yarn Report not found - using empty dataset")
                self.expected_yarn = pd.DataFrame()
            
            # Load current yarn inventory
            self.yarn_inventory = pd.read_excel(self.data_path / "yarn_inventory (2).xlsx")
            print(f"Loaded yarn inventory: {len(self.yarn_inventory)} yarns")
            
            # Load BOM for yarn calculations
            self.style_bom = pd.read_csv(self.data_path / "Style_BOM.csv")
            print(f"Loaded BOM: {len(self.style_bom)} entries")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def analyze_yarn_criticality(self):
        """Analyze yarn criticality based on multiple factors"""
        if self.yarn_inventory is None or self.yarn_demand is None:
            return {}
        
        critical_analysis = []
        
        # Merge inventory and demand data
        for idx, yarn_inv in self.yarn_inventory.iterrows():
            # Use flexible column detection for yarn ID
            yarn_id_col = find_column(pd.DataFrame([yarn_inv]), YARN_ID_VARIATIONS)
            yarn_id = yarn_inv.get(yarn_id_col, yarn_inv.get('Desc#', 'Unknown'))
            description = yarn_inv.get('Description', '')
            supplier = yarn_inv.get('Supplier', 'Unknown')
            
            # Get correct inventory values (handle NaN)
            theoretical_balance = yarn_inv.get('Theoretical Balance', 0)
            allocated = yarn_inv.get('Allocated', 0)
            on_order = yarn_inv.get('On Order', 0)
            planning_balance = yarn_inv.get('Planning Balance', 0)
            
            # Handle NaN values
            if pd.isna(theoretical_balance):
                theoretical_balance = 0
            if pd.isna(allocated):
                allocated = 0
            if pd.isna(on_order):
                on_order = 0
            if pd.isna(planning_balance):
                planning_balance = 0
            
            # CORRECT FORMULA: Planning Balance = Theoretical Balance - Allocated + On Order
            # If Planning Balance is negative, we have a shortage
            shortage = max(0, -planning_balance)  # Convert negative balance to positive shortage
            
            # Get weekly demand from demand data
            weekly_demand = 0
            if 'Desc#' in self.yarn_demand.columns:
                demand_row = self.yarn_demand[self.yarn_demand['Desc#'] == yarn_id]
                if not demand_row.empty:
                    weekly_demand = demand_row['Demand This Week'].iloc[0] if 'Demand This Week' in demand_row.columns else 0
            
            # Calculate weeks of supply based on planning balance
            if weekly_demand > 0:
                # If planning balance is positive, calculate weeks of supply
                # If planning balance is negative, we have 0 weeks of supply
                if planning_balance > 0:
                    weeks_of_supply = planning_balance / weekly_demand
                else:
                    weeks_of_supply = 0
            else:
                weeks_of_supply = 999 if planning_balance > 0 else 0
            
            # Determine criticality score (0-100)
            criticality_score = 0
            
            # Factor 1: Shortage amount (40 points)
            if shortage > 0:
                if shortage > 10000:
                    criticality_score += 40
                elif shortage > 5000:
                    criticality_score += 30
                elif shortage > 1000:
                    criticality_score += 20
                else:
                    criticality_score += 10
            
            # Factor 2: Weeks of supply (30 points)
            if weeks_of_supply < 1:
                criticality_score += 30
            elif weeks_of_supply < 2:
                criticality_score += 20
            elif weeks_of_supply < 4:
                criticality_score += 10
            
            # Factor 3: No receipts scheduled (20 points)
            if on_order == 0 and shortage > 0:
                criticality_score += 20
            
            # Factor 4: High weekly demand (10 points)
            if weekly_demand > 5000:
                criticality_score += 10
            
            # Determine risk level
            if criticality_score >= 70:
                risk_level = 'CRITICAL'
            elif criticality_score >= 50:
                risk_level = 'HIGH'
            elif criticality_score >= 30:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            # Only include yarns with shortages or low supply
            if shortage > 0 or weeks_of_supply < 30:
                critical_analysis.append({
                    'yarn_id': str(yarn_id),
                    'description': str(description[:50]) if description else '',
                    'supplier': str(supplier),
                    'theoretical_balance': float(theoretical_balance) if not pd.isna(theoretical_balance) else 0.0,
                    'allocated': float(allocated) if not pd.isna(allocated) else 0.0,
                    'on_order': float(on_order) if not pd.isna(on_order) else 0.0,
                    'planning_balance': float(planning_balance) if not pd.isna(planning_balance) else 0.0,
                    'balance': float(planning_balance) if not pd.isna(planning_balance) else 0.0,  # Keep for compatibility
                    'shortage': float(shortage) if not pd.isna(shortage) else 0.0,
                    'weekly_demand': float(weekly_demand) if not pd.isna(weekly_demand) else 0.0,
                    'weeks_of_supply': float(weeks_of_supply) if not pd.isna(weeks_of_supply) else 0.0,
                    'criticality_score': float(criticality_score) if not pd.isna(criticality_score) else 0.0,
                    'risk_level': risk_level
                })
        
        # Sort by criticality score
        critical_analysis.sort(key=lambda x: x['criticality_score'], reverse=True)
        
        return {
            'yarns': critical_analysis,
            'summary': {
                'total_yarns': int(len(critical_analysis)),
                'critical_count': int(sum(1 for y in critical_analysis if y['risk_level'] == 'CRITICAL')),
                'high_count': int(sum(1 for y in critical_analysis if y['risk_level'] == 'HIGH')),
                'total_shortage': float(sum(y['shortage'] for y in critical_analysis)),
                'yarns_with_shortage': int(sum(1 for y in critical_analysis if y['shortage'] > 0))
            }
        }
    
    def analyze_weekly_demand_pattern(self):
        """Analyze weekly demand patterns for planning"""
        if self.yarn_demand is None:
            return {}
        
        weekly_analysis = {}
        
        # Extract week columns
        week_cols = []
        for col in self.yarn_demand.columns:
            if 'Demand Week' in col or col == 'Demand This Week':
                week_cols.append(col)
        
        # Calculate total demand by week
        for week_col in week_cols:
            week_demand = self.yarn_demand[week_col].sum()
            week_name = week_col.replace('Demand ', '')
            weekly_analysis[week_name] = {
                'total_demand': float(week_demand),
                'yarns_needed': int((self.yarn_demand[week_col] > 0).sum())
            }
        
        # Find peak demand week
        if weekly_analysis:
            peak_week = max(weekly_analysis.items(), key=lambda x: x[1]['total_demand'])
            avg_weekly_demand = sum(w['total_demand'] for w in weekly_analysis.values()) / len(weekly_analysis)
        else:
            peak_week = None
            avg_weekly_demand = 0
        
        return {
            'weekly_demand': weekly_analysis,
            'peak_week': peak_week[0] if peak_week else None,
            'peak_demand': float(peak_week[1]['total_demand']) if peak_week else 0,
            'average_weekly_demand': float(avg_weekly_demand),
            'weeks_analyzed': int(len(weekly_analysis))
        }
    
    def analyze_style_yarn_requirements(self):
        """Analyze yarn requirements by style"""
        if self.yarn_by_style is None:
            return {}
        
        style_analysis = []
        
        # Group by style - check column name
        style_col = 'Style#' if 'Style#' in self.yarn_by_style.columns else 'Style'
        styles = self.yarn_by_style[style_col].unique()
        
        for style in styles[:20]:  # Top 20 styles
            style_data = self.yarn_by_style[self.yarn_by_style[style_col] == style]
            
            # Calculate total yarn requirement
            total_requirement = style_data['Total'].sum()
            yarn_count = len(style_data)
            
            # Get primary yarn (highest percentage)
            if 'Percentage' in style_data.columns:
                primary_yarn = style_data.nlargest(1, 'Percentage')
                if not primary_yarn.empty:
                    # Check which column has the yarn ID
                    yarn_id_col = 'Desc#' if 'Desc#' in primary_yarn.columns else 'Yarn'
                    primary_yarn_id = primary_yarn[yarn_id_col].iloc[0]
                    primary_percentage = primary_yarn['Percentage'].iloc[0]
                else:
                    primary_yarn_id = None
                    primary_percentage = 0
            else:
                primary_yarn_id = None
                primary_percentage = 0
            
            style_analysis.append({
                'style': str(style),
                'total_yarn_requirement': float(total_requirement),
                'yarn_types_count': int(yarn_count),
                'primary_yarn': str(primary_yarn_id) if primary_yarn_id else None,
                'primary_percentage': float(primary_percentage),
                'this_week_demand': float(style_data['This Week'].sum()) if 'This Week' in style_data.columns else 0
            })
        
        # Sort by total requirement
        style_analysis.sort(key=lambda x: x['total_yarn_requirement'], reverse=True)
        
        return {
            'styles': style_analysis,
            'summary': {
                'total_styles': int(len(styles)),
                'total_yarn_demand': float(self.yarn_by_style['Total'].sum()),
                'unique_yarns': int(self.yarn_by_style['Yarn'].nunique() if 'Yarn' in self.yarn_by_style.columns else 0)
            }
        }
    
    def analyze_expected_deliveries(self):
        """Analyze expected yarn deliveries from POs"""
        if self.expected_yarn is None:
            return {}
        
        delivery_analysis = []
        
        # Group by yarn
        if 'Desc#' in self.expected_yarn.columns:
            yarn_groups = self.expected_yarn.groupby('Desc#')
            
            for yarn_id, group in yarn_groups:
                total_on_order = group['PO Order Amt'].sum() if 'PO Order Amt' in group.columns else 0
                total_received = group['Rec&#39;d'].sum() if 'Rec&#39;d' in group.columns else 0
                pending = total_on_order - total_received
                
                # Get delivery schedule
                deliveries = []
                for col in ['This Week', 'Unscheduled or Past Due']:
                    if col in group.columns:
                        amount = group[col].sum()
                        if amount > 0:
                            deliveries.append({
                                'period': col,
                                'amount': float(amount)
                            })
                
                delivery_analysis.append({
                    'yarn_id': str(yarn_id),
                    'total_on_order': float(total_on_order),
                    'received': float(total_received),
                    'pending': float(pending),
                    'po_count': int(len(group)),
                    'deliveries': deliveries
                })
        
        # Sort by pending amount
        delivery_analysis.sort(key=lambda x: x['pending'], reverse=True)
        
        return {
            'deliveries': delivery_analysis[:20],  # Top 20
            'summary': {
                'total_pos': int(len(self.expected_yarn)),
                'unique_yarns': int(self.expected_yarn['Desc#'].nunique()) if 'Desc#' in self.expected_yarn.columns else 0,
                'total_on_order': float(self.expected_yarn['PO Order Amt'].sum()) if 'PO Order Amt' in self.expected_yarn.columns else 0,
                'past_due': float(self.expected_yarn['Unscheduled or Past Due'].sum()) if 'Unscheduled or Past Due' in self.expected_yarn.columns else 0
            }
        }
    
    def generate_procurement_recommendations(self):
        """Generate smart procurement recommendations"""
        critical = self.analyze_yarn_criticality()
        weekly = self.analyze_weekly_demand_pattern()
        expected = self.analyze_expected_deliveries()
        
        recommendations = []
        
        if critical and 'yarns' in critical:
            for yarn in critical['yarns'][:10]:  # Top 10 critical yarns
                if yarn['risk_level'] in ['CRITICAL', 'HIGH']:
                    # Calculate order quantity
                    # Order for 4 weeks of supply + safety stock
                    order_qty = max(
                        yarn['weekly_demand'] * 4 * 1.2,  # 4 weeks + 20% safety
                        yarn['shortage'] * 1.5  # Or 150% of shortage
                    )
                    
                    # Check if already on order
                    on_order = 0
                    if expected and 'deliveries' in expected:
                        for delivery in expected['deliveries']:
                            if delivery['yarn_id'] == yarn['yarn_id']:
                                on_order = delivery['pending']
                                break
                    
                    net_order = max(0, order_qty - on_order)
                    
                    if net_order > 0:
                        recommendations.append({
                            'yarn_id': yarn['yarn_id'],
                            'description': yarn['description'],
                            'supplier': yarn['supplier'],
                            'current_balance': float(yarn['balance']),
                            'weekly_demand': float(yarn['weekly_demand']),
                            'weeks_of_supply': float(yarn['weeks_of_supply']),
                            'shortage': float(yarn['shortage']),
                            'on_order': float(on_order),
                            'recommended_order': float(net_order),
                            'urgency': yarn['risk_level'],
                            'estimated_cost': float(net_order * 3.50)  # Assume $3.50/lb
                        })
        
        # Sort by urgency and shortage
        recommendations.sort(key=lambda x: (
            0 if x['urgency'] == 'CRITICAL' else 1,
            -x['shortage']
        ))
        
        return {
            'recommendations': recommendations,
            'summary': {
                'items_to_order': int(len(recommendations)),
                'total_order_quantity': float(sum(r['recommended_order'] for r in recommendations)),
                'total_estimated_cost': float(sum(r['estimated_cost'] for r in recommendations)),
                'critical_orders': int(sum(1 for r in recommendations if r['urgency'] == 'CRITICAL')),
                'high_priority_orders': int(sum(1 for r in recommendations if r['urgency'] == 'HIGH'))
            }
        }
    
    def find_substitution_opportunities(self):
        """
        Find yarns with negative balance that have industry-standard compatible substitutes.
        Uses proper yarn structure analysis to prevent spun/filament mixing.
        """
        if self.yarn_inventory is None:
            return {}
        
        # Import our industry-standard analyzer
        import sys
        sys.path.append('D:\\Agent-MCP-1-ddd')
        from yarn_interchangeability_analyzer import YarnInterchangeabilityAnalyzer
        
        # Initialize the industry-standard analyzer
        analyzer = YarnInterchangeabilityAnalyzer()
        
        substitution_opportunities = []
        
        # Get yarns with negative planning balance
        # Find Planning Balance column
        planning_col = find_column(self.yarn_inventory, PLANNING_BALANCE_VARIATIONS)
        if not planning_col:
            return []
        
        negative_yarns = self.yarn_inventory[self.yarn_inventory[planning_col] < 0].copy()
        
        for _, neg_yarn in negative_yarns.iterrows():
            yarn_id = neg_yarn['Desc#']
            description = neg_yarn['Description']
            planning_balance = neg_yarn['Planning Balance']
            shortage_amount = abs(planning_balance)
            
            # Parse yarn structure using industry standards
            target_yarn_info = analyzer.parse_yarn_size(description)
            yarn_structure = analyzer._determine_yarn_structure(description)
            
            # Skip if we can't determine yarn structure
            if yarn_structure == 'unknown':
                continue
            
            # Find potential substitutes with positive balance
            positive_yarns = self.yarn_inventory[
                (self.yarn_inventory['Planning Balance'] > 0) &
                (self.yarn_inventory['Desc#'] != yarn_id)
            ].copy()
            
            compatible_substitutes = []
            
            for _, pos_yarn in positive_yarns.iterrows():
                pos_description = pos_yarn['Description']
                pos_yarn_info = analyzer.parse_yarn_size(pos_description)
                pos_structure = analyzer._determine_yarn_structure(pos_description)
                
                # CRITICAL: Only consider yarns with same structure (spun vs spun, filament vs filament)
                if yarn_structure != pos_structure:
                    continue  # Skip incompatible structures
                
                # Check if yarns are within tolerance using industry standards
                if analyzer._sizes_within_tolerance(target_yarn_info, pos_yarn_info):
                    # Additional checks for material compatibility
                    if self._are_materials_compatible(description, pos_description):
                        # Additional color compatibility check
                        if self._are_colors_compatible(description, pos_description):
                            compatible_substitutes.append({
                                'yarn_id': pos_yarn['Desc#'],
                                'description': pos_description,
                                'available_balance': pos_yarn['Planning Balance'],
                                'supplier': pos_yarn['Supplier'],
                                'color': self._extract_color(pos_description),
                                'yarn_structure': pos_structure,
                                'compatibility_score': self._calculate_compatibility_score(
                                    target_yarn_info, pos_yarn_info
                                )
                            })
            
            # Sort substitutes by compatibility score and availability
            if compatible_substitutes:
                compatible_substitutes.sort(
                    key=lambda x: (-x['compatibility_score'], -x['available_balance'])
                )
                
                # Create properly structured substitution opportunity
                material_type = self._extract_primary_material(description)
                
                substitution_opportunities.append({
                    'shortage_yarn_id': str(yarn_id),
                    'shortage_description': str(description[:80]) if description else '',
                    'planning_balance': float(planning_balance),
                    'shortage_amount': float(shortage_amount),
                    'material_type': material_type,
                    'yarn_structure': yarn_structure,
                    'substitute_options': compatible_substitutes[:5],  # Top 5 substitutes
                    'best_substitute': compatible_substitutes[0]['yarn_id'],
                    'best_substitute_balance': compatible_substitutes[0]['available_balance'],
                    'validation': {
                        'structure_matched': True,
                        'size_compatible': True,
                        'material_compatible': True,
                        'color_compatible': True
                    }
                })
        
        # Sort by shortage amount (largest first)
        substitution_opportunities.sort(key=lambda x: x['shortage_amount'], reverse=True)
        
        return {
            'opportunities': substitution_opportunities[:15],  # Top 15 opportunities
            'summary': {
                'total_opportunities': len(substitution_opportunities),
                'total_shortage_with_substitutes': sum(s['shortage_amount'] for s in substitution_opportunities),
                'yarns_with_substitutes': len(substitution_opportunities)
            }
        }
    
    def get_comprehensive_yarn_intelligence(self):
        """Get all yarn intelligence in one call"""
        result = {
            'criticality_analysis': self.analyze_yarn_criticality(),
            'weekly_patterns': self.analyze_weekly_demand_pattern(),
            'style_requirements': self.analyze_style_yarn_requirements(),
            'expected_deliveries': self.analyze_expected_deliveries(),
            'procurement_recommendations': self.generate_procurement_recommendations(),
            'substitution_opportunities': self.find_substitution_opportunities(),
            'timestamp': datetime.now().isoformat()
        }
        # Clean NaN values before returning
        return clean_for_json(result)
    
    def _are_materials_compatible(self, desc1: str, desc2: str) -> bool:
        """Check if two yarns have compatible materials"""
        if not desc1 or not desc2:
            return False
        
        desc1_lower = str(desc1).lower()
        desc2_lower = str(desc2).lower()
        
        # Define material compatibility rules
        synthetic_materials = ['polyester', 'nylon', 'acrylic', 'polypropylene']
        natural_materials = ['cotton', 'wool', 'bamboo', 'linen']
        cellulosic_materials = ['rayon', 'tencel', 'viscose', 'modal']
        
        # Get primary materials
        material1 = self._get_primary_material_family(desc1_lower)
        material2 = self._get_primary_material_family(desc2_lower)
        
        # Same material family = compatible
        if material1 == material2:
            return True
        
        # Cross-family compatibility rules
        if material1 == 'synthetic' and material2 == 'synthetic':
            return True
        elif material1 == 'natural' and material2 == 'natural':
            return True
        elif material1 == 'cellulosic' and material2 == 'cellulosic':
            return True
        
        return False
    
    def _get_primary_material_family(self, description: str) -> str:
        """Get the primary material family"""
        synthetic_materials = ['polyester', 'nylon', 'acrylic', 'polypropylene', 'polyethylene']
        natural_materials = ['cotton', 'wool', 'bamboo', 'linen']
        cellulosic_materials = ['rayon', 'tencel', 'viscose', 'modal']
        
        for material in synthetic_materials:
            if material in description:
                return 'synthetic'
        
        for material in natural_materials:
            if material in description:
                return 'natural'
        
        for material in cellulosic_materials:
            if material in description:
                return 'cellulosic'
        
        return 'unknown'
    
    def _are_colors_compatible(self, desc1: str, desc2: str) -> bool:
        """Check if two yarns have compatible colors"""
        if not desc1 or not desc2:
            return False
        
        color1 = self._extract_color(desc1)
        color2 = self._extract_color(desc2)
        
        # Same color = always compatible
        if color1.lower() == color2.lower():
            return True
        
        # Define color equivalencies
        color_families = {
            'natural': ['natural', 'nat', 'crudo', 'buff', 'organic', 'raw'],
            'black': ['black', 'anil', 'anthracite', 'midnight'],
            'white': ['white', 'ivory', 'cream', 'pearl'],
            'grey': ['grey', 'gray', 'silver', 'stealth'],
            'blue': ['blue', 'indigo', 'navy', 'capri'],
        }
        
        for family, colors in color_families.items():
            if color1.lower() in colors and color2.lower() in colors:
                return True
        
        return False
    
    def _extract_color(self, description: str) -> str:
        """Extract color from yarn description"""
        if not description:
            return 'unknown'
        
        desc_lower = str(description).lower()
        
        # Common color patterns
        colors = ['natural', 'black', 'white', 'grey', 'gray', 'blue', 'red', 'green', 
                 'yellow', 'orange', 'purple', 'brown', 'navy', 'indigo', 'anil', 
                 'anthracite', 'crudo', 'buff']
        
        for color in colors:
            if color in desc_lower:
                return color
        
        return 'unknown'
    
    def _extract_primary_material(self, description: str) -> str:
        """Extract primary material type from description"""
        if not description:
            return 'Unknown'
        
        desc_lower = str(description).lower()
        
        # Priority order for material detection
        if 'cotton' in desc_lower:
            return 'Cotton'
        elif 'polyester' in desc_lower:
            return 'Polyester'
        elif 'nylon' in desc_lower:
            return 'Nylon'
        elif 'rayon' in desc_lower:
            return 'Rayon'
        elif 'bamboo' in desc_lower:
            return 'Bamboo'
        elif 'wool' in desc_lower:
            return 'Wool'
        elif 'acrylic' in desc_lower:
            return 'Acrylic'
        elif 'spandex' in desc_lower or 'elastane' in desc_lower:
            return 'Spandex'
        elif 'modacrylic' in desc_lower:
            return 'Modacrylic'
        
        return 'Mixed'
    
    def _calculate_compatibility_score(self, yarn1_info: dict, yarn2_info: dict) -> float:
        """Calculate compatibility score between two yarns (0-100)"""
        score = 0.0
        
        # Same yarn system bonus (40 points)
        if yarn1_info.get('yarn_system') == yarn2_info.get('yarn_system'):
            score += 40
        
        # Same weight category bonus (30 points)
        if yarn1_info.get('weight_category') == yarn2_info.get('weight_category'):
            score += 30
        
        # TEX equivalency bonus (20 points based on closeness)
        tex1 = yarn1_info.get('tex_equivalent')
        tex2 = yarn2_info.get('tex_equivalent')
        if tex1 and tex2:
            tex_diff = abs(tex1 - tex2) / max(tex1, tex2)
            if tex_diff <= 0.05:  # Within 5%
                score += 20
            elif tex_diff <= 0.10:  # Within 10%
                score += 15
            elif tex_diff <= 0.20:  # Within 20%
                score += 10
        
        # Same interchangeability group bonus (10 points)
        if yarn1_info.get('interchangeability_group') == yarn2_info.get('interchangeability_group'):
            score += 10
        
        return min(100.0, score)  # Cap at 100

# API endpoint function
def get_yarn_intelligence():
    """Function to be called by Flask API"""
    # Use correct Windows path
    data_path = "D:\\Agent-MCP-1-ddd\\Agent-MCP-1-dd\\ERP Data\\prompts\\5"
    engine = YarnIntelligenceEngine(data_path=data_path)
    if engine.load_all_yarn_data():
        return engine.get_comprehensive_yarn_intelligence()
    else:
        return {"error": "Failed to load yarn data"}

if __name__ == "__main__":
    print("Beverly Knits Yarn Intelligence Engine")
    print("=" * 60)
    
    engine = YarnIntelligenceEngine()
    if engine.load_all_yarn_data():
        intelligence = engine.get_comprehensive_yarn_intelligence()
        
        # Display summary
        print("\nYARN INTELLIGENCE SUMMARY")
        print("-" * 40)
        
        if 'criticality_analysis' in intelligence:
            summary = intelligence['criticality_analysis']['summary']
            print(f"Critical Yarns: {summary['critical_count']}")
            print(f"High Risk Yarns: {summary['high_count']}")
            print(f"Total Shortage: {summary['total_shortage']:,.0f} lbs")
        
        if 'procurement_recommendations' in intelligence:
            proc_summary = intelligence['procurement_recommendations']['summary']
            print(f"\nProcurement Needed:")
            print(f"  Items to Order: {proc_summary['items_to_order']}")
            print(f"  Total Quantity: {proc_summary['total_order_quantity']:,.0f} lbs")
            print(f"  Estimated Cost: ${proc_summary['total_estimated_cost']:,.2f}")
        
        # Save to JSON
        with open('yarn_intelligence.json', 'w') as f:
            json.dump(intelligence, f, indent=2, default=str)
        print("\nFull report saved to yarn_intelligence.json")