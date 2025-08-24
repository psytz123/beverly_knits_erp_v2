#!/usr/bin/env python3
"""
Yarn Aggregation Intelligence Module
=====================================
Integrates yarn interchangeability analysis with shortage calculations
to identify alternative yarns that can be used when primary yarns are short.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('D:\\Agent-MCP-1-ddd')
from yarn_interchangeability_analyzer import YarnInterchangeabilityAnalyzer

class YarnAggregationIntelligence:
    def __init__(self, data_path="D:\\Agent-MCP-1-ddd\\Agent-MCP-1-dd\\ERP Data\\prompts\\5"):
        self.data_path = Path(data_path)
        self.analyzer = YarnInterchangeabilityAnalyzer()
        self.yarn_inventory = None
        self.yarn_groups = {}
        self.aggregated_balances = {}
        
    def load_data(self):
        """Load yarn inventory and BOM data for analysis"""
        try:
            # Load yarn inventory
            self.yarn_inventory = pd.read_excel(self.data_path / "yarn_inventory (2).xlsx")
            print(f"✓ Loaded yarn inventory: {len(self.yarn_inventory)} yarns")
            
            # Prepare data for interchangeability analyzer
            # Create yarn data file format
            yarn_data = self.yarn_inventory[['Desc#', 'Description', 'Supplier', 'Color']].copy()
            yarn_data.columns = ['desc_id', 'description', 'supplier', 'color']
            
            # Extract material type from description
            yarn_data['type'] = yarn_data['description'].apply(self._extract_material_type)
            yarn_data['blend'] = yarn_data['description'].apply(self._extract_blend)
            
            # Save temporary file for analyzer
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                temp_file_path = tmp_file.name
                yarn_data.to_csv(temp_file_path, index=False)
            
            # Load BOM if available
            bom_file = self.data_path / "Style_BOM.csv"
            if bom_file.exists():
                self.analyzer.load_bom_files([str(bom_file)])
            
            # Load yarn data
            self.analyzer.load_yarn_files([temp_file_path])
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _extract_material_type(self, description):
        """Extract material type from description"""
        if pd.isna(description):
            return 'Unknown'
        
        desc_lower = str(description).lower()
        
        # Common material types
        materials = {
            'polyester': 'Polyester',
            'cotton': 'Cotton',
            'nylon': 'Nylon',
            'spandex': 'Spandex',
            'modacrylic': 'Modacrylic',
            'rayon': 'Rayon',
            'bamboo': 'Bamboo',
            'wool': 'Wool',
            'acrylic': 'Acrylic'
        }
        
        for keyword, material in materials.items():
            if keyword in desc_lower:
                return material
        
        return 'Other'
    
    def _extract_blend(self, description):
        """Extract blend percentage from description"""
        if pd.isna(description):
            return '100%'
        
        import re
        desc_str = str(description)
        
        # Look for percentage patterns
        pattern = r'(\d+)[/%](\d+)'
        match = re.search(pattern, desc_str)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        
        # Look for 100% patterns
        if '100%' in desc_str:
            return '100%'
        
        return 'Unknown'
    
    def find_interchangeable_groups(self):
        """Find groups of interchangeable yarns"""
        # Use the analyzer to find groups
        self.yarn_groups = self.analyzer.find_interchangeable_yarns()
        
        print(f"\n📊 Found {len(self.yarn_groups)} interchangeable yarn groups")
        
        # Analyze each group
        group_analysis = []
        for group_id, yarns in self.yarn_groups.items():
            # Check if yarns is a list or dict
            if isinstance(yarns, list) and len(yarns) > 1:
                group_info = {
                    'group_id': group_id,
                    'yarn_count': len(yarns),
                    'yarns': [y['desc_id'] if isinstance(y, dict) else str(y) for y in yarns],
                    'material': group_id.split('|')[0] if '|' in group_id else 'Unknown',
                    'color': group_id.split('|')[-1] if '|' in group_id else 'Unknown'
                }
                group_analysis.append(group_info)
            elif isinstance(yarns, dict):
                # Handle dict format
                yarn_list = yarns.get('yarns', [])
                if len(yarn_list) > 1:
                    group_info = {
                        'group_id': group_id,
                        'yarn_count': len(yarn_list),
                        'yarns': yarn_list,
                        'material': group_id.split('|')[0] if '|' in group_id else 'Unknown',
                        'color': group_id.split('|')[-1] if '|' in group_id else 'Unknown'
                    }
                    group_analysis.append(group_info)
        
        return group_analysis
    
    def calculate_aggregated_balances(self):
        """Calculate aggregated planning balances for interchangeable yarn groups"""
        aggregated = {}
        
        for group_id, yarns in self.yarn_groups.items():
            # Handle different data formats
            if isinstance(yarns, list):
                if len(yarns) <= 1:
                    continue
                yarn_ids = [y['desc_id'] if isinstance(y, dict) else str(y) for y in yarns]
            elif isinstance(yarns, dict):
                yarn_list = yarns.get('yarns', [])
                if len(yarn_list) <= 1:
                    continue
                yarn_ids = yarn_list
            else:
                continue
            
            # Get inventory data for these yarns
            group_inventory = self.yarn_inventory[self.yarn_inventory['Desc#'].isin(yarn_ids)]
            
            if not group_inventory.empty:
                # Aggregate balances
                total_theoretical = group_inventory['Theoretical Balance'].sum()
                total_allocated = group_inventory['Allocated'].sum()
                total_on_order = group_inventory['On Order'].sum()
                total_planning = group_inventory['Planning Balance'].sum()
                
                # Calculate aggregated shortage
                aggregated_shortage = max(0, -total_planning)
                
                # Get individual shortages
                individual_shortages = []
                for _, yarn in group_inventory.iterrows():
                    if yarn['Planning Balance'] < 0:
                        individual_shortages.append({
                            'yarn_id': yarn['Desc#'],
                            'description': yarn['Description'],
                            'shortage': -yarn['Planning Balance']
                        })
                
                aggregated[group_id] = {
                    'group_id': group_id,
                    'yarn_count': len(group_inventory),
                    'yarns': list(group_inventory['Desc#']),
                    'descriptions': list(group_inventory['Description']),
                    'total_theoretical': total_theoretical,
                    'total_allocated': total_allocated,
                    'total_on_order': total_on_order,
                    'total_planning': total_planning,
                    'aggregated_shortage': aggregated_shortage,
                    'individual_shortages': individual_shortages,
                    'can_substitute': aggregated_shortage == 0 and len(individual_shortages) > 0
                }
        
        self.aggregated_balances = aggregated
        return aggregated
    
    def get_substitution_opportunities(self):
        """Identify yarns with negative balance that can be substituted with positive balance yarns"""
        opportunities = []
        
        # First, find all yarns with negative planning balance
        negative_yarns = self.yarn_inventory[self.yarn_inventory['Planning Balance'] < 0].copy()
        print(f"Found {len(negative_yarns)} yarns with negative balance")
        
        # For each negative yarn, check if it belongs to a group with positive alternatives
        for _, neg_yarn in negative_yarns.iterrows():
            yarn_id = neg_yarn['Desc#']
            
            # Find which group this yarn belongs to
            for group_id, yarns in self.yarn_groups.items():
                # Get yarn IDs in this group
                if isinstance(yarns, list):
                    group_yarn_ids = [y['desc_id'] if isinstance(y, dict) else str(y) for y in yarns]
                elif isinstance(yarns, dict):
                    group_yarn_ids = yarns.get('yarns', [])
                else:
                    continue
                
                # Check if our negative yarn is in this group
                if str(yarn_id) in [str(g) for g in group_yarn_ids]:
                    # Found the group - now check for positive alternatives
                    group_inventory = self.yarn_inventory[
                        self.yarn_inventory['Desc#'].isin([int(g) if str(g).isdigit() else g for g in group_yarn_ids])
                    ]
                    
                    # Find yarns with positive balance in the same group
                    positive_yarns = group_inventory[
                        (group_inventory['Planning Balance'] > 0) & 
                        (group_inventory['Desc#'] != yarn_id)
                    ]
                    
                    if not positive_yarns.empty:
                        # We found substitutes!
                        opportunities.append({
                            'shortage_yarn': yarn_id,
                            'shortage_description': neg_yarn['Description'],
                            'planning_balance': neg_yarn['Planning Balance'],
                            'shortage_amount': abs(neg_yarn['Planning Balance']),
                            'substitute_options': [
                                {
                                    'yarn_id': row['Desc#'],
                                    'description': row['Description'],
                                    'available_balance': row['Planning Balance'],
                                    'supplier': row['Supplier']
                                }
                                for _, row in positive_yarns.iterrows()
                            ],
                            'group_id': group_id,
                            'total_group_balance': group_inventory['Planning Balance'].sum(),
                            'recommendation': f"SUBSTITUTE: Use yarn {positive_yarns.iloc[0]['Desc#']} ({positive_yarns.iloc[0]['Planning Balance']:.2f} lbs available)"
                        })
                        break  # Found a group, no need to check others
        
        # Sort by shortage amount (largest first)
        opportunities.sort(key=lambda x: x['shortage_amount'], reverse=True)
        
        return opportunities
    
    def generate_aggregation_report(self):
        """Generate comprehensive aggregation analysis report"""
        # Find groups
        groups = self.find_interchangeable_groups()
        
        # Calculate aggregated balances
        aggregated = self.calculate_aggregated_balances()
        
        # Find substitution opportunities
        opportunities = self.get_substitution_opportunities()
        
        report = {
            'summary': {
                'total_groups': len(groups),
                'groups_with_alternatives': sum(1 for g in groups if g['yarn_count'] > 1),
                'substitution_opportunities': len(opportunities),
                'potential_shortage_reduction': sum(
                    opp['shortage_amount'] for opp in opportunities
                )
            },
            'interchangeable_groups': groups[:10],  # Top 10 groups
            'aggregated_balances': list(aggregated.values())[:10],  # Top 10
            'substitution_opportunities': opportunities[:10],  # Top 10
            'recommendations': self._generate_recommendations(opportunities)
        }
        
        return report
    
    def _generate_recommendations(self, opportunities):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if opportunities:
            total_resolvable = sum(opp['shortage_amount'] for opp in opportunities)
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Implement Yarn Substitutions',
                'impact': f"Resolve {len(opportunities)} shortages totaling {total_resolvable:,.2f} lbs",
                'details': f"Use interchangeable yarns to avoid {len(opportunities)} stockouts"
            })
        
        # Find groups with all yarns short
        all_short_groups = []
        for group_id, data in self.aggregated_balances.items():
            if data['aggregated_shortage'] > 0 and not data['can_substitute']:
                all_short_groups.append(data)
        
        if all_short_groups:
            total_shortage = sum(g['aggregated_shortage'] for g in all_short_groups)
            recommendations.append({
                'priority': 'CRITICAL',
                'action': 'Order Entire Yarn Groups',
                'impact': f"Address {len(all_short_groups)} groups with {total_shortage:,.2f} lbs shortage",
                'details': "These groups have no available substitutes - order any yarn in the group"
            })
        
        return recommendations

# API Integration Function
def get_yarn_aggregation_analysis():
    """Function to be called by Flask API"""
    analyzer = YarnAggregationIntelligence()
    
    if analyzer.load_data():
        return analyzer.generate_aggregation_report()
    else:
        return {"error": "Failed to load data"}

if __name__ == "__main__":
    print("🧵 Yarn Aggregation Intelligence Analyzer")
    print("=" * 60)
    
    analyzer = YarnAggregationIntelligence()
    if analyzer.load_data():
        report = analyzer.generate_aggregation_report()
        
        print("\n📊 AGGREGATION ANALYSIS SUMMARY")
        print("-" * 40)
        print(f"Total Interchangeable Groups: {report['summary']['total_groups']}")
        print(f"Groups with Alternatives: {report['summary']['groups_with_alternatives']}")
        print(f"Substitution Opportunities: {report['summary']['substitution_opportunities']}")
        print(f"Potential Shortage Reduction: {report['summary']['potential_shortage_reduction']:,.2f} lbs")
        
        if report['substitution_opportunities']:
            print("\n🔄 TOP SUBSTITUTION OPPORTUNITIES:")
            print("-" * 40)
            for opp in report['substitution_opportunities'][:3]:
                print(f"\nShortage: {opp['shortage_yarn']} ({opp['shortage_amount']:.2f} lbs)")
                print(f"Can substitute with:")
                for sub in opp['substitute_options'][:2]:
                    print(f"  - {sub['yarn_id']}: {sub['available']:.2f} lbs available")
        
        if report['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            print("-" * 40)
            for rec in report['recommendations']:
                print(f"\n{rec['priority']}: {rec['action']}")
                print(f"Impact: {rec['impact']}")
                print(f"Details: {rec['details']}")