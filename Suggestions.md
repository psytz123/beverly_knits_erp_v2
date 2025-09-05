Suggestions.md

import pandas as pd
from typing import Dict, List, Tuple

class TimePhasedPlanning:
    """
    Calculates weekly planning balances, identifies shortages, and generates
    recommendations based on inventory, demand, and PO deliveries.

    This class represents an existing component of the Beverly Knits ERP v2
    system and is not modified by the eFab.ai API integration.
    """

    def __init__(self, initial_inventory: pd.DataFrame, demand_forecast: pd.DataFrame, po_deliveries: pd.DataFrame):
        """
        Initializes the planning engine with necessary data.

        Args:
            initial_inventory (pd.DataFrame): DataFrame with current inventory levels.
                                              Expected columns: ['yarn_id', 'planning_balance']
            demand_forecast (pd.DataFrame): DataFrame with weekly demand.
                                            Expected columns: ['yarn_id', 'week_36', 'week_37', ...]
            po_deliveries (pd.DataFrame): DataFrame with weekly PO deliveries.
                                          Expected columns: ['yarn_id', 'week_36', 'week_37', ...]
        """
        self.initial_inventory = initial_inventory
        self.demand_forecast = demand_forecast
        self.po_deliveries = po_deliveries
        self.planning_horizon = [col for col in demand_forecast.columns if col.startswith('week_')]

    def calculate_planning_balance(self) -> pd.DataFrame:
        """
        Calculates the time-phased planning balance for each yarn over the planning horizon.

        Returns:
            pd.DataFrame: A DataFrame with the projected inventory balance for each week.
        """
        # Merge dataframes
        merged_df = pd.merge(self.initial_inventory, self.demand_forecast, on='yarn_id', how='left')
        merged_df = pd.merge(merged_df, self.po_deliveries, on='yarn_id', how='left', suffixes=('_demand', '_supply'))
        merged_df = merged_df.fillna(0)

        balance_df = merged_df[['yarn_id']].copy()
        current_balance = merged_df['planning_balance']

        for week in self.planning_horizon:
            demand_col = f"{week}_demand"
            supply_col = f"{week}_supply"
            
            # Ensure columns exist, default to 0 if not
            weekly_demand = merged_df[demand_col] if demand_col in merged_df else 0
            weekly_supply = merged_df[supply_col] if supply_col in merged_df else 0

            # Beginning of week balance is the end of last week's balance
            balance_df[f'beg_balance_{week}'] = current_balance
            # End of week balance
            current_balance = current_balance - weekly_demand + weekly_supply
            balance_df[f'end_balance_{week}'] = current_balance

        return balance_df

    def identify_shortages(self, planning_balance_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identifies weeks where a shortage is projected to occur for each yarn.

        Args:
            planning_balance_df (pd.DataFrame): The output from calculate_planning_balance.

        Returns:
            Dict[str, List[str]]: A dictionary mapping yarn_id to a list of weeks with shortages.
        """
        shortages = {}
        for index, row in planning_balance_df.iterrows():
            yarn_id = row['yarn_id']
            shortage_weeks = []
            for week in self.planning_horizon:
                if row[f'end_balance_{week}'] < 0:
                    shortage_weeks.append(week)
            if shortage_weeks:
                shortages[yarn_id] = shortage_weeks
        return shortages

    def generate_expedite_recommendations(self, shortages: Dict[str, List[str]]) -> List[Dict]:
        """
        Generates recommendations to expedite POs to cover projected shortages.

        Args:
            shortages (Dict[str, List[str]]): The output from identify_shortages.

        Returns:
            List[Dict]: A list of dictionaries, each representing an expedite recommendation.
        """
        recommendations = []
        for yarn_id, weeks in shortages.items():
            first_shortage_week = weeks[0]
            recommendations.append({
                'yarn_id': yarn_id,
                'action': 'EXPEDITE',
                'details': f"Projected shortage in {first_shortage_week}. Review incoming POs.",
                'priority': 'HIGH'
            })
        return recommendations

    def calculate_coverage_weeks(self, planning_balance_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the number of weeks of forward coverage for each yarn.

        Args:
            planning_balance_df (pd.DataFrame): The output from calculate_planning_balance.

        Returns:
            pd.DataFrame: DataFrame with yarn_id and its corresponding weeks of coverage.
        """
        coverage_data = []
        for index, row in planning_balance_df.iterrows():
            yarn_id = row['yarn_id']
            coverage = 0
            for week in self.planning_horizon:
                if row[f'end_balance_{week}'] > 0:
                    coverage += 1
                else:
                    break  # Stop counting at the first shortage
            coverage_data.append({'yarn_id': yarn_id, 'coverage_weeks': coverage})
        
        return pd.DataFrame(coverage_data)

    def run_full_plan(self) -> Dict:
        """
        Executes the entire time-phased planning process.

        Returns:
            Dict: A dictionary containing all planning results.
        """
        planning