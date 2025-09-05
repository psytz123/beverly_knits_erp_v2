"""
Business Rules and Critical Calculations
Extracted from beverly_comprehensive_erp.py
PRESERVED EXACTLY - All custom calculations maintained as-is

CRITICAL: These business rules MUST NOT BE MODIFIED without validation
They represent core business logic for Beverly Knits operations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


class BusinessRules:
    """
    Centralized business rules and calculations
    ALL formulas are preserved exactly from original implementation
    """
    
    # Column name variations for flexible detection
    PLANNING_BALANCE_VARIATIONS = [
        'Planning Balance', 'Planning_Balance', 
        'Planning_Ballance', 'planning_balance'
    ]
    ON_ORDER_VARIATIONS = ['On Order', 'On_Order', 'on_order']
    ALLOCATED_VARIATIONS = ['Allocated', 'allocated']
    THEORETICAL_BALANCE_VARIATIONS = [
        'Theoretical Balance', 'Theoretical_Balance', 
        'Theoratical_Balance', 'theoretical_balance'
    ]
    
    @staticmethod
    def calculate_planning_balance(theoretical_balance: float, 
                                  allocated: float, 
                                  on_order: float) -> float:
        """
        CRITICAL FORMULA: Planning Balance calculation
        
        Planning_Balance = Theoretical_Balance + Allocated + On_Order
        
        IMPORTANT: Allocated values are ALREADY NEGATIVE in source data files
        Do NOT subtract or apply abs() to Allocated values
        
        Args:
            theoretical_balance: Current theoretical inventory
            allocated: Allocated quantity (ALREADY NEGATIVE in data)
            on_order: Quantity on order
            
        Returns:
            Planning balance value
        """
        return theoretical_balance + allocated + on_order
    
    @staticmethod
    def calculate_weekly_demand(consumed_data: Optional[float] = None,
                               allocated_qty: Optional[float] = None,
                               monthly_consumed: Optional[float] = None) -> float:
        """
        Calculate weekly demand based on available data
        
        Priority order:
        1. If consumed data exists: monthly_consumed / 4.3
        2. If allocated exists: allocated_qty / 8 (8-week production cycle)
        3. Default: 10 (minimal default)
        
        Args:
            consumed_data: Historical consumption data
            allocated_qty: Allocated quantity
            monthly_consumed: Monthly consumption value
            
        Returns:
            Weekly demand value
        """
        if consumed_data is not None and monthly_consumed is not None:
            # Use historical consumption
            return abs(monthly_consumed) / 4.3
        elif allocated_qty is not None:
            # Use allocated quantity over 8-week cycle
            return allocated_qty / 8
        else:
            # Minimal default
            return 10
    
    @staticmethod
    def calculate_yarn_substitution_score(color_match: float,
                                         composition_match: float,
                                         weight_match: float) -> float:
        """
        Calculate yarn substitution similarity score
        
        Formula:
        similarity_score = (color_match * 0.3) + 
                         (composition_match * 0.4) + 
                         (weight_match * 0.3)
        
        Args:
            color_match: Color matching score (0-1)
            composition_match: Composition matching score (0-1)
            weight_match: Weight matching score (0-1)
            
        Returns:
            Overall similarity score (0-1)
        """
        return (color_match * 0.3 + 
                composition_match * 0.4 + 
                weight_match * 0.3)
    
    @staticmethod
    def calculate_shortage_risk(days_of_supply: float) -> str:
        """
        Calculate shortage risk level based on days of supply
        
        Risk Levels:
        - CRITICAL: < 7 days
        - HIGH: 7-14 days
        - MEDIUM: 14-30 days
        - LOW: > 30 days
        
        Args:
            days_of_supply: Number of days of supply available
            
        Returns:
            Risk level string
        """
        if days_of_supply < 7:
            return 'CRITICAL'
        elif days_of_supply < 14:
            return 'HIGH'
        elif days_of_supply < 30:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    @staticmethod
    def calculate_reorder_point(daily_demand: float,
                               lead_time_days: int = 30,
                               safety_stock_multiplier: float = 1.5) -> float:
        """
        Calculate reorder point with safety stock
        
        Formula:
        reorder_point = daily_demand * lead_time_days * safety_stock_multiplier
        
        Args:
            daily_demand: Average daily demand
            lead_time_days: Lead time in days (default 30)
            safety_stock_multiplier: Safety stock multiplier (default 1.5)
            
        Returns:
            Reorder point quantity
        """
        return daily_demand * lead_time_days * safety_stock_multiplier
    
    @staticmethod
    def convert_yards_to_pounds(yards: float, 
                               fabric_weight_oz_per_yard: float = 5.5) -> float:
        """
        Convert fabric yards to pounds
        
        Formula:
        pounds = (yards * fabric_weight_oz_per_yard) / 16
        
        Args:
            yards: Quantity in yards
            fabric_weight_oz_per_yard: Fabric weight in oz/yard (default 5.5)
            
        Returns:
            Quantity in pounds
        """
        return (yards * fabric_weight_oz_per_yard) / 16
    
    @staticmethod
    def calculate_production_efficiency(actual_output: float,
                                       standard_output: float) -> float:
        """
        Calculate production efficiency percentage
        
        Formula:
        efficiency = (actual_output / standard_output) * 100
        
        Args:
            actual_output: Actual production output
            standard_output: Standard/expected output
            
        Returns:
            Efficiency percentage
        """
        if standard_output <= 0:
            return 0
        return (actual_output / standard_output) * 100
    
    @staticmethod
    def calculate_oee(availability: float,
                      performance: float,
                      quality: float) -> float:
        """
        Calculate Overall Equipment Effectiveness (OEE)
        
        Formula:
        OEE = Availability * Performance * Quality
        
        Args:
            availability: Machine availability rate (0-1)
            performance: Performance rate (0-1)
            quality: Quality rate (0-1)
            
        Returns:
            OEE percentage
        """
        return availability * performance * quality * 100
    
    @staticmethod
    def calculate_economic_order_quantity(annual_demand: float,
                                         ordering_cost: float = 100,
                                         holding_cost_rate: float = 0.25,
                                         unit_cost: float = 10) -> float:
        """
        Calculate Economic Order Quantity (EOQ)
        
        Formula:
        EOQ = sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        where holding_cost = holding_cost_rate * unit_cost
        
        Args:
            annual_demand: Annual demand quantity
            ordering_cost: Cost per order (default 100)
            holding_cost_rate: Holding cost rate (default 0.25)
            unit_cost: Unit cost (default 10)
            
        Returns:
            EOQ value
        """
        if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_rate <= 0 or unit_cost <= 0:
            return 0
        
        holding_cost = holding_cost_rate * unit_cost
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return round(eoq, 2)
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize common column name variations
        
        Handles typos and variations in column names
        
        Args:
            df: DataFrame with potentially inconsistent column names
            
        Returns:
            DataFrame with standardized column names
        """
        column_mappings = {
            # Planning Balance variations
            'Planning_Ballance': 'Planning_Balance',
            'Planning Ballance': 'Planning_Balance',
            'planning_ballance': 'Planning_Balance',
            'Planning Balance': 'Planning_Balance',
            
            # Theoretical Balance variations
            'Theoratical_Balance': 'Theoretical_Balance',
            'Theoretical Balance': 'Theoretical_Balance',
            
            # Style variations
            'fStyle#': 'Style',
            'Style#': 'Style',
            
            # Yarn ID variations
            'Yarn_ID': 'Desc#',
            'YarnID': 'Desc#',
            'Yarn ID': 'Desc#',
            'Desc': 'Desc#',
            
            # Other common variations
            'Qty Shipped': 'Quantity',
            'qty_shipped': 'Quantity',
            'Units': 'Quantity'
        }
        
        # Apply mappings
        df = df.rename(columns=column_mappings)
        return df
    
    @staticmethod
    def validate_planning_balance(row: pd.Series) -> bool:
        """
        Validate that Planning Balance formula is correctly applied
        
        Checks:
        Planning_Balance = Theoretical_Balance + Allocated + On_Order
        (Remember: Allocated is already negative)
        
        Args:
            row: DataFrame row with planning balance components
            
        Returns:
            True if validation passes
        """
        try:
            theoretical = float(row.get('Theoretical_Balance', 0))
            allocated = float(row.get('Allocated', 0))  # Already negative
            on_order = float(row.get('On_Order', 0))
            planning = float(row.get('Planning_Balance', 0))
            
            calculated = theoretical + allocated + on_order
            
            # Allow small floating point differences
            return abs(calculated - planning) < 0.01
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def get_machine_capacity(machine_id: int) -> float:
        """
        Get machine capacity in lbs per day
        
        Based on historical production data
        
        Args:
            machine_id: Machine ID number
            
        Returns:
            Capacity in lbs/day
        """
        machine_capacities = {
            45: 150,   # Machine 45: 150 lbs/day
            88: 200,   # Machine 88: 200 lbs/day
            127: 250,  # Machine 127: 250 lbs/day
            147: 180,  # Machine 147: 180 lbs/day
            161: 220,  # Machine 161: 220 lbs/day
            224: 195,  # Machine 224: 195 lbs/day
            110: 175,  # Machine 110: 175 lbs/day
        }
        
        return machine_capacities.get(machine_id, 100)  # Default 100 lbs/day
    
    @staticmethod
    def parse_work_center_code(work_center: str) -> Dict[str, any]:
        """
        Parse work center pattern: x.xx.xx.X
        
        Pattern breakdown:
        - First digit = knit construction
        - Second pair = machine diameter  
        - Third pair = needle cut
        - Letter = type (F/M/C/V etc.)
        
        Example: '9.38.20.F' = construction 9, diameter 38, needle 20, type F
        
        Args:
            work_center: Work center code string
            
        Returns:
            Dictionary with parsed components
        """
        try:
            parts = work_center.split('.')
            if len(parts) == 4:
                return {
                    'construction': int(parts[0]),
                    'diameter': int(parts[1]),
                    'needle_cut': int(parts[2]),
                    'type': parts[3],
                    'full_code': work_center
                }
        except (ValueError, IndexError):
            pass
        
        return {
            'construction': None,
            'diameter': None,
            'needle_cut': None,
            'type': None,
            'full_code': work_center
        }
    
    @staticmethod
    def calculate_fabric_weight(width_inches: float,
                               length_yards: float,
                               gsm: float) -> float:
        """
        Calculate fabric weight in pounds from dimensions and GSM
        
        Formula:
        weight_lbs = (width_inches * length_yards * 36 * gsm) / (39.37 * 39.37 * 453.592)
        
        Args:
            width_inches: Fabric width in inches
            length_yards: Fabric length in yards
            gsm: Grams per square meter
            
        Returns:
            Weight in pounds
        """
        # Convert to metric, calculate area, then weight
        width_meters = width_inches / 39.37
        length_meters = length_yards * 0.9144
        area_m2 = width_meters * length_meters
        weight_grams = area_m2 * gsm
        weight_lbs = weight_grams / 453.592
        
        return round(weight_lbs, 2)


class ValidationRules:
    """Data validation rules for critical business data"""
    
    @staticmethod
    def validate_inventory_data(df: pd.DataFrame) -> List[str]:
        """
        Validate inventory data integrity
        
        Args:
            df: Inventory DataFrame
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check required columns
        required_columns = ['Desc#', 'Planning_Balance', 'Theoretical_Balance']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check for negative theoretical balance (usually an error)
        if 'Theoretical_Balance' in df.columns:
            negative_theoretical = df[df['Theoretical_Balance'] < 0]
            if not negative_theoretical.empty:
                errors.append(f"Found {len(negative_theoretical)} items with negative Theoretical Balance")
        
        # Check Planning Balance calculation
        if all(col in df.columns for col in ['Planning_Balance', 'Theoretical_Balance', 'Allocated', 'On_Order']):
            invalid_rows = []
            for idx, row in df.iterrows():
                if not BusinessRules.validate_planning_balance(row):
                    invalid_rows.append(idx)
            
            if invalid_rows:
                errors.append(f"Planning Balance calculation invalid for {len(invalid_rows)} rows")
        
        return errors
    
    @staticmethod
    def validate_bom_data(df: pd.DataFrame) -> List[str]:
        """
        Validate Bill of Materials data
        
        Args:
            df: BOM DataFrame
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for required columns
        required_columns = ['Style', 'Yarn_ID', 'Usage']
        for col in required_columns:
            found = False
            for variation in [col, col.lower(), col.upper()]:
                if variation in df.columns:
                    found = True
                    break
            if not found:
                errors.append(f"Missing required BOM column: {col}")
        
        # Check for invalid usage values
        if 'Usage' in df.columns:
            invalid_usage = df[df['Usage'] <= 0]
            if not invalid_usage.empty:
                errors.append(f"Found {len(invalid_usage)} BOM entries with invalid usage <= 0")
        
        return errors