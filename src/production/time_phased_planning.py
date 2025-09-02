#!/usr/bin/env python3
"""
Time-Phased Planning Engine
Calculates weekly planning balance and shortage timeline for materials
Integrates PO delivery schedules with production demand to predict when shortages occur
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)


class TimePhasedPlanning:
    """
    Calculates weekly planning balance and shortage timeline
    Matches manual Excel calculations for yarn shortage prediction
    """
    
    def __init__(self):
        """Initialize Time-Phased Planning Engine"""
        
        # Week configuration
        self.current_week = 36  # Starting week number
        self.planning_horizon = 13  # Weeks to plan ahead
        self.confirmed_demand_horizon = 9  # Weeks with confirmed orders
        
        # Safety stock configuration
        self.safety_stock_weeks = 2  # Weeks of safety stock
        self.safety_stock_multiplier = 1.2
        
        # Week number to date mapping (as per Expected_Yarn_Report structure)
        self.week_dates = {
            36: '9/5/2025',
            37: '9/12/2025', 
            38: '9/19/2025',
            39: '9/26/2025',
            40: '10/3/2025',
            41: '10/10/2025',
            42: '10/17/2025',
            43: '10/24/2025',
            44: '10/31/2025'  # Extended for later weeks
        }
    
    def calculate_weekly_balance(self, yarn_id: str, starting_balance: float, 
                               weekly_receipts: Dict[str, float], 
                               weekly_demand: Dict[str, float],
                               start_week: int = 36, horizon: int = 9) -> Dict[str, float]:
        """
        Calculate rolling balance for each week
        Balance[Week N] = Balance[Week N-1] + Receipts[N] - Demand[N]
        
        Args:
            yarn_id: Yarn identifier
            starting_balance: Current theoretical balance
            weekly_receipts: PO receipts by week {week_36: amount, etc.}
            weekly_demand: Production demand by week
            start_week: First week number to calculate
            horizon: Number of weeks to calculate
            
        Returns:
            Weekly balance amounts {week_36: balance, etc.}
        """
        weekly_balances = {}
        current_balance = starting_balance
        
        # Add past due receipts to starting balance
        past_due = weekly_receipts.get('past_due', 0)
        current_balance += past_due
        
        logger.debug(f"Yarn {yarn_id}: Starting balance {starting_balance}, Past due +{past_due} = {current_balance}")
        
        # Calculate balance for each week
        for week_offset in range(horizon):
            week_num = start_week + week_offset
            week_key = f'week_{week_num}'
            
            # Get receipts and demand for this week
            receipts = weekly_receipts.get(week_key, 0)
            demand = weekly_demand.get(week_key, 0)
            
            # Calculate weekly balance: Previous + Receipts - Demand  
            current_balance += receipts - demand
            weekly_balances[week_key] = current_balance
            
            logger.debug(f"  {week_key}: +{receipts} -{demand} = {current_balance}")
        
        return weekly_balances
    
    def identify_shortage_periods(self, yarn_id: str, weekly_balances: Dict[str, float]) -> List[Tuple[str, float, Optional[str]]]:
        """
        Find weeks where balance < 0 (shortage periods)
        
        Args:
            yarn_id: Yarn identifier
            weekly_balances: Weekly balance amounts
            
        Returns:
            List of (week_num, shortage_amount, recovery_week) tuples
        """
        shortage_periods = []
        
        for week_key, balance in weekly_balances.items():
            if balance < 0:
                shortage_amount = abs(balance)
                recovery_week = self._find_recovery_week(weekly_balances, week_key)
                
                shortage_periods.append((week_key, shortage_amount, recovery_week))
        
        if shortage_periods:
            logger.info(f"Yarn {yarn_id}: {len(shortage_periods)} shortage periods identified")
        
        return shortage_periods
    
    def _find_recovery_week(self, weekly_balances: Dict[str, float], shortage_week: str) -> Optional[str]:
        """
        Find the week when balance returns to positive after a shortage
        
        Args:
            weekly_balances: Weekly balance amounts
            shortage_week: Week with shortage
            
        Returns:
            First week with positive balance after shortage, or None
        """
        shortage_week_num = int(shortage_week.split('_')[1])
        
        # Check subsequent weeks for recovery
        for week_offset in range(1, 10):
            check_week_num = shortage_week_num + week_offset
            check_week_key = f'week_{check_week_num}'
            
            if check_week_key in weekly_balances:
                if weekly_balances[check_week_key] > 0:
                    return check_week_key
        
        return None
    
    def calculate_expedite_requirements(self, yarn_id: str, shortage_timeline: List[Tuple[str, float, Optional[str]]], 
                                      weekly_receipts: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Determine which POs need expediting to prevent shortages
        
        Args:
            yarn_id: Yarn identifier
            shortage_timeline: List of shortage periods
            weekly_receipts: PO receipt schedule
            
        Returns:
            List of expedite recommendations
        """
        expedite_recommendations = []
        
        for shortage_week, shortage_amount, recovery_week in shortage_timeline:
            # Find the best receipt to expedite
            expedite_candidate = self._find_best_expedite_candidate(
                shortage_week, shortage_amount, weekly_receipts
            )
            
            if expedite_candidate:
                recommendation = {
                    'yarn_id': yarn_id,
                    'shortage_week': shortage_week,
                    'shortage_amount': shortage_amount,
                    'expedite_from_week': expedite_candidate['from_week'],
                    'expedite_to_week': shortage_week,
                    'expedite_amount': min(shortage_amount, expedite_candidate['amount']),
                    'recovery_week': recovery_week
                }
                expedite_recommendations.append(recommendation)
        
        return expedite_recommendations
    
    def _find_best_expedite_candidate(self, target_week: str, needed_amount: float, 
                                    weekly_receipts: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Find the best PO receipt to expedite to cover a shortage
        
        Args:
            target_week: Week needing material
            needed_amount: Amount needed
            weekly_receipts: Available receipts by week
            
        Returns:
            Best expedite candidate or None
        """
        target_week_num = int(target_week.split('_')[1])
        candidates = []
        
        # Check later weeks for receipts to expedite
        for week_offset in range(1, 10):
            check_week_num = target_week_num + week_offset
            check_week_key = f'week_{check_week_num}'
            
            if check_week_key in weekly_receipts:
                receipt_amount = weekly_receipts[check_week_key]
                if receipt_amount > 0:
                    candidates.append({
                        'from_week': check_week_key,
                        'amount': receipt_amount,
                        'weeks_moved': week_offset
                    })
        
        # Check 'later' bucket
        if 'later' in weekly_receipts and weekly_receipts['later'] > 0:
            candidates.append({
                'from_week': 'later',
                'amount': weekly_receipts['later'],
                'weeks_moved': 99  # High number to prefer nearer weeks
            })
        
        # Sort by fewest weeks moved (prefer expediting from nearest future week)
        candidates.sort(key=lambda x: x['weeks_moved'])
        
        return candidates[0] if candidates else None
    
    def calculate_yarn_coverage_weeks(self, yarn_id: str, current_balance: float, 
                                    weekly_demand: Dict[str, float]) -> float:
        """
        Calculate how many weeks current inventory will last
        
        Args:
            yarn_id: Yarn identifier
            current_balance: Current inventory balance
            weekly_demand: Demand schedule by week
            
        Returns:
            Number of weeks of coverage
        """
        if current_balance <= 0:
            return 0.0
        
        total_demand = sum(weekly_demand.values())
        if total_demand <= 0:
            return 999.0  # Infinite coverage if no demand
        
        weekly_avg_demand = total_demand / len(weekly_demand)
        coverage_weeks = current_balance / weekly_avg_demand if weekly_avg_demand > 0 else 999.0
        
        return round(coverage_weeks, 2)
    
    def generate_shortage_summary(self, yarn_id: str, weekly_balances: Dict[str, float], 
                                weekly_receipts: Dict[str, float], 
                                weekly_demand: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate comprehensive shortage analysis summary
        
        Args:
            yarn_id: Yarn identifier
            weekly_balances: Weekly balance projections
            weekly_receipts: PO receipt schedule
            weekly_demand: Demand schedule
            
        Returns:
            Complete shortage analysis
        """
        shortage_periods = self.identify_shortage_periods(yarn_id, weekly_balances)
        expedite_recs = self.calculate_expedite_requirements(yarn_id, shortage_periods, weekly_receipts)
        
        # Find first shortage week
        first_shortage_week = None
        if shortage_periods:
            first_shortage_week = shortage_periods[0][0]
        
        # Calculate total shortage amount
        total_shortage = sum(amount for _, amount, _ in shortage_periods)
        
        # Find next receipt week
        next_receipt_week = None
        next_receipt_amount = 0
        for week_num in range(36, 50):
            week_key = f'week_{week_num}'
            if weekly_receipts.get(week_key, 0) > 0:
                next_receipt_week = week_key
                next_receipt_amount = weekly_receipts[week_key]
                break
        
        if not next_receipt_week and weekly_receipts.get('later', 0) > 0:
            next_receipt_week = 'later'
            next_receipt_amount = weekly_receipts['later']
        
        # Calculate coverage weeks
        current_balance = list(weekly_balances.values())[0] if weekly_balances else 0
        coverage_weeks = self.calculate_yarn_coverage_weeks(yarn_id, current_balance, weekly_demand)
        
        return {
            'yarn_id': yarn_id,
            'has_shortage': len(shortage_periods) > 0,
            'shortage_count': len(shortage_periods),
            'first_shortage_week': first_shortage_week,
            'total_shortage_amount': total_shortage,
            'shortage_periods': shortage_periods,
            'next_receipt_week': next_receipt_week,
            'next_receipt_amount': next_receipt_amount,
            'coverage_weeks': coverage_weeks,
            'expedite_recommendations': expedite_recs,
            'weekly_balances': weekly_balances
        }
    
    def process_yarn_time_phased(self, yarn_data: Dict[str, Any], 
                               weekly_receipts: Dict[str, float],
                               weekly_demand: Dict[str, float]) -> Dict[str, Any]:
        """
        Process complete time-phased analysis for a single yarn
        
        Args:
            yarn_data: Current yarn inventory data
            weekly_receipts: PO delivery schedule
            weekly_demand: Production demand schedule
            
        Returns:
            Complete time-phased analysis
        """
        yarn_id = str(yarn_data.get('yarn_id', yarn_data.get('Desc#', 'unknown')))
        
        # Get starting balance (theoretical balance before allocated)
        theoretical_balance = yarn_data.get('theoretical_balance', 
                                          yarn_data.get('Theoretical Balance', 0))
        
        # Calculate weekly balances
        weekly_balances = self.calculate_weekly_balance(
            yarn_id=yarn_id,
            starting_balance=theoretical_balance,
            weekly_receipts=weekly_receipts,
            weekly_demand=weekly_demand
        )
        
        # Generate shortage summary
        shortage_summary = self.generate_shortage_summary(
            yarn_id, weekly_balances, weekly_receipts, weekly_demand
        )
        
        # Combine with original yarn data
        result = yarn_data.copy()
        result.update(shortage_summary)
        result['time_phased_enabled'] = True
        
        return result


def create_mock_demand_schedule(yarn_id: str, total_allocated: float, horizon_weeks: int = 9) -> Dict[str, float]:
    """
    Create mock weekly demand schedule for testing
    Distributes total allocated amount across planning horizon
    
    Args:
        yarn_id: Yarn identifier
        total_allocated: Total allocated amount (negative value)
        horizon_weeks: Number of weeks to spread demand
        
    Returns:
        Weekly demand schedule
    """
    weekly_demand = {}
    
    # Convert allocated to positive demand
    total_demand = abs(total_allocated)
    
    # Simple distribution: equal amounts per week
    weekly_amount = total_demand / horizon_weeks if horizon_weeks > 0 else 0
    
    for week_offset in range(horizon_weeks):
        week_num = 36 + week_offset  # Start at week 36
        week_key = f'week_{week_num}'
        weekly_demand[week_key] = weekly_amount
    
    return weekly_demand


def main():
    """Test function for Time-Phased Planning Engine"""
    
    # Initialize planning engine
    planner = TimePhasedPlanning()
    
    # Test with sample yarn data (matching plan document example)
    test_yarn = {
        'yarn_id': '18884',
        'theoretical_balance': 2506.18,
        'allocated': -30859.80,
        'planning_balance': 7807.68
    }
    
    # Test weekly receipts (from plan document example)
    test_receipts = {
        'past_due': 20161.30,
        'week_36': 0,
        'week_37': 0,
        'week_38': 0,
        'week_39': 0,
        'week_40': 0,
        'week_41': 0,
        'week_42': 0,
        'week_43': 4000,
        'week_44': 4000,
        'later': 8000
    }
    
    # Create mock demand schedule
    test_demand = create_mock_demand_schedule('18884', test_yarn['allocated'], 9)
    
    print(f"Testing yarn {test_yarn['yarn_id']}")
    print(f"Theoretical Balance: {test_yarn['theoretical_balance']:,.2f}")
    print(f"Total Allocated: {test_yarn['allocated']:,.2f}")
    print(f"Planning Balance: {test_yarn['planning_balance']:,.2f}")
    print(f"Total Receipts: {sum(test_receipts.values()):,.2f}")
    print(f"Weekly Demand: {test_demand}")
    
    # Process time-phased analysis
    result = planner.process_yarn_time_phased(test_yarn, test_receipts, test_demand)
    
    print(f"\n=== Time-Phased Analysis Results ===")
    print(f"Has Shortage: {result['has_shortage']}")
    print(f"First Shortage Week: {result['first_shortage_week']}")
    print(f"Total Shortage Amount: {result['total_shortage_amount']:,.2f}")
    print(f"Next Receipt Week: {result['next_receipt_week']}")
    print(f"Coverage Weeks: {result['coverage_weeks']}")
    
    print(f"\n=== Weekly Balances ===")
    for week, balance in result['weekly_balances'].items():
        status = "SHORTAGE" if balance < 0 else "OK"
        print(f"{week}: {balance:,.2f} lbs ({status})")
    
    if result['expedite_recommendations']:
        print(f"\n=== Expedite Recommendations ===")
        for rec in result['expedite_recommendations']:
            print(f"Expedite {rec['expedite_amount']:,.2f} lbs from {rec['expedite_from_week']} to {rec['expedite_to_week']}")


if __name__ == "__main__":
    main()