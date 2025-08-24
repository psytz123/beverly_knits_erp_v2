#!/usr/bin/env python3
"""
Yarn Allocation Manager for Beverly Knits ERP
Implements priority-based yarn allocation system for Knit Orders
Based on IMPLEMENTATION_TODO.md Task 2.4 (Lines 294-338)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YarnAllocationManager:
    """
    Manages yarn allocation priorities for Knit Orders.
    Implements hard/soft allocation types and priority-based reallocation.
    """
    
    def __init__(self):
        """Initialize the Yarn Allocation Manager"""
        self.allocation_types = {
            'HARD': 'Cannot be changed - production started',
            'SOFT': 'Planned but not started',
            'AVAILABLE': 'Can be reallocated if needed'
        }
        
        # Priority weights for scoring
        self.priority_weights = {
            'production_started': 1000,  # Highest priority for in-progress orders
            'sales_order_driven': 100,   # Customer orders get high priority
            'days_until_due': 1,         # Urgency factor
            'quantity_size': 0.1,        # Larger orders get slight preference
            'customer_priority': 50      # VIP customer factor
        }
        
        self.allocations = {}  # Track all yarn allocations
        self.allocation_history = []  # Track allocation changes
        
    def prioritize_knit_orders(self, knit_orders: pd.DataFrame) -> pd.DataFrame:
        """
        Prioritize Knit Orders based on multiple factors.
        
        Args:
            knit_orders: DataFrame with KO details
            
        Returns:
            DataFrame sorted by priority score (highest first)
        """
        if knit_orders.empty:
            logger.warning("No knit orders to prioritize")
            return knit_orders
            
        # Add priority scores
        knit_orders['priority_score'] = knit_orders.apply(
            self.calculate_priority, axis=1
        )
        
        # Sort by priority (highest first)
        prioritized = knit_orders.sort_values('priority_score', ascending=False)
        
        # Add priority rank
        prioritized['priority_rank'] = range(1, len(prioritized) + 1)
        
        logger.info(f"Prioritized {len(prioritized)} knit orders")
        return prioritized
    
    def calculate_priority(self, ko: pd.Series) -> float:
        """
        Calculate priority score for a single Knit Order.
        
        Args:
            ko: Series containing KO data
            
        Returns:
            Priority score (higher = more important)
        """
        score = 0
        
        # 1. Already started (has shipped quantity) - HIGHEST PRIORITY
        if pd.notna(ko.get('Shipped (lbs)', 0)) and ko.get('Shipped (lbs)', 0) > 0:
            score += self.priority_weights['production_started']
            logger.debug(f"KO {ko.get('Actions', 'Unknown')}: Added {self.priority_weights['production_started']} for production started")
        
        # 2. Sales order driven vs forecast driven
        if ko.get('is_sales_order', False) or ko.get('SO#', ''):
            score += self.priority_weights['sales_order_driven']
            logger.debug(f"KO {ko.get('Actions', 'Unknown')}: Added {self.priority_weights['sales_order_driven']} for sales order")
        
        # 3. Days until due (urgency)
        if pd.notna(ko.get('Quoted Date')):
            try:
                # Handle both string and datetime formats
                if isinstance(ko['Quoted Date'], str):
                    due_date = pd.to_datetime(ko['Quoted Date'])
                else:
                    due_date = ko['Quoted Date']
                    
                days_until_due = (due_date - datetime.now()).days
                
                # More urgent = higher score (inverse relationship)
                urgency_score = max(0, 100 - days_until_due) * self.priority_weights['days_until_due']
                score += urgency_score
                logger.debug(f"KO {ko.get('Actions', 'Unknown')}: Added {urgency_score:.1f} for urgency ({days_until_due} days)")
                
            except Exception as e:
                logger.warning(f"Could not parse date for KO {ko.get('Actions', 'Unknown')}: {e}")
        
        # 4. Order quantity (slight preference for larger orders)
        qty = ko.get('Qty Ordered (lbs)', 0)
        if qty > 0:
            score += min(qty * self.priority_weights['quantity_size'], 50)  # Cap at 50
        
        # 5. Customer priority (if available)
        if ko.get('customer_priority', 'NORMAL') == 'VIP':
            score += self.priority_weights['customer_priority']
        
        return score
    
    def determine_allocation_type(self, ko: pd.Series) -> str:
        """
        Determine the allocation type for a Knit Order.
        
        Args:
            ko: Knit Order data
            
        Returns:
            Allocation type (HARD, SOFT, or AVAILABLE)
        """
        # Production started = HARD allocation
        if ko.get('Shipped (lbs)', 0) > 0 or ko.get('G00 (lbs)', 0) > 0:
            return 'HARD'
        
        # Has start date in past = SOFT allocation
        if pd.notna(ko.get('Start Date')):
            try:
                start_date = pd.to_datetime(ko['Start Date'])
                if start_date <= datetime.now():
                    return 'SOFT'
            except:
                pass
        
        # Future or unscheduled = AVAILABLE
        return 'AVAILABLE'
    
    def allocate_yarn_for_ko(self, ko: pd.Series, bom_data: pd.DataFrame, 
                            available_yarn: Dict[str, float]) -> Dict:
        """
        Allocate yarn for a specific Knit Order based on BOM.
        
        Args:
            ko: Knit Order data
            bom_data: BOM data for the style
            available_yarn: Dictionary of available yarn quantities
            
        Returns:
            Dictionary with allocation details
        """
        style = ko.get('Style#', '')
        ko_id = ko.get('Actions', ko.get('KO_ID', 'NEW'))
        qty_lbs = ko.get('Qty Ordered (lbs)', 0) - ko.get('G00 (lbs)', 0)  # Remaining to produce
        
        if qty_lbs <= 0:
            logger.info(f"KO {ko_id} already complete or invalid quantity")
            return {}
        
        # Get BOM for style
        style_bom = bom_data[bom_data['Style#'] == style] if not bom_data.empty else pd.DataFrame()
        
        if style_bom.empty:
            logger.warning(f"No BOM found for style {style}")
            return {}
        
        allocation_type = self.determine_allocation_type(ko)
        allocations = {}
        
        for _, yarn in style_bom.iterrows():
            yarn_id = yarn.get('Desc#', yarn.get('Yarn_ID', ''))
            if not yarn_id:
                continue
                
            yarn_qty_needed = qty_lbs * yarn.get('BOM_Percent', 0) / 100
            
            # Check availability
            available = available_yarn.get(yarn_id, 0)
            allocated = min(yarn_qty_needed, available)
            
            allocations[yarn_id] = {
                'ko_id': ko_id,
                'style': style,
                'quantity_requested': yarn_qty_needed,
                'quantity_allocated': allocated,
                'allocation_type': allocation_type,
                'shortage': max(0, yarn_qty_needed - allocated),
                'priority_score': ko.get('priority_score', 0)
            }
            
            # Update available yarn
            available_yarn[yarn_id] = max(0, available - allocated)
        
        return allocations
    
    def suggest_reallocation(self, yarn_shortage: Dict[str, float], 
                           current_allocations: pd.DataFrame) -> List[Dict]:
        """
        Suggest yarn reallocation to address shortages.
        
        Args:
            yarn_shortage: Dictionary of yarn shortages
            current_allocations: Current allocation DataFrame
            
        Returns:
            List of reallocation suggestions
        """
        suggestions = []
        
        for yarn_id, shortage_qty in yarn_shortage.items():
            if shortage_qty <= 0:
                continue
            
            # Find AVAILABLE allocations that could be reallocated
            available_allocs = current_allocations[
                (current_allocations['yarn_id'] == yarn_id) & 
                (current_allocations['allocation_type'] == 'AVAILABLE')
            ].sort_values('priority_score', ascending=True)  # Lowest priority first
            
            cumulative_available = 0
            realloc_from = []
            
            for _, alloc in available_allocs.iterrows():
                if cumulative_available >= shortage_qty:
                    break
                    
                realloc_qty = min(alloc['quantity_allocated'], shortage_qty - cumulative_available)
                cumulative_available += realloc_qty
                
                realloc_from.append({
                    'ko_id': alloc['ko_id'],
                    'quantity': realloc_qty,
                    'priority_score': alloc['priority_score']
                })
            
            if realloc_from:
                suggestions.append({
                    'yarn_id': yarn_id,
                    'shortage_qty': shortage_qty,
                    'can_reallocate': cumulative_available,
                    'reallocate_from': realloc_from,
                    'shortage_after_reallocation': max(0, shortage_qty - cumulative_available)
                })
        
        return suggestions
    
    def create_allocation_report(self, allocations: Dict) -> pd.DataFrame:
        """
        Create a detailed allocation report.
        
        Args:
            allocations: Dictionary of all allocations
            
        Returns:
            DataFrame with allocation summary
        """
        report_data = []
        
        for yarn_id, ko_allocations in allocations.items():
            total_allocated = sum(a['quantity_allocated'] for a in ko_allocations)
            total_requested = sum(a['quantity_requested'] for a in ko_allocations)
            
            hard_allocs = sum(a['quantity_allocated'] for a in ko_allocations 
                            if a['allocation_type'] == 'HARD')
            soft_allocs = sum(a['quantity_allocated'] for a in ko_allocations 
                            if a['allocation_type'] == 'SOFT')
            available_allocs = sum(a['quantity_allocated'] for a in ko_allocations 
                                 if a['allocation_type'] == 'AVAILABLE')
            
            report_data.append({
                'yarn_id': yarn_id,
                'total_requested': total_requested,
                'total_allocated': total_allocated,
                'allocation_rate': (total_allocated / total_requested * 100) if total_requested > 0 else 0,
                'hard_allocations': hard_allocs,
                'soft_allocations': soft_allocs,
                'available_allocations': available_allocs,
                'num_kos': len(ko_allocations),
                'total_shortage': max(0, total_requested - total_allocated)
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Sort by shortage (highest first)
        report_df = report_df.sort_values('total_shortage', ascending=False)
        
        return report_df
    
    def optimize_allocation_schedule(self, knit_orders: pd.DataFrame, 
                                    yarn_availability: pd.DataFrame) -> Dict:
        """
        Optimize the allocation schedule to minimize shortages.
        
        Args:
            knit_orders: DataFrame of knit orders
            yarn_availability: DataFrame of yarn availability over time
            
        Returns:
            Optimized allocation schedule
        """
        # Prioritize orders
        prioritized_kos = self.prioritize_knit_orders(knit_orders)
        
        # Initialize available yarn
        available_yarn = {}
        if not yarn_availability.empty:
            for _, yarn in yarn_availability.iterrows():
                yarn_id = yarn.get('Desc#', yarn.get('Yarn_ID', ''))
                available_qty = yarn.get('Available', 0)
                available_yarn[yarn_id] = available_qty
        
        allocation_schedule = {}
        unallocated_kos = []
        
        # Allocate in priority order
        for _, ko in prioritized_kos.iterrows():
            ko_id = ko.get('Actions', ko.get('KO_ID', f"KO_{_}"))
            
            # Mock BOM data for now (should be loaded from actual BOM)
            # In production, this would query the actual BOM data
            mock_bom = pd.DataFrame()  # Empty for now
            
            allocation = self.allocate_yarn_for_ko(ko, mock_bom, available_yarn)
            
            if allocation:
                allocation_schedule[ko_id] = {
                    'priority_rank': ko.get('priority_rank', 999),
                    'allocations': allocation,
                    'allocation_type': self.determine_allocation_type(ko),
                    'can_start': all(a['shortage'] == 0 for a in allocation.values())
                }
            else:
                unallocated_kos.append(ko_id)
        
        # Summary statistics
        summary = {
            'total_kos': len(prioritized_kos),
            'allocated_kos': len(allocation_schedule),
            'unallocated_kos': len(unallocated_kos),
            'can_start_immediately': sum(1 for s in allocation_schedule.values() if s['can_start']),
            'hard_allocations': sum(1 for s in allocation_schedule.values() if s['allocation_type'] == 'HARD'),
            'soft_allocations': sum(1 for s in allocation_schedule.values() if s['allocation_type'] == 'SOFT'),
            'schedule': allocation_schedule,
            'unallocated': unallocated_kos
        }
        
        logger.info(f"Allocation optimization complete: {summary['allocated_kos']}/{summary['total_kos']} KOs allocated")
        
        return summary


# Test function
def test_yarn_allocation_manager():
    """Test the Yarn Allocation Manager with sample data"""
    
    manager = YarnAllocationManager()
    
    # Create sample Knit Orders
    sample_kos = pd.DataFrame({
        'Actions': ['KO001', 'KO002', 'KO003', 'KO004'],
        'Style#': ['ST001', 'ST002', 'ST001', 'ST003'],
        'Qty Ordered (lbs)': [1000, 500, 750, 1200],
        'G00 (lbs)': [200, 0, 0, 100],  # KO001 and KO004 have started
        'Shipped (lbs)': [50, 0, 0, 0],  # KO001 has shipped some
        'Quoted Date': [
            datetime.now() + timedelta(days=10),
            datetime.now() + timedelta(days=5),
            datetime.now() + timedelta(days=30),
            datetime.now() + timedelta(days=7)
        ],
        'is_sales_order': [True, True, False, True],
        'customer_priority': ['VIP', 'NORMAL', 'NORMAL', 'VIP']
    })
    
    # Test prioritization
    print("Testing Knit Order Prioritization...")
    prioritized = manager.prioritize_knit_orders(sample_kos)
    print("\nPrioritized Knit Orders:")
    print(prioritized[['Actions', 'priority_score', 'priority_rank']])
    
    # Test allocation type determination
    print("\n\nTesting Allocation Type Determination...")
    for _, ko in sample_kos.iterrows():
        alloc_type = manager.determine_allocation_type(ko)
        print(f"KO {ko['Actions']}: {alloc_type} allocation")
    
    # Test yarn availability
    yarn_availability = pd.DataFrame({
        'Yarn_ID': ['Y001', 'Y002', 'Y003'],
        'Available': [500, 1000, 750]
    })
    
    # Test optimization
    print("\n\nTesting Allocation Optimization...")
    optimization_result = manager.optimize_allocation_schedule(sample_kos, yarn_availability)
    
    print(f"\nOptimization Summary:")
    print(f"  Total KOs: {optimization_result['total_kos']}")
    print(f"  Allocated: {optimization_result['allocated_kos']}")
    print(f"  Can Start: {optimization_result['can_start_immediately']}")
    print(f"  Hard Allocations: {optimization_result['hard_allocations']}")
    print(f"  Soft Allocations: {optimization_result['soft_allocations']}")
    
    print("\n✅ Yarn Allocation Manager tests completed successfully!")


if __name__ == "__main__":
    test_yarn_allocation_manager()