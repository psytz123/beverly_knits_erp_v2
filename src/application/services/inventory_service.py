"""Application service for inventory management using dependency injection."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from src.domain.entities.yarn import Yarn
from src.domain.interfaces.yarn_repository import IYarnRepository
from src.utils.cache_manager import UnifiedCacheManager


class InventoryService:
    """Service for managing inventory operations."""
    
    def __init__(self, repository: IYarnRepository, cache: UnifiedCacheManager):
        """Initialize inventory service with dependencies."""
        self.repository = repository
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    async def get_inventory_status(self, include_forecast: bool = False) -> Dict[str, Any]:
        """Get current inventory status overview."""
        cache_key = f"inventory_status:{include_forecast}"
        
        # Check cache
        if cached := await self.cache.get(cache_key):
            return cached
        
        # Get all yarns
        yarns = await self.repository.get_all(limit=10000)
        
        # Calculate metrics
        total_yarns = len(yarns)
        shortages = [y for y in yarns if y.has_shortage()]
        critical_shortages = [y for y in yarns if y.planning_balance < 0]
        
        status = {
            'summary': {
                'total_yarns': total_yarns,
                'shortage_count': len(shortages),
                'critical_shortage_count': len(critical_shortages),
                'healthy_count': total_yarns - len(shortages),
                'shortage_percentage': (len(shortages) / total_yarns * 100) if total_yarns > 0 else 0
            },
            'inventory_value': {
                'theoretical_total': sum(y.theoretical_balance * y.cost_per_unit for y in yarns),
                'available_total': sum(y.available_balance * y.cost_per_unit for y in yarns),
                'planning_total': sum(y.planning_balance * y.cost_per_unit for y in yarns),
                'on_order_total': sum(y.on_order * y.cost_per_unit for y in yarns)
            },
            'critical_items': [
                {
                    'yarn_id': y.yarn_id,
                    'description': y.description,
                    'planning_balance': y.planning_balance,
                    'shortage_amount': y.get_shortage_amount(),
                    'days_remaining': y.days_of_stock(1.0)  # Default usage
                }
                for y in sorted(critical_shortages, key=lambda x: x.planning_balance)[:20]
            ],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Cache result
        await self.cache.set(cache_key, status, ttl=300)
        
        return status
    
    async def calculate_planning_balance(self, yarn_id: str) -> Optional[Dict[str, Any]]:
        """Calculate planning balance for a specific yarn."""
        cache_key = f"planning_balance:{yarn_id}"
        
        # Check cache
        if cached := await self.cache.get(cache_key):
            return cached
        
        # Get yarn from repository
        yarn = await self.repository.get_by_id(yarn_id)
        
        if not yarn:
            return None
        
        balance = {
            'yarn_id': yarn.yarn_id,
            'description': yarn.description,
            'theoretical_balance': yarn.theoretical_balance,
            'allocated': yarn.allocated,
            'on_order': yarn.on_order,
            'planning_balance': yarn.planning_balance,
            'available_balance': yarn.available_balance,
            'has_shortage': yarn.has_shortage(),
            'shortage_amount': yarn.get_shortage_amount(),
            'min_stock_level': yarn.min_stock_level,
            'unit_of_measure': yarn.unit_of_measure,
            'calculated_at': datetime.utcnow().isoformat()
        }
        
        # Cache result
        await self.cache.set(cache_key, balance, ttl=300)
        
        return balance
    
    async def detect_shortages(self, threshold: float = 0) -> List[Dict[str, Any]]:
        """Detect yarns with shortages."""
        cache_key = f"shortages:{threshold}"
        
        # Check cache
        if cached := await self.cache.get(cache_key):
            return cached
        
        # Get shortages from repository
        shortage_yarns = await self.repository.get_shortages(threshold)
        
        shortages = [
            {
                'yarn_id': y.yarn_id,
                'description': y.description,
                'planning_balance': y.planning_balance,
                'shortage_amount': y.get_shortage_amount(),
                'min_stock_level': y.min_stock_level,
                'theoretical_balance': y.theoretical_balance,
                'allocated': y.allocated,
                'on_order': y.on_order,
                'supplier': y.supplier,
                'lead_time_days': y.lead_time_days,
                'severity': 'critical' if y.planning_balance < 0 else 'warning'
            }
            for y in shortage_yarns
        ]
        
        # Cache result
        await self.cache.set(cache_key, shortages, ttl=300)
        
        return shortages
    
    async def get_reorder_suggestions(self, daily_usage_map: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get reorder suggestions based on usage patterns."""
        cache_key = f"reorder_suggestions:{hash(tuple(sorted(daily_usage_map.items())))}"
        
        # Check cache
        if cached := await self.cache.get(cache_key):
            return cached
        
        # Get yarns needing reorder
        yarns_to_reorder = await self.repository.get_yarns_needing_reorder(daily_usage_map)
        
        suggestions = []
        for yarn in yarns_to_reorder:
            daily_usage = daily_usage_map.get(yarn.yarn_id, 0)
            
            suggestion = {
                'yarn_id': yarn.yarn_id,
                'description': yarn.description,
                'current_balance': yarn.planning_balance,
                'daily_usage': daily_usage,
                'days_remaining': yarn.days_of_stock(daily_usage),
                'lead_time_days': yarn.lead_time_days,
                'suggested_quantity': yarn.calculate_reorder_quantity(daily_usage),
                'urgency': 'high' if yarn.days_of_stock(daily_usage) < 7 else 'medium',
                'supplier': yarn.supplier,
                'estimated_cost': yarn.calculate_reorder_quantity(daily_usage) * yarn.cost_per_unit
            }
            suggestions.append(suggestion)
        
        # Cache result
        await self.cache.set(cache_key, suggestions, ttl=600)
        
        return suggestions
    
    async def perform_yarn_netting(self, style_id: str, quantity: float) -> Dict[str, Any]:
        """Perform yarn netting calculation for a style and quantity."""
        cache_key = f"yarn_netting:{style_id}:{quantity}"
        
        # Check cache
        if cached := await self.cache.get(cache_key):
            return cached
        
        # This would integrate with BOM data
        # For now, returning a placeholder structure
        netting_result = {
            'style_id': style_id,
            'quantity_requested': quantity,
            'yarn_requirements': [],
            'availability_status': 'checking',
            'shortages': [],
            'warnings': [],
            'calculated_at': datetime.utcnow().isoformat()
        }
        
        # Cache result
        await self.cache.set(cache_key, netting_result, ttl=300)
        
        return netting_result
    
    async def get_inventory_by_filter(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get inventory with custom filters."""
        yarns = await self.repository.get_all(limit=10000)
        
        # Apply filters
        filtered_yarns = yarns
        
        if 'yarn_type' in filters:
            filtered_yarns = [y for y in filtered_yarns if y.yarn_type == filters['yarn_type']]
        
        if 'supplier' in filters:
            filtered_yarns = [y for y in filtered_yarns if y.supplier == filters['supplier']]
        
        if 'has_shortage' in filters and filters['has_shortage']:
            filtered_yarns = [y for y in filtered_yarns if y.has_shortage()]
        
        if 'min_balance' in filters:
            min_balance = float(filters['min_balance'])
            filtered_yarns = [y for y in filtered_yarns if y.planning_balance >= min_balance]
        
        if 'max_balance' in filters:
            max_balance = float(filters['max_balance'])
            filtered_yarns = [y for y in filtered_yarns if y.planning_balance <= max_balance]
        
        # Convert to dict format
        return [yarn.to_dict() for yarn in filtered_yarns]
    
    async def bulk_update_balances(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk update yarn balances."""
        success_count = 0
        error_count = 0
        errors = []
        
        for update in updates:
            try:
                yarn_id = update.get('yarn_id')
                yarn = await self.repository.get_by_id(yarn_id)
                
                if yarn:
                    # Update fields if provided
                    if 'theoretical_balance' in update:
                        yarn.theoretical_balance = float(update['theoretical_balance'])
                    if 'allocated' in update:
                        yarn.allocated = float(update['allocated'])
                    if 'on_order' in update:
                        yarn.on_order = float(update['on_order'])
                    
                    # Save update
                    await self.repository.update(yarn)
                    success_count += 1
                else:
                    error_count += 1
                    errors.append(f"Yarn {yarn_id} not found")
                    
            except Exception as e:
                error_count += 1
                errors.append(str(e))
        
        # Clear relevant caches
        await self.cache.delete_pattern("inventory_status:*")
        await self.cache.delete_pattern("planning_balance:*")
        await self.cache.delete_pattern("shortages:*")
        
        return {
            'success_count': success_count,
            'error_count': error_count,
            'errors': errors[:10],  # Limit error messages
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_inventory_analytics(self) -> Dict[str, Any]:
        """Get comprehensive inventory analytics."""
        stats = await self.repository.get_summary_stats()
        
        # Add additional analytics
        yarns = await self.repository.get_all(limit=10000)
        
        # Calculate ABC analysis
        yarns_with_value = [
            (y, y.planning_balance * y.cost_per_unit) 
            for y in yarns
        ]
        yarns_with_value.sort(key=lambda x: x[1], reverse=True)
        
        total_value = sum(v for _, v in yarns_with_value)
        cumulative_value = 0
        
        a_items = []
        b_items = []
        c_items = []
        
        for yarn, value in yarns_with_value:
            cumulative_value += value
            percentage = (cumulative_value / total_value * 100) if total_value > 0 else 0
            
            if percentage <= 80:
                a_items.append(yarn.yarn_id)
            elif percentage <= 95:
                b_items.append(yarn.yarn_id)
            else:
                c_items.append(yarn.yarn_id)
        
        analytics = {
            **stats,
            'abc_analysis': {
                'a_items_count': len(a_items),
                'b_items_count': len(b_items),
                'c_items_count': len(c_items),
                'a_items_percentage': len(a_items) / len(yarns) * 100 if yarns else 0,
                'b_items_percentage': len(b_items) / len(yarns) * 100 if yarns else 0,
                'c_items_percentage': len(c_items) / len(yarns) * 100 if yarns else 0
            },
            'inventory_health': {
                'healthy_percentage': (len([y for y in yarns if not y.has_shortage()]) / len(yarns) * 100) if yarns else 0,
                'overstocked_count': len([y for y in yarns if y.planning_balance > y.min_stock_level * 3]),
                'understocked_count': len([y for y in yarns if y.has_shortage()]),
                'zero_stock_count': len([y for y in yarns if y.planning_balance <= 0])
            }
        }
        
        return analytics