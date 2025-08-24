#!/usr/bin/env python3
"""
AI Inventory Optimization Module
Advanced AI-powered inventory optimization with reinforcement learning and dynamic safety stock
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML libraries not available for AI optimization")


@dataclass
class OptimizationConfig:
    """Configuration for AI inventory optimization"""
    min_safety_stock_days: int = 7
    max_safety_stock_days: int = 45
    service_level_target: float = 0.95
    holding_cost_per_unit: float = 0.1
    stockout_cost_per_unit: float = 5.0
    optimization_horizon_days: int = 90
    rl_learning_rate: float = 0.01
    rl_discount_factor: float = 0.95


class InventoryIntelligenceAPI:
    """
    High-level API for inventory intelligence operations
    """
    
    def __init__(self, optimizer: Optional['AIInventoryOptimizer'] = None):
        self.optimizer = optimizer or AIInventoryOptimizer()
        logger.info("InventoryIntelligenceAPI initialized")
    
    def get_optimization_recommendations(self, 
                                       inventory_data: pd.DataFrame,
                                       sales_history: pd.DataFrame,
                                       forecast_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get comprehensive optimization recommendations
        """
        try:
            # Prepare data
            current_levels = self._prepare_inventory_data(inventory_data)
            demand_history = self._prepare_sales_history(sales_history)
            
            # Run optimization
            recommendations = self.optimizer.optimize_inventory_levels(
                current_inventory=current_levels,
                demand_history=demand_history,
                forecast_data=forecast_data
            )
            
            return {
                'status': 'success',
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_inventory_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert inventory DataFrame to list of dicts"""
        inventory_list = []
        for _, row in df.iterrows():
            inventory_list.append({
                'product_id': row.get('product_id', row.get('id', '')),
                'current_stock': row.get('quantity', row.get('stock', 0)),
                'location': row.get('location', 'main'),
                'last_updated': row.get('last_updated', datetime.now())
            })
        return inventory_list
    
    def _prepare_sales_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare sales history for analysis"""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df


class AIInventoryOptimizer:
    """
    Core AI-powered inventory optimization engine
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.ml_model = None
        self.scaler = StandardScaler()
        self._initialize_ml_model()
        logger.info("AIInventoryOptimizer initialized")
    
    def _initialize_ml_model(self):
        """Initialize ML model for demand prediction"""
        if ML_AVAILABLE:
            self.ml_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
    
    def optimize_inventory_levels(self,
                                current_inventory: List[Dict[str, Any]],
                                demand_history: pd.DataFrame,
                                forecast_data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Optimize inventory levels using AI
        """
        recommendations = []
        
        for item in current_inventory:
            product_id = item['product_id']
            current_stock = item['current_stock']
            
            # Get demand statistics
            product_demand = demand_history[demand_history['product_id'] == product_id] if 'product_id' in demand_history.columns else pd.DataFrame()
            
            if not product_demand.empty:
                # Calculate optimal levels
                optimal_levels = self._calculate_optimal_levels(
                    product_id=product_id,
                    current_stock=current_stock,
                    demand_data=product_demand,
                    forecast=forecast_data
                )
                
                recommendations.append(optimal_levels)
            else:
                # Default recommendation for items without history
                recommendations.append({
                    'product_id': product_id,
                    'current_stock': current_stock,
                    'recommended_min': current_stock * 0.8,
                    'recommended_max': current_stock * 1.5,
                    'reorder_point': current_stock * 0.9,
                    'order_quantity': max(10, current_stock * 0.5),
                    'confidence': 0.5,
                    'optimization_method': 'default'
                })
        
        return recommendations
    
    def _calculate_optimal_levels(self,
                                product_id: str,
                                current_stock: float,
                                demand_data: pd.DataFrame,
                                forecast: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate optimal inventory levels for a product
        """
        # Extract demand statistics
        daily_demand = demand_data.groupby('date')['quantity'].sum() if 'quantity' in demand_data.columns else pd.Series([current_stock/30])
        avg_demand = daily_demand.mean()
        std_demand = daily_demand.std() if len(daily_demand) > 1 else avg_demand * 0.2
        
        # Calculate safety stock using service level
        z_score = 1.645  # 95% service level
        lead_time_days = 14  # Default lead time
        safety_stock = z_score * std_demand * np.sqrt(lead_time_days)
        
        # Calculate reorder point
        reorder_point = (avg_demand * lead_time_days) + safety_stock
        
        # Economic order quantity (simplified)
        holding_cost = self.config.holding_cost_per_unit
        ordering_cost = 50  # Fixed ordering cost
        eoq = np.sqrt((2 * avg_demand * 365 * ordering_cost) / holding_cost) if holding_cost > 0 else avg_demand * 30
        
        return {
            'product_id': product_id,
            'current_stock': current_stock,
            'recommended_min': safety_stock,
            'recommended_max': reorder_point + eoq,
            'reorder_point': reorder_point,
            'order_quantity': eoq,
            'avg_daily_demand': avg_demand,
            'demand_std_dev': std_demand,
            'safety_stock': safety_stock,
            'confidence': min(0.95, len(demand_data) / 30),  # Confidence based on data availability
            'optimization_method': 'ai_statistical'
        }


class DynamicSafetyStockOptimizer:
    """
    Dynamic safety stock optimization based on demand variability
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.optimization_history = []
        logger.info("DynamicSafetyStockOptimizer initialized")
    
    def optimize_safety_stock(self,
                            product_id: str,
                            demand_history: pd.DataFrame,
                            current_safety_stock: float,
                            service_level_actual: float) -> Dict[str, Any]:
        """
        Dynamically adjust safety stock based on performance
        """
        # Calculate demand variability
        demand_cv = self._calculate_demand_variability(demand_history)
        
        # Adjust safety stock based on service level performance
        adjustment_factor = 1.0
        if service_level_actual < self.config.service_level_target - 0.05:
            adjustment_factor = 1.2  # Increase safety stock
        elif service_level_actual > self.config.service_level_target + 0.05:
            adjustment_factor = 0.9  # Decrease safety stock
        
        # Apply variability-based adjustment
        variability_factor = 1 + (demand_cv - 0.3) * 0.5  # Baseline CV of 0.3
        
        new_safety_stock = current_safety_stock * adjustment_factor * variability_factor
        
        # Apply bounds
        new_safety_stock = max(
            self.config.min_safety_stock_days * demand_history['quantity'].mean(),
            min(new_safety_stock, self.config.max_safety_stock_days * demand_history['quantity'].mean())
        )
        
        return {
            'product_id': product_id,
            'current_safety_stock': current_safety_stock,
            'recommended_safety_stock': new_safety_stock,
            'adjustment_factor': adjustment_factor * variability_factor,
            'demand_variability': demand_cv,
            'service_level_gap': service_level_actual - self.config.service_level_target
        }
    
    def _calculate_demand_variability(self, demand_history: pd.DataFrame) -> float:
        """Calculate coefficient of variation for demand"""
        if 'quantity' in demand_history.columns and len(demand_history) > 1:
            mean_demand = demand_history['quantity'].mean()
            std_demand = demand_history['quantity'].std()
            return std_demand / mean_demand if mean_demand > 0 else 1.0
        return 0.5  # Default moderate variability


class ReinforcementLearningOptimizer:
    """
    Reinforcement learning-based inventory optimization
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.q_table = {}  # State-action value table
        self.learning_rate = config.rl_learning_rate if config else 0.01
        self.discount_factor = config.rl_discount_factor if config else 0.95
        self.epsilon = 0.1  # Exploration rate
        logger.info("ReinforcementLearningOptimizer initialized")
    
    def get_action(self, state: Tuple[int, int, int]) -> int:
        """
        Get optimal action for given state using epsilon-greedy
        State: (inventory_level_bucket, demand_level_bucket, season)
        Action: order quantity bucket (0-10)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 11)  # Explore
        
        # Exploit: choose best action from Q-table
        state_actions = self.q_table.get(state, {})
        if not state_actions:
            return 5  # Default middle action
        
        return max(state_actions.items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """Update Q-value based on experience"""
        current_q = self.q_table.get(state, {}).get(action, 0)
        next_max_q = max(self.q_table.get(next_state, {}).values(), default=0)
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q
    
    def calculate_reward(self, inventory_level: float, demand: float, 
                        holding_cost: float, stockout_cost: float) -> float:
        """Calculate reward based on inventory performance"""
        if inventory_level >= demand:
            # Fulfilled demand but incurred holding cost
            reward = -holding_cost * (inventory_level - demand)
        else:
            # Stockout occurred
            reward = -stockout_cost * (demand - inventory_level)
        
        return reward
    
    def train_on_history(self, history_data: pd.DataFrame):
        """Train the RL model on historical data"""
        for i in range(len(history_data) - 1):
            # Extract state and action from history
            state = self._extract_state(history_data.iloc[i])
            action = self._extract_action(history_data.iloc[i])
            next_state = self._extract_state(history_data.iloc[i + 1])
            
            # Calculate reward
            reward = self.calculate_reward(
                inventory_level=history_data.iloc[i]['inventory'],
                demand=history_data.iloc[i]['demand'],
                holding_cost=self.config.holding_cost_per_unit,
                stockout_cost=self.config.stockout_cost_per_unit
            )
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state)
    
    def _extract_state(self, row: pd.Series) -> Tuple[int, int, int]:
        """Extract state representation from data row"""
        # Bucket inventory level (0-10)
        inv_bucket = min(int(row.get('inventory', 0) / 100), 10)
        # Bucket demand level (0-10)
        demand_bucket = min(int(row.get('demand', 0) / 50), 10)
        # Season (0-3)
        season = row.get('season', 0) if 'season' in row else 0
        
        return (inv_bucket, demand_bucket, season)
    
    def _extract_action(self, row: pd.Series) -> int:
        """Extract action (order quantity) from data row"""
        return min(int(row.get('order_quantity', 0) / 100), 10)


# Module exports
__all__ = [
    'InventoryIntelligenceAPI',
    'AIInventoryOptimizer', 
    'DynamicSafetyStockOptimizer',
    'ReinforcementLearningOptimizer',
    'OptimizationConfig'
]