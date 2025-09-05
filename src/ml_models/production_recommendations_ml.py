"""
ML-Based Production Recommendations System for Beverly Knits ERP
Uses machine learning to optimize production planning and scheduling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ProductionRecommendation:
    """Container for production recommendations"""
    style_id: str
    recommended_quantity: float
    priority_score: float
    estimated_profit: float
    risk_score: float
    yarn_availability: Dict[str, float]
    machine_assignment: str
    start_date: datetime
    completion_date: datetime
    confidence_score: float
    reasoning: str


@dataclass
class OptimizationResult:
    """Result of production optimization"""
    total_profit: float
    total_quantity: float
    machine_utilization: float
    yarn_utilization: float
    recommendations: List[ProductionRecommendation]
    constraints_met: bool
    optimization_time: float


class ProductionRecommendationsML:
    """
    ML-based production recommendation system that optimizes:
    - Production scheduling
    - Machine assignments
    - Yarn allocation
    - Profit maximization
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 retrain_threshold: int = 100):
        """
        Initialize the ML recommendation system
        
        Args:
            model_path: Path to saved model
            retrain_threshold: Number of new samples before retraining
        """
        self.model_path = model_path
        self.retrain_threshold = retrain_threshold
        self.samples_since_training = 0
        
        # Initialize models
        self.demand_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.profit_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.risk_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Load model if path provided
        if model_path:
            self.load_models(model_path)
        
        logger.info("Production Recommendations ML system initialized")
    
    def train_models(self, 
                     historical_data: pd.DataFrame,
                     target_columns: Dict[str, str]):
        """
        Train all recommendation models
        
        Args:
            historical_data: Historical production data
            target_columns: Mapping of model to target column
        """
        logger.info("Training production recommendation models...")
        
        # Prepare features
        features = self._prepare_features(historical_data)
        self.feature_columns = features.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Train demand model
        if 'demand' in target_columns:
            y_demand = historical_data[target_columns['demand']]
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_demand, test_size=0.2, random_state=42
            )
            self.demand_model.fit(X_train, y_train)
            demand_score = self.demand_model.score(X_test, y_test)
            logger.info(f"Demand model R² score: {demand_score:.4f}")
        
        # Train profit model
        if 'profit' in target_columns:
            y_profit = historical_data[target_columns['profit']]
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_profit, test_size=0.2, random_state=42
            )
            self.profit_model.fit(X_train, y_train)
            profit_score = self.profit_model.score(X_test, y_test)
            logger.info(f"Profit model R² score: {profit_score:.4f}")
        
        # Train risk model
        if 'risk' in target_columns:
            y_risk = historical_data[target_columns['risk']]
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_risk, test_size=0.2, random_state=42
            )
            self.risk_model.fit(X_train, y_train)
            risk_score = self.risk_model.score(X_test, y_test)
            logger.info(f"Risk model R² score: {risk_score:.4f}")
        
        self.is_trained = True
        self.samples_since_training = 0
        
        logger.info("Model training completed")
    
    def generate_recommendations(self,
                                current_inventory: pd.DataFrame,
                                pending_orders: pd.DataFrame,
                                machine_capacity: pd.DataFrame,
                                constraints: Optional[Dict] = None) -> List[ProductionRecommendation]:
        """
        Generate ML-based production recommendations
        
        Args:
            current_inventory: Current yarn inventory
            pending_orders: Pending production orders
            machine_capacity: Available machine capacity
            constraints: Production constraints
            
        Returns:
            List of production recommendations
        """
        if not self.is_trained:
            logger.warning("Models not trained, returning basic recommendations")
            return self._generate_basic_recommendations(pending_orders)
        
        recommendations = []
        
        # Process each pending order
        for _, order in pending_orders.iterrows():
            try:
                # Prepare features for prediction
                features = self._prepare_order_features(
                    order, 
                    current_inventory,
                    machine_capacity
                )
                
                # Scale features
                X_scaled = self.scaler.transform([features])
                
                # Predict demand, profit, and risk
                predicted_demand = self.demand_model.predict(X_scaled)[0]
                predicted_profit = self.profit_model.predict(X_scaled)[0]
                predicted_risk = self.risk_model.predict(X_scaled)[0]
                
                # Calculate priority score
                priority_score = self._calculate_priority_score(
                    predicted_demand,
                    predicted_profit,
                    predicted_risk
                )
                
                # Check yarn availability
                yarn_availability = self._check_yarn_availability(
                    order['style_id'],
                    order['quantity'],
                    current_inventory
                )
                
                # Assign to best available machine
                machine_assignment = self._assign_machine(
                    order,
                    machine_capacity,
                    predicted_risk
                )
                
                # Calculate dates
                start_date, completion_date = self._calculate_dates(
                    order['quantity'],
                    machine_assignment,
                    machine_capacity
                )
                
                # Create recommendation
                rec = ProductionRecommendation(
                    style_id=order['style_id'],
                    recommended_quantity=min(order['quantity'], predicted_demand),
                    priority_score=priority_score,
                    estimated_profit=predicted_profit,
                    risk_score=predicted_risk,
                    yarn_availability=yarn_availability,
                    machine_assignment=machine_assignment,
                    start_date=start_date,
                    completion_date=completion_date,
                    confidence_score=self._calculate_confidence(features),
                    reasoning=self._generate_reasoning(
                        predicted_demand,
                        predicted_profit,
                        predicted_risk,
                        yarn_availability
                    )
                )
                
                recommendations.append(rec)
                
            except Exception as e:
                logger.error(f"Error generating recommendation for order {order.get('order_id')}: {e}")
                continue
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Apply constraints if provided
        if constraints:
            recommendations = self._apply_constraints(recommendations, constraints)
        
        return recommendations
    
    def optimize_production_schedule(self,
                                    recommendations: List[ProductionRecommendation],
                                    optimization_objective: str = 'profit') -> OptimizationResult:
        """
        Optimize production schedule based on recommendations
        
        Args:
            recommendations: List of production recommendations
            optimization_objective: 'profit', 'quantity', or 'efficiency'
            
        Returns:
            Optimized production schedule
        """
        start_time = datetime.now()
        
        # Initialize optimization variables
        selected_recommendations = []
        total_profit = 0
        total_quantity = 0
        machine_workload = {}
        yarn_usage = {}
        
        # Greedy optimization based on objective
        for rec in recommendations:
            # Check if we can accommodate this recommendation
            can_add = True
            
            # Check machine capacity
            if rec.machine_assignment in machine_workload:
                if machine_workload[rec.machine_assignment] > 0.9:  # 90% utilization limit
                    can_add = False
            
            # Check yarn availability
            for yarn_id, required in rec.yarn_availability.items():
                if yarn_id in yarn_usage:
                    if yarn_usage[yarn_id] + required > 1.0:  # Can't exceed available
                        can_add = False
                        break
            
            if can_add:
                selected_recommendations.append(rec)
                total_profit += rec.estimated_profit
                total_quantity += rec.recommended_quantity
                
                # Update machine workload
                if rec.machine_assignment not in machine_workload:
                    machine_workload[rec.machine_assignment] = 0
                machine_workload[rec.machine_assignment] += 0.1  # Simplified workload calc
                
                # Update yarn usage
                for yarn_id, usage in rec.yarn_availability.items():
                    if yarn_id not in yarn_usage:
                        yarn_usage[yarn_id] = 0
                    yarn_usage[yarn_id] += usage
        
        # Calculate utilization metrics
        avg_machine_util = np.mean(list(machine_workload.values())) if machine_workload else 0
        avg_yarn_util = np.mean(list(yarn_usage.values())) if yarn_usage else 0
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            total_profit=total_profit,
            total_quantity=total_quantity,
            machine_utilization=avg_machine_util,
            yarn_utilization=avg_yarn_util,
            recommendations=selected_recommendations,
            constraints_met=True,
            optimization_time=optimization_time
        )
    
    def update_with_feedback(self,
                           recommendation: ProductionRecommendation,
                           actual_outcome: Dict[str, Any]):
        """
        Update models with actual production outcomes
        
        Args:
            recommendation: The recommendation that was executed
            actual_outcome: Actual results
        """
        self.samples_since_training += 1
        
        # Store feedback for future training
        # In production, this would append to a training dataset
        
        # Retrain if threshold reached
        if self.samples_since_training >= self.retrain_threshold:
            logger.info("Retraining threshold reached, triggering model update")
            # In production, this would trigger a retraining pipeline
    
    # Private helper methods
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training"""
        features = pd.DataFrame()
        
        # Time-based features
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'])
            features['day_of_week'] = df['order_date'].dt.dayofweek
            features['month'] = df['order_date'].dt.month
            features['quarter'] = df['order_date'].dt.quarter
        
        # Quantity features
        if 'quantity' in df.columns:
            features['quantity'] = df['quantity']
            features['quantity_log'] = np.log1p(df['quantity'])
        
        # Style features (simplified)
        if 'style_id' in df.columns:
            # Extract numeric part of style ID
            features['style_numeric'] = df['style_id'].str.extract('(\d+)').astype(float).fillna(0)
        
        # Lead time features
        if 'lead_time' in df.columns:
            features['lead_time'] = df['lead_time']
        
        return features.fillna(0)
    
    def _prepare_order_features(self,
                               order: pd.Series,
                               inventory: pd.DataFrame,
                               capacity: pd.DataFrame) -> List[float]:
        """Prepare features for a single order"""
        features = []
        
        # Order features
        features.append(order.get('quantity', 0))
        features.append(np.log1p(order.get('quantity', 0)))
        
        # Time features
        order_date = pd.to_datetime(order.get('order_date', datetime.now()))
        features.append(order_date.dayofweek)
        features.append(order_date.month)
        features.append(order_date.quarter)
        
        # Inventory features
        total_inventory = inventory['planning_balance'].sum() if 'planning_balance' in inventory else 0
        features.append(total_inventory)
        
        # Capacity features
        total_capacity = capacity['available_hours'].sum() if 'available_hours' in capacity else 480
        features.append(total_capacity)
        
        # Style features
        style_numeric = 0
        if 'style_id' in order:
            try:
                style_numeric = float(''.join(filter(str.isdigit, str(order['style_id']))))
            except ValueError:
                pass
        features.append(style_numeric)
        
        # Lead time
        features.append(order.get('lead_time', 14))
        
        # Pad to match training features if needed
        while len(features) < len(self.feature_columns):
            features.append(0)
        
        return features[:len(self.feature_columns)]
    
    def _calculate_priority_score(self,
                                 demand: float,
                                 profit: float,
                                 risk: float) -> float:
        """Calculate priority score for recommendation"""
        # Weighted combination (higher is better)
        demand_weight = 0.3
        profit_weight = 0.5
        risk_weight = 0.2
        
        # Normalize risk (lower is better, so invert)
        risk_score = 1 - min(risk / 100, 1)
        
        priority = (
            demand_weight * min(demand / 1000, 1) +
            profit_weight * min(profit / 10000, 1) +
            risk_weight * risk_score
        )
        
        return priority * 100  # Scale to 0-100
    
    def _check_yarn_availability(self,
                                style_id: str,
                                quantity: float,
                                inventory: pd.DataFrame) -> Dict[str, float]:
        """Check yarn availability for style"""
        availability = {}
        
        # Simplified - in production would use BOM explosion
        required_yarns = ['YARN001', 'YARN002']  # Example
        
        for yarn in required_yarns:
            if yarn in inventory['yarn_id'].values:
                yarn_data = inventory[inventory['yarn_id'] == yarn].iloc[0]
                available = yarn_data.get('planning_balance', 0)
                required = quantity * 0.1  # Simplified calculation
                availability[yarn] = min(available / required, 1.0) if required > 0 else 1.0
            else:
                availability[yarn] = 0.0
        
        return availability
    
    def _assign_machine(self,
                       order: pd.Series,
                       capacity: pd.DataFrame,
                       risk: float) -> str:
        """Assign order to best available machine"""
        # Get machines with available capacity
        available_machines = capacity[capacity['available_hours'] > 0]
        
        if available_machines.empty:
            return "PENDING"
        
        # Prefer machines with lower utilization for high-risk orders
        if risk > 50:
            # Choose least utilized machine
            best_machine = available_machines.nlargest(1, 'available_hours')
        else:
            # Choose most efficient machine (simplified)
            best_machine = available_machines.nsmallest(1, 'available_hours')
        
        if not best_machine.empty:
            return str(best_machine.iloc[0].get('machine_id', 'M001'))
        
        return "M001"  # Default
    
    def _calculate_dates(self,
                        quantity: float,
                        machine: str,
                        capacity: pd.DataFrame) -> Tuple[datetime, datetime]:
        """Calculate start and completion dates"""
        start_date = datetime.now() + timedelta(days=1)
        
        # Estimate production time (simplified)
        production_hours = quantity / 100  # 100 units per hour
        production_days = production_hours / 8  # 8 hours per day
        
        completion_date = start_date + timedelta(days=max(1, production_days))
        
        return start_date, completion_date
    
    def _calculate_confidence(self, features: List[float]) -> float:
        """Calculate confidence score for prediction"""
        # Simplified confidence based on feature completeness
        non_zero_features = sum(1 for f in features if f != 0)
        confidence = non_zero_features / len(features) if features else 0
        return min(confidence * 100, 95)  # Cap at 95%
    
    def _generate_reasoning(self,
                          demand: float,
                          profit: float,
                          risk: float,
                          availability: Dict[str, float]) -> str:
        """Generate human-readable reasoning for recommendation"""
        reasons = []
        
        if demand > 500:
            reasons.append(f"High demand forecast ({demand:.0f} units)")
        
        if profit > 5000:
            reasons.append(f"High profit potential (${profit:.2f})")
        
        if risk < 30:
            reasons.append(f"Low risk score ({risk:.1f}%)")
        elif risk > 70:
            reasons.append(f"High risk - consider mitigation ({risk:.1f}%)")
        
        yarn_issues = [y for y, a in availability.items() if a < 0.5]
        if yarn_issues:
            reasons.append(f"Yarn shortage warning: {', '.join(yarn_issues)}")
        
        return "; ".join(reasons) if reasons else "Standard production recommendation"
    
    def _generate_basic_recommendations(self,
                                      orders: pd.DataFrame) -> List[ProductionRecommendation]:
        """Generate basic recommendations without ML"""
        recommendations = []
        
        for _, order in orders.iterrows():
            rec = ProductionRecommendation(
                style_id=order.get('style_id', 'UNKNOWN'),
                recommended_quantity=order.get('quantity', 0),
                priority_score=50.0,  # Default priority
                estimated_profit=order.get('quantity', 0) * 10,  # Simple estimate
                risk_score=30.0,  # Default risk
                yarn_availability={},
                machine_assignment="M001",
                start_date=datetime.now() + timedelta(days=1),
                completion_date=datetime.now() + timedelta(days=7),
                confidence_score=50.0,
                reasoning="Basic recommendation (models not trained)"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _apply_constraints(self,
                         recommendations: List[ProductionRecommendation],
                         constraints: Dict) -> List[ProductionRecommendation]:
        """Apply business constraints to recommendations"""
        filtered = recommendations.copy()
        
        # Max quantity constraint
        if 'max_quantity' in constraints:
            total_qty = 0
            final_recs = []
            for rec in filtered:
                if total_qty + rec.recommended_quantity <= constraints['max_quantity']:
                    final_recs.append(rec)
                    total_qty += rec.recommended_quantity
            filtered = final_recs
        
        # Min profit constraint
        if 'min_profit' in constraints:
            filtered = [r for r in filtered if r.estimated_profit >= constraints['min_profit']]
        
        # Max risk constraint
        if 'max_risk' in constraints:
            filtered = [r for r in filtered if r.risk_score <= constraints['max_risk']]
        
        return filtered
    
    def save_models(self, path: str):
        """Save trained models to disk"""
        if not self.is_trained:
            logger.warning("Models not trained, nothing to save")
            return
        
        model_data = {
            'demand_model': self.demand_model,
            'profit_model': self.profit_model,
            'risk_model': self.risk_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(path)
            self.demand_model = model_data['demand_model']
            self.profit_model = model_data['profit_model']
            self.risk_model = model_data['risk_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            logger.info(f"Models loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize system
    ml_system = ProductionRecommendationsML()
    
    # Create sample data
    historical_data = pd.DataFrame({
        'order_date': pd.date_range('2024-01-01', periods=100),
        'style_id': ['S' + str(i % 20) for i in range(100)],
        'quantity': np.random.randint(100, 1000, 100),
        'lead_time': np.random.randint(7, 30, 100),
        'demand': np.random.randint(500, 2000, 100),
        'profit': np.random.uniform(1000, 10000, 100),
        'risk': np.random.uniform(10, 80, 100)
    })
    
    # Train models
    ml_system.train_models(
        historical_data,
        {'demand': 'demand', 'profit': 'profit', 'risk': 'risk'}
    )
    
    # Generate recommendations
    current_inventory = pd.DataFrame({
        'yarn_id': ['YARN001', 'YARN002'],
        'planning_balance': [1000, 500]
    })
    
    pending_orders = pd.DataFrame({
        'order_id': ['ORD001', 'ORD002'],
        'style_id': ['S001', 'S002'],
        'quantity': [500, 300],
        'order_date': [datetime.now(), datetime.now()],
        'lead_time': [14, 21]
    })
    
    machine_capacity = pd.DataFrame({
        'machine_id': ['M001', 'M002'],
        'available_hours': [40, 60]
    })
    
    recommendations = ml_system.generate_recommendations(
        current_inventory,
        pending_orders,
        machine_capacity
    )
    
    # Optimize schedule
    result = ml_system.optimize_production_schedule(recommendations)
    
    print(f"Generated {len(recommendations)} recommendations")
    print(f"Optimized profit: ${result.total_profit:,.2f}")
    print(f"Total quantity: {result.total_quantity:,.0f} units")
    print(f"Machine utilization: {result.machine_utilization:.1%}")
    
    # Print top recommendation
    if recommendations:
        top_rec = recommendations[0]
        print(f"\nTop Recommendation:")
        print(f"  Style: {top_rec.style_id}")
        print(f"  Quantity: {top_rec.recommended_quantity:.0f}")
        print(f"  Priority: {top_rec.priority_score:.1f}")
        print(f"  Reasoning: {top_rec.reasoning}")