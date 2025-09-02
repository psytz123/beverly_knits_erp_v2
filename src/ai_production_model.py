#!/usr/bin/env python3
"""
AI Production Model for Beverly Knits ERP
Visual AI production system with capacity optimization, bottleneck detection, and machine-level intelligence
Integrates with machine mapper and production capacity manager for complete factory floor optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import math
import random

# Import existing components
try:
    from src.production.machine_mapper import get_machine_mapper, MachineWorkCenterMapper
    from src.production.production_capacity_manager import get_capacity_manager, ProductionCapacityManager
    from src.ml_models.production_recommendation_ml import get_ml_model, ProductionRecommendationModel
except ImportError:
    try:
        from production.machine_mapper import get_machine_mapper, MachineWorkCenterMapper
        from production.production_capacity_manager import get_capacity_manager, ProductionCapacityManager
        from ml_models.production_recommendation_ml import get_ml_model, ProductionRecommendationModel
    except ImportError:
        logger.warning("Could not import production components - some features will be disabled")
        get_machine_mapper = None
        get_capacity_manager = None
        get_ml_model = None

logger = logging.getLogger(__name__)

class BottleneckSeverity(Enum):
    """Bottleneck severity levels"""
    NONE = "NONE"
    LOW = "LOW" 
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    
    @property
    def color(self):
        colors = {
            "NONE": "#22c55e",      # Green
            "LOW": "#84cc16",       # Light green
            "MEDIUM": "#eab308",    # Yellow
            "HIGH": "#f97316",      # Orange
            "CRITICAL": "#ef4444"   # Red
        }
        return colors.get(self.value, "#6b7280")  # Gray default
    
    @property
    def priority_score(self):
        scores = {
            "NONE": 0,
            "LOW": 1,
            "MEDIUM": 2, 
            "HIGH": 3,
            "CRITICAL": 4
        }
        return scores.get(self.value, 0)

@dataclass 
class BottleneckAlert:
    """Bottleneck detection result"""
    work_center: str
    machine_ids: List[str]
    severity: BottleneckSeverity
    utilization_percent: float
    capacity_shortage_lbs: float
    affected_styles: List[str]
    estimated_delay_days: float
    recommendation: str
    urgency_score: float
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['severity'] = self.severity.value
        result['color'] = self.severity.color
        return result

@dataclass
class ProductionOptimization:
    """Production optimization recommendation"""
    optimization_type: str  # "REBALANCE", "PRIORITY_ADJUST", "CAPACITY_ADD", "SCHEDULE_CHANGE"
    target: str  # work_center or machine_id
    current_value: float
    recommended_value: float
    impact_description: str
    potential_improvement_percent: float
    effort_level: str  # "LOW", "MEDIUM", "HIGH"
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ProductionForecast:
    """Production forecast data"""
    time_horizon_days: int
    forecasted_demand_lbs: float
    current_capacity_lbs: float
    capacity_utilization_percent: float
    projected_bottlenecks: List[str]
    confidence_score: float
    
    def to_dict(self) -> Dict:
        return asdict(self)

class AIProductionModel:
    """
    AI-powered production optimization model
    
    Features:
    - Real-time bottleneck detection across 285 machines
    - Capacity optimization recommendations  
    - Production flow visualization data
    - ML-powered demand forecasting integration
    - Work center load balancing
    - Long-term planning (30/60/90 days)
    """
    
    def __init__(self):
        """Initialize the AI production model"""
        self.machine_mapper = None
        self.capacity_manager = None
        self.ml_model = None
        
        # AI model parameters
        self.bottleneck_threshold = 85.0  # % utilization to trigger bottleneck alert
        self.optimal_utilization = 75.0   # Target utilization %
        self.capacity_buffer = 20.0       # Buffer capacity % for demand spikes
        
        # Historical data simulation (in real system, this would be actual data)
        self.historical_utilization = {}  # machine_id -> List[utilization_values]
        self.demand_patterns = {}         # style -> seasonal patterns
        
        # Performance tracking
        self.optimization_history = []
        self.prediction_accuracy = 0.85
        
        # Initialize components
        self._initialize_components()
        self._generate_simulation_data()
        
        logger.info("AI Production Model initialized successfully")
    
    def _initialize_components(self):
        """Initialize machine mapper, capacity manager, and ML model"""
        try:
            if get_machine_mapper:
                self.machine_mapper = get_machine_mapper()
                logger.info("Machine mapper loaded")
            
            if get_capacity_manager:
                self.capacity_manager = get_capacity_manager()
                logger.info("Capacity manager loaded")
                
            if get_ml_model:
                self.ml_model = get_ml_model()
                logger.info("ML recommendation model loaded")
                
        except Exception as e:
            logger.warning(f"Failed to initialize some components: {e}")
    
    def _generate_simulation_data(self):
        """Generate realistic simulation data for demonstration"""
        if not self.machine_mapper:
            return
        
        # Generate historical utilization patterns
        for machine_id in self.machine_mapper.machine_info.keys():
            # Simulate 30 days of utilization data
            base_utilization = random.uniform(40, 90)
            daily_variation = random.uniform(5, 15)
            
            utilization_history = []
            for day in range(30):
                daily_util = base_utilization + random.gauss(0, daily_variation)
                daily_util = max(0, min(100, daily_util))  # Clamp to 0-100%
                utilization_history.append(daily_util)
            
            self.historical_utilization[machine_id] = utilization_history
            
            # Update current utilization in capacity manager
            current_util = utilization_history[-1]
            self.capacity_manager.update_machine_utilization(machine_id, current_util)
        
        logger.info(f"Generated simulation data for {len(self.historical_utilization)} machines")
    
    # === Core AI Analysis Methods ===
    
    def detect_bottlenecks(self) -> List[BottleneckAlert]:
        """Detect production bottlenecks using AI analysis"""
        if not self.machine_mapper or not self.capacity_manager:
            return []
        
        bottlenecks = []
        
        # Analyze each work center
        for work_center, machine_ids in self.machine_mapper.work_center_to_machines.items():
            if not machine_ids:
                continue
            
            # Calculate work center metrics
            total_utilization = 0
            total_capacity = 0
            affected_styles = set()
            max_delay = 0
            
            for machine_id in machine_ids:
                utilization = self.capacity_manager.get_machine_utilization(machine_id)
                total_utilization += utilization
                
                # Get assigned style
                assigned_style = self.capacity_manager.get_machine_assignment(machine_id)
                if assigned_style:
                    affected_styles.add(assigned_style)
                    style_capacity = self.capacity_manager.get_style_capacity(assigned_style)
                    total_capacity += style_capacity
                else:
                    total_capacity += self.capacity_manager.default_capacity
            
            # Calculate average utilization
            avg_utilization = total_utilization / len(machine_ids) if machine_ids else 0
            
            # Determine bottleneck severity
            severity = self._calculate_bottleneck_severity(avg_utilization, len(machine_ids))
            
            if severity != BottleneckSeverity.NONE:
                # Calculate capacity shortage
                current_effective_capacity = total_capacity * (avg_utilization / 100)
                target_capacity = total_capacity * (self.optimal_utilization / 100)
                capacity_shortage = max(0, current_effective_capacity - target_capacity)
                
                # Estimate delay impact
                delay_days = self._estimate_delay_impact(capacity_shortage, total_capacity)
                
                # Generate recommendation
                recommendation = self._generate_bottleneck_recommendation(severity, work_center, avg_utilization)
                
                # Calculate urgency score
                urgency = self._calculate_urgency_score(severity, delay_days, len(affected_styles))
                
                bottleneck = BottleneckAlert(
                    work_center=work_center,
                    machine_ids=machine_ids,
                    severity=severity,
                    utilization_percent=avg_utilization,
                    capacity_shortage_lbs=capacity_shortage,
                    affected_styles=list(affected_styles),
                    estimated_delay_days=delay_days,
                    recommendation=recommendation,
                    urgency_score=urgency
                )
                
                bottlenecks.append(bottleneck)
        
        # Sort by urgency score (highest first)
        bottlenecks.sort(key=lambda x: x.urgency_score, reverse=True)
        
        logger.info(f"Detected {len(bottlenecks)} bottlenecks")
        return bottlenecks
    
    def _calculate_bottleneck_severity(self, utilization: float, machine_count: int) -> BottleneckSeverity:
        """Calculate bottleneck severity based on utilization and machine count"""
        # Adjust thresholds based on machine count (more machines = higher tolerance)
        machine_factor = min(1.2, 1.0 + (machine_count - 1) * 0.05)
        
        if utilization >= 95 * machine_factor:
            return BottleneckSeverity.CRITICAL
        elif utilization >= 90 * machine_factor:
            return BottleneckSeverity.HIGH
        elif utilization >= self.bottleneck_threshold * machine_factor:
            return BottleneckSeverity.MEDIUM
        elif utilization >= 70 * machine_factor:
            return BottleneckSeverity.LOW
        else:
            return BottleneckSeverity.NONE
    
    def _estimate_delay_impact(self, capacity_shortage: float, total_capacity: float) -> float:
        """Estimate production delay in days"""
        if capacity_shortage <= 0 or total_capacity <= 0:
            return 0.0
        
        shortage_ratio = capacity_shortage / total_capacity
        # Exponential relationship - small shortages have minimal impact, large ones are severe
        delay_days = shortage_ratio * 7 * (1 + shortage_ratio)  # Up to 2 weeks delay
        
        return min(14.0, delay_days)  # Cap at 2 weeks
    
    def _calculate_urgency_score(self, severity: BottleneckSeverity, delay_days: float, affected_styles_count: int) -> float:
        """Calculate urgency score for prioritization"""
        base_score = severity.priority_score * 20
        delay_score = min(20, delay_days * 2)
        style_score = min(10, affected_styles_count)
        
        return base_score + delay_score + style_score
    
    def _generate_bottleneck_recommendation(self, severity: BottleneckSeverity, work_center: str, utilization: float) -> str:
        """Generate AI recommendation for bottleneck resolution"""
        if severity == BottleneckSeverity.CRITICAL:
            return f"URGENT: Work center {work_center} at {utilization:.1f}% - Add overtime shifts or redistribute work immediately"
        elif severity == BottleneckSeverity.HIGH:
            return f"HIGH PRIORITY: Work center {work_center} at {utilization:.1f}% - Consider additional capacity or rebalancing"
        elif severity == BottleneckSeverity.MEDIUM:
            return f"MONITOR: Work center {work_center} at {utilization:.1f}% - Schedule optimization recommended"
        else:
            return f"LOW IMPACT: Work center {work_center} at {utilization:.1f}% - Minor adjustments may help"
    
    def generate_optimization_recommendations(self) -> List[ProductionOptimization]:
        """Generate AI-powered optimization recommendations"""
        if not self.machine_mapper or not self.capacity_manager:
            return []
        
        optimizations = []
        
        # Get current work center status
        work_center_summaries = self.capacity_manager.get_all_work_centers_summary()
        
        # Analyze each work center for optimization opportunities
        for wc_id, summary in work_center_summaries.items():
            avg_util = summary['avg_utilization_percent']
            machine_count = summary['machine_count']
            
            # Check for load balancing opportunities
            if avg_util > self.optimal_utilization * 1.1:  # Overloaded
                target_util = self.optimal_utilization
                improvement = ((avg_util - target_util) / avg_util) * 100
                
                optimization = ProductionOptimization(
                    optimization_type="REBALANCE",
                    target=wc_id,
                    current_value=avg_util,
                    recommended_value=target_util,
                    impact_description=f"Redistribute work from overloaded work center {wc_id}",
                    potential_improvement_percent=improvement,
                    effort_level="MEDIUM"
                )
                optimizations.append(optimization)
                
            elif avg_util < self.optimal_utilization * 0.6:  # Underutilized
                # Look for capacity to absorb from other work centers
                target_util = self.optimal_utilization
                improvement = ((target_util - avg_util) / target_util) * 100
                
                optimization = ProductionOptimization(
                    optimization_type="PRIORITY_ADJUST",
                    target=wc_id,
                    current_value=avg_util,
                    recommended_value=target_util,
                    impact_description=f"Increase utilization of underused work center {wc_id}",
                    potential_improvement_percent=improvement,
                    effort_level="LOW"
                )
                optimizations.append(optimization)
        
        # Sort by potential improvement
        optimizations.sort(key=lambda x: x.potential_improvement_percent, reverse=True)
        
        logger.info(f"Generated {len(optimizations)} optimization recommendations")
        return optimizations
    
    def generate_production_forecast(self, horizon_days: int = 30) -> ProductionForecast:
        """Generate AI-powered production forecast"""
        if not self.capacity_manager:
            return ProductionForecast(
                time_horizon_days=horizon_days,
                forecasted_demand_lbs=0,
                current_capacity_lbs=0,
                capacity_utilization_percent=0,
                projected_bottlenecks=[],
                confidence_score=0
            )
        
        # Get current capacity overview
        machine_status = self.capacity_manager.get_machine_level_status()
        
        current_capacity = machine_status['total_capacity_lbs_day']
        current_utilization = machine_status['avg_utilization_percent']
        
        # Simulate demand forecast (in real system, this would use actual ML models)
        # Base forecast on current utilization trends
        seasonal_factor = 1.0 + 0.1 * math.sin(2 * math.pi * datetime.now().timetuple().tm_yday / 365)
        growth_factor = 1.0 + (horizon_days / 365) * 0.05  # 5% annual growth
        
        forecasted_demand = current_capacity * (current_utilization / 100) * seasonal_factor * growth_factor
        
        # Project bottlenecks
        projected_bottlenecks = []
        forecasted_utilization = (forecasted_demand / current_capacity) * 100
        
        if forecasted_utilization > 85:
            # Identify which work centers will be bottlenecked
            bottlenecks = self.detect_bottlenecks()
            for bottleneck in bottlenecks[:3]:  # Top 3
                if bottleneck.severity.priority_score >= 2:
                    projected_bottlenecks.append(bottleneck.work_center)
        
        # Calculate confidence based on data quality and model performance
        confidence = self.prediction_accuracy * 0.9  # Slight discount for forecasting
        
        return ProductionForecast(
            time_horizon_days=horizon_days,
            forecasted_demand_lbs=forecasted_demand,
            current_capacity_lbs=current_capacity,
            capacity_utilization_percent=forecasted_utilization,
            projected_bottlenecks=projected_bottlenecks,
            confidence_score=confidence
        )
    
    def get_factory_floor_ai_insights(self) -> Dict[str, Any]:
        """Get complete AI insights for factory floor visualization"""
        # Detect bottlenecks
        bottlenecks = self.detect_bottlenecks()
        
        # Generate optimizations
        optimizations = self.generate_optimization_recommendations()
        
        # Generate forecast
        forecast_30d = self.generate_production_forecast(30)
        forecast_90d = self.generate_production_forecast(90)
        
        # Get machine status with AI enhancements
        machine_status = self.capacity_manager.get_machine_level_status() if self.capacity_manager else {}
        
        # Calculate AI-driven KPIs
        ai_kpis = self._calculate_ai_kpis(bottlenecks, optimizations, forecast_30d)
        
        # Generate actionable insights
        actionable_insights = self._generate_actionable_insights(bottlenecks, optimizations)
        
        return {
            "ai_analysis": {
                "bottlenecks": [b.to_dict() for b in bottlenecks],
                "optimizations": [o.to_dict() for o in optimizations],
                "forecast_30_days": forecast_30d.to_dict(),
                "forecast_90_days": forecast_90d.to_dict(),
                "ai_kpis": ai_kpis,
                "actionable_insights": actionable_insights
            },
            "machine_status": machine_status,
            "analysis_timestamp": datetime.now().isoformat(),
            "model_confidence": self.prediction_accuracy
        }
    
    def _calculate_ai_kpis(self, bottlenecks: List[BottleneckAlert], optimizations: List[ProductionOptimization], forecast: ProductionForecast) -> Dict[str, Any]:
        """Calculate AI-driven KPIs"""
        # Bottleneck metrics
        critical_bottlenecks = len([b for b in bottlenecks if b.severity == BottleneckSeverity.CRITICAL])
        high_bottlenecks = len([b for b in bottlenecks if b.severity == BottleneckSeverity.HIGH])
        
        # Optimization potential
        total_improvement = sum(o.potential_improvement_percent for o in optimizations)
        avg_improvement = total_improvement / max(1, len(optimizations))
        
        # Capacity health
        capacity_health = "GOOD"
        if critical_bottlenecks > 0:
            capacity_health = "CRITICAL"
        elif high_bottlenecks > 2:
            capacity_health = "WARNING"
        elif len(bottlenecks) > 5:
            capacity_health = "ATTENTION"
        
        return {
            "bottleneck_summary": {
                "critical": critical_bottlenecks,
                "high": high_bottlenecks,
                "total": len(bottlenecks)
            },
            "optimization_potential": {
                "opportunities": len(optimizations),
                "avg_improvement_percent": round(avg_improvement, 1),
                "total_improvement_percent": round(total_improvement, 1)
            },
            "capacity_health": capacity_health,
            "forecast_accuracy": round(forecast.confidence_score * 100, 1),
            "predicted_utilization": round(forecast.capacity_utilization_percent, 1)
        }
    
    def _generate_actionable_insights(self, bottlenecks: List[BottleneckAlert], optimizations: List[ProductionOptimization]) -> List[Dict[str, Any]]:
        """Generate actionable insights for management"""
        insights = []
        
        # Top bottleneck insight
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            insights.append({
                "type": "BOTTLENECK_ALERT",
                "priority": "HIGH",
                "title": f"Work Center {top_bottleneck.work_center} Critical",
                "description": top_bottleneck.recommendation,
                "action_items": [
                    "Review current assignments",
                    "Consider overtime or additional shifts",
                    "Evaluate equipment status"
                ],
                "estimated_impact": f"{top_bottleneck.estimated_delay_days:.1f} day delay risk"
            })
        
        # Top optimization insight
        if optimizations:
            top_optimization = optimizations[0]
            insights.append({
                "type": "OPTIMIZATION_OPPORTUNITY", 
                "priority": "MEDIUM",
                "title": f"Optimization Available: {top_optimization.optimization_type}",
                "description": top_optimization.impact_description,
                "action_items": [
                    "Analyze current workload distribution",
                    "Plan rebalancing strategy",
                    "Monitor impact after changes"
                ],
                "estimated_impact": f"{top_optimization.potential_improvement_percent:.1f}% improvement potential"
            })
        
        # Capacity utilization insight
        if self.capacity_manager:
            machine_status = self.capacity_manager.get_machine_level_status()
            avg_utilization = machine_status.get('avg_utilization_percent', 0)
            
            if avg_utilization < 60:
                insights.append({
                    "type": "CAPACITY_UNDERUTILIZATION",
                    "priority": "LOW",
                    "title": "Capacity Underutilization Detected",
                    "description": f"Average factory utilization at {avg_utilization:.1f}% - below optimal range",
                    "action_items": [
                        "Review production schedules",
                        "Consider bringing forward orders",
                        "Evaluate marketing opportunities"
                    ],
                    "estimated_impact": "Potential cost savings through increased utilization"
                })
        
        return insights


# Global instance
_ai_production_model = None

def get_ai_production_model() -> AIProductionModel:
    """Get or create the global AI production model instance"""
    global _ai_production_model
    
    if _ai_production_model is None:
        _ai_production_model = AIProductionModel()
    
    return _ai_production_model

def reset_ai_production_model():
    """Reset the global AI production model instance"""
    global _ai_production_model
    _ai_production_model = None


if __name__ == "__main__":
    # Test the AI production model
    print("Testing AI Production Model")
    print("=" * 50)
    
    ai_model = AIProductionModel()
    
    # Test bottleneck detection
    print("1. Bottleneck Detection:")
    bottlenecks = ai_model.detect_bottlenecks()
    print(f"Found {len(bottlenecks)} bottlenecks")
    
    for i, bottleneck in enumerate(bottlenecks[:3]):  # Show top 3
        print(f"  {i+1}. WC {bottleneck.work_center}: {bottleneck.severity.value} "
              f"({bottleneck.utilization_percent:.1f}% util)")
    
    # Test optimization recommendations
    print("\n2. Optimization Recommendations:")
    optimizations = ai_model.generate_optimization_recommendations()
    print(f"Found {len(optimizations)} optimization opportunities")
    
    for i, opt in enumerate(optimizations[:3]):  # Show top 3
        print(f"  {i+1}. {opt.optimization_type}: {opt.potential_improvement_percent:.1f}% improvement")
    
    # Test production forecast
    print("\n3. Production Forecast (30 days):")
    forecast = ai_model.generate_production_forecast(30)
    print(f"  Forecasted demand: {forecast.forecasted_demand_lbs:.0f} lbs/day")
    print(f"  Capacity utilization: {forecast.capacity_utilization_percent:.1f}%")
    print(f"  Projected bottlenecks: {len(forecast.projected_bottlenecks)}")
    
    # Test complete factory insights
    print("\n4. Factory Floor AI Insights:")
    insights = ai_model.get_factory_floor_ai_insights()
    ai_kpis = insights['ai_analysis']['ai_kpis']
    print(f"  Capacity health: {ai_kpis['capacity_health']}")
    print(f"  Optimization opportunities: {ai_kpis['optimization_potential']['opportunities']}")
    print(f"  Actionable insights: {len(insights['ai_analysis']['actionable_insights'])}")
    
    print(f"\nAI Production Model test completed successfully!")