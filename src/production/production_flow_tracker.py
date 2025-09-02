#!/usr/bin/env python3
"""
Production Flow Tracking System for Beverly Knits ERP
Implements the complete production flow from yarn to finished goods
Based on: YARN → G00 (Knit) → G02 (Finishing) → I01 (Inspection) → F01 (Available) → P01 (Allocated)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum

class ProductionStage(Enum):
    """Production stage enumeration"""
    YARN = "YARN"  # Raw yarn inventory
    G00 = "G00"    # Knitting - yarn consumption point
    G02 = "G02"    # Finishing/Dyeing
    I01 = "I01"    # Quality Inspection
    F01 = "F01"    # Finished Goods (Available)
    P01 = "P01"    # Allocated to customer orders
    
    @property
    def description(self):
        descriptions = {
            "YARN": "Raw Yarn Inventory",
            "G00": "Fabric Knitting (Raw Knitted Fabric)",
            "G02": "Fabric Finishing (Dyeing/Processing)",
            "I01": "Final Inspection (Quality Control)",
            "F01": "Available Fabric (Finished Goods)",
            "P01": "Allocated Fabric (Customer Orders)"
        }
        return descriptions.get(self.value, self.value)
    
    @property
    def is_wip(self):
        """Check if stage is work-in-process"""
        return self in [ProductionStage.G00, ProductionStage.G02, ProductionStage.I01]
    
    @property
    def is_finished(self):
        """Check if stage is finished goods"""
        return self in [ProductionStage.F01, ProductionStage.P01]


@dataclass
class StageTransition:
    """Represents a transition between production stages"""
    from_stage: ProductionStage
    to_stage: ProductionStage
    quantity: float
    style_id: str
    timestamp: datetime
    batch_id: Optional[str] = None
    notes: Optional[str] = None
    
    @property
    def lead_time_days(self) -> int:
        """Standard lead time for this transition"""
        lead_times = {
            (ProductionStage.YARN, ProductionStage.G00): 1,
            (ProductionStage.G00, ProductionStage.G02): 7,
            (ProductionStage.G02, ProductionStage.I01): 3,
            (ProductionStage.I01, ProductionStage.F01): 1,
            (ProductionStage.F01, ProductionStage.P01): 0
        }
        return lead_times.get((self.from_stage, self.to_stage), 0)


@dataclass
class ProductionBatch:
    """Represents a production batch moving through stages"""
    batch_id: str
    style_id: str
    start_quantity: float
    current_quantity: float
    current_stage: ProductionStage
    start_date: datetime
    target_date: datetime
    yarn_consumed: Dict[str, float]  # Yarn ID -> quantity consumed
    stage_history: List[StageTransition]
    quality_metrics: Dict[str, Any]
    
    @property
    def yield_rate(self) -> float:
        """Calculate yield rate from start to current"""
        if self.start_quantity > 0:
            return self.current_quantity / self.start_quantity
        return 0
    
    @property
    def days_in_production(self) -> int:
        """Calculate days in production"""
        return (datetime.now() - self.start_date).days
    
    def get_stage_quantity(self, stage: ProductionStage) -> float:
        """Get quantity at a specific stage"""
        for transition in reversed(self.stage_history):
            if transition.to_stage == stage:
                return transition.quantity
        return 0


class ProductionFlowTracker:
    """Main production flow tracking system"""
    
    def __init__(self, data_path: str = None):
        self.data_path = Path(data_path) if data_path else Path(".")
        self.production_batches: Dict[str, ProductionBatch] = {}
        self.stage_inventory: Dict[ProductionStage, Dict[str, float]] = {
            stage: {} for stage in ProductionStage
        }
        self.yarn_allocations: Dict[str, float] = {}  # Yarn ID -> allocated quantity
        self.capacity_constraints = self._initialize_capacity_constraints()
        
    def _initialize_capacity_constraints(self) -> Dict[ProductionStage, float]:
        """Initialize daily capacity constraints by stage"""
        return {
            ProductionStage.G00: 5000,  # lbs/day knitting capacity
            ProductionStage.G02: 4000,  # lbs/day finishing capacity
            ProductionStage.I01: 6000,  # lbs/day inspection capacity
            ProductionStage.F01: float('inf'),  # No capacity limit
            ProductionStage.P01: float('inf')   # No capacity limit
        }
    
    def create_production_order(self, style_id: str, quantity: float, 
                               bom_data: Dict[str, float], 
                               target_date: datetime) -> ProductionBatch:
        """
        Create a new production order and allocate yarn
        
        Args:
            style_id: Style identifier
            quantity: Fabric quantity to produce (yards)
            bom_data: Dict of yarn_id -> percentage for this style
            target_date: Target completion date
            
        Returns:
            ProductionBatch object
        """
        batch_id = f"PB-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{style_id}"
        
        # Calculate yarn requirements
        yarn_requirements = self._calculate_yarn_requirements(quantity, bom_data)
        
        # Check yarn availability
        yarn_available = self._check_yarn_availability(yarn_requirements)
        if not yarn_available:
            raise ValueError(f"Insufficient yarn for style {style_id}")
        
        # Allocate yarn
        self._allocate_yarn(yarn_requirements, batch_id)
        
        # Create production batch
        batch = ProductionBatch(
            batch_id=batch_id,
            style_id=style_id,
            start_quantity=quantity,
            current_quantity=quantity,
            current_stage=ProductionStage.YARN,
            start_date=datetime.now(),
            target_date=target_date,
            yarn_consumed=yarn_requirements,
            stage_history=[],
            quality_metrics={}
        )
        
        self.production_batches[batch_id] = batch
        return batch
    
    def _calculate_yarn_requirements(self, fabric_quantity: float, 
                                    bom_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate yarn requirements based on BOM"""
        requirements = {}
        for yarn_id, percentage in bom_data.items():
            # Convert fabric yards to yarn pounds using standard conversion
            yarn_pounds = fabric_quantity * percentage * 0.45  # Standard conversion factor
            requirements[yarn_id] = yarn_pounds
        return requirements
    
    def _check_yarn_availability(self, requirements: Dict[str, float]) -> bool:
        """Check if required yarn is available"""
        for yarn_id, required_qty in requirements.items():
            available = self.get_yarn_planning_balance(yarn_id)
            if available < required_qty:
                return False
        return True
    
    def _allocate_yarn(self, requirements: Dict[str, float], batch_id: str):
        """Allocate yarn to production batch"""
        for yarn_id, quantity in requirements.items():
            if yarn_id not in self.yarn_allocations:
                self.yarn_allocations[yarn_id] = 0
            self.yarn_allocations[yarn_id] += quantity
            
            # Update stage inventory
            if yarn_id not in self.stage_inventory[ProductionStage.YARN]:
                self.stage_inventory[ProductionStage.YARN][yarn_id] = 0
            self.stage_inventory[ProductionStage.YARN][yarn_id] -= quantity
    
    def get_yarn_planning_balance(self, yarn_id: str) -> float:
        """
        Get yarn planning balance
        Formula: Current_Inventory + On_Order - Allocated_to_G00_WIP
        """
        # This would integrate with the main inventory system
        # For now, return a mock value
        return 1000.0 - self.yarn_allocations.get(yarn_id, 0)
    
    def move_batch_to_stage(self, batch_id: str, target_stage: ProductionStage,
                           quantity: Optional[float] = None,
                           quality_pass_rate: float = 1.0) -> StageTransition:
        """
        Move a production batch to the next stage
        
        Args:
            batch_id: Batch identifier
            target_stage: Target production stage
            quantity: Quantity to move (None = all)
            quality_pass_rate: Quality pass rate (for I01 -> F01)
            
        Returns:
            StageTransition object
        """
        if batch_id not in self.production_batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        batch = self.production_batches[batch_id]
        
        # Validate stage transition
        valid_transitions = self._get_valid_transitions(batch.current_stage)
        if target_stage not in valid_transitions:
            raise ValueError(f"Invalid transition from {batch.current_stage} to {target_stage}")
        
        # Check capacity constraints
        if not self._check_capacity(target_stage, quantity or batch.current_quantity):
            raise ValueError(f"Insufficient capacity at stage {target_stage}")
        
        # Calculate quantity to move
        move_quantity = quantity or batch.current_quantity
        
        # Apply quality loss if moving to F01
        if target_stage == ProductionStage.F01:
            move_quantity *= quality_pass_rate
        
        # Create transition record
        transition = StageTransition(
            from_stage=batch.current_stage,
            to_stage=target_stage,
            quantity=move_quantity,
            style_id=batch.style_id,
            timestamp=datetime.now(),
            batch_id=batch_id
        )
        
        # Update batch
        batch.current_stage = target_stage
        batch.current_quantity = move_quantity
        batch.stage_history.append(transition)
        
        # Update stage inventory
        self._update_stage_inventory(batch.style_id, batch.current_stage, 
                                    target_stage, move_quantity)
        
        # Special handling for G00 (yarn consumption point)
        if target_stage == ProductionStage.G00:
            self._record_yarn_consumption(batch)
        
        return transition
    
    def _get_valid_transitions(self, current_stage: ProductionStage) -> List[ProductionStage]:
        """Get valid next stages from current stage"""
        transitions = {
            ProductionStage.YARN: [ProductionStage.G00],
            ProductionStage.G00: [ProductionStage.G02],
            ProductionStage.G02: [ProductionStage.I01],
            ProductionStage.I01: [ProductionStage.F01],
            ProductionStage.F01: [ProductionStage.P01],
            ProductionStage.P01: []
        }
        return transitions.get(current_stage, [])
    
    def _check_capacity(self, stage: ProductionStage, quantity: float) -> bool:
        """Check if stage has capacity for quantity"""
        daily_capacity = self.capacity_constraints.get(stage, float('inf'))
        current_wip = sum(self.stage_inventory.get(stage, {}).values())
        return (current_wip + quantity) <= daily_capacity
    
    def _update_stage_inventory(self, style_id: str, from_stage: ProductionStage,
                               to_stage: ProductionStage, quantity: float):
        """Update inventory levels at stages"""
        # Remove from source stage
        if from_stage != ProductionStage.YARN:  # Yarn already handled in allocation
            if style_id in self.stage_inventory[from_stage]:
                self.stage_inventory[from_stage][style_id] -= quantity
                if self.stage_inventory[from_stage][style_id] <= 0:
                    del self.stage_inventory[from_stage][style_id]
        
        # Add to target stage
        if style_id not in self.stage_inventory[to_stage]:
            self.stage_inventory[to_stage][style_id] = 0
        self.stage_inventory[to_stage][style_id] += quantity
    
    def _record_yarn_consumption(self, batch: ProductionBatch):
        """Record actual yarn consumption when entering G00"""
        # This is where yarn is physically consumed
        # Update inventory system with actual consumption
        for yarn_id, quantity in batch.yarn_consumed.items():
            # Record consumption in tracking system
            pass  # Would integrate with main inventory system
    
    def get_production_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of production pipeline"""
        pipeline_status = {
            "stages": {},
            "wip_summary": {},
            "capacity_utilization": {},
            "bottlenecks": [],
            "total_wip_value": 0
        }
        
        # Calculate stage-wise inventory
        for stage in ProductionStage:
            stage_data = self.stage_inventory.get(stage, {})
            total_quantity = sum(stage_data.values())
            
            pipeline_status["stages"][stage.value] = {
                "description": stage.description,
                "total_quantity": total_quantity,
                "styles": len(stage_data),
                "is_wip": stage.is_wip,
                "details": stage_data
            }
            
            if stage.is_wip:
                pipeline_status["wip_summary"][stage.value] = total_quantity
            
            # Calculate capacity utilization
            if stage in self.capacity_constraints:
                capacity = self.capacity_constraints[stage]
                utilization = (total_quantity / capacity * 100) if capacity > 0 else 0
                pipeline_status["capacity_utilization"][stage.value] = utilization
                
                # Identify bottlenecks (>80% utilization)
                if utilization > 80:
                    pipeline_status["bottlenecks"].append({
                        "stage": stage.value,
                        "utilization": utilization,
                        "quantity": total_quantity,
                        "capacity": capacity
                    })
        
        return pipeline_status
    
    def get_batch_tracking(self, batch_id: str) -> Dict[str, Any]:
        """Get detailed tracking for a specific batch"""
        if batch_id not in self.production_batches:
            return {"error": f"Batch {batch_id} not found"}
        
        batch = self.production_batches[batch_id]
        
        return {
            "batch_id": batch.batch_id,
            "style_id": batch.style_id,
            "current_stage": batch.current_stage.value,
            "current_quantity": batch.current_quantity,
            "start_quantity": batch.start_quantity,
            "yield_rate": batch.yield_rate,
            "days_in_production": batch.days_in_production,
            "target_date": batch.target_date.isoformat(),
            "yarn_consumed": batch.yarn_consumed,
            "stage_history": [
                {
                    "from": t.from_stage.value,
                    "to": t.to_stage.value,
                    "quantity": t.quantity,
                    "timestamp": t.timestamp.isoformat()
                }
                for t in batch.stage_history
            ]
        }
    
    def calculate_f01_replenishment_needs(self, safety_stock_days: int = 20) -> Dict[str, float]:
        """
        Calculate F01 replenishment needs based on safety stock
        
        Args:
            safety_stock_days: Days of safety stock required at F01
            
        Returns:
            Dict of style_id -> replenishment quantity needed
        """
        replenishment_needs = {}
        
        # Get current F01 inventory
        f01_inventory = self.stage_inventory.get(ProductionStage.F01, {})
        
        for style_id, current_stock in f01_inventory.items():
            # Calculate daily demand (would come from sales forecasting)
            daily_demand = self._get_daily_demand(style_id)
            
            # Calculate required safety stock
            required_stock = daily_demand * safety_stock_days
            
            # Calculate replenishment need
            if current_stock < required_stock:
                replenishment_needs[style_id] = required_stock - current_stock
        
        return replenishment_needs
    
    def _get_daily_demand(self, style_id: str) -> float:
        """Get forecasted daily demand for a style"""
        # This would integrate with the forecasting engine
        # For now, return a mock value
        return 50.0
    
    def generate_production_plan(self, replenishment_needs: Dict[str, float],
                                bom_data: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Generate production plan based on F01 replenishment needs
        
        Args:
            replenishment_needs: Dict of style_id -> quantity needed
            bom_data: Dict of style_id -> yarn requirements
            
        Returns:
            List of production orders to create
        """
        production_plan = []
        
        for style_id, quantity_needed in replenishment_needs.items():
            # Check yarn availability for this style
            if style_id not in bom_data:
                continue
            
            yarn_requirements = self._calculate_yarn_requirements(
                quantity_needed, bom_data[style_id]
            )
            
            # Calculate production timeline
            total_lead_time = 12  # Days from G00 to F01
            start_date = datetime.now()
            target_date = start_date + timedelta(days=total_lead_time)
            
            production_order = {
                "style_id": style_id,
                "quantity": quantity_needed,
                "yarn_requirements": yarn_requirements,
                "start_date": start_date.isoformat(),
                "target_date": target_date.isoformat(),
                "priority": self._calculate_priority(style_id, quantity_needed)
            }
            
            production_plan.append(production_order)
        
        # Sort by priority
        production_plan.sort(key=lambda x: x["priority"], reverse=True)
        
        return production_plan
    
    def _calculate_priority(self, style_id: str, quantity: float) -> int:
        """Calculate production priority"""
        # Priority based on quantity and current stock levels
        f01_stock = self.stage_inventory.get(ProductionStage.F01, {}).get(style_id, 0)
        
        if f01_stock == 0:
            return 100  # Critical - no stock
        elif f01_stock < quantity * 0.2:
            return 80   # High - low stock
        elif f01_stock < quantity * 0.5:
            return 60   # Medium
        else:
            return 40   # Low
    
    def get_yarn_impact_analysis(self, production_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze yarn impact of production plan
        
        Args:
            production_plan: List of production orders
            
        Returns:
            Yarn requirement analysis
        """
        total_yarn_requirements = {}
        yarn_shortages = {}
        
        for order in production_plan:
            for yarn_id, quantity in order["yarn_requirements"].items():
                if yarn_id not in total_yarn_requirements:
                    total_yarn_requirements[yarn_id] = 0
                total_yarn_requirements[yarn_id] += quantity
        
        # Check against available yarn
        for yarn_id, required in total_yarn_requirements.items():
            available = self.get_yarn_planning_balance(yarn_id)
            if available < required:
                yarn_shortages[yarn_id] = {
                    "required": required,
                    "available": available,
                    "shortage": required - available
                }
        
        return {
            "total_requirements": total_yarn_requirements,
            "shortages": yarn_shortages,
            "can_execute": len(yarn_shortages) == 0
        }
    
    def export_production_metrics(self) -> Dict[str, Any]:
        """Export comprehensive production metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_status": self.get_production_pipeline_status(),
            "active_batches": len(self.production_batches),
            "stage_metrics": {},
            "lead_times": {},
            "quality_metrics": {}
        }
        
        # Calculate stage-specific metrics
        for stage in ProductionStage:
            if stage.is_wip:
                batches_in_stage = [
                    b for b in self.production_batches.values() 
                    if b.current_stage == stage
                ]
                
                metrics["stage_metrics"][stage.value] = {
                    "batch_count": len(batches_in_stage),
                    "total_quantity": sum(b.current_quantity for b in batches_in_stage),
                    "average_age_days": np.mean([b.days_in_production for b in batches_in_stage]) if batches_in_stage else 0
                }
        
        # Calculate average lead times
        for batch in self.production_batches.values():
            for i in range(len(batch.stage_history) - 1):
                transition = batch.stage_history[i]
                next_transition = batch.stage_history[i + 1]
                lead_time = (next_transition.timestamp - transition.timestamp).days
                
                key = f"{transition.to_stage.value}_to_{next_transition.to_stage.value}"
                if key not in metrics["lead_times"]:
                    metrics["lead_times"][key] = []
                metrics["lead_times"][key].append(lead_time)
        
        # Average lead times
        for key, times in metrics["lead_times"].items():
            metrics["lead_times"][key] = {
                "average": np.mean(times),
                "min": min(times),
                "max": max(times)
            }
        
        # Calculate quality metrics
        total_yield = []
        for batch in self.production_batches.values():
            if batch.current_stage in [ProductionStage.F01, ProductionStage.P01]:
                total_yield.append(batch.yield_rate)
        
        metrics["quality_metrics"] = {
            "average_yield": np.mean(total_yield) if total_yield else 1.0,
            "batches_inspected": len(total_yield)
        }
        
        return metrics


if __name__ == "__main__":
    """Test the production flow tracking system"""
    
    print("Production Flow Tracking System Test")
    print("=" * 50)
    
    # Initialize tracker
    tracker = ProductionFlowTracker()
    
    # Create sample BOM data
    bom_data = {
        "STYLE001": {
            "19003": 0.45,  # 45% of yarn 19003
            "19004": 0.35,  # 35% of yarn 19004
            "19005": 0.20   # 20% of yarn 19005
        }
    }
    
    # Create a production order
    print("\n1. Creating production order...")
    try:
        batch = tracker.create_production_order(
            style_id="STYLE001",
            quantity=1000,  # 1000 yards of fabric
            bom_data=bom_data["STYLE001"],
            target_date=datetime.now() + timedelta(days=14)
        )
        print(f"   Created batch: {batch.batch_id}")
        print(f"   Yarn consumed: {batch.yarn_consumed}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Move batch through stages
    print("\n2. Moving batch through production stages...")
    stages = [ProductionStage.G00, ProductionStage.G02, ProductionStage.I01, ProductionStage.F01]
    
    for stage in stages:
        try:
            transition = tracker.move_batch_to_stage(
                batch.batch_id, 
                stage,
                quality_pass_rate=0.95 if stage == ProductionStage.F01 else 1.0
            )
            print(f"   Moved to {stage.value}: {transition.quantity} units")
        except Exception as e:
            print(f"   Error moving to {stage.value}: {e}")
    
    # Get production pipeline status
    print("\n3. Production Pipeline Status:")
    status = tracker.get_production_pipeline_status()
    for stage_name, stage_data in status["stages"].items():
        if stage_data["total_quantity"] > 0:
            print(f"   {stage_name}: {stage_data['total_quantity']} units")
    
    # Calculate F01 replenishment needs
    print("\n4. F01 Replenishment Analysis:")
    replenishment = tracker.calculate_f01_replenishment_needs(safety_stock_days=20)
    for style, quantity in replenishment.items():
        print(f"   {style}: Need {quantity} units")
    
    # Generate production plan
    print("\n5. Production Plan Generation:")
    plan = tracker.generate_production_plan(replenishment, bom_data)
    for order in plan[:3]:  # Show first 3 orders
        print(f"   Style {order['style_id']}: {order['quantity']} units, Priority: {order['priority']}")
    
    # Export metrics
    print("\n6. Production Metrics:")
    metrics = tracker.export_production_metrics()
    print(f"   Active batches: {metrics['active_batches']}")
    print(f"   Quality yield: {metrics['quality_metrics']['average_yield']:.2%}")
    
    print("\n✅ Production Flow Tracking System Test Complete!")