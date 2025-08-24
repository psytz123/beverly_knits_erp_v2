#!/usr/bin/env python3
"""
Planning Data API Module
Provides a unified API for production planning data and operations
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanningPhase(Enum):
    """Planning phases enumeration"""
    DEMAND_ANALYSIS = "demand_analysis"
    MATERIAL_PLANNING = "material_planning"
    CAPACITY_PLANNING = "capacity_planning"
    PRODUCTION_SCHEDULING = "production_scheduling"
    PROCUREMENT_PLANNING = "procurement_planning"
    EXECUTION_MONITORING = "execution_monitoring"


@dataclass
class PlanningRequest:
    """Standard planning request structure"""
    request_id: str
    request_type: str
    planning_horizon_days: int
    products: Optional[List[str]] = None
    constraints: Optional[Dict[str, Any]] = None
    priority: str = "normal"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PlanningResponse:
    """Standard planning response structure"""
    request_id: str
    status: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PlanningDataAPI:
    """
    Unified API for production planning data and operations
    """
    
    def __init__(self):
        self.planning_cache = {}
        self.active_plans = {}
        self.planning_history = []
        logger.info("PlanningDataAPI initialized")
    
    def get_planning_data(self, request: PlanningRequest) -> PlanningResponse:
        """
        Get planning data based on request parameters
        """
        try:
            # Validate request
            validation_errors = self._validate_request(request)
            if validation_errors:
                return PlanningResponse(
                    request_id=request.request_id,
                    status="error",
                    data={},
                    metadata={"request_type": request.request_type},
                    errors=validation_errors,
                    warnings=[]
                )
            
            # Route to appropriate handler
            if request.request_type == "demand_forecast":
                response_data = self._get_demand_forecast(request)
            elif request.request_type == "material_requirements":
                response_data = self._get_material_requirements(request)
            elif request.request_type == "capacity_analysis":
                response_data = self._get_capacity_analysis(request)
            elif request.request_type == "production_schedule":
                response_data = self._get_production_schedule(request)
            elif request.request_type == "procurement_plan":
                response_data = self._get_procurement_plan(request)
            else:
                response_data = self._get_generic_planning_data(request)
            
            # Create response
            response = PlanningResponse(
                request_id=request.request_id,
                status="success",
                data=response_data,
                metadata={
                    "request_type": request.request_type,
                    "planning_horizon": request.planning_horizon_days,
                    "products_count": len(request.products) if request.products else 0
                },
                errors=[],
                warnings=self._check_warnings(response_data)
            )
            
            # Cache response
            self.planning_cache[request.request_id] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing planning request: {str(e)}")
            return PlanningResponse(
                request_id=request.request_id,
                status="error",
                data={},
                metadata={"request_type": request.request_type},
                errors=[str(e)],
                warnings=[]
            )
    
    def _validate_request(self, request: PlanningRequest) -> List[str]:
        """Validate planning request"""
        errors = []
        
        if not request.request_id:
            errors.append("Request ID is required")
        
        if not request.request_type:
            errors.append("Request type is required")
        
        if request.planning_horizon_days <= 0:
            errors.append("Planning horizon must be positive")
        
        if request.planning_horizon_days > 365:
            errors.append("Planning horizon cannot exceed 365 days")
        
        return errors
    
    def _get_demand_forecast(self, request: PlanningRequest) -> Dict[str, Any]:
        """Get demand forecast data"""
        forecast_data = {
            "forecast_date": datetime.now().isoformat(),
            "horizon_days": request.planning_horizon_days,
            "forecasts": []
        }
        
        # Generate sample forecast data
        products = request.products or [f"PROD_{i:03d}" for i in range(1, 11)]
        
        for product in products:
            forecast = {
                "product_id": product,
                "current_demand": np.random.randint(100, 1000),
                "forecast_periods": []
            }
            
            # Generate daily forecasts
            for day in range(request.planning_horizon_days):
                date = datetime.now() + timedelta(days=day)
                base_demand = forecast["current_demand"]
                
                # Add some variability
                daily_demand = base_demand * (1 + np.random.normal(0, 0.1))
                daily_demand = max(0, int(daily_demand))
                
                forecast["forecast_periods"].append({
                    "date": date.isoformat(),
                    "demand": daily_demand,
                    "confidence_lower": int(daily_demand * 0.8),
                    "confidence_upper": int(daily_demand * 1.2)
                })
            
            # Summary statistics
            all_demands = [p["demand"] for p in forecast["forecast_periods"]]
            forecast["summary"] = {
                "total_demand": sum(all_demands),
                "avg_daily_demand": np.mean(all_demands),
                "peak_demand": max(all_demands),
                "min_demand": min(all_demands)
            }
            
            forecast_data["forecasts"].append(forecast)
        
        # Overall summary
        forecast_data["summary"] = {
            "total_products": len(products),
            "total_forecasted_demand": sum(f["summary"]["total_demand"] for f in forecast_data["forecasts"]),
            "forecast_method": "ensemble",
            "confidence_level": 0.95
        }
        
        return forecast_data
    
    def _get_material_requirements(self, request: PlanningRequest) -> Dict[str, Any]:
        """Get material requirements planning data"""
        mrp_data = {
            "planning_date": datetime.now().isoformat(),
            "horizon_days": request.planning_horizon_days,
            "requirements": []
        }
        
        # Generate sample MRP data
        products = request.products or [f"PROD_{i:03d}" for i in range(1, 6)]
        
        for product in products:
            # Simulate BOM explosion
            materials = [
                {"material_id": f"MAT_{product}_{i}", "quantity_per_unit": np.random.randint(1, 5)}
                for i in range(1, 4)
            ]
            
            requirement = {
                "product_id": product,
                "planned_quantity": np.random.randint(100, 500),
                "material_requirements": []
            }
            
            for material in materials:
                mat_req = {
                    "material_id": material["material_id"],
                    "required_quantity": requirement["planned_quantity"] * material["quantity_per_unit"],
                    "current_stock": np.random.randint(0, 1000),
                    "on_order": np.random.randint(0, 500),
                    "lead_time_days": np.random.randint(7, 30)
                }
                
                # Calculate shortage
                available = mat_req["current_stock"] + mat_req["on_order"]
                mat_req["shortage"] = max(0, mat_req["required_quantity"] - available)
                mat_req["order_date"] = (datetime.now() - timedelta(days=mat_req["lead_time_days"])).isoformat()
                
                requirement["material_requirements"].append(mat_req)
            
            mrp_data["requirements"].append(requirement)
        
        # Summary
        all_materials = []
        for req in mrp_data["requirements"]:
            all_materials.extend(req["material_requirements"])
        
        mrp_data["summary"] = {
            "total_products": len(products),
            "total_materials": len(set(m["material_id"] for m in all_materials)),
            "materials_with_shortage": sum(1 for m in all_materials if m["shortage"] > 0),
            "total_shortage_value": sum(m["shortage"] for m in all_materials)
        }
        
        return mrp_data
    
    def _get_capacity_analysis(self, request: PlanningRequest) -> Dict[str, Any]:
        """Get capacity planning analysis"""
        capacity_data = {
            "analysis_date": datetime.now().isoformat(),
            "horizon_days": request.planning_horizon_days,
            "capacity_analysis": []
        }
        
        # Define work centers
        work_centers = ["WC_CUTTING", "WC_SEWING", "WC_FINISHING", "WC_PACKAGING"]
        
        for wc in work_centers:
            analysis = {
                "work_center": wc,
                "capacity_hours": 8 * request.planning_horizon_days,  # 8 hours per day
                "planned_load": 0,
                "utilization": 0,
                "daily_capacity": []
            }
            
            # Generate daily capacity data
            for day in range(request.planning_horizon_days):
                date = datetime.now() + timedelta(days=day)
                daily_data = {
                    "date": date.isoformat(),
                    "available_hours": 8,
                    "planned_hours": np.random.uniform(4, 10),  # Random load
                    "utilization_pct": 0
                }
                
                daily_data["utilization_pct"] = min(100, (daily_data["planned_hours"] / daily_data["available_hours"]) * 100)
                analysis["daily_capacity"].append(daily_data)
                analysis["planned_load"] += daily_data["planned_hours"]
            
            analysis["utilization"] = (analysis["planned_load"] / analysis["capacity_hours"]) * 100
            analysis["status"] = "overloaded" if analysis["utilization"] > 100 else "optimal" if analysis["utilization"] > 70 else "underutilized"
            
            capacity_data["capacity_analysis"].append(analysis)
        
        # Summary
        capacity_data["summary"] = {
            "work_centers_count": len(work_centers),
            "avg_utilization": np.mean([a["utilization"] for a in capacity_data["capacity_analysis"]]),
            "overloaded_centers": sum(1 for a in capacity_data["capacity_analysis"] if a["status"] == "overloaded"),
            "total_capacity_hours": sum(a["capacity_hours"] for a in capacity_data["capacity_analysis"]),
            "total_planned_hours": sum(a["planned_load"] for a in capacity_data["capacity_analysis"])
        }
        
        return capacity_data
    
    def _get_production_schedule(self, request: PlanningRequest) -> Dict[str, Any]:
        """Get production schedule"""
        schedule_data = {
            "schedule_date": datetime.now().isoformat(),
            "horizon_days": request.planning_horizon_days,
            "scheduled_orders": []
        }
        
        # Generate sample production orders
        products = request.products or [f"PROD_{i:03d}" for i in range(1, 8)]
        
        order_id = 1000
        for product in products:
            # Create multiple orders for each product
            num_orders = np.random.randint(1, 4)
            
            for _ in range(num_orders):
                start_day = np.random.randint(0, max(1, request.planning_horizon_days - 5))
                duration = np.random.randint(1, 5)
                
                order = {
                    "order_id": f"PO_{order_id}",
                    "product_id": product,
                    "quantity": np.random.randint(50, 300),
                    "start_date": (datetime.now() + timedelta(days=start_day)).isoformat(),
                    "end_date": (datetime.now() + timedelta(days=start_day + duration)).isoformat(),
                    "duration_days": duration,
                    "status": np.random.choice(["planned", "confirmed", "in_progress"], p=[0.5, 0.3, 0.2]),
                    "priority": np.random.choice(["low", "normal", "high"], p=[0.2, 0.6, 0.2]),
                    "work_center": np.random.choice(["WC_CUTTING", "WC_SEWING", "WC_FINISHING"])
                }
                
                schedule_data["scheduled_orders"].append(order)
                order_id += 1
        
        # Sort by start date
        schedule_data["scheduled_orders"].sort(key=lambda x: x["start_date"])
        
        # Summary
        schedule_data["summary"] = {
            "total_orders": len(schedule_data["scheduled_orders"]),
            "total_quantity": sum(o["quantity"] for o in schedule_data["scheduled_orders"]),
            "orders_by_status": {},
            "orders_by_priority": {}
        }
        
        # Count by status
        for status in ["planned", "confirmed", "in_progress"]:
            count = sum(1 for o in schedule_data["scheduled_orders"] if o["status"] == status)
            schedule_data["summary"]["orders_by_status"][status] = count
        
        # Count by priority
        for priority in ["low", "normal", "high"]:
            count = sum(1 for o in schedule_data["scheduled_orders"] if o["priority"] == priority)
            schedule_data["summary"]["orders_by_priority"][priority] = count
        
        return schedule_data
    
    def _get_procurement_plan(self, request: PlanningRequest) -> Dict[str, Any]:
        """Get procurement planning data"""
        procurement_data = {
            "plan_date": datetime.now().isoformat(),
            "horizon_days": request.planning_horizon_days,
            "purchase_orders": []
        }
        
        # Generate sample procurement requirements
        materials = [f"MAT_{i:03d}" for i in range(1, 11)]
        
        po_id = 5000
        for material in materials:
            # Determine if procurement is needed
            if np.random.random() > 0.3:  # 70% chance of needing procurement
                po = {
                    "po_id": f"PUR_{po_id}",
                    "material_id": material,
                    "quantity": np.random.randint(100, 1000),
                    "unit_price": np.random.uniform(1, 50),
                    "total_value": 0,
                    "supplier": f"SUPP_{np.random.randint(1, 6):03d}",
                    "order_date": datetime.now().isoformat(),
                    "requested_date": (datetime.now() + timedelta(days=np.random.randint(7, 30))).isoformat(),
                    "lead_time_days": np.random.randint(7, 30),
                    "status": np.random.choice(["draft", "approved", "sent", "confirmed"], p=[0.2, 0.3, 0.3, 0.2]),
                    "urgency": np.random.choice(["normal", "urgent", "critical"], p=[0.7, 0.2, 0.1])
                }
                
                po["total_value"] = po["quantity"] * po["unit_price"]
                procurement_data["purchase_orders"].append(po)
                po_id += 1
        
        # Sort by requested date
        procurement_data["purchase_orders"].sort(key=lambda x: x["requested_date"])
        
        # Summary
        procurement_data["summary"] = {
            "total_orders": len(procurement_data["purchase_orders"]),
            "total_value": sum(po["total_value"] for po in procurement_data["purchase_orders"]),
            "unique_materials": len(set(po["material_id"] for po in procurement_data["purchase_orders"])),
            "unique_suppliers": len(set(po["supplier"] for po in procurement_data["purchase_orders"])),
            "orders_by_status": {},
            "urgent_orders": sum(1 for po in procurement_data["purchase_orders"] if po["urgency"] in ["urgent", "critical"])
        }
        
        # Count by status
        for status in ["draft", "approved", "sent", "confirmed"]:
            count = sum(1 for po in procurement_data["purchase_orders"] if po["status"] == status)
            procurement_data["summary"]["orders_by_status"][status] = count
        
        return procurement_data
    
    def _get_generic_planning_data(self, request: PlanningRequest) -> Dict[str, Any]:
        """Get generic planning data for unspecified request types"""
        return {
            "message": f"Generic planning data for request type: {request.request_type}",
            "request_id": request.request_id,
            "planning_horizon": request.planning_horizon_days,
            "data": {
                "placeholder": "This would contain specific planning data",
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _check_warnings(self, data: Dict[str, Any]) -> List[str]:
        """Check for warnings in planning data"""
        warnings = []
        
        # Check for capacity warnings
        if "capacity_analysis" in data:
            overloaded = data.get("summary", {}).get("overloaded_centers", 0)
            if overloaded > 0:
                warnings.append(f"{overloaded} work centers are overloaded")
        
        # Check for material shortages
        if "requirements" in data:
            shortages = data.get("summary", {}).get("materials_with_shortage", 0)
            if shortages > 0:
                warnings.append(f"{shortages} materials have shortages")
        
        # Check for urgent orders
        if "purchase_orders" in data:
            urgent = data.get("summary", {}).get("urgent_orders", 0)
            if urgent > 0:
                warnings.append(f"{urgent} purchase orders are urgent or critical")
        
        return warnings
    
    def create_planning_session(self, session_name: str, parameters: Dict[str, Any]) -> str:
        """Create a new planning session"""
        session_id = f"SESS_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        session = {
            "session_id": session_id,
            "session_name": session_name,
            "created_at": datetime.now().isoformat(),
            "parameters": parameters,
            "status": "active",
            "phases_completed": []
        }
        
        self.active_plans[session_id] = session
        logger.info(f"Created planning session: {session_id}")
        
        return session_id
    
    def get_planning_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get planning session details"""
        return self.active_plans.get(session_id)
    
    def update_planning_phase(self, session_id: str, phase: PlanningPhase, phase_data: Dict[str, Any]) -> bool:
        """Update planning session with phase results"""
        if session_id not in self.active_plans:
            logger.error(f"Session {session_id} not found")
            return False
        
        session = self.active_plans[session_id]
        
        # Add phase data
        phase_result = {
            "phase": phase.value,
            "completed_at": datetime.now().isoformat(),
            "data": phase_data
        }
        
        session["phases_completed"].append(phase_result)
        
        # Check if all phases complete
        if len(session["phases_completed"]) >= len(PlanningPhase):
            session["status"] = "completed"
            session["completed_at"] = datetime.now().isoformat()
        
        return True
    
    def get_planning_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent planning history"""
        return self.planning_history[-limit:]
    
    def export_planning_data(self, session_id: str, format: str = "json") -> Union[str, pd.DataFrame]:
        """Export planning session data"""
        session = self.get_planning_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if format == "json":
            return json.dumps(session, indent=2, default=str)
        elif format == "dataframe":
            # Convert to DataFrame format
            rows = []
            for phase in session.get("phases_completed", []):
                row = {
                    "session_id": session_id,
                    "session_name": session["session_name"],
                    "phase": phase["phase"],
                    "completed_at": phase["completed_at"]
                }
                rows.append(row)
            return pd.DataFrame(rows)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience functions
def create_planning_request(request_type: str, 
                          horizon_days: int,
                          products: Optional[List[str]] = None,
                          **kwargs) -> PlanningRequest:
    """Create a planning request"""
    return PlanningRequest(
        request_id=f"REQ_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        request_type=request_type,
        planning_horizon_days=horizon_days,
        products=products,
        constraints=kwargs.get("constraints"),
        priority=kwargs.get("priority", "normal")
    )


def get_planning_api() -> PlanningDataAPI:
    """Get or create planning API instance"""
    global _planning_api
    if '_planning_api' not in globals():
        _planning_api = PlanningDataAPI()
    return _planning_api


# Module exports
__all__ = [
    'PlanningDataAPI',
    'PlanningRequest',
    'PlanningResponse',
    'PlanningPhase',
    'create_planning_request',
    'get_planning_api'
]