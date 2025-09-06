#!/usr/bin/env python3
"""
Furniture Manufacturing Specialization Agent
Industry-specific ERP intelligence for furniture manufacturing operations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.agent_base import BaseAgent, AgentMessage, AgentCapability, MessageType, Priority
from ..core.state_manager import CustomerProfile

# Setup logging
logger = logging.getLogger(__name__)


class FurnitureProductType(Enum):
    """Furniture product categories"""
    SEATING = "SEATING"                # Chairs, sofas, benches
    TABLES = "TABLES"                  # Dining, coffee, work tables
    STORAGE = "STORAGE"                # Cabinets, dressers, wardrobes
    BEDROOM = "BEDROOM"                # Beds, nightstands, mattresses
    OFFICE = "OFFICE"                  # Desks, office chairs, filing
    OUTDOOR = "OUTDOOR"                # Patio, garden furniture
    CUSTOM = "CUSTOM"                  # Custom/bespoke pieces


class WoodType(Enum):
    """Wood material classifications"""
    HARDWOOD = "HARDWOOD"              # Oak, maple, cherry, walnut
    SOFTWOOD = "SOFTWOOD"              # Pine, cedar, fir
    ENGINEERED = "ENGINEERED"          # Plywood, MDF, particleboard
    RECLAIMED = "RECLAIMED"            # Recycled wood materials
    COMPOSITE = "COMPOSITE"            # Wood-plastic composites


class FinishType(Enum):
    """Furniture finish categories"""
    STAIN = "STAIN"                    # Wood stains
    PAINT = "PAINT"                    # Painted finishes
    LACQUER = "LACQUER"                # Clear protective coatings
    VENEER = "VENEER"                  # Wood veneer applications
    UPHOLSTERY = "UPHOLSTERY"          # Fabric/leather coverings
    NATURAL = "NATURAL"                # No finish applied


@dataclass
class FurnitureMaterial:
    """Furniture manufacturing material specification"""
    material_id: str
    material_name: str
    material_type: str  # wood, hardware, fabric, foam, etc.
    wood_type: Optional[WoodType] = None
    grade: str = "A"  # A, B, C quality grades
    board_feet_per_unit: float = 0.0
    moisture_content: float = 0.0
    supplier: str = ""
    lead_time_days: int = 7
    cost_per_unit: float = 0.0
    minimum_order_quantity: int = 1
    waste_factor: float = 0.1  # 10% waste factor


@dataclass
class FurnitureDesign:
    """Furniture design specification"""
    design_id: str
    product_name: str
    product_type: FurnitureProductType
    dimensions: Dict[str, float]  # length, width, height in inches
    materials: List[FurnitureMaterial]
    hardware_list: List[Dict[str, Any]]
    finish_requirements: List[FinishType]
    assembly_time_hours: float = 0.0
    skill_level_required: str = "INTERMEDIATE"  # BEGINNER, INTERMEDIATE, ADVANCED
    custom_options: List[str] = field(default_factory=list)


class FurnitureManufacturingAgent(BaseAgent):
    """
    Furniture Manufacturing Specialization Agent
    
    Capabilities:
    - Furniture-specific BOM management and wood yield optimization
    - Manufacturing process planning for woodworking operations
    - Finish scheduling and quality control workflows
    - Custom order configuration and engineering
    - Wood inventory optimization with moisture content tracking
    - Seasonal demand forecasting for furniture categories
    - Supplier management for lumber and hardware vendors
    - Compliance tracking for furniture safety standards
    """
    
    def __init__(self, agent_id: str = "furniture_manufacturing_agent"):
        """Initialize Furniture Manufacturing Agent"""
        super().__init__(
            agent_id=agent_id,
            agent_name="Furniture Manufacturing Specialist",
            agent_description="Industry-specific intelligence for furniture manufacturing operations"
        )
        
        # Furniture-specific knowledge base
        self.wood_yield_factors = {
            WoodType.HARDWOOD: 0.85,     # 85% yield after milling
            WoodType.SOFTWOOD: 0.90,     # 90% yield
            WoodType.ENGINEERED: 0.95,   # 95% yield
            WoodType.RECLAIMED: 0.75,    # 75% yield (more waste)
            WoodType.COMPOSITE: 0.92     # 92% yield
        }
        
        self.finish_cure_times = {
            FinishType.STAIN: 4,         # 4 hours
            FinishType.PAINT: 6,         # 6 hours
            FinishType.LACQUER: 2,       # 2 hours
            FinishType.VENEER: 24,       # 24 hours (glue cure)
            FinishType.UPHOLSTERY: 1,    # 1 hour
            FinishType.NATURAL: 0        # No cure time
        }
        
        self.seasonal_demand_patterns = {
            FurnitureProductType.OUTDOOR: {
                "spring": 1.5, "summer": 1.8, "fall": 0.8, "winter": 0.3
            },
            FurnitureProductType.BEDROOM: {
                "spring": 1.2, "summer": 0.9, "fall": 1.1, "winter": 1.0
            },
            FurnitureProductType.OFFICE: {
                "spring": 1.1, "summer": 0.8, "fall": 1.4, "winter": 1.0
            }
        }
        
        # Manufacturing workflows
        self.woodworking_operations = [
            "lumber_selection", "rough_milling", "kiln_drying",
            "final_milling", "joinery", "assembly", "sanding",
            "finishing", "quality_inspection", "packaging"
        ]
        
        self.quality_checkpoints = {
            "lumber_selection": ["moisture_content", "grain_direction", "defect_inspection"],
            "rough_milling": ["dimension_accuracy", "square_check", "surface_quality"],
            "joinery": ["joint_fit", "glue_line_quality", "alignment"],
            "finishing": ["coverage_uniformity", "cure_completion", "color_match"],
            "final_inspection": ["dimensional_accuracy", "finish_quality", "hardware_function"]
        }
    
    def _initialize(self):
        """Initialize furniture manufacturing capabilities"""
        # Register furniture-specific capabilities
        self.register_capability(AgentCapability(
            name="wood_yield_optimization",
            description="Optimize wood material usage and minimize waste",
            input_schema={
                "type": "object",
                "properties": {
                    "design_specification": {"type": "object"},
                    "available_lumber": {"type": "array"},
                    "waste_tolerance": {"type": "number"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "optimized_cutting_plan": {"type": "object"},
                    "material_requirements": {"type": "array"},
                    "estimated_waste_percentage": {"type": "number"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="furniture_bom_generation",
            description="Generate detailed bill of materials for furniture designs",
            input_schema={
                "type": "object",
                "properties": {
                    "furniture_design": {"type": "object"},
                    "quantity": {"type": "integer"},
                    "customization_options": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "detailed_bom": {"type": "array"},
                    "material_costs": {"type": "object"},
                    "lead_time_analysis": {"type": "object"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="finish_scheduling",
            description="Schedule finishing operations based on cure times and capacity",
            input_schema={
                "type": "object",
                "properties": {
                    "production_orders": {"type": "array"},
                    "finish_booth_capacity": {"type": "object"},
                    "environmental_conditions": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "finish_schedule": {"type": "array"},
                    "bottleneck_analysis": {"type": "object"},
                    "capacity_utilization": {"type": "number"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="custom_configuration",
            description="Configure custom furniture orders with engineering validation",
            input_schema={
                "type": "object",
                "properties": {
                    "base_design": {"type": "object"},
                    "customization_requests": {"type": "array"},
                    "budget_constraints": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "feasibility_analysis": {"type": "object"},
                    "engineering_drawings": {"type": "object"},
                    "cost_impact": {"type": "object"}
                }
            }
        ))
        
        # Register message handlers
        self.register_message_handler(MessageType.REQUEST, self._handle_furniture_request)
    
    async def _handle_furniture_request(self, message: AgentMessage) -> AgentMessage:
        """Handle furniture manufacturing requests"""
        action = message.payload.get("action")
        
        try:
            if action == "optimize_wood_yield":
                result = await self._optimize_wood_yield(message.payload)
            elif action == "generate_furniture_bom":
                result = await self._generate_furniture_bom(message.payload)
            elif action == "schedule_finishing":
                result = await self._schedule_finishing(message.payload)
            elif action == "configure_custom_order":
                result = await self._configure_custom_order(message.payload)
            elif action == "analyze_seasonal_demand":
                result = await self._analyze_seasonal_demand(message.payload)
            else:
                result = {"error": "Unsupported action", "action": action}
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload=result,
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            self.logger.error(f"Error handling furniture request: {str(e)}")
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                correlation_id=message.correlation_id
            )
    
    async def _optimize_wood_yield(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize wood material usage and cutting plans"""
        design = payload.get("design_specification", {})
        available_lumber = payload.get("available_lumber", [])
        waste_tolerance = payload.get("waste_tolerance", 0.1)
        
        # Simulate wood yield optimization
        cutting_plan = {
            "boards_required": [],
            "cutting_sequence": [],
            "waste_pieces": []
        }
        
        # Calculate material requirements with yield factors
        material_requirements = []
        for material in design.get("materials", []):
            if "wood_type" in material:
                wood_type = WoodType(material["wood_type"])
                yield_factor = self.wood_yield_factors.get(wood_type, 0.85)
                
                required_board_feet = material.get("board_feet_per_unit", 0)
                adjusted_requirement = required_board_feet / yield_factor
                
                material_requirements.append({
                    "material_id": material["material_id"],
                    "raw_board_feet_needed": adjusted_requirement,
                    "finished_board_feet": required_board_feet,
                    "yield_factor": yield_factor,
                    "estimated_waste": adjusted_requirement - required_board_feet
                })
        
        total_waste = sum(mat["estimated_waste"] for mat in material_requirements)
        total_material = sum(mat["raw_board_feet_needed"] for mat in material_requirements)
        waste_percentage = (total_waste / total_material * 100) if total_material > 0 else 0
        
        return {
            "optimized_cutting_plan": cutting_plan,
            "material_requirements": material_requirements,
            "estimated_waste_percentage": waste_percentage,
            "yield_optimization_suggestions": [
                "Use engineered lumber for non-visible components",
                "Optimize grain direction for strength requirements",
                "Consider board width optimization for yield improvement"
            ]
        }
    
    async def _generate_furniture_bom(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed bill of materials for furniture"""
        furniture_design = payload.get("furniture_design", {})
        quantity = payload.get("quantity", 1)
        customizations = payload.get("customization_options", {})
        
        # Base BOM from design
        base_materials = furniture_design.get("materials", [])
        hardware_list = furniture_design.get("hardware_list", [])
        
        # Calculate quantities
        detailed_bom = []
        total_cost = 0.0
        
        for material in base_materials:
            unit_cost = material.get("cost_per_unit", 0.0)
            units_needed = material.get("units_per_piece", 1) * quantity
            extended_cost = unit_cost * units_needed
            
            detailed_bom.append({
                "item_id": material.get("material_id"),
                "description": material.get("material_name"),
                "category": material.get("material_type"),
                "quantity_needed": units_needed,
                "unit_cost": unit_cost,
                "extended_cost": extended_cost,
                "supplier": material.get("supplier", ""),
                "lead_time_days": material.get("lead_time_days", 0)
            })
            
            total_cost += extended_cost
        
        # Add hardware
        for hardware in hardware_list:
            hardware_cost = hardware.get("cost_per_unit", 0.0) * hardware.get("quantity_per_piece", 1) * quantity
            total_cost += hardware_cost
            
            detailed_bom.append({
                "item_id": hardware.get("hardware_id"),
                "description": hardware.get("description"),
                "category": "HARDWARE",
                "quantity_needed": hardware.get("quantity_per_piece", 1) * quantity,
                "unit_cost": hardware.get("cost_per_unit", 0.0),
                "extended_cost": hardware_cost,
                "supplier": hardware.get("supplier", ""),
                "lead_time_days": hardware.get("lead_time_days", 0)
            })
        
        # Lead time analysis
        max_lead_time = max(
            (item.get("lead_time_days", 0) for item in detailed_bom), 
            default=0
        )
        
        return {
            "detailed_bom": detailed_bom,
            "material_costs": {
                "total_material_cost": total_cost,
                "cost_per_unit": total_cost / quantity if quantity > 0 else 0,
                "material_cost_breakdown": {
                    "wood": sum(item["extended_cost"] for item in detailed_bom if "wood" in item["category"].lower()),
                    "hardware": sum(item["extended_cost"] for item in detailed_bom if item["category"] == "HARDWARE"),
                    "other": sum(item["extended_cost"] for item in detailed_bom if item["category"] not in ["HARDWARE"] and "wood" not in item["category"].lower())
                }
            },
            "lead_time_analysis": {
                "critical_path_days": max_lead_time,
                "procurement_schedule": sorted(
                    [{"item": item["description"], "order_by": datetime.now() + timedelta(days=max_lead_time - item["lead_time_days"])}
                     for item in detailed_bom],
                    key=lambda x: x["order_by"]
                )
            }
        }
    
    async def _schedule_finishing(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule finishing operations"""
        production_orders = payload.get("production_orders", [])
        booth_capacity = payload.get("finish_booth_capacity", {"booths": 2, "pieces_per_booth": 10})
        environmental = payload.get("environmental_conditions", {"temperature": 70, "humidity": 45})
        
        finish_schedule = []
        current_time = datetime.now()
        
        for order in production_orders:
            pieces = order.get("quantity", 1)
            finish_type = FinishType(order.get("finish_type", "STAIN"))
            cure_hours = self.finish_cure_times.get(finish_type, 4)
            
            # Adjust cure time for environmental conditions
            temp_factor = 1.0 if environmental["temperature"] >= 65 else 1.2
            humidity_factor = 1.0 if environmental["humidity"] <= 50 else 1.1
            adjusted_cure_hours = cure_hours * temp_factor * humidity_factor
            
            finish_schedule.append({
                "order_id": order.get("order_id"),
                "finish_type": finish_type.value,
                "pieces": pieces,
                "estimated_finish_time": adjusted_cure_hours,
                "start_time": current_time.isoformat(),
                "completion_time": (current_time + timedelta(hours=adjusted_cure_hours)).isoformat(),
                "booth_assignment": f"BOOTH_{(len(finish_schedule) % booth_capacity['booths']) + 1}"
            })
        
        # Calculate capacity utilization
        total_booth_hours = booth_capacity["booths"] * 8  # 8 hour workday
        used_hours = sum(item["estimated_finish_time"] for item in finish_schedule)
        utilization = (used_hours / total_booth_hours * 100) if total_booth_hours > 0 else 0
        
        return {
            "finish_schedule": finish_schedule,
            "bottleneck_analysis": {
                "critical_finish_type": max(
                    set(item["finish_type"] for item in finish_schedule),
                    key=lambda x: sum(item["estimated_finish_time"] for item in finish_schedule if item["finish_type"] == x),
                    default="NONE"
                ),
                "environmental_impact": f"Cure times extended by {((temp_factor * humidity_factor - 1) * 100):.1f}% due to conditions"
            },
            "capacity_utilization": utilization
        }
    
    async def _configure_custom_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Configure custom furniture order"""
        base_design = payload.get("base_design", {})
        customizations = payload.get("customization_requests", [])
        budget = payload.get("budget_constraints", {})
        
        feasibility_issues = []
        cost_impact = {"base_cost": 1000, "customization_cost": 0, "total_cost": 1000}
        
        # Analyze each customization
        for customization in customizations:
            custom_type = customization.get("type")
            
            if custom_type == "dimension_change":
                size_change = customization.get("size_change_percentage", 0)
                if abs(size_change) > 50:
                    feasibility_issues.append("Dimension change exceeds 50% - may require design re-engineering")
                cost_impact["customization_cost"] += abs(size_change) * 5  # $5 per % change
            
            elif custom_type == "material_upgrade":
                from_material = customization.get("from_material")
                to_material = customization.get("to_material")
                if to_material == "HARDWOOD" and from_material == "ENGINEERED":
                    cost_impact["customization_cost"] += 200  # $200 upgrade
            
            elif custom_type == "finish_change":
                finish_complexity = customization.get("complexity", "STANDARD")
                if finish_complexity == "COMPLEX":
                    cost_impact["customization_cost"] += 150
                    feasibility_issues.append("Complex finish may extend lead time by 1-2 weeks")
        
        cost_impact["total_cost"] = cost_impact["base_cost"] + cost_impact["customization_cost"]
        
        # Check budget constraints
        max_budget = budget.get("maximum_budget", float('inf'))
        if cost_impact["total_cost"] > max_budget:
            feasibility_issues.append(f"Total cost ${cost_impact['total_cost']} exceeds budget ${max_budget}")
        
        return {
            "feasibility_analysis": {
                "is_feasible": len(feasibility_issues) == 0,
                "issues": feasibility_issues,
                "complexity_rating": "HIGH" if len(customizations) > 3 else "MEDIUM" if len(customizations) > 1 else "LOW"
            },
            "engineering_drawings": {
                "required_drawings": ["dimension_sheet", "material_specification", "finish_schedule"],
                "estimated_engineering_hours": len(customizations) * 2
            },
            "cost_impact": cost_impact
        }
    
    async def _analyze_seasonal_demand(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonal demand patterns for furniture"""
        product_mix = payload.get("product_mix", {})
        historical_data = payload.get("historical_sales", [])
        current_season = payload.get("current_season", "spring")
        
        seasonal_forecast = {}
        
        for product_type_str, quantity in product_mix.items():
            try:
                product_type = FurnitureProductType(product_type_str.upper())
                pattern = self.seasonal_demand_patterns.get(product_type, {"spring": 1.0, "summer": 1.0, "fall": 1.0, "winter": 1.0})
                seasonal_multiplier = pattern.get(current_season, 1.0)
                
                seasonal_forecast[product_type_str] = {
                    "base_demand": quantity,
                    "seasonal_multiplier": seasonal_multiplier,
                    "adjusted_demand": quantity * seasonal_multiplier,
                    "seasonal_pattern": pattern
                }
            except ValueError:
                # Unknown product type
                seasonal_forecast[product_type_str] = {
                    "base_demand": quantity,
                    "seasonal_multiplier": 1.0,
                    "adjusted_demand": quantity,
                    "note": "No seasonal pattern data available"
                }
        
        return {
            "seasonal_demand_forecast": seasonal_forecast,
            "current_season": current_season,
            "peak_season_products": [
                product for product, data in seasonal_forecast.items()
                if data.get("seasonal_multiplier", 1.0) > 1.2
            ],
            "off_season_products": [
                product for product, data in seasonal_forecast.items()
                if data.get("seasonal_multiplier", 1.0) < 0.8
            ],
            "inventory_recommendations": [
                "Build inventory for peak season products 6-8 weeks in advance",
                "Consider promotional pricing for off-season products",
                "Plan wood drying schedules based on seasonal production needs"
            ]
        }


# Export main component
__all__ = ["FurnitureManufacturingAgent", "FurnitureProductType", "WoodType", "FinishType"]