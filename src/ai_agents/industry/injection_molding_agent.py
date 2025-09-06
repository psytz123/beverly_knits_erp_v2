#!/usr/bin/env python3
"""
Injection Molding Manufacturing Specialization Agent
Industry-specific ERP intelligence for injection molding operations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from ..core.agent_base import BaseAgent, AgentMessage, AgentCapability, MessageType, Priority
from ..core.state_manager import CustomerProfile

# Setup logging
logger = logging.getLogger(__name__)


class PlasticType(Enum):
    """Plastic material classifications"""
    THERMOPLASTIC = "THERMOPLASTIC"    # ABS, PC, PP, PE, PS, POM
    THERMOSET = "THERMOSET"           # Epoxy, polyurethane, phenolic
    ENGINEERING = "ENGINEERING"        # PEEK, PEI, PAI, LCP
    COMMODITY = "COMMODITY"           # PE, PP, PS, PVC
    BIODEGRADABLE = "BIODEGRADABLE"   # PLA, PHA, starch-based


class MoldType(Enum):
    """Injection mold classifications"""
    SINGLE_CAVITY = "SINGLE_CAVITY"       # One part per cycle
    MULTI_CAVITY = "MULTI_CAVITY"         # Multiple identical parts
    FAMILY_MOLD = "FAMILY_MOLD"           # Multiple different parts
    STACK_MOLD = "STACK_MOLD"             # Stacked parting lines
    HOT_RUNNER = "HOT_RUNNER"             # Heated runner system
    COLD_RUNNER = "COLD_RUNNER"           # Unheated runner system


class DefectType(Enum):
    """Common injection molding defects"""
    SHORT_SHOT = "SHORT_SHOT"             # Incomplete fill
    FLASH = "FLASH"                       # Excess material
    SINK_MARKS = "SINK_MARKS"             # Surface depressions
    WARPAGE = "WARPAGE"                   # Part distortion
    BURN_MARKS = "BURN_MARKS"             # Material degradation
    WELD_LINES = "WELD_LINES"             # Visible flow fronts
    EJECTOR_MARKS = "EJECTOR_MARKS"       # Pin impressions
    COLOR_VARIATION = "COLOR_VARIATION"    # Inconsistent color


@dataclass
class PlasticMaterial:
    """Injection molding material specification"""
    material_id: str
    material_name: str
    plastic_type: PlasticType
    grade: str = "STANDARD"
    melt_temperature_f: int = 400
    mold_temperature_f: int = 180
    injection_pressure_psi: int = 15000
    shrinkage_rate: float = 0.005  # 0.5% typical
    density_g_cm3: float = 1.0
    cost_per_pound: float = 2.50
    supplier: str = ""
    lead_time_days: int = 14
    moisture_sensitivity: bool = True
    drying_temperature_f: int = 180
    drying_time_hours: int = 4


@dataclass
class MoldSpecification:
    """Injection mold specification"""
    mold_id: str
    mold_name: str
    mold_type: MoldType
    cavity_count: int = 1
    part_weight_grams: float = 50.0
    cycle_time_seconds: float = 30.0
    runner_weight_grams: float = 10.0
    clamping_force_tons: int = 100
    shot_size_ounces: float = 2.0
    cooling_time_seconds: float = 15.0
    maintenance_cycles: int = 100000
    current_cycle_count: int = 0
    last_maintenance: Optional[datetime] = None


@dataclass
class InjectionMoldingJob:
    """Injection molding production job"""
    job_id: str
    part_number: str
    material: PlasticMaterial
    mold: MoldSpecification
    quantity_ordered: int
    quantity_produced: int = 0
    scheduled_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    machine_assignment: Optional[str] = None
    quality_requirements: Dict[str, Any] = field(default_factory=dict)
    setup_time_minutes: int = 60
    teardown_time_minutes: int = 30


class InjectionMoldingAgent(BaseAgent):
    """
    Injection Molding Manufacturing Specialization Agent
    
    Capabilities:
    - Injection molding process parameter optimization
    - Cycle time analysis and improvement recommendations
    - Material drying and conditioning schedules
    - Mold maintenance prediction and scheduling
    - Quality control and defect analysis
    - Machine capacity planning and scheduling
    - Energy consumption optimization
    - Scrap rate analysis and reduction strategies
    """
    
    def __init__(self, agent_id: str = "injection_molding_agent"):
        """Initialize Injection Molding Agent"""
        super().__init__(
            agent_id=agent_id,
            agent_name="Injection Molding Specialist",
            agent_description="Industry-specific intelligence for injection molding manufacturing operations"
        )
        
        # Injection molding knowledge base
        self.material_properties = {
            "ABS": {"melt_temp": 450, "mold_temp": 160, "shrinkage": 0.006, "density": 1.05},
            "PP": {"melt_temp": 420, "mold_temp": 100, "shrinkage": 0.015, "density": 0.90},
            "PE": {"melt_temp": 380, "mold_temp": 80, "shrinkage": 0.020, "density": 0.92},
            "PC": {"melt_temp": 570, "mold_temp": 190, "shrinkage": 0.005, "density": 1.20},
            "POM": {"melt_temp": 410, "mold_temp": 180, "shrinkage": 0.020, "density": 1.42}
        }
        
        self.defect_causes = {
            DefectType.SHORT_SHOT: [
                "Insufficient injection pressure", "Low material temperature", 
                "Inadequate venting", "Too fast cooling"
            ],
            DefectType.FLASH: [
                "Excessive injection pressure", "Worn mold parting line", 
                "Insufficient clamping force", "Mold misalignment"
            ],
            DefectType.SINK_MARKS: [
                "Insufficient packing pressure", "Thick sections cooling unevenly",
                "Short packing time", "Low mold temperature"
            ],
            DefectType.WARPAGE: [
                "Uneven cooling", "Residual stress", "Part design issues", 
                "Incorrect gate location"
            ]
        }
        
        self.machine_efficiency_factors = {
            "setup_optimization": 0.15,      # 15% time reduction
            "preventive_maintenance": 0.10,   # 10% efficiency gain
            "parameter_optimization": 0.08,   # 8% cycle time reduction
            "material_handling": 0.05         # 5% efficiency gain
        }
    
    def _initialize(self):
        """Initialize injection molding capabilities"""
        # Register injection molding capabilities
        self.register_capability(AgentCapability(
            name="optimize_process_parameters",
            description="Optimize injection molding process parameters for quality and efficiency",
            input_schema={
                "type": "object",
                "properties": {
                    "material_specification": {"type": "object"},
                    "part_geometry": {"type": "object"},
                    "quality_requirements": {"type": "object"},
                    "production_constraints": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "optimized_parameters": {"type": "object"},
                    "expected_cycle_time": {"type": "number"},
                    "quality_predictions": {"type": "object"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="schedule_mold_maintenance",
            description="Predict and schedule mold maintenance based on cycle count and wear",
            input_schema={
                "type": "object",
                "properties": {
                    "mold_inventory": {"type": "array"},
                    "production_schedule": {"type": "array"},
                    "maintenance_capacity": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "maintenance_schedule": {"type": "array"},
                    "risk_analysis": {"type": "object"},
                    "downtime_impact": {"type": "object"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="analyze_defects",
            description="Analyze injection molding defects and recommend corrective actions",
            input_schema={
                "type": "object",
                "properties": {
                    "defect_data": {"type": "array"},
                    "process_parameters": {"type": "object"},
                    "material_conditions": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "root_cause_analysis": {"type": "object"},
                    "corrective_actions": {"type": "array"},
                    "prevention_strategies": {"type": "array"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="optimize_material_usage",
            description="Optimize plastic material usage and minimize waste",
            input_schema={
                "type": "object",
                "properties": {
                    "production_jobs": {"type": "array"},
                    "material_inventory": {"type": "object"},
                    "recycling_capabilities": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "material_plan": {"type": "object"},
                    "waste_reduction_strategies": {"type": "array"},
                    "cost_savings": {"type": "object"}
                }
            }
        ))
        
        # Register message handlers
        self.register_message_handler(MessageType.REQUEST, self._handle_molding_request)
    
    async def _handle_molding_request(self, message: AgentMessage) -> AgentMessage:
        """Handle injection molding requests"""
        action = message.payload.get("action")
        
        try:
            if action == "optimize_process_parameters":
                result = await self._optimize_process_parameters(message.payload)
            elif action == "schedule_mold_maintenance":
                result = await self._schedule_mold_maintenance(message.payload)
            elif action == "analyze_defects":
                result = await self._analyze_defects(message.payload)
            elif action == "optimize_material_usage":
                result = await self._optimize_material_usage(message.payload)
            elif action == "calculate_cycle_time":
                result = await self._calculate_cycle_time(message.payload)
            elif action == "energy_optimization":
                result = await self._optimize_energy_consumption(message.payload)
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
            self.logger.error(f"Error handling molding request: {str(e)}")
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                correlation_id=message.correlation_id
            )
    
    async def _optimize_process_parameters(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize injection molding process parameters"""
        material_spec = payload.get("material_specification", {})
        part_geometry = payload.get("part_geometry", {})
        quality_reqs = payload.get("quality_requirements", {})
        constraints = payload.get("production_constraints", {})
        
        # Get material properties
        material_name = material_spec.get("material_name", "ABS")
        base_properties = self.material_properties.get(material_name, self.material_properties["ABS"])
        
        # Calculate optimal parameters
        part_weight = part_geometry.get("weight_grams", 50.0)
        wall_thickness = part_geometry.get("wall_thickness_mm", 2.5)
        
        # Temperature optimization
        melt_temp = base_properties["melt_temp"]
        if wall_thickness > 4.0:
            melt_temp += 20  # Higher temp for thick walls
        
        mold_temp = base_properties["mold_temp"]
        if quality_reqs.get("surface_finish") == "HIGH":
            mold_temp += 20  # Higher temp for better finish
        
        # Pressure optimization
        injection_pressure = 15000  # Base pressure (psi)
        if part_weight > 100:
            injection_pressure += 5000  # Higher pressure for larger parts
        
        packing_pressure = injection_pressure * 0.8
        
        # Time optimization
        cooling_time = max(wall_thickness * 2, 10)  # Rule of thumb: 2 sec per mm
        injection_time = part_weight / 50  # Rough estimate
        
        cycle_time = injection_time + cooling_time + 5  # Add 5 sec for other operations
        
        optimized_parameters = {
            "melt_temperature_f": melt_temp,
            "mold_temperature_f": mold_temp,
            "injection_pressure_psi": injection_pressure,
            "packing_pressure_psi": packing_pressure,
            "injection_time_seconds": injection_time,
            "cooling_time_seconds": cooling_time,
            "cycle_time_seconds": cycle_time
        }
        
        # Quality predictions
        quality_predictions = {
            "dimensional_accuracy": "±0.002 inches" if mold_temp > 160 else "±0.005 inches",
            "surface_finish": "Excellent" if mold_temp > base_properties["mold_temp"] + 15 else "Good",
            "warpage_risk": "Low" if cooling_time > wall_thickness * 2 else "Medium",
            "expected_defect_rate": "< 2%" if all([
                melt_temp > base_properties["melt_temp"] - 20,
                injection_pressure > 12000,
                cooling_time > wall_thickness * 1.5
            ]) else "2-5%"
        }
        
        return {
            "optimized_parameters": optimized_parameters,
            "expected_cycle_time": cycle_time,
            "quality_predictions": quality_predictions,
            "optimization_notes": [
                f"Melt temperature optimized for {wall_thickness}mm wall thickness",
                f"Mold temperature adjusted for {quality_reqs.get('surface_finish', 'standard')} surface finish",
                f"Cycle time estimated at {cycle_time:.1f} seconds"
            ]
        }
    
    async def _schedule_mold_maintenance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule mold maintenance"""
        mold_inventory = payload.get("mold_inventory", [])
        production_schedule = payload.get("production_schedule", [])
        maintenance_capacity = payload.get("maintenance_capacity", {"hours_per_week": 40})
        
        maintenance_schedule = []
        risk_analysis = {}
        total_downtime_hours = 0
        
        for mold in mold_inventory:
            mold_id = mold.get("mold_id")
            current_cycles = mold.get("current_cycle_count", 0)
            maintenance_cycles = mold.get("maintenance_cycles", 100000)
            last_maintenance = mold.get("last_maintenance")
            
            # Calculate maintenance urgency
            cycles_remaining = maintenance_cycles - (current_cycles % maintenance_cycles)
            urgency_percentage = ((maintenance_cycles - cycles_remaining) / maintenance_cycles) * 100
            
            # Estimate cycles per week from production schedule
            weekly_cycles = 0
            for job in production_schedule:
                if job.get("mold_id") == mold_id:
                    job_cycles = job.get("quantity", 0) // mold.get("cavity_count", 1)
                    weekly_cycles += job_cycles
            
            # Predict when maintenance will be needed
            if weekly_cycles > 0:
                weeks_until_maintenance = cycles_remaining / weekly_cycles
                maintenance_date = datetime.now() + timedelta(weeks=weeks_until_maintenance)
            else:
                maintenance_date = datetime.now() + timedelta(weeks=52)  # Default 1 year
            
            # Determine maintenance type and duration
            if urgency_percentage > 90:
                maintenance_type = "CRITICAL"
                maintenance_hours = 8
                priority = Priority.CRITICAL
            elif urgency_percentage > 70:
                maintenance_type = "PREVENTIVE"
                maintenance_hours = 4
                priority = Priority.HIGH
            else:
                maintenance_type = "ROUTINE"
                maintenance_hours = 2
                priority = Priority.MEDIUM
            
            maintenance_item = {
                "mold_id": mold_id,
                "maintenance_type": maintenance_type,
                "urgency_percentage": urgency_percentage,
                "scheduled_date": maintenance_date.isoformat(),
                "estimated_hours": maintenance_hours,
                "priority": priority.value,
                "cycles_remaining": cycles_remaining,
                "weeks_until_due": weeks_until_maintenance if weekly_cycles > 0 else 52
            }
            
            maintenance_schedule.append(maintenance_item)
            total_downtime_hours += maintenance_hours
            
            # Risk analysis
            risk_level = "HIGH" if urgency_percentage > 85 else "MEDIUM" if urgency_percentage > 60 else "LOW"
            risk_analysis[mold_id] = {
                "risk_level": risk_level,
                "failure_probability": urgency_percentage / 100,
                "impact_if_failed": f"Production shutdown for {maintenance_hours * 2} hours" # Failure takes longer than maintenance
            }
        
        # Sort schedule by priority and urgency
        maintenance_schedule.sort(key=lambda x: (Priority(x["priority"]).score, -x["urgency_percentage"]), reverse=True)
        
        return {
            "maintenance_schedule": maintenance_schedule,
            "risk_analysis": risk_analysis,
            "downtime_impact": {
                "total_maintenance_hours": total_downtime_hours,
                "weeks_of_capacity_needed": total_downtime_hours / maintenance_capacity.get("hours_per_week", 40),
                "production_impact": f"Estimated {total_downtime_hours} hours of production downtime",
                "cost_impact": f"${total_downtime_hours * 150:.2f} in maintenance costs"  # $150/hour estimate
            }
        }
    
    async def _analyze_defects(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze injection molding defects"""
        defect_data = payload.get("defect_data", [])
        process_params = payload.get("process_parameters", {})
        material_conditions = payload.get("material_conditions", {})
        
        defect_summary = {}
        root_causes = {}
        corrective_actions = []
        prevention_strategies = []
        
        # Analyze defect patterns
        for defect in defect_data:
            defect_type = defect.get("defect_type")
            frequency = defect.get("frequency", 1)
            
            if defect_type in defect_summary:
                defect_summary[defect_type] += frequency
            else:
                defect_summary[defect_type] = frequency
        
        # Find most common defects
        most_common_defect = max(defect_summary.keys(), key=defect_summary.get) if defect_summary else None
        
        if most_common_defect:
            try:
                defect_enum = DefectType(most_common_defect)
                possible_causes = self.defect_causes.get(defect_enum, ["Unknown causes"])
                
                # Analyze process parameters to identify likely causes
                likely_causes = []
                
                if defect_enum == DefectType.SHORT_SHOT:
                    if process_params.get("injection_pressure", 0) < 12000:
                        likely_causes.append("Insufficient injection pressure")
                    if process_params.get("melt_temperature", 0) < 400:
                        likely_causes.append("Low material temperature")
                
                elif defect_enum == DefectType.FLASH:
                    if process_params.get("injection_pressure", 0) > 20000:
                        likely_causes.append("Excessive injection pressure")
                    if process_params.get("clamping_force", 0) < 50:
                        likely_causes.append("Insufficient clamping force")
                
                elif defect_enum == DefectType.SINK_MARKS:
                    if process_params.get("packing_pressure", 0) < 8000:
                        likely_causes.append("Insufficient packing pressure")
                    if process_params.get("cooling_time", 0) < 10:
                        likely_causes.append("Insufficient cooling time")
                
                root_causes[most_common_defect] = {
                    "possible_causes": possible_causes,
                    "likely_causes": likely_causes or possible_causes[:2],
                    "frequency": defect_summary[most_common_defect],
                    "percentage_of_total": (defect_summary[most_common_defect] / sum(defect_summary.values())) * 100
                }
                
                # Generate corrective actions
                if defect_enum == DefectType.SHORT_SHOT:
                    corrective_actions.extend([
                        "Increase injection pressure by 10-15%",
                        "Raise melt temperature by 10-20°F",
                        "Check and clean mold vents",
                        "Extend injection time"
                    ])
                
                elif defect_enum == DefectType.FLASH:
                    corrective_actions.extend([
                        "Reduce injection pressure by 5-10%",
                        "Increase clamping force",
                        "Inspect mold parting line for wear",
                        "Check mold alignment"
                    ])
                
                elif defect_enum == DefectType.SINK_MARKS:
                    corrective_actions.extend([
                        "Increase packing pressure",
                        "Extend packing time",
                        "Increase mold temperature",
                        "Optimize gate location"
                    ])
                
            except ValueError:
                root_causes[most_common_defect] = {"possible_causes": ["Unknown defect type"]}
        
        # General prevention strategies
        prevention_strategies = [
            "Implement statistical process control (SPC)",
            "Regular preventive maintenance schedule",
            "Operator training on process parameters",
            "Material moisture control and drying",
            "Mold temperature monitoring",
            "First article inspection protocols"
        ]
        
        return {
            "root_cause_analysis": {
                "defect_summary": defect_summary,
                "most_common_defect": most_common_defect,
                "detailed_analysis": root_causes,
                "total_defect_rate": sum(defect_summary.values()) / len(defect_data) * 100 if defect_data else 0
            },
            "corrective_actions": corrective_actions,
            "prevention_strategies": prevention_strategies,
            "recommendations": [
                "Focus immediate attention on reducing " + (most_common_defect or "defects"),
                "Monitor process parameters more closely",
                "Consider parameter optimization study",
                "Review material handling procedures"
            ]
        }
    
    async def _optimize_material_usage(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize plastic material usage"""
        production_jobs = payload.get("production_jobs", [])
        material_inventory = payload.get("material_inventory", {})
        recycling_capabilities = payload.get("recycling_capabilities", {})
        
        material_requirements = {}
        waste_analysis = {}
        cost_savings = {"total_savings": 0, "waste_reduction": 0, "recycling_value": 0}
        
        # Calculate material requirements
        for job in production_jobs:
            material = job.get("material_name", "Unknown")
            part_weight = job.get("part_weight_grams", 50)
            runner_weight = job.get("runner_weight_grams", 10)
            quantity = job.get("quantity", 1)
            cavity_count = job.get("cavity_count", 1)
            
            # Calculate material needed per shot
            material_per_shot = (part_weight * cavity_count) + runner_weight
            shots_needed = quantity // cavity_count
            total_material_grams = material_per_shot * shots_needed
            
            if material in material_requirements:
                material_requirements[material] += total_material_grams
            else:
                material_requirements[material] = total_material_grams
        
        # Analyze waste streams
        for material, total_grams in material_requirements.items():
            # Typical waste sources
            startup_waste = total_grams * 0.02  # 2% startup waste
            runner_waste = sum(
                job.get("runner_weight_grams", 10) * (job.get("quantity", 1) // job.get("cavity_count", 1))
                for job in production_jobs 
                if job.get("material_name") == material
            )
            
            defect_waste = total_grams * 0.03  # 3% defect rate assumption
            
            total_waste = startup_waste + runner_waste + defect_waste
            
            # Recycling potential
            can_recycle_runners = recycling_capabilities.get("regrind_capability", False)
            can_recycle_defects = recycling_capabilities.get("defect_recycling", False)
            
            recyclable_material = 0
            if can_recycle_runners:
                recyclable_material += runner_waste * 0.8  # 80% recovery rate
            if can_recycle_defects:
                recyclable_material += defect_waste * 0.7   # 70% recovery rate
            
            waste_analysis[material] = {
                "total_required_grams": total_grams,
                "total_waste_grams": total_waste,
                "waste_percentage": (total_waste / total_grams) * 100,
                "recyclable_grams": recyclable_material,
                "net_waste_grams": total_waste - recyclable_material
            }
            
            # Calculate cost savings
            material_cost_per_gram = material_inventory.get(material, {}).get("cost_per_gram", 0.003)  # $0.003/gram default
            recycling_value = recyclable_material * material_cost_per_gram * 0.3  # 30% value recovery
            waste_cost_avoided = recyclable_material * 0.001  # $0.001/gram disposal cost avoided
            
            cost_savings["recycling_value"] += recycling_value
            cost_savings["waste_reduction"] += waste_cost_avoided
        
        cost_savings["total_savings"] = cost_savings["recycling_value"] + cost_savings["waste_reduction"]
        
        # Generate waste reduction strategies
        waste_reduction_strategies = [
            "Implement runner recycling system - can recover 80% of runner waste",
            "Optimize gate and runner design to minimize material usage",
            "Implement real-time process monitoring to reduce defects",
            "Use hot runner systems to eliminate cold runner waste",
            "Establish material drying procedures to prevent moisture defects",
            "Train operators on proper startup procedures to minimize purge waste"
        ]
        
        material_plan = {
            "requirements_by_material": material_requirements,
            "total_material_needed_kg": sum(material_requirements.values()) / 1000,
            "total_waste_kg": sum(data["total_waste_grams"] for data in waste_analysis.values()) / 1000,
            "recycling_potential_kg": sum(data["recyclable_grams"] for data in waste_analysis.values()) / 1000,
            "net_waste_kg": sum(data["net_waste_grams"] for data in waste_analysis.values()) / 1000
        }
        
        return {
            "material_plan": material_plan,
            "waste_analysis": waste_analysis,
            "waste_reduction_strategies": waste_reduction_strategies,
            "cost_savings": cost_savings,
            "recommendations": [
                "Prioritize runner system optimization for highest volume materials",
                "Implement material tracking system for better waste visibility",
                "Consider blending recycled content up to 25% for non-critical applications",
                "Establish waste reduction KPIs and regular monitoring"
            ]
        }
    
    async def _calculate_cycle_time(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimized cycle time"""
        part_specs = payload.get("part_specifications", {})
        material_props = payload.get("material_properties", {})
        quality_reqs = payload.get("quality_requirements", {})
        
        # Part geometry factors
        wall_thickness = part_specs.get("wall_thickness_mm", 2.5)
        part_weight = part_specs.get("weight_grams", 50.0)
        surface_area = part_specs.get("surface_area_cm2", 100.0)
        
        # Material factors
        material_name = material_props.get("material_name", "ABS")
        material_data = self.material_properties.get(material_name, self.material_properties["ABS"])
        
        # Calculate cycle components
        injection_time = part_weight / 100  # Rough estimate: 100g per second
        
        # Cooling time is typically the longest component
        # Rule of thumb: 2 seconds per mm of wall thickness
        base_cooling_time = wall_thickness * 2
        
        # Adjust for material thermal properties
        if material_name in ["PC", "POM"]:  # Higher temp materials
            base_cooling_time *= 1.2
        elif material_name in ["PP", "PE"]:  # Lower temp materials
            base_cooling_time *= 0.8
        
        # Quality adjustments
        if quality_reqs.get("dimensional_tolerance") == "TIGHT":
            base_cooling_time *= 1.3  # Longer cooling for better dimensions
        
        if quality_reqs.get("surface_finish") == "HIGH":
            base_cooling_time *= 1.1  # Longer for better surface
        
        # Other cycle components
        mold_open_close_time = 2.0  # seconds
        part_ejection_time = 1.0    # seconds
        
        total_cycle_time = injection_time + base_cooling_time + mold_open_close_time + part_ejection_time
        
        # Calculate production rates
        parts_per_hour = 3600 / total_cycle_time
        parts_per_shift = parts_per_hour * 8  # 8 hour shift
        
        return {
            "cycle_time_breakdown": {
                "injection_time_seconds": injection_time,
                "cooling_time_seconds": base_cooling_time,
                "mold_operation_time_seconds": mold_open_close_time,
                "ejection_time_seconds": part_ejection_time,
                "total_cycle_time_seconds": total_cycle_time
            },
            "production_rates": {
                "parts_per_hour": round(parts_per_hour),
                "parts_per_shift": round(parts_per_shift),
                "parts_per_day": round(parts_per_shift * 3)  # 3 shifts
            },
            "optimization_opportunities": [
                f"Cooling time is {(base_cooling_time/total_cycle_time)*100:.1f}% of cycle - focus area for reduction",
                "Consider conformal cooling for faster heat removal",
                "Optimize mold open/close speed settings",
                "Evaluate part ejection system efficiency"
            ]
        }
    
    async def _optimize_energy_consumption(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize energy consumption"""
        machine_specs = payload.get("machine_specifications", {})
        production_schedule = payload.get("production_schedule", [])
        energy_rates = payload.get("energy_rates", {"peak": 0.12, "off_peak": 0.08})  # $/kWh
        
        # Energy consumption factors
        base_machine_power_kw = machine_specs.get("rated_power_kw", 25)
        heating_power_kw = base_machine_power_kw * 0.3  # 30% for heating
        hydraulic_power_kw = base_machine_power_kw * 0.4  # 40% for hydraulics
        auxiliary_power_kw = base_machine_power_kw * 0.3  # 30% for auxiliaries
        
        energy_optimization = []
        total_energy_savings = 0
        
        for job in production_schedule:
            cycle_time = job.get("cycle_time_seconds", 30)
            quantity = job.get("quantity", 1)
            
            # Energy per cycle
            energy_per_cycle_kwh = (base_machine_power_kw * cycle_time) / 3600
            total_energy_kwh = energy_per_cycle_kwh * quantity
            
            # Optimization strategies
            optimizations = []
            
            # Heating optimization
            if job.get("material_temperature", 450) > 500:
                heating_savings = heating_power_kw * 0.1  # 10% savings with optimized heating
                optimizations.append({
                    "strategy": "Optimize barrel heating zones",
                    "energy_savings_kwh": heating_savings * (cycle_time * quantity) / 3600,
                    "cost_savings": heating_savings * 0.1 * quantity  # Rough estimate
                })
            
            # Hydraulic optimization
            hydraulic_savings = hydraulic_power_kw * 0.15  # 15% with variable speed drives
            optimizations.append({
                "strategy": "Install variable frequency drives",
                "energy_savings_kwh": hydraulic_savings * (cycle_time * quantity) / 3600,
                "cost_savings": hydraulic_savings * 0.1 * quantity
            })
            
            # Schedule optimization
            if job.get("priority", "NORMAL") != "URGENT":
                schedule_savings = total_energy_kwh * (energy_rates["peak"] - energy_rates["off_peak"])
                optimizations.append({
                    "strategy": "Schedule during off-peak hours",
                    "energy_savings_kwh": 0,  # Same energy, different rate
                    "cost_savings": schedule_savings
                })
            
            job_savings = sum(opt["cost_savings"] for opt in optimizations)
            total_energy_savings += job_savings
            
            energy_optimization.append({
                "job_id": job.get("job_id"),
                "current_energy_kwh": total_energy_kwh,
                "current_cost": total_energy_kwh * energy_rates["peak"],
                "optimizations": optimizations,
                "potential_savings": job_savings
            })
        
        return {
            "energy_analysis": {
                "total_consumption_kwh": sum(job["current_energy_kwh"] for job in energy_optimization),
                "total_cost": sum(job["current_cost"] for job in energy_optimization),
                "potential_savings": total_energy_savings,
                "savings_percentage": (total_energy_savings / sum(job["current_cost"] for job in energy_optimization)) * 100 if energy_optimization else 0
            },
            "optimization_strategies": [
                "Install variable frequency drives on hydraulic pumps",
                "Optimize heating zone temperatures and timing",
                "Schedule non-urgent jobs during off-peak hours",
                "Implement machine idle-mode power reduction",
                "Consider servo-driven machines for high-volume jobs"
            ],
            "job_specific_recommendations": energy_optimization,
            "payback_analysis": {
                "annual_savings": total_energy_savings * 250,  # 250 working days
                "vfd_investment": 15000,  # Typical VFD cost
                "payback_months": (15000 / (total_energy_savings * 250 / 12)) if total_energy_savings > 0 else 0
            }
        }


# Export main component
__all__ = ["InjectionMoldingAgent", "PlasticType", "MoldType", "DefectType"]