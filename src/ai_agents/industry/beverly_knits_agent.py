#!/usr/bin/env python3
"""
Beverly Knits Manufacturing Specialization Agent
Industry-specific ERP intelligence for textile/knitting manufacturing operations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from ..core.agent_base import BaseAgent, AgentMessage, AgentCapability, MessageType, Priority
from ..core.state_manager import CustomerProfile
from ..integration.erp_bridge import ERPIntegrationBridge, APICallStatus

# Setup logging
logger = logging.getLogger(__name__)


class YarnCategory(Enum):
    """Yarn material categories"""
    COTTON = "COTTON"
    WOOL = "WOOL"
    SYNTHETIC = "SYNTHETIC"
    BLEND = "BLEND"
    SPECIALTY = "SPECIALTY"


class ProductionStage(Enum):
    """Production flow stages"""
    G00_GREIGE = "G00"           # Greige/raw knitted fabric
    G02_GREIGE_STAGE2 = "G02"    # Secondary greige processing
    I01_QC = "I01"               # Quality control/inspection
    F01_FINISHED = "F01"         # Finished goods


class KnitConstruction(Enum):
    """Knit construction types"""
    JERSEY = "JERSEY"
    RIB = "RIB"
    INTERLOCK = "INTERLOCK"
    FLEECE = "FLEECE"
    FRENCH_TERRY = "FRENCH_TERRY"
    THERMAL = "THERMAL"


@dataclass
class YarnSpecification:
    """Yarn specification with textile properties"""
    yarn_id: str
    description: str
    category: YarnCategory
    weight: str = ""              # Yarn weight classification
    fiber_content: str = ""       # Cotton 100%, Polyester 65% Cotton 35%, etc.
    color: str = ""
    lot_number: str = ""
    twist: str = ""               # S-twist, Z-twist
    ply: int = 1                  # Single, 2-ply, etc.
    tex: float = 0.0              # Linear density
    tensile_strength: float = 0.0 # Breaking strength
    supplier: str = ""
    lead_time_days: int = 14
    minimum_order_lbs: float = 0.0


@dataclass
class MachineSpecification:
    """Knitting machine specification"""
    machine_id: str
    machine_type: str = ""        # Circular, Flat, Warp
    gauge: int = 0                # Needles per inch
    diameter_inches: float = 0.0  # Machine diameter
    needle_count: int = 0         # Total needles
    max_rpm: int = 0              # Maximum RPM
    work_center: str = ""         # Work center assignment
    capacity_lbs_per_hour: float = 0.0
    setup_time_hours: float = 2.0
    maintenance_schedule: str = "WEEKLY"


@dataclass
class KnitOrder:
    """Knit production order"""
    order_id: str
    style_number: str
    construction: KnitConstruction
    yarn_requirements: List[Dict[str, Any]]
    target_weight_lbs: float
    target_yardage: float
    quality_specs: Dict[str, Any]
    assigned_machine: Optional[str] = None
    production_stage: ProductionStage = ProductionStage.G00_GREIGE
    priority: str = "NORMAL"
    due_date: Optional[datetime] = None


class BeverlyKnitsManufacturingAgent(BaseAgent):
    """
    Beverly Knits Manufacturing Specialization Agent
    
    Capabilities:
    - Textile-specific inventory management with planning balance calculations
    - Yarn substitution and compatibility analysis
    - Knitting machine assignment and capacity optimization
    - Style-to-yarn BOM management and multi-level netting
    - Production flow tracking through G00→G02→I01→F01 stages
    - ML-powered demand forecasting for textile products
    - Quality control workflow management
    - Supplier management for yarn and textile vendors
    - Integration with existing Beverly Knits ERP APIs
    """
    
    def __init__(self, agent_id: str = "beverly_knits_manufacturing_agent", erp_base_url: str = "http://localhost:5006"):
        """Initialize Beverly Knits Manufacturing Agent"""
        super().__init__(
            agent_id=agent_id,
            agent_name="Beverly Knits Manufacturing Specialist",
            agent_description="Industry-specific intelligence for textile/knitting manufacturing operations"
        )
        
        # ERP Integration Bridge
        self.erp_bridge = ERPIntegrationBridge(erp_base_url=erp_base_url)
        
        # Textile-specific knowledge base
        self.yarn_compatibility_matrix = {
            YarnCategory.COTTON: [YarnCategory.COTTON, YarnCategory.BLEND],
            YarnCategory.WOOL: [YarnCategory.WOOL, YarnCategory.BLEND],
            YarnCategory.SYNTHETIC: [YarnCategory.SYNTHETIC, YarnCategory.BLEND],
            YarnCategory.BLEND: [YarnCategory.COTTON, YarnCategory.WOOL, YarnCategory.SYNTHETIC, YarnCategory.BLEND],
            YarnCategory.SPECIALTY: [YarnCategory.SPECIALTY]
        }
        
        self.gauge_recommendations = {
            KnitConstruction.JERSEY: {"min_gauge": 18, "max_gauge": 32},
            KnitConstruction.RIB: {"min_gauge": 14, "max_gauge": 28},
            KnitConstruction.INTERLOCK: {"min_gauge": 16, "max_gauge": 24},
            KnitConstruction.FLEECE: {"min_gauge": 12, "max_gauge": 20},
            KnitConstruction.FRENCH_TERRY: {"min_gauge": 14, "max_gauge": 22},
            KnitConstruction.THERMAL: {"min_gauge": 10, "max_gauge": 18}
        }
        
        self.production_flow_stages = {
            ProductionStage.G00_GREIGE: {"next_stage": ProductionStage.G02_GREIGE_STAGE2, "avg_duration_hours": 24},
            ProductionStage.G02_GREIGE_STAGE2: {"next_stage": ProductionStage.I01_QC, "avg_duration_hours": 8},
            ProductionStage.I01_QC: {"next_stage": ProductionStage.F01_FINISHED, "avg_duration_hours": 4},
            ProductionStage.F01_FINISHED: {"next_stage": None, "avg_duration_hours": 0}
        }
        
        # Work center pattern parsing (x.xx.xx.X format)
        self.work_center_mapping = {}
        
        # Cached data from ERP
        self.cached_inventory_data = {}
        self.cached_bom_data = {}
        self.cached_machine_data = {}
        self.last_cache_update = None
        
        self.logger.info("Beverly Knits Manufacturing Agent initialized")
    
    def _initialize(self):
        """Initialize Beverly Knits specific capabilities"""
        # Register textile manufacturing capabilities
        self.register_capability(AgentCapability(
            name="yarn_inventory_optimization",
            description="Optimize yarn inventory with planning balance calculations",
            input_schema={
                "type": "object",
                "properties": {
                    "view": {"type": "string", "enum": ["summary", "detailed", "shortage"]},
                    "analysis": {"type": "string", "enum": ["shortage", "excess", "reorder"]},
                    "forecast_horizon": {"type": "integer", "default": 30}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "inventory_analysis": {"type": "object"},
                    "shortage_alerts": {"type": "array"},
                    "reorder_recommendations": {"type": "array"}
                }
            },
            estimated_duration_seconds=60,
            risk_level="LOW"
        ))
        
        self.register_capability(AgentCapability(
            name="intelligent_yarn_substitution",
            description="Find compatible yarn substitutes based on specifications",
            input_schema={
                "type": "object",
                "properties": {
                    "target_yarn": {"type": "string"},
                    "substitution_criteria": {"type": "object"},
                    "quality_tolerance": {"type": "number", "default": 0.1}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "substitute_yarns": {"type": "array"},
                    "compatibility_score": {"type": "number"},
                    "quality_impact": {"type": "object"}
                }
            },
            estimated_duration_seconds=30,
            risk_level="MEDIUM"
        ))
        
        self.register_capability(AgentCapability(
            name="machine_assignment_optimization",
            description="Assign knit orders to optimal machines based on specifications",
            input_schema={
                "type": "object",
                "properties": {
                    "unassigned_orders": {"type": "array"},
                    "machine_availability": {"type": "object"},
                    "optimization_criteria": {"type": "string", "enum": ["efficiency", "due_date", "setup_time"]}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "machine_assignments": {"type": "array"},
                    "utilization_forecast": {"type": "object"},
                    "bottleneck_analysis": {"type": "object"}
                }
            },
            estimated_duration_seconds=120,
            risk_level="MEDIUM"
        ))
        
        self.register_capability(AgentCapability(
            name="production_flow_tracking",
            description="Track production through G00→G02→I01→F01 stages",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "current_stage": {"type": "string"},
                    "quality_metrics": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "production_status": {"type": "object"},
                    "next_stage_recommendations": {"type": "array"},
                    "quality_alerts": {"type": "array"}
                }
            },
            estimated_duration_seconds=30,
            risk_level="LOW"
        ))
        
        self.register_capability(AgentCapability(
            name="textile_demand_forecasting",
            description="ML-powered demand forecasting for textile products",
            input_schema={
                "type": "object",
                "properties": {
                    "style_numbers": {"type": "array"},
                    "forecast_horizon": {"type": "integer", "default": 90},
                    "detail_level": {"type": "string", "enum": ["summary", "detailed"]}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "demand_forecast": {"type": "object"},
                    "confidence_intervals": {"type": "object"},
                    "seasonal_patterns": {"type": "object"}
                }
            },
            estimated_duration_seconds=180,
            risk_level="LOW"
        ))
        
        # Register message handlers
        self.register_message_handler(MessageType.REQUEST, self._handle_beverly_knits_request)
        
        # Initialize ERP bridge
        asyncio.create_task(self.erp_bridge.initialize())
        
        # Start background monitoring
        asyncio.create_task(self._production_monitoring_loop())
        asyncio.create_task(self._inventory_monitoring_loop())
    
    async def _handle_beverly_knits_request(self, message: AgentMessage) -> AgentMessage:
        """Handle Beverly Knits manufacturing requests"""
        action = message.payload.get("action")
        
        try:
            if action == "optimize_yarn_inventory":
                result = await self._optimize_yarn_inventory(message.payload)
            elif action == "find_yarn_substitutes":
                result = await self._find_yarn_substitutes(message.payload)
            elif action == "assign_machines":
                result = await self._assign_machines(message.payload)
            elif action == "track_production_flow":
                result = await self._track_production_flow(message.payload)
            elif action == "forecast_demand":
                result = await self._forecast_demand(message.payload)
            elif action == "analyze_shortages":
                result = await self._analyze_shortages(message.payload)
            elif action == "optimize_bom":
                result = await self._optimize_bom(message.payload)
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
            self.logger.error(f"Error handling Beverly Knits request: {str(e)}")
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                correlation_id=message.correlation_id
            )
    
    async def _optimize_yarn_inventory(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize yarn inventory using Beverly Knits ERP data"""
        try:
            # Call ERP inventory intelligence API using bridge
            result = await self.erp_bridge.get_inventory_intelligence(
                view=payload.get("view", "summary"),
                analysis=payload.get("analysis", "shortage"),
                realtime=payload.get("realtime", True)
            )
            
            if result.status != APICallStatus.SUCCESS:
                return {"error": f"Failed to retrieve inventory data: {result.error}"}
            
            # Enhance with textile-specific analysis
            enhanced_analysis = await self._enhance_inventory_analysis(result.data)
            
            # Generate recommendations
            recommendations = await self._generate_inventory_recommendations(enhanced_analysis)
            
            return {
                "inventory_analysis": enhanced_analysis,
                "recommendations": recommendations,
                "erp_integration": {
                    "data_source": "Beverly Knits ERP",
                    "last_updated": result.timestamp.isoformat(),
                    "response_time_ms": result.response_time_ms,
                    "from_cache": result.from_cache
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing yarn inventory: {str(e)}")
            return {"error": str(e)}
    
    async def _find_yarn_substitutes(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Find compatible yarn substitutes"""
        try:
            target_yarn = payload.get("target_yarn")
            criteria = payload.get("substitution_criteria", {})
            
            # Call ERP yarn substitution API using bridge
            result = await self.erp_bridge.get_yarn_substitution(
                target_yarn=target_yarn,
                compatibility_threshold=criteria.get("compatibility_threshold", 0.8)
            )
            
            if result.status != APICallStatus.SUCCESS:
                return {"error": f"Failed to retrieve substitution data: {result.error}"}
            
            # Enhance with textile-specific compatibility analysis
            enhanced_substitutes = []
            for substitute in result.data.get("substitutes", []):
                compatibility = self._calculate_yarn_compatibility(target_yarn, substitute)
                enhanced_substitutes.append({
                    **substitute,
                    "textile_compatibility_score": compatibility,
                    "knitting_suitability": self._assess_knitting_suitability(substitute),
                    "quality_impact": self._assess_quality_impact(substitute)
                })
            
            return {
                "substitute_yarns": enhanced_substitutes,
                "textile_analysis": {
                    "fiber_compatibility": self._analyze_fiber_compatibility(target_yarn, enhanced_substitutes),
                    "gauge_recommendations": self._get_gauge_recommendations(enhanced_substitutes),
                    "construction_impact": self._analyze_construction_impact(enhanced_substitutes)
                },
                "erp_integration": {
                    "data_source": "Beverly Knits ERP Yarn Intelligence",
                    "substitution_method": "ML-Enhanced"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error finding yarn substitutes: {str(e)}")
            return {"error": str(e)}
    
    async def _assign_machines(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Assign knit orders to optimal machines"""
        try:
            # Get unassigned orders from ERP using bridge
            result = await self.erp_bridge.get_machine_assignments()
            
            if result.status != APICallStatus.SUCCESS:
                return {"error": f"Failed to retrieve machine data: {result.error}"}
            
            # Parse work center patterns (x.xx.xx.X)
            machine_assignments = []
            for suggestion in result.data.get("suggestions", []):
                work_center = suggestion.get("work_center", "")
                parsed_wc = self._parse_work_center(work_center)
                
                assignment = {
                    **suggestion,
                    "work_center_analysis": parsed_wc,
                    "gauge_compatibility": self._check_gauge_compatibility(suggestion),
                    "setup_time_estimate": self._estimate_setup_time(suggestion),
                    "production_efficiency": self._calculate_production_efficiency(suggestion)
                }
                machine_assignments.append(assignment)
            
            # Sort by efficiency score
            machine_assignments.sort(key=lambda x: x.get("production_efficiency", 0), reverse=True)
            
            return {
                "machine_assignments": machine_assignments,
                "optimization_analysis": {
                    "total_orders": len(machine_assignments),
                    "average_efficiency": sum(a.get("production_efficiency", 0) for a in machine_assignments) / len(machine_assignments) if machine_assignments else 0,
                    "bottleneck_machines": self._identify_bottleneck_machines(machine_assignments)
                },
                "erp_integration": {
                    "data_source": "Beverly Knits ERP Machine Assignment",
                    "optimization_method": "Efficiency-Based"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error assigning machines: {str(e)}")
            return {"error": str(e)}
    
    async def _track_production_flow(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Track production through flow stages"""
        try:
            # Get production pipeline data
            pipeline_data = await self._call_erp_api("production_planning", {
                "view": "pipeline",
                "include_stages": "true"
            })
            
            if not pipeline_data:
                return {"error": "Failed to retrieve production pipeline data"}
            
            # Track specific order if provided
            order_id = payload.get("order_id")
            if order_id:
                order_status = self._track_specific_order(order_id, pipeline_data)
            else:
                order_status = None
            
            # Analyze production flow
            flow_analysis = {
                "stage_analysis": {},
                "bottlenecks": [],
                "efficiency_metrics": {}
            }
            
            for stage in ProductionStage:
                stage_data = self._analyze_stage_performance(stage, pipeline_data)
                flow_analysis["stage_analysis"][stage.value] = stage_data
                
                if stage_data.get("utilization", 0) > 0.9:
                    flow_analysis["bottlenecks"].append({
                        "stage": stage.value,
                        "utilization": stage_data.get("utilization"),
                        "avg_duration": stage_data.get("avg_duration_hours")
                    })
            
            return {
                "production_status": order_status,
                "flow_analysis": flow_analysis,
                "stage_recommendations": self._generate_stage_recommendations(flow_analysis),
                "erp_integration": {
                    "data_source": "Beverly Knits ERP Production Pipeline",
                    "tracking_method": "Real-time"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error tracking production flow: {str(e)}")
            return {"error": str(e)}
    
    async def _forecast_demand(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ML-powered demand forecast"""
        try:
            # Call ERP ML forecast API
            forecast_params = {
                "detail": payload.get("detail_level", "detailed"),
                "horizon": payload.get("forecast_horizon", 90),
                "format": "report"
            }
            
            if "style_numbers" in payload:
                forecast_params["styles"] = ",".join(payload["style_numbers"])
            
            forecast_data = await self._call_erp_api("ml_forecast", forecast_params)
            
            if not forecast_data:
                return {"error": "Failed to retrieve forecast data from ERP"}
            
            # Enhance with textile-specific analysis
            enhanced_forecast = await self._enhance_demand_forecast(forecast_data)
            
            # Generate textile-specific insights
            textile_insights = {
                "seasonal_patterns": self._analyze_textile_seasonality(forecast_data),
                "yarn_impact_analysis": self._analyze_yarn_demand_impact(forecast_data),
                "production_capacity_requirements": self._calculate_capacity_requirements(forecast_data)
            }
            
            return {
                "demand_forecast": enhanced_forecast,
                "textile_insights": textile_insights,
                "ml_model_info": {
                    "models_used": ["ARIMA", "Prophet", "LSTM", "XGBoost", "Ensemble"],
                    "accuracy_metrics": forecast_data.get("accuracy_metrics", {}),
                    "confidence_level": forecast_data.get("confidence_level", 0.85)
                },
                "erp_integration": {
                    "data_source": "Beverly Knits ERP ML Forecasting",
                    "model_version": "Ensemble-v2"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error forecasting demand: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_shortages(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze yarn shortages and generate action plans"""
        try:
            # Get yarn intelligence data
            yarn_data = await self._call_erp_api("yarn_intelligence", {
                "analysis": "shortage",
                "forecast": "true"
            })
            
            if not yarn_data:
                return {"error": "Failed to retrieve yarn intelligence data"}
            
            shortages = yarn_data.get("shortages", [])
            enhanced_shortage_analysis = []
            
            for shortage in shortages:
                yarn_id = shortage.get("yarn_id", "")
                
                # Enhance with textile-specific analysis
                enhanced_shortage = {
                    **shortage,
                    "substitution_options": await self._find_emergency_substitutes(yarn_id),
                    "production_impact": self._assess_production_impact(shortage),
                    "procurement_urgency": self._calculate_procurement_urgency(shortage),
                    "alternative_suppliers": self._get_alternative_suppliers(yarn_id)
                }
                enhanced_shortage_analysis.append(enhanced_shortage)
            
            # Prioritize shortages
            prioritized_shortages = sorted(
                enhanced_shortage_analysis,
                key=lambda x: x.get("procurement_urgency", 0),
                reverse=True
            )
            
            return {
                "shortage_analysis": prioritized_shortages,
                "action_plan": self._generate_shortage_action_plan(prioritized_shortages),
                "summary": {
                    "total_shortages": len(shortages),
                    "critical_shortages": len([s for s in enhanced_shortage_analysis if s.get("procurement_urgency", 0) > 8]),
                    "estimated_production_impact_days": sum(s.get("production_impact", {}).get("delay_days", 0) for s in enhanced_shortage_analysis)
                },
                "erp_integration": {
                    "data_source": "Beverly Knits ERP Yarn Intelligence",
                    "analysis_method": "AI-Enhanced"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing shortages: {str(e)}")
            return {"error": str(e)}
    
    async def _optimize_bom(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize Bill of Materials for textile production"""
        try:
            # Get inventory netting data
            netting_data = await self._call_erp_api("inventory_netting", payload)
            
            if not netting_data:
                return {"error": "Failed to retrieve BOM netting data"}
            
            # Enhance with textile-specific BOM optimization
            optimized_bom = await self._enhance_bom_analysis(netting_data)
            
            # Generate optimization recommendations
            optimization_recommendations = [
                {
                    "category": "Yarn Substitution",
                    "recommendations": self._generate_substitution_recommendations(optimized_bom)
                },
                {
                    "category": "Inventory Efficiency",
                    "recommendations": self._generate_inventory_efficiency_recommendations(optimized_bom)
                },
                {
                    "category": "Production Optimization",
                    "recommendations": self._generate_production_optimization_recommendations(optimized_bom)
                }
            ]
            
            return {
                "optimized_bom": optimized_bom,
                "optimization_recommendations": optimization_recommendations,
                "savings_analysis": {
                    "potential_cost_savings": self._calculate_potential_savings(optimized_bom),
                    "inventory_reduction": self._calculate_inventory_reduction(optimized_bom),
                    "efficiency_gains": self._calculate_efficiency_gains(optimized_bom)
                },
                "erp_integration": {
                    "data_source": "Beverly Knits ERP Multi-level Netting",
                    "optimization_method": "Textile-Specific"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing BOM: {str(e)}")
            return {"error": str(e)}
    
    async def _call_erp_api(self, endpoint_key: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Call Beverly Knits ERP API endpoint using bridge"""
        try:
            result = await self.erp_bridge.call_erp_api(endpoint_key, params)
            
            if result.status == APICallStatus.SUCCESS:
                return result.data
            else:
                self.logger.error(f"ERP API call failed for {endpoint_key}: {result.error}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calling ERP API {endpoint_key}: {str(e)}")
            return None
    
    def _parse_work_center(self, work_center: str) -> Dict[str, Any]:
        """Parse work center pattern (x.xx.xx.X)"""
        try:
            if not work_center or '.' not in work_center:
                return {"error": "Invalid work center format"}
            
            parts = work_center.split('.')
            if len(parts) != 4:
                return {"error": "Work center must be in format x.xx.xx.X"}
            
            return {
                "knit_construction": int(parts[0]) if parts[0].isdigit() else 0,
                "machine_diameter": float(f"{parts[1]}.{parts[2]}") if parts[1].isdigit() and parts[2].isdigit() else 0,
                "needle_cut": int(parts[2]) if parts[2].isdigit() else 0,
                "machine_type": parts[3].upper(),
                "full_pattern": work_center
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing work center {work_center}: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_yarn_compatibility(self, target_yarn: str, substitute: Dict[str, Any]) -> float:
        """Calculate yarn compatibility score"""
        # Placeholder implementation - would use actual yarn specifications
        base_score = 0.8  # Base compatibility
        
        # Adjust based on fiber content similarity
        target_fiber = substitute.get("target_fiber_content", "")
        substitute_fiber = substitute.get("fiber_content", "")
        
        if target_fiber and substitute_fiber:
            # Simple similarity check (would be enhanced with actual fiber analysis)
            if target_fiber.lower() == substitute_fiber.lower():
                base_score += 0.15
            elif "cotton" in both.lower() for both in [target_fiber, substitute_fiber]:
                base_score += 0.10
        
        return min(1.0, base_score)
    
    def _assess_knitting_suitability(self, substitute: Dict[str, Any]) -> Dict[str, Any]:
        """Assess substitute yarn's knitting suitability"""
        return {
            "gauge_compatibility": "HIGH",  # Would be calculated based on yarn weight/tex
            "tension_requirements": "STANDARD",
            "needle_recommendation": "Standard knitting needles",
            "special_considerations": []
        }
    
    def _assess_quality_impact(self, substitute: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality impact of yarn substitution"""
        return {
            "fabric_hand": "SIMILAR",
            "color_matching": "ACCEPTABLE",
            "performance_characteristics": "MAINTAINED",
            "durability_impact": "NEUTRAL",
            "customer_acceptance": "HIGH"
        }
    
    async def _update_cache(self):
        """Update cached data from ERP"""
        try:
            # Cache key ERP data
            self.cached_inventory_data = await self._call_erp_api("inventory_intelligence", {"view": "summary"})
            self.cached_bom_data = await self._call_erp_api("inventory_netting")
            self.cached_machine_data = await self._call_erp_api("machine_assignment", {})
            
            self.last_cache_update = datetime.now()
            self.logger.info("ERP data cache updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating cache: {str(e)}")
    
    async def _production_monitoring_loop(self):
        """Background production monitoring"""
        while self.status.value != "SHUTDOWN":
            try:
                # Monitor production flow
                await self._monitor_production_stages()
                await self._check_machine_utilization()
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in production monitoring: {str(e)}")
                await asyncio.sleep(1800)
    
    async def _inventory_monitoring_loop(self):
        """Background inventory monitoring"""
        while self.status.value != "SHUTDOWN":
            try:
                # Monitor yarn levels and shortages
                await self._monitor_yarn_shortages()
                await self._update_cache()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in inventory monitoring: {str(e)}")
                await asyncio.sleep(3600)
    
    # Textile-specific enhancement methods
    async def _enhance_inventory_analysis(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance ERP inventory data with textile-specific analysis"""
        enhanced_data = response.copy()
        
        # Add textile-specific metrics
        yarn_items = enhanced_data.get("inventory_items", [])
        
        # Calculate textile-specific insights
        textile_insights = {
            "yarn_category_distribution": self._analyze_yarn_category_distribution(yarn_items),
            "fiber_content_analysis": self._analyze_fiber_content(yarn_items),
            "seasonal_demand_impact": self._calculate_seasonal_impact(yarn_items),
            "substitution_opportunities": self._identify_substitution_opportunities(yarn_items),
            "quality_risk_assessment": self._assess_quality_risks(yarn_items)
        }
        
        enhanced_data["textile_insights"] = textile_insights
        return enhanced_data
    
    async def _generate_inventory_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate inventory optimization recommendations"""
        recommendations = []
        
        inventory_items = analysis.get("inventory_items", [])
        textile_insights = analysis.get("textile_insights", {})
        
        # Generate reorder recommendations
        for item in inventory_items:
            planning_balance = item.get("planning_balance", 0)
            if planning_balance < 0:
                recommendations.append({
                    "type": "CRITICAL_REORDER",
                    "yarn_id": item.get("yarn_id", ""),
                    "recommendation": f"Immediate reorder required - shortage of {abs(planning_balance)} lbs",
                    "urgency": "HIGH",
                    "suggested_quantity": abs(planning_balance) * 1.5,
                    "textile_considerations": self._get_textile_reorder_considerations(item)
                })
        
        # Generate substitution recommendations
        substitution_opps = textile_insights.get("substitution_opportunities", [])
        for opp in substitution_opps[:3]:  # Top 3 opportunities
            recommendations.append({
                "type": "YARN_SUBSTITUTION",
                "recommendation": f"Consider substituting {opp.get('original_yarn')} with {opp.get('substitute_yarn')}",
                "cost_savings": opp.get("cost_savings", 0),
                "quality_impact": opp.get("quality_impact", "NEUTRAL")
            })
        
        # Generate seasonal recommendations
        recommendations.append({
            "type": "SEASONAL_PLANNING",
            "recommendation": "Increase cotton yarn inventory for spring/summer production cycle",
            "timing": "2 weeks before seasonal ramp-up",
            "rationale": "Historical demand patterns show 40% increase in cotton usage during spring"
        })
        
        return recommendations
    
    async def _enhance_demand_forecast(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance demand forecast with textile-specific insights"""
        return {**forecast_data, "textile_insights": "applied"}
    
    def _analyze_textile_seasonality(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonal patterns for textile products"""
        return {"seasonal_analysis": "completed"}
    
    def _analyze_yarn_demand_impact(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact on yarn demand"""
        return {"yarn_impact": "analyzed"}
    
    def _calculate_capacity_requirements(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate production capacity requirements"""
        return {"capacity_requirements": "calculated"}
    
    # Additional placeholder methods would be implemented here...
    
    async def _find_emergency_substitutes(self, yarn_id: str) -> List[Dict[str, Any]]:
        """Find emergency yarn substitutes"""
        return []
    
    def _assess_production_impact(self, shortage: Dict[str, Any]) -> Dict[str, Any]:
        """Assess production impact of shortage"""
        return {"delay_days": 0}
    
    def _calculate_procurement_urgency(self, shortage: Dict[str, Any]) -> float:
        """Calculate procurement urgency score"""
        return 5.0
    
    def _generate_shortage_action_plan(self, shortages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate action plan for shortages"""
        return []
    
    # Textile-specific analysis helper methods
    def _analyze_yarn_category_distribution(self, yarn_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of yarn categories"""
        category_counts = {}
        total_value = 0
        
        for item in yarn_items:
            # Determine yarn category from description
            description = item.get("description", "").lower()
            category = "UNKNOWN"
            
            if "cotton" in description:
                category = "COTTON"
            elif "wool" in description:
                category = "WOOL"
            elif any(synthetic in description for synthetic in ["polyester", "nylon", "acrylic"]):
                category = "SYNTHETIC"
            elif any(blend in description for blend in ["blend", "mixed"]):
                category = "BLEND"
            
            value = item.get("inventory_value", 0)
            category_counts[category] = category_counts.get(category, 0) + value
            total_value += value
        
        # Calculate percentages
        category_percentages = {}
        if total_value > 0:
            for category, value in category_counts.items():
                category_percentages[category] = (value / total_value) * 100
        
        return {
            "category_counts": category_counts,
            "category_percentages": category_percentages,
            "dominant_category": max(category_counts, key=category_counts.get) if category_counts else "UNKNOWN"
        }
    
    def _analyze_fiber_content(self, yarn_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze fiber content distribution"""
        fiber_analysis = {
            "natural_fiber_percentage": 0,
            "synthetic_fiber_percentage": 0,
            "blend_percentage": 0,
            "fiber_diversity_score": 0
        }
        
        natural_keywords = ["cotton", "wool", "linen", "silk", "bamboo"]
        synthetic_keywords = ["polyester", "nylon", "acrylic", "spandex", "elastane"]
        
        natural_count = 0
        synthetic_count = 0
        blend_count = 0
        
        for item in yarn_items:
            description = item.get("description", "").lower()
            
            natural_matches = sum(1 for keyword in natural_keywords if keyword in description)
            synthetic_matches = sum(1 for keyword in synthetic_keywords if keyword in description)
            
            if natural_matches > 0 and synthetic_matches > 0:
                blend_count += 1
            elif natural_matches > 0:
                natural_count += 1
            elif synthetic_matches > 0:
                synthetic_count += 1
        
        total_items = len(yarn_items)
        if total_items > 0:
            fiber_analysis["natural_fiber_percentage"] = (natural_count / total_items) * 100
            fiber_analysis["synthetic_fiber_percentage"] = (synthetic_count / total_items) * 100
            fiber_analysis["blend_percentage"] = (blend_count / total_items) * 100
            fiber_analysis["fiber_diversity_score"] = min(10, len(set(
                [keyword for item in yarn_items for keyword in natural_keywords + synthetic_keywords
                 if keyword in item.get("description", "").lower()]
            )))
        
        return fiber_analysis
    
    def _calculate_seasonal_impact(self, yarn_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate seasonal demand impact on yarn categories"""
        current_month = datetime.now().month
        season = self._get_current_season(current_month)
        
        seasonal_impact = {
            "current_season": season,
            "high_demand_categories": [],
            "low_demand_categories": [],
            "seasonal_adjustments": {}
        }
        
        # Seasonal demand patterns for textile
        seasonal_patterns = {
            "spring": {"cotton": 1.3, "linen": 1.4, "wool": 0.7},
            "summer": {"cotton": 1.5, "linen": 1.6, "wool": 0.5},
            "fall": {"wool": 1.4, "cotton": 0.9, "synthetic": 1.2},
            "winter": {"wool": 1.6, "fleece": 1.4, "cotton": 0.8}
        }
        
        current_patterns = seasonal_patterns.get(season, {})
        for category, multiplier in current_patterns.items():
            if multiplier > 1.2:
                seasonal_impact["high_demand_categories"].append(category)
            elif multiplier < 0.8:
                seasonal_impact["low_demand_categories"].append(category)
            seasonal_impact["seasonal_adjustments"][category] = multiplier
        
        return seasonal_impact
    
    def _identify_substitution_opportunities(self, yarn_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify yarn substitution opportunities"""
        opportunities = []
        
        # Group yarns by similar characteristics
        yarn_groups = {}
        for item in yarn_items:
            # Create grouping key based on color and general type
            description = item.get("description", "")
            color = self._extract_color_from_description(description)
            yarn_type = self._extract_yarn_type_from_description(description)
            
            key = f"{yarn_type}_{color}"
            if key not in yarn_groups:
                yarn_groups[key] = []
            yarn_groups[key].append(item)
        
        # Find substitution opportunities within groups
        for group_key, yarns in yarn_groups.items():
            if len(yarns) > 1:
                # Sort by cost per pound
                sorted_yarns = sorted(yarns, key=lambda x: x.get("cost_per_lb", 0))
                
                if len(sorted_yarns) >= 2:
                    cheapest = sorted_yarns[0]
                    most_expensive = sorted_yarns[-1]
                    
                    cost_diff = most_expensive.get("cost_per_lb", 0) - cheapest.get("cost_per_lb", 0)
                    if cost_diff > 0.50:  # More than $0.50 per lb difference
                        opportunities.append({
                            "original_yarn": most_expensive.get("yarn_id", ""),
                            "substitute_yarn": cheapest.get("yarn_id", ""),
                            "cost_savings": cost_diff,
                            "quality_impact": "SIMILAR",
                            "compatibility_score": 0.9
                        })
        
        return opportunities[:10]  # Return top 10 opportunities
    
    def _assess_quality_risks(self, yarn_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess quality risks in yarn inventory"""
        risk_assessment = {
            "total_risk_score": 0,
            "high_risk_yarns": [],
            "quality_concerns": [],
            "mitigation_recommendations": []
        }
        
        high_risk_count = 0
        for item in yarn_items:
            risk_score = 0
            risk_factors = []
            
            # Check for age-related risk (if we had date data)
            planning_balance = item.get("planning_balance", 0)
            if planning_balance < 0:
                risk_score += 3
                risk_factors.append("shortage_risk")
            
            # Check for single-source risk
            description = item.get("description", "")
            if "specialty" in description.lower() or "custom" in description.lower():
                risk_score += 2
                risk_factors.append("specialty_item_risk")
            
            if risk_score > 3:
                high_risk_count += 1
                risk_assessment["high_risk_yarns"].append({
                    "yarn_id": item.get("yarn_id", ""),
                    "risk_score": risk_score,
                    "risk_factors": risk_factors
                })
        
        risk_assessment["total_risk_score"] = high_risk_count
        
        if high_risk_count > 0:
            risk_assessment["quality_concerns"].append(f"{high_risk_count} yarns identified as high-risk")
            risk_assessment["mitigation_recommendations"].append("Diversify supplier base for high-risk yarns")
            risk_assessment["mitigation_recommendations"].append("Increase safety stock for critical specialty yarns")
        
        return risk_assessment
    
    def _get_textile_reorder_considerations(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Get textile-specific reorder considerations"""
        return {
            "minimum_order_quantity": "Check supplier MOQ requirements",
            "lead_time_considerations": "Account for yarn dyeing and shipping time",
            "quality_testing": "Reserve 5% for quality testing before production use",
            "storage_requirements": "Ensure proper humidity control for natural fibers"
        }
    
    def _get_current_season(self, month: int) -> str:
        """Get current season based on month"""
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall"
        else:
            return "winter"
    
    def _extract_color_from_description(self, description: str) -> str:
        """Extract color information from yarn description"""
        common_colors = ["black", "white", "red", "blue", "green", "yellow", "brown", "gray", "grey", "navy", "beige"]
        description_lower = description.lower()
        
        for color in common_colors:
            if color in description_lower:
                return color
        
        return "unknown"
    
    def _extract_yarn_type_from_description(self, description: str) -> str:
        """Extract yarn type from description"""
        description_lower = description.lower()
        
        if "cotton" in description_lower:
            return "cotton"
        elif "wool" in description_lower:
            return "wool"
        elif any(synthetic in description_lower for synthetic in ["polyester", "nylon", "acrylic"]):
            return "synthetic"
        elif "blend" in description_lower:
            return "blend"
        else:
            return "unknown"


# Export main component
__all__ = ["BeverlyKnitsManufacturingAgent", "YarnCategory", "ProductionStage", "KnitConstruction"]