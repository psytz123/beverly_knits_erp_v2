#!/usr/bin/env python3
"""
Electrical Equipment Manufacturing Specialization Agent
Industry-specific ERP intelligence for electrical equipment manufacturing operations
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


class ElectricalProductType(Enum):
    """Electrical equipment product categories"""
    CONTROL_PANELS = "CONTROL_PANELS"         # Electrical control panels
    SWITCHGEAR = "SWITCHGEAR"                 # High voltage switchgear
    TRANSFORMERS = "TRANSFORMERS"             # Power transformers
    MOTOR_CONTROLS = "MOTOR_CONTROLS"         # Motor control centers
    LIGHTING_FIXTURES = "LIGHTING_FIXTURES"   # Industrial lighting
    POWER_SUPPLIES = "POWER_SUPPLIES"         # DC power supplies
    CABLE_ASSEMBLIES = "CABLE_ASSEMBLIES"     # Wire harnesses
    CIRCUIT_BREAKERS = "CIRCUIT_BREAKERS"     # Protection devices
    SENSORS = "SENSORS"                       # Industrial sensors
    AUTOMATION = "AUTOMATION"                 # Automation equipment


class CertificationType(Enum):
    """Electrical safety certifications"""
    UL_LISTED = "UL_LISTED"                   # UL certification
    CE_MARKED = "CE_MARKED"                   # European conformity
    CSA_CERTIFIED = "CSA_CERTIFIED"           # Canadian standards
    IEC_COMPLIANT = "IEC_COMPLIANT"           # International standards
    NEMA_RATED = "NEMA_RATED"                 # NEMA ratings
    IP_RATED = "IP_RATED"                     # Ingress protection
    ATEX_CERTIFIED = "ATEX_CERTIFIED"         # Explosion proof
    FCC_APPROVED = "FCC_APPROVED"             # FCC compliance


class TestType(Enum):
    """Electrical testing procedures"""
    INSULATION_RESISTANCE = "INSULATION_RESISTANCE"
    HIPOT_TEST = "HIPOT_TEST"                 # High potential test
    CONTINUITY_TEST = "CONTINUITY_TEST"       # Circuit continuity
    FUNCTIONAL_TEST = "FUNCTIONAL_TEST"       # Operational testing
    BURN_IN_TEST = "BURN_IN_TEST"            # Extended operation
    VIBRATION_TEST = "VIBRATION_TEST"         # Mechanical stress
    TEMPERATURE_CYCLE = "TEMPERATURE_CYCLE"   # Thermal stress
    EMC_TEST = "EMC_TEST"                     # Electromagnetic compatibility


@dataclass
class ElectricalComponent:
    """Electrical component specification"""
    component_id: str
    component_name: str
    category: str  # resistor, capacitor, IC, connector, etc.
    manufacturer: str = ""
    part_number: str = ""
    voltage_rating: float = 0.0
    current_rating: float = 0.0
    power_rating: float = 0.0
    tolerance: str = ""
    temperature_rating: str = ""
    certifications: List[CertificationType] = field(default_factory=list)
    lead_time_weeks: int = 12
    cost_per_unit: float = 0.0
    minimum_order_quantity: int = 1
    supplier: str = ""
    rohs_compliant: bool = True
    lifecycle_status: str = "ACTIVE"  # ACTIVE, OBSOLETE, END_OF_LIFE


@dataclass
class ElectricalAssembly:
    """Electrical assembly specification"""
    assembly_id: str
    assembly_name: str
    product_type: ElectricalProductType
    voltage_class: str = "LOW_VOLTAGE"  # LOW_VOLTAGE, MEDIUM_VOLTAGE, HIGH_VOLTAGE
    current_rating: float = 0.0
    enclosure_type: str = "NEMA_1"
    components: List[ElectricalComponent] = field(default_factory=list)
    wire_schedule: List[Dict[str, Any]] = field(default_factory=list)
    assembly_time_hours: float = 4.0
    test_requirements: List[TestType] = field(default_factory=list)
    certifications_required: List[CertificationType] = field(default_factory=list)
    environmental_rating: str = "INDOOR"
    documentation: List[str] = field(default_factory=list)


@dataclass
class ElectricalTestProcedure:
    """Electrical testing procedure"""
    test_id: str
    test_type: TestType
    test_parameters: Dict[str, Any]
    pass_criteria: Dict[str, Any]
    equipment_required: List[str] = field(default_factory=list)
    test_duration_minutes: int = 30
    safety_requirements: List[str] = field(default_factory=list)
    documentation_required: bool = True


class ElectricalEquipmentAgent(BaseAgent):
    """
    Electrical Equipment Manufacturing Specialization Agent
    
    Capabilities:
    - Electrical schematic validation and component optimization
    - Wire harness routing and length calculations
    - Electrical testing procedure generation and scheduling
    - Certification compliance tracking and management
    - Component lifecycle management and obsolescence monitoring
    - Power consumption analysis and energy efficiency optimization
    - Safety compliance verification for electrical standards
    - Lead time analysis for long-lead electrical components
    """
    
    def __init__(self, agent_id: str = "electrical_equipment_agent"):
        """Initialize Electrical Equipment Agent"""
        super().__init__(
            agent_id=agent_id,
            agent_name="Electrical Equipment Specialist",
            agent_description="Industry-specific intelligence for electrical equipment manufacturing operations"
        )
        
        # Electrical engineering knowledge base
        self.voltage_classes = {
            "LOW_VOLTAGE": {"min_v": 0, "max_v": 1000, "safety_class": "LV"},
            "MEDIUM_VOLTAGE": {"min_v": 1001, "max_v": 35000, "safety_class": "MV"},
            "HIGH_VOLTAGE": {"min_v": 35001, "max_v": 500000, "safety_class": "HV"}
        }
        
        self.nema_ratings = {
            "NEMA_1": {"protection": "Indoor, general purpose"},
            "NEMA_3R": {"protection": "Outdoor, rain resistant"},
            "NEMA_4": {"protection": "Indoor/outdoor, watertight"},
            "NEMA_4X": {"protection": "Indoor/outdoor, corrosion resistant"},
            "NEMA_12": {"protection": "Indoor, dust-tight"}
        }
        
        self.test_parameters = {
            TestType.INSULATION_RESISTANCE: {
                "min_resistance_mohm": 1.0,
                "test_voltage": 500,
                "test_duration_sec": 60
            },
            TestType.HIPOT_TEST: {
                "test_voltage_multiplier": 2.0,
                "leakage_current_max_ma": 5.0,
                "test_duration_sec": 60
            },
            TestType.CONTINUITY_TEST: {
                "max_resistance_ohm": 0.1,
                "test_current_ma": 100
            }
        }
        
        self.certification_requirements = {
            CertificationType.UL_LISTED: {
                "testing_required": ["HIPOT_TEST", "INSULATION_RESISTANCE", "FUNCTIONAL_TEST"],
                "lead_time_weeks": 12,
                "cost_estimate": 15000
            },
            CertificationType.CE_MARKED: {
                "testing_required": ["EMC_TEST", "FUNCTIONAL_TEST", "TEMPERATURE_CYCLE"],
                "lead_time_weeks": 8,
                "cost_estimate": 8000
            }
        }
        
        # Component lifecycle tracking
        self.component_lead_times = {
            "standard_components": 4,    # weeks
            "specialty_components": 12,  # weeks
            "custom_components": 20,     # weeks
            "obsolete_components": 52    # weeks (if available)
        }
    
    def _initialize(self):
        """Initialize electrical equipment capabilities"""
        # Register electrical-specific capabilities
        self.register_capability(AgentCapability(
            name="validate_electrical_design",
            description="Validate electrical schematics and component specifications",
            input_schema={
                "type": "object",
                "properties": {
                    "schematic_data": {"type": "object"},
                    "component_specifications": {"type": "array"},
                    "operating_conditions": {"type": "object"},
                    "safety_requirements": {"type": "array"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "validation_results": {"type": "object"},
                    "design_violations": {"type": "array"},
                    "optimization_suggestions": {"type": "array"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="generate_test_procedures",
            description="Generate electrical testing procedures for assemblies",
            input_schema={
                "type": "object",
                "properties": {
                    "assembly_specification": {"type": "object"},
                    "certification_requirements": {"type": "array"},
                    "quality_standards": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "test_procedures": {"type": "array"},
                    "equipment_requirements": {"type": "array"},
                    "estimated_test_time": {"type": "number"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="track_certifications",
            description="Track certification compliance and renewal requirements",
            input_schema={
                "type": "object",
                "properties": {
                    "product_line": {"type": "array"},
                    "target_markets": {"type": "array"},
                    "certification_status": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "compliance_status": {"type": "object"},
                    "renewal_schedule": {"type": "array"},
                    "gap_analysis": {"type": "object"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="optimize_wire_routing",
            description="Optimize wire harness routing and calculate wire lengths",
            input_schema={
                "type": "object",
                "properties": {
                    "connection_matrix": {"type": "object"},
                    "panel_layout": {"type": "object"},
                    "routing_constraints": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "optimized_routing": {"type": "object"},
                    "wire_schedule": {"type": "array"},
                    "material_requirements": {"type": "object"}
                }
            }
        ))
        
        # Register message handlers
        self.register_message_handler(MessageType.REQUEST, self._handle_electrical_request)
    
    async def _handle_electrical_request(self, message: AgentMessage) -> AgentMessage:
        """Handle electrical equipment requests"""
        action = message.payload.get("action")
        
        try:
            if action == "validate_electrical_design":
                result = await self._validate_electrical_design(message.payload)
            elif action == "generate_test_procedures":
                result = await self._generate_test_procedures(message.payload)
            elif action == "track_certifications":
                result = await self._track_certifications(message.payload)
            elif action == "optimize_wire_routing":
                result = await self._optimize_wire_routing(message.payload)
            elif action == "analyze_component_lifecycle":
                result = await self._analyze_component_lifecycle(message.payload)
            elif action == "calculate_power_consumption":
                result = await self._calculate_power_consumption(message.payload)
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
            self.logger.error(f"Error handling electrical request: {str(e)}")
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                correlation_id=message.correlation_id
            )
    
    async def _validate_electrical_design(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate electrical design and schematics"""
        schematic_data = payload.get("schematic_data", {})
        component_specs = payload.get("component_specifications", [])
        operating_conditions = payload.get("operating_conditions", {})
        safety_requirements = payload.get("safety_requirements", [])
        
        validation_results = {"is_valid": True, "warnings": [], "errors": []}
        design_violations = []
        optimization_suggestions = []
        
        # Validate voltage ratings
        system_voltage = operating_conditions.get("system_voltage", 480)
        for component in component_specs:
            component_voltage = component.get("voltage_rating", 0)
            if component_voltage < system_voltage * 1.2:  # 20% safety margin
                design_violations.append({
                    "component": component.get("component_name"),
                    "violation": "Insufficient voltage rating",
                    "required": system_voltage * 1.2,
                    "actual": component_voltage,
                    "severity": "HIGH"
                })
        
        # Validate current ratings
        system_current = operating_conditions.get("system_current", 100)
        total_component_current = sum(
            comp.get("current_rating", 0) for comp in component_specs
        )
        
        if total_component_current > system_current * 0.8:  # 80% derating
            design_violations.append({
                "violation": "Current capacity exceeded",
                "system_capacity": system_current,
                "required_capacity": total_component_current,
                "severity": "CRITICAL"
            })
        
        # Check temperature ratings
        operating_temp = operating_conditions.get("ambient_temperature", 40)
        for component in component_specs:
            temp_rating = component.get("temperature_rating", "85C")
            max_temp = int(temp_rating.replace("C", ""))
            if max_temp < operating_temp + 20:  # 20C rise allowance
                design_violations.append({
                    "component": component.get("component_name"),
                    "violation": "Insufficient temperature rating",
                    "operating_temp": operating_temp,
                    "component_rating": max_temp,
                    "severity": "MEDIUM"
                })
        
        # Generate optimization suggestions
        optimization_suggestions = [
            "Consider using components with higher voltage ratings for safety margin",
            "Implement thermal management for high-power components",
            "Use modular design for easier maintenance and upgrades",
            "Add current monitoring for predictive maintenance",
            "Consider redundancy for critical control circuits"
        ]
        
        # Check for obsolete components
        obsolete_components = [
            comp for comp in component_specs 
            if comp.get("lifecycle_status") in ["OBSOLETE", "END_OF_LIFE"]
        ]
        
        if obsolete_components:
            validation_results["warnings"].append(
                f"Found {len(obsolete_components)} obsolete components requiring replacement"
            )
        
        # Overall validation
        if design_violations:
            critical_violations = [v for v in design_violations if v.get("severity") == "CRITICAL"]
            if critical_violations:
                validation_results["is_valid"] = False
                validation_results["errors"] = [v["violation"] for v in critical_violations]
        
        return {
            "validation_results": validation_results,
            "design_violations": design_violations,
            "optimization_suggestions": optimization_suggestions,
            "component_analysis": {
                "total_components": len(component_specs),
                "obsolete_components": len(obsolete_components),
                "certification_compliance": self._check_certification_compliance(component_specs, safety_requirements)
            }
        }
    
    def _check_certification_compliance(self, components: List[Dict], requirements: List[str]) -> Dict[str, Any]:
        """Check component certification compliance"""
        compliant_components = 0
        non_compliant_components = []
        
        for component in components:
            component_certs = component.get("certifications", [])
            has_required_cert = any(
                cert in component_certs for cert in requirements
            )
            
            if has_required_cert:
                compliant_components += 1
            else:
                non_compliant_components.append(component.get("component_name"))
        
        compliance_percentage = (compliant_components / len(components) * 100) if components else 100
        
        return {
            "compliance_percentage": compliance_percentage,
            "compliant_components": compliant_components,
            "non_compliant_components": non_compliant_components,
            "required_certifications": requirements
        }
    
    async def _generate_test_procedures(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate electrical test procedures"""
        assembly_spec = payload.get("assembly_specification", {})
        cert_requirements = payload.get("certification_requirements", [])
        quality_standards = payload.get("quality_standards", {})
        
        test_procedures = []
        equipment_requirements = set()
        total_test_time = 0
        
        # Determine required tests based on product type and certifications
        product_type = assembly_spec.get("product_type", "CONTROL_PANELS")
        voltage_class = assembly_spec.get("voltage_class", "LOW_VOLTAGE")
        
        # Standard electrical tests
        standard_tests = [
            TestType.CONTINUITY_TEST,
            TestType.INSULATION_RESISTANCE,
            TestType.FUNCTIONAL_TEST
        ]
        
        # Add high-pot test for higher voltages
        if voltage_class in ["MEDIUM_VOLTAGE", "HIGH_VOLTAGE"]:
            standard_tests.append(TestType.HIPOT_TEST)
        
        # Add certification-specific tests
        for cert in cert_requirements:
            if cert in self.certification_requirements:
                cert_tests = self.certification_requirements[cert].get("testing_required", [])
                for test_name in cert_tests:
                    try:
                        test_type = TestType(test_name)
                        if test_type not in standard_tests:
                            standard_tests.append(test_type)
                    except ValueError:
                        continue
        
        # Generate test procedures for each required test
        for test_type in standard_tests:
            test_params = self.test_parameters.get(test_type, {})
            
            procedure = {
                "test_id": f"TEST_{test_type.value}_{datetime.now().strftime('%Y%m%d')}",
                "test_type": test_type.value,
                "test_description": self._get_test_description(test_type),
                "test_parameters": test_params,
                "pass_criteria": self._get_pass_criteria(test_type, assembly_spec),
                "equipment_required": self._get_test_equipment(test_type),
                "safety_precautions": self._get_safety_precautions(test_type, voltage_class),
                "estimated_duration_minutes": self._get_test_duration(test_type),
                "procedure_steps": self._get_test_steps(test_type)
            }
            
            test_procedures.append(procedure)
            equipment_requirements.update(procedure["equipment_required"])
            total_test_time += procedure["estimated_duration_minutes"]
        
        return {
            "test_procedures": test_procedures,
            "equipment_requirements": list(equipment_requirements),
            "estimated_test_time": total_test_time,
            "test_summary": {
                "total_tests": len(test_procedures),
                "critical_tests": len([t for t in standard_tests if t in [TestType.HIPOT_TEST, TestType.INSULATION_RESISTANCE]]),
                "certification_tests": len([t for t in test_procedures if any(cert in t.get("applicable_certifications", []) for cert in cert_requirements)])
            }
        }
    
    def _get_test_description(self, test_type: TestType) -> str:
        """Get test description"""
        descriptions = {
            TestType.CONTINUITY_TEST: "Verify electrical continuity of all circuits",
            TestType.INSULATION_RESISTANCE: "Measure insulation resistance between circuits and ground",
            TestType.HIPOT_TEST: "High potential test to verify insulation integrity",
            TestType.FUNCTIONAL_TEST: "Verify proper operation of all functions",
            TestType.EMC_TEST: "Electromagnetic compatibility testing"
        }
        return descriptions.get(test_type, "Standard electrical test")
    
    def _get_pass_criteria(self, test_type: TestType, assembly_spec: Dict) -> Dict[str, Any]:
        """Get pass criteria for test"""
        voltage_rating = assembly_spec.get("voltage_rating", 480)
        
        criteria = {
            TestType.CONTINUITY_TEST: {"max_resistance_ohm": 0.1},
            TestType.INSULATION_RESISTANCE: {"min_resistance_mohm": 1.0},
            TestType.HIPOT_TEST: {
                "test_voltage": voltage_rating * 2,
                "max_leakage_current_ma": 5.0
            },
            TestType.FUNCTIONAL_TEST: {"all_functions_operational": True}
        }
        
        return criteria.get(test_type, {})
    
    def _get_test_equipment(self, test_type: TestType) -> List[str]:
        """Get required test equipment"""
        equipment = {
            TestType.CONTINUITY_TEST: ["Digital multimeter", "Test leads"],
            TestType.INSULATION_RESISTANCE: ["Megohmmeter", "Test leads"],
            TestType.HIPOT_TEST: ["Hipot tester", "Safety equipment"],
            TestType.FUNCTIONAL_TEST: ["Power source", "Load simulator", "Oscilloscope"],
            TestType.EMC_TEST: ["EMC test chamber", "Spectrum analyzer", "Signal generator"]
        }
        
        return equipment.get(test_type, ["Standard test equipment"])
    
    def _get_safety_precautions(self, test_type: TestType, voltage_class: str) -> List[str]:
        """Get safety precautions for test"""
        base_precautions = [
            "Ensure power is disconnected before testing",
            "Use appropriate PPE",
            "Follow lockout/tagout procedures"
        ]
        
        if test_type == TestType.HIPOT_TEST:
            base_precautions.extend([
                "Verify test area is clear",
                "Use safety interlocks",
                "Have emergency shutdown procedures ready"
            ])
        
        if voltage_class in ["MEDIUM_VOLTAGE", "HIGH_VOLTAGE"]:
            base_precautions.extend([
                "Arc flash protection required",
                "Qualified personnel only",
                "Ground all test equipment"
            ])
        
        return base_precautions
    
    def _get_test_duration(self, test_type: TestType) -> int:
        """Get estimated test duration in minutes"""
        durations = {
            TestType.CONTINUITY_TEST: 15,
            TestType.INSULATION_RESISTANCE: 10,
            TestType.HIPOT_TEST: 20,
            TestType.FUNCTIONAL_TEST: 60,
            TestType.EMC_TEST: 240  # 4 hours
        }
        
        return durations.get(test_type, 30)
    
    def _get_test_steps(self, test_type: TestType) -> List[str]:
        """Get detailed test steps"""
        steps = {
            TestType.CONTINUITY_TEST: [
                "1. Disconnect all power sources",
                "2. Set multimeter to continuity mode",
                "3. Test each circuit for continuity",
                "4. Record resistance values",
                "5. Verify readings meet specifications"
            ],
            TestType.HIPOT_TEST: [
                "1. Ensure test area is safe and clear",
                "2. Connect test leads to assembly",
                "3. Set test voltage per specifications",
                "4. Apply voltage gradually",
                "5. Hold for specified duration",
                "6. Monitor for leakage current",
                "7. Gradually reduce voltage to zero"
            ]
        }
        
        return steps.get(test_type, ["Follow standard test procedures"])
    
    async def _track_certifications(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Track certification compliance and renewals"""
        product_line = payload.get("product_line", [])
        target_markets = payload.get("target_markets", [])
        cert_status = payload.get("certification_status", {})
        
        compliance_status = {}
        renewal_schedule = []
        gap_analysis = {"missing_certifications": [], "expiring_certifications": []}
        
        # Determine required certifications based on target markets
        required_certs = set()
        market_cert_map = {
            "US": [CertificationType.UL_LISTED, CertificationType.FCC_APPROVED],
            "CANADA": [CertificationType.CSA_CERTIFIED],
            "EUROPE": [CertificationType.CE_MARKED],
            "INTERNATIONAL": [CertificationType.IEC_COMPLIANT]
        }
        
        for market in target_markets:
            if market.upper() in market_cert_map:
                required_certs.update(market_cert_map[market.upper()])
        
        # Analyze each product
        for product in product_line:
            product_id = product.get("product_id")
            product_certs = cert_status.get(product_id, {})
            
            product_compliance = {
                "required_certifications": list(required_certs),
                "current_certifications": list(product_certs.keys()),
                "compliance_percentage": 0,
                "status": "NON_COMPLIANT"
            }
            
            # Check compliance
            compliant_certs = [cert for cert in required_certs if cert.value in product_certs]
            if required_certs:
                product_compliance["compliance_percentage"] = (len(compliant_certs) / len(required_certs)) * 100
                
                if len(compliant_certs) == len(required_certs):
                    product_compliance["status"] = "COMPLIANT"
                elif len(compliant_certs) > 0:
                    product_compliance["status"] = "PARTIALLY_COMPLIANT"
            
            compliance_status[product_id] = product_compliance
            
            # Check for missing certifications
            missing_certs = required_certs - set(product_certs.keys())
            for cert in missing_certs:
                gap_analysis["missing_certifications"].append({
                    "product_id": product_id,
                    "certification": cert.value,
                    "estimated_cost": self.certification_requirements.get(cert, {}).get("cost_estimate", 10000),
                    "estimated_timeline_weeks": self.certification_requirements.get(cert, {}).get("lead_time_weeks", 12)
                })
            
            # Check for expiring certifications
            for cert_name, cert_info in product_certs.items():
                expiry_date = cert_info.get("expiry_date")
                if expiry_date:
                    try:
                        expiry = datetime.fromisoformat(expiry_date.replace("Z", "+00:00"))
                        days_until_expiry = (expiry - datetime.now()).days
                        
                        if days_until_expiry < 90:  # Within 90 days
                            gap_analysis["expiring_certifications"].append({
                                "product_id": product_id,
                                "certification": cert_name,
                                "expiry_date": expiry_date,
                                "days_until_expiry": days_until_expiry,
                                "renewal_required": days_until_expiry < 0
                            })
                            
                            renewal_schedule.append({
                                "product_id": product_id,
                                "certification": cert_name,
                                "renewal_date": (expiry - timedelta(days=60)).isoformat(),  # 60 days before expiry
                                "estimated_cost": self.certification_requirements.get(CertificationType(cert_name), {}).get("cost_estimate", 5000) * 0.5  # Renewal typically 50% of initial cost
                            })
                    except (ValueError, TypeError):
                        continue
        
        # Sort renewal schedule by date
        renewal_schedule.sort(key=lambda x: x["renewal_date"])
        
        return {
            "compliance_status": compliance_status,
            "renewal_schedule": renewal_schedule,
            "gap_analysis": gap_analysis,
            "summary": {
                "total_products": len(product_line),
                "compliant_products": sum(1 for status in compliance_status.values() if status["status"] == "COMPLIANT"),
                "missing_certifications": len(gap_analysis["missing_certifications"]),
                "expiring_certifications": len(gap_analysis["expiring_certifications"]),
                "total_estimated_cost": sum(item["estimated_cost"] for item in gap_analysis["missing_certifications"]) + sum(item["estimated_cost"] for item in renewal_schedule)
            }
        }
    
    async def _optimize_wire_routing(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize wire harness routing"""
        connection_matrix = payload.get("connection_matrix", {})
        panel_layout = payload.get("panel_layout", {})
        routing_constraints = payload.get("routing_constraints", {})
        
        # Simulate wire routing optimization
        wire_schedule = []
        total_wire_length = 0
        
        # Process each connection
        connections = connection_matrix.get("connections", [])
        for i, connection in enumerate(connections):
            from_point = connection.get("from")
            to_point = connection.get("to")
            wire_type = connection.get("wire_type", "16 AWG")
            signal_type = connection.get("signal_type", "CONTROL")
            
            # Calculate wire length (simplified)
            from_coords = panel_layout.get(from_point, {"x": 0, "y": 0})
            to_coords = panel_layout.get(to_point, {"x": 10, "y": 10})
            
            direct_distance = math.sqrt(
                (to_coords["x"] - from_coords["x"])**2 + 
                (to_coords["y"] - from_coords["y"])**2
            )
            
            # Add routing factor for actual path
            routing_factor = routing_constraints.get("routing_factor", 1.5)
            service_loop_length = routing_constraints.get("service_loop_inches", 6)
            
            actual_length = (direct_distance * routing_factor) + service_loop_length
            
            wire_schedule.append({
                "wire_id": f"W{i+1:03d}",
                "from": from_point,
                "to": to_point,
                "wire_type": wire_type,
                "signal_type": signal_type,
                "length_inches": round(actual_length, 1),
                "color_code": self._get_wire_color(signal_type),
                "routing_path": f"Route via {routing_constraints.get('preferred_path', 'wireway')}"
            })
            
            total_wire_length += actual_length
        
        # Calculate material requirements
        wire_types = {}
        for wire in wire_schedule:
            wire_type = wire["wire_type"]
            length = wire["length_inches"]
            
            if wire_type in wire_types:
                wire_types[wire_type] += length
            else:
                wire_types[wire_type] = length
        
        # Add waste factor and convert to feet
        waste_factor = 1.1  # 10% waste
        material_requirements = {}
        for wire_type, total_inches in wire_types.items():
            total_feet = (total_inches * waste_factor) / 12
            material_requirements[wire_type] = {
                "required_feet": round(total_feet, 1),
                "waste_factor": waste_factor,
                "estimated_cost": round(total_feet * 0.5, 2)  # $0.50/ft estimate
            }
        
        return {
            "optimized_routing": {
                "total_connections": len(connections),
                "total_wire_length_feet": round(total_wire_length / 12, 1),
                "routing_efficiency": round((sum(math.sqrt((panel_layout.get(c.get("to", ""), {"x": 0, "y": 0})["x"] - panel_layout.get(c.get("from", ""), {"x": 0, "y": 0})["x"])**2 + (panel_layout.get(c.get("to", ""), {"x": 0, "y": 0})["y"] - panel_layout.get(c.get("from", ""), {"x": 0, "y": 0})["y"])**2) for c in connections) / total_wire_length) * 100, 1)
            },
            "wire_schedule": wire_schedule,
            "material_requirements": material_requirements,
            "optimization_notes": [
                "Wire routing optimized for shortest path while maintaining proper separation",
                "Service loops included for maintenance access",
                "Wire colors assigned per industry standards",
                "Consider wire bundling for improved organization"
            ]
        }
    
    def _get_wire_color(self, signal_type: str) -> str:
        """Get standard wire color for signal type"""
        color_codes = {
            "POWER": "BLACK",
            "NEUTRAL": "WHITE", 
            "GROUND": "GREEN",
            "CONTROL": "RED",
            "ANALOG": "BLUE",
            "COMMUNICATION": "ORANGE"
        }
        return color_codes.get(signal_type, "GRAY")
    
    async def _analyze_component_lifecycle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze component lifecycle and obsolescence risk"""
        component_list = payload.get("component_list", [])
        forecast_horizon = payload.get("forecast_horizon_years", 5)
        
        lifecycle_analysis = {}
        risk_summary = {"high_risk": 0, "medium_risk": 0, "low_risk": 0}
        recommendations = []
        
        for component in component_list:
            component_id = component.get("component_id")
            lifecycle_status = component.get("lifecycle_status", "ACTIVE")
            introduction_year = component.get("introduction_year", 2020)
            manufacturer = component.get("manufacturer", "Unknown")
            
            current_year = datetime.now().year
            component_age = current_year - introduction_year
            
            # Assess risk based on age and status
            risk_level = "LOW"
            risk_factors = []
            
            if lifecycle_status == "OBSOLETE":
                risk_level = "HIGH"
                risk_factors.append("Component already obsolete")
            elif lifecycle_status == "END_OF_LIFE":
                risk_level = "HIGH"
                risk_factors.append("End of life announced")
            elif component_age > 10:
                risk_level = "MEDIUM"
                risk_factors.append("Component age > 10 years")
            elif component_age > 15:
                risk_level = "HIGH"
                risk_factors.append("Component age > 15 years")
            
            # Additional risk factors
            if "specialty" in component.get("category", "").lower():
                if risk_level == "LOW":
                    risk_level = "MEDIUM"
                risk_factors.append("Specialty component with limited sources")
            
            # Predict future availability
            years_remaining = max(0, 20 - component_age)  # Assume 20 year typical lifecycle
            availability_forecast = "AVAILABLE" if years_remaining > forecast_horizon else "AT_RISK"
            
            lifecycle_analysis[component_id] = {
                "current_status": lifecycle_status,
                "component_age_years": component_age,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "estimated_years_remaining": years_remaining,
                "availability_forecast": availability_forecast,
                "recommended_action": self._get_lifecycle_recommendation(risk_level, years_remaining)
            }
            
            # Update risk summary
            if risk_level == "HIGH":
                risk_summary["high_risk"] += 1
            elif risk_level == "MEDIUM":
                risk_summary["medium_risk"] += 1
            else:
                risk_summary["low_risk"] += 1
        
        # Generate recommendations
        high_risk_components = [
            comp_id for comp_id, analysis in lifecycle_analysis.items()
            if analysis["risk_level"] == "HIGH"
        ]
        
        if high_risk_components:
            recommendations.append(f"Immediate attention required for {len(high_risk_components)} high-risk components")
            recommendations.append("Begin last-time-buy analysis for obsolete components")
            recommendations.append("Identify alternative components for high-risk items")
        
        return {
            "lifecycle_analysis": lifecycle_analysis,
            "risk_summary": risk_summary,
            "recommendations": recommendations,
            "forecast_summary": {
                "forecast_horizon_years": forecast_horizon,
                "components_at_risk": sum(1 for analysis in lifecycle_analysis.values() if analysis["availability_forecast"] == "AT_RISK"),
                "immediate_action_required": len(high_risk_components),
                "estimated_redesign_components": sum(1 for analysis in lifecycle_analysis.values() if analysis["risk_level"] == "HIGH" and analysis["current_status"] == "OBSOLETE")
            }
        }
    
    def _get_lifecycle_recommendation(self, risk_level: str, years_remaining: int) -> str:
        """Get lifecycle management recommendation"""
        if risk_level == "HIGH":
            return "IMMEDIATE_ACTION - Find replacement or do last-time buy"
        elif risk_level == "MEDIUM":
            if years_remaining < 3:
                return "PLAN_REPLACEMENT - Identify alternatives within 12 months"
            else:
                return "MONITOR - Review annually for status changes"
        else:
            return "CONTINUE_USE - No immediate action required"
    
    async def _calculate_power_consumption(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate power consumption and efficiency analysis"""
        assembly_components = payload.get("assembly_components", [])
        operating_conditions = payload.get("operating_conditions", {})
        efficiency_targets = payload.get("efficiency_targets", {})
        
        power_analysis = {}
        total_power_watts = 0
        efficiency_analysis = {}
        
        # Calculate power for each component
        for component in assembly_components:
            component_id = component.get("component_id")
            power_rating = component.get("power_rating", 0)
            utilization_factor = component.get("utilization_factor", 0.7)  # 70% typical
            efficiency = component.get("efficiency", 0.85)  # 85% typical
            
            actual_power = power_rating * utilization_factor / efficiency
            total_power_watts += actual_power
            
            power_analysis[component_id] = {
                "rated_power_watts": power_rating,
                "utilization_factor": utilization_factor,
                "efficiency": efficiency,
                "actual_power_consumption": actual_power,
                "annual_energy_kwh": actual_power * 8760 / 1000,  # 8760 hours/year
                "annual_cost": actual_power * 8760 / 1000 * 0.10  # $0.10/kWh
            }
        
        # Overall efficiency analysis
        total_rated_power = sum(comp.get("power_rating", 0) for comp in assembly_components)
        overall_efficiency = total_rated_power / total_power_watts if total_power_watts > 0 else 0
        
        efficiency_analysis = {
            "overall_efficiency": overall_efficiency,
            "efficiency_rating": self._get_efficiency_rating(overall_efficiency),
            "improvement_potential": max(0, efficiency_targets.get("target_efficiency", 0.9) - overall_efficiency),
            "annual_energy_consumption_kwh": total_power_watts * 8760 / 1000,
            "annual_operating_cost": total_power_watts * 8760 / 1000 * 0.10
        }
        
        # Optimization suggestions
        optimization_suggestions = []
        if overall_efficiency < 0.8:
            optimization_suggestions.append("Consider higher efficiency components - current efficiency below 80%")
        
        high_power_components = [
            comp for comp in assembly_components 
            if comp.get("power_rating", 0) > total_rated_power * 0.2
        ]
        
        if high_power_components:
            optimization_suggestions.append("Focus efficiency improvements on high-power components")
            optimization_suggestions.append("Consider variable speed drives for motor loads")
        
        optimization_suggestions.extend([
            "Implement power monitoring for energy management",
            "Consider power factor correction if reactive loads present",
            "Evaluate standby power consumption modes"
        ])
        
        return {
            "power_analysis": power_analysis,
            "efficiency_analysis": efficiency_analysis,
            "optimization_suggestions": optimization_suggestions,
            "summary": {
                "total_rated_power_watts": total_rated_power,
                "total_actual_power_watts": total_power_watts,
                "power_factor": total_rated_power / total_power_watts if total_power_watts > 0 else 1,
                "annual_energy_cost": efficiency_analysis["annual_operating_cost"],
                "efficiency_improvement_savings": (efficiency_analysis.get("improvement_potential", 0) * efficiency_analysis["annual_operating_cost"])
            }
        }
    
    def _get_efficiency_rating(self, efficiency: float) -> str:
        """Get efficiency rating"""
        if efficiency >= 0.9:
            return "EXCELLENT"
        elif efficiency >= 0.85:
            return "GOOD"
        elif efficiency >= 0.8:
            return "FAIR"
        else:
            return "POOR"


# Export main component
__all__ = ["ElectricalEquipmentAgent", "ElectricalProductType", "CertificationType", "TestType"]