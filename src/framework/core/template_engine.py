#!/usr/bin/env python3
"""
Industry Template Engine
Intelligent template management system for generating industry-specific ERP configurations
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import json
import yaml
import copy
from pathlib import Path

from .abstract_manufacturing import IndustryType, ManufacturingComplexity, ManufacturingFramework

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types of configuration templates"""
    SYSTEM_CORE = "SYSTEM_CORE"
    BUSINESS_RULES = "BUSINESS_RULES" 
    INDUSTRY_SPECIFIC = "INDUSTRY_SPECIFIC"
    INTEGRATION = "INTEGRATION"
    COMPLIANCE = "COMPLIANCE"
    PERFORMANCE = "PERFORMANCE"
    COMPLETE_SOLUTION = "COMPLETE_SOLUTION"


@dataclass
class ConfigurationTemplate:
    """Industry-specific configuration template"""
    template_id: str
    template_name: str
    industry_type: IndustryType
    template_type: TemplateType
    complexity_level: ManufacturingComplexity
    base_configuration: Dict[str, Any]
    customization_rules: List[Dict[str, Any]] = field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    estimated_implementation_hours: float = 8.0
    confidence_score: float = 0.95
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class TemplateCustomization:
    """Customization applied to a template"""
    customization_id: str
    rule_name: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    priority: int = 1
    description: str = ""


class IndustryTemplateEngine:
    """
    Engine for managing and generating industry-specific configuration templates
    Provides intelligent template selection and customization
    """
    
    def __init__(self):
        self.logger = logging.getLogger("IndustryTemplateEngine")
        
        # Template storage
        self.templates: Dict[str, ConfigurationTemplate] = {}
        self.template_index: Dict[IndustryType, List[str]] = {}
        
        # Customization intelligence
        self.customization_patterns = self._load_customization_patterns()
        self.industry_best_practices = self._load_industry_best_practices()
        self.template_relationships = self._build_template_relationships()
        
        # Usage analytics
        self.usage_analytics = {
            "template_usage_count": {},
            "customization_success_rates": {},
            "performance_metrics": {}
        }
        
        self._initialize_core_templates()
    
    def _load_customization_patterns(self) -> Dict[str, Any]:
        """Load intelligent customization patterns"""
        return {
            "company_size_patterns": {
                "small": {
                    "user_limits": 50,
                    "concurrent_sessions": 25,
                    "data_retention_years": 3,
                    "backup_strategy": "daily",
                    "reporting_complexity": "basic"
                },
                "medium": {
                    "user_limits": 250,
                    "concurrent_sessions": 100,
                    "data_retention_years": 7,
                    "backup_strategy": "every_6_hours",
                    "reporting_complexity": "advanced"
                },
                "large": {
                    "user_limits": 1000,
                    "concurrent_sessions": 500,
                    "data_retention_years": 10,
                    "backup_strategy": "continuous",
                    "reporting_complexity": "enterprise"
                },
                "enterprise": {
                    "user_limits": "unlimited",
                    "concurrent_sessions": 2000,
                    "data_retention_years": 15,
                    "backup_strategy": "real_time_replication",
                    "reporting_complexity": "custom_analytics"
                }
            },
            "regulatory_patterns": {
                "fda_regulated": {
                    "audit_trail": "comprehensive",
                    "electronic_signatures": "required",
                    "change_control": "validated",
                    "batch_records": "complete_genealogy"
                },
                "iso_certified": {
                    "document_control": "version_managed",
                    "training_records": "competency_based",
                    "corrective_actions": "systematic_approach"
                },
                "sox_compliant": {
                    "financial_controls": "segregation_of_duties",
                    "approval_workflows": "dual_authorization",
                    "audit_logging": "immutable_records"
                }
            },
            "integration_patterns": {
                "erp_integration": {
                    "data_sync_frequency": "real_time",
                    "conflict_resolution": "priority_based",
                    "error_handling": "retry_with_escalation"
                },
                "manufacturing_systems": {
                    "machine_connectivity": "opc_ua",
                    "data_collection": "automatic",
                    "real_time_monitoring": "dashboard_alerts"
                }
            }
        }
    
    def _load_industry_best_practices(self) -> Dict[IndustryType, Dict]:
        """Load industry-specific best practices"""
        return {
            IndustryType.FURNITURE: {
                "inventory_management": {
                    "wood_moisture_tracking": "required",
                    "lumber_grade_classification": ["A", "B", "C"],
                    "finish_inventory_controls": "temperature_humidity_controlled",
                    "hardware_compatibility_matrix": "fastener_material_matching"
                },
                "production_control": {
                    "cutting_optimization": "yield_maximization_algorithms",
                    "assembly_sequencing": "component_availability_driven",
                    "quality_checkpoints": ["material_inspection", "assembly_fit", "finish_quality"],
                    "environmental_controls": "dust_collection_humidity_control"
                },
                "compliance_requirements": {
                    "carb_phase_2": "formaldehyde_emission_standards",
                    "lacey_act": "wood_species_documentation",
                    "osha_safety": "machinery_guarding_lockout_tagout"
                }
            },
            
            IndustryType.INJECTION_MOLDING: {
                "material_management": {
                    "resin_storage": "temperature_humidity_controlled",
                    "batch_tracking": "lot_genealogy_complete",
                    "material_testing": "melt_flow_index_monitoring",
                    "regrind_management": "percentage_limits_quality_impact"
                },
                "process_control": {
                    "shot_weight_monitoring": "real_time_spc_charts",
                    "cycle_time_optimization": "automatic_parameter_adjustment",
                    "pressure_profiling": "injection_hold_pack_optimization",
                    "temperature_control": "zone_by_zone_pid_control"
                },
                "quality_systems": {
                    "dimensional_inspection": "coordinate_measuring_machine",
                    "defect_classification": "flash_sink_marks_warpage",
                    "statistical_control": "cpk_monitoring_alerts",
                    "first_article_inspection": "full_dimensional_layout"
                }
            },
            
            IndustryType.ELECTRICAL_EQUIPMENT: {
                "component_traceability": {
                    "serial_number_tracking": "component_level_genealogy",
                    "supplier_certifications": "certificate_of_compliance",
                    "critical_component_controls": "approved_vendor_list",
                    "counterfeit_prevention": "authorized_distributor_verification"
                },
                "testing_protocols": {
                    "electrical_testing": ["hipot", "continuity", "functional"],
                    "safety_testing": ["insulation_resistance", "ground_bond", "leakage_current"],
                    "environmental_testing": ["temperature_cycling", "vibration", "humidity"],
                    "performance_validation": ["load_testing", "efficiency_measurement"]
                },
                "compliance_management": {
                    "ul_certification": "listing_and_recognition_programs",
                    "ce_marking": "emc_and_low_voltage_directives",
                    "fcc_compliance": "emissions_and_immunity_testing",
                    "rohs_compliance": "restricted_substances_verification"
                }
            },
            
            IndustryType.TEXTILE: {
                "yarn_management": {
                    "fiber_tracking": "cotton_synthetic_blend_ratios",
                    "color_matching": "pantone_color_standards",
                    "lot_control": "dye_lot_segregation",
                    "substitution_rules": "yarn_weight_compatibility"
                },
                "production_planning": {
                    "loom_scheduling": "fabric_type_changeover_optimization",
                    "knitting_machine_assignment": "gauge_compatibility",
                    "dyeing_batch_optimization": "color_sequence_planning",
                    "finishing_operations": "heat_setting_chemical_application"
                },
                "quality_control": {
                    "fabric_inspection": "four_point_system",
                    "color_fastness_testing": "wash_light_rub_fastness",
                    "dimensional_stability": "shrinkage_skew_bow_testing",
                    "performance_testing": "pilling_abrasion_strength"
                }
            }
        }
    
    def _build_template_relationships(self) -> Dict[str, List[str]]:
        """Build relationships between templates for dependency management"""
        return {
            "core_system_dependencies": [
                "database_configuration",
                "user_management", 
                "security_framework",
                "audit_logging"
            ],
            "industry_specific_dependencies": [
                "core_system",
                "business_rule_engine",
                "reporting_framework"
            ],
            "integration_dependencies": [
                "core_system",
                "security_framework",
                "data_mapping_engine"
            ],
            "compliance_dependencies": [
                "audit_logging",
                "document_management",
                "workflow_engine",
                "security_framework"
            ]
        }
    
    def _initialize_core_templates(self):
        """Initialize core system templates for all industries"""
        self.logger.info("Initializing core industry templates...")
        
        # Create templates for each industry
        for industry in IndustryType:
            self._create_industry_templates(industry)
        
        # Build template index
        self._build_template_index()
        
        self.logger.info(f"Initialized {len(self.templates)} templates across {len(IndustryType)} industries")
    
    def _create_industry_templates(self, industry: IndustryType):
        """Create comprehensive templates for a specific industry"""
        
        # Core System Template
        core_template = self._create_core_system_template(industry)
        self.templates[core_template.template_id] = core_template
        
        # Business Rules Template
        business_template = self._create_business_rules_template(industry)
        self.templates[business_template.template_id] = business_template
        
        # Industry-Specific Features Template
        industry_template = self._create_industry_specific_template(industry)
        self.templates[industry_template.template_id] = industry_template
        
        # Integration Template
        integration_template = self._create_integration_template(industry)
        self.templates[integration_template.template_id] = integration_template
        
        # Compliance Template
        compliance_template = self._create_compliance_template(industry)
        self.templates[compliance_template.template_id] = compliance_template
        
        # Complete Solution Template (combines all)
        complete_template = self._create_complete_solution_template(industry)
        self.templates[complete_template.template_id] = complete_template
    
    def _create_core_system_template(self, industry: IndustryType) -> ConfigurationTemplate:
        """Create core system template for industry"""
        
        base_config = {
            "system_core": {
                "modules": ["inventory", "production", "sales", "purchasing", "finance"],
                "database": {
                    "type": "postgresql",
                    "connection_pooling": True,
                    "backup_strategy": "continuous",
                    "replication": "master_slave"
                },
                "caching": {
                    "type": "redis",
                    "cache_levels": ["application", "database", "session"],
                    "ttl_policies": {
                        "inventory_data": 300,
                        "production_data": 600,
                        "reporting_data": 1800
                    }
                },
                "security": {
                    "authentication": "multi_factor",
                    "authorization": "role_based_access_control",
                    "encryption": "aes_256_at_rest_tls_in_transit",
                    "session_management": "secure_token_based"
                }
            },
            "user_management": {
                "roles": ["admin", "manager", "operator", "viewer"],
                "permissions": "granular_module_based",
                "password_policy": "strong_complexity_requirements",
                "login_monitoring": "failed_attempts_lockout"
            },
            "audit_logging": {
                "scope": "all_transactions",
                "retention": "7_years",
                "integrity": "cryptographic_hashing",
                "compliance": "sox_gdpr_ready"
            }
        }
        
        return ConfigurationTemplate(
            template_id=f"{industry.value.lower()}_core_system",
            template_name=f"{industry.value.replace('_', ' ').title()} Core System",
            industry_type=industry,
            template_type=TemplateType.SYSTEM_CORE,
            complexity_level=ManufacturingComplexity.MODERATE,
            base_configuration=base_config,
            estimated_implementation_hours=16.0,
            confidence_score=0.98
        )
    
    def _create_business_rules_template(self, industry: IndustryType) -> ConfigurationTemplate:
        """Create business rules template for industry"""
        
        base_config = {
            "workflow_engine": {
                "approval_workflows": {
                    "purchase_orders": "amount_based_escalation",
                    "production_changes": "engineering_approval",
                    "quality_deviations": "qc_manager_approval"
                },
                "business_rules": {
                    "inventory_reorder": "min_max_planning",
                    "production_scheduling": "capacity_constrained",
                    "quality_control": "statistical_process_control"
                },
                "notifications": {
                    "channels": ["email", "sms", "dashboard"],
                    "escalation_rules": "time_based_priority",
                    "delivery_confirmation": True
                }
            },
            "calculation_engine": {
                "costing_methods": ["standard", "actual", "average"],
                "planning_algorithms": ["mrp", "jit", "constraint_based"],
                "forecasting_models": ["arima", "exponential_smoothing", "ml_ensemble"]
            }
        }
        
        return ConfigurationTemplate(
            template_id=f"{industry.value.lower()}_business_rules",
            template_name=f"{industry.value.replace('_', ' ').title()} Business Rules",
            industry_type=industry,
            template_type=TemplateType.BUSINESS_RULES,
            complexity_level=ManufacturingComplexity.COMPLEX,
            base_configuration=base_config,
            estimated_implementation_hours=24.0,
            confidence_score=0.92
        )
    
    def _create_industry_specific_template(self, industry: IndustryType) -> ConfigurationTemplate:
        """Create industry-specific features template"""
        
        if industry in self.industry_best_practices:
            base_config = copy.deepcopy(self.industry_best_practices[industry])
        else:
            base_config = self._create_generic_industry_config()
        
        # Add common industry enhancements
        base_config.update({
            "reporting": {
                "industry_kpis": self._get_industry_kpis(industry),
                "dashboards": self._get_industry_dashboards(industry),
                "compliance_reports": self._get_compliance_reports(industry)
            },
            "data_analytics": {
                "predictive_models": self._get_predictive_models(industry),
                "optimization_algorithms": self._get_optimization_algorithms(industry)
            }
        })
        
        return ConfigurationTemplate(
            template_id=f"{industry.value.lower()}_industry_specific",
            template_name=f"{industry.value.replace('_', ' ').title()} Industry Features",
            industry_type=industry,
            template_type=TemplateType.INDUSTRY_SPECIFIC,
            complexity_level=ManufacturingComplexity.COMPLEX,
            base_configuration=base_config,
            estimated_implementation_hours=32.0,
            confidence_score=0.89
        )
    
    def _create_integration_template(self, industry: IndustryType) -> ConfigurationTemplate:
        """Create integration template for industry"""
        
        base_config = {
            "erp_integrations": {
                "supported_systems": ["sap", "oracle", "quickbooks", "dynamics"],
                "integration_patterns": ["real_time_sync", "batch_processing", "event_driven"],
                "data_mapping": "intelligent_field_matching",
                "conflict_resolution": "priority_based_rules"
            },
            "manufacturing_systems": {
                "mes_connectivity": "opc_ua_protocol",
                "machine_interfaces": ["modbus", "ethernet_ip", "profinet"],
                "data_collection": "automatic_real_time",
                "control_integration": "recipe_parameter_management"
            },
            "third_party_services": {
                "shipping_carriers": ["ups", "fedex", "dhl"],
                "payment_processors": ["stripe", "paypal", "authorize_net"],
                "document_management": ["sharepoint", "box", "google_drive"]
            }
        }
        
        return ConfigurationTemplate(
            template_id=f"{industry.value.lower()}_integration",
            template_name=f"{industry.value.replace('_', ' ').title()} Integration",
            industry_type=industry,
            template_type=TemplateType.INTEGRATION,
            complexity_level=ManufacturingComplexity.COMPLEX,
            base_configuration=base_config,
            estimated_implementation_hours=20.0,
            confidence_score=0.85
        )
    
    def _create_compliance_template(self, industry: IndustryType) -> ConfigurationTemplate:
        """Create compliance template for industry"""
        
        base_config = {
            "regulatory_compliance": {
                "audit_trails": "comprehensive_change_tracking",
                "document_control": "version_management_approval",
                "training_management": "competency_based_records",
                "corrective_actions": "systematic_root_cause_analysis"
            },
            "data_protection": {
                "gdpr_compliance": "consent_management_right_to_forget",
                "data_classification": "confidential_internal_public",
                "backup_encryption": "aes_256_key_management",
                "access_logging": "who_what_when_where"
            },
            "industry_standards": self._get_industry_compliance_standards(industry)
        }
        
        return ConfigurationTemplate(
            template_id=f"{industry.value.lower()}_compliance",
            template_name=f"{industry.value.replace('_', ' ').title()} Compliance",
            industry_type=industry,
            template_type=TemplateType.COMPLIANCE,
            complexity_level=ManufacturingComplexity.ENTERPRISE,
            base_configuration=base_config,
            estimated_implementation_hours=28.0,
            confidence_score=0.95
        )
    
    def _create_complete_solution_template(self, industry: IndustryType) -> ConfigurationTemplate:
        """Create complete solution template combining all components"""
        
        # Merge all component templates
        component_templates = [
            f"{industry.value.lower()}_core_system",
            f"{industry.value.lower()}_business_rules", 
            f"{industry.value.lower()}_industry_specific",
            f"{industry.value.lower()}_integration",
            f"{industry.value.lower()}_compliance"
        ]
        
        base_config = {}
        total_hours = 0.0
        
        for template_id in component_templates:
            if template_id in self.templates:
                template = self.templates[template_id]
                base_config.update(template.base_configuration)
                total_hours += template.estimated_implementation_hours
        
        return ConfigurationTemplate(
            template_id=f"{industry.value.lower()}_complete_solution",
            template_name=f"{industry.value.replace('_', ' ').title()} Complete Solution",
            industry_type=industry,
            template_type=TemplateType.COMPLETE_SOLUTION,
            complexity_level=ManufacturingComplexity.ENTERPRISE,
            base_configuration=base_config,
            dependencies=component_templates,
            estimated_implementation_hours=total_hours * 0.8,  # 20% efficiency gain from integrated approach
            confidence_score=0.92
        )
    
    def _create_generic_industry_config(self) -> Dict[str, Any]:
        """Create generic industry configuration for unsupported industries"""
        return {
            "inventory_management": {
                "tracking_methods": ["lot_number", "serial_number", "batch_tracking"],
                "valuation_methods": ["fifo", "lifo", "weighted_average"],
                "cycle_counting": "abc_classification_based"
            },
            "production_control": {
                "scheduling_methods": ["forward", "backward", "constraint_based"],
                "capacity_planning": "finite_capacity_scheduling",
                "quality_control": "inspection_point_management"
            }
        }
    
    def _get_industry_kpis(self, industry: IndustryType) -> List[str]:
        """Get industry-specific KPIs"""
        kpi_mapping = {
            IndustryType.FURNITURE: [
                "lumber_yield_percentage", "assembly_efficiency", "finish_quality_rate",
                "delivery_performance", "customer_satisfaction_score"
            ],
            IndustryType.INJECTION_MOLDING: [
                "machine_utilization", "cycle_time_variance", "first_pass_yield",
                "material_waste_percentage", "changeover_time"
            ],
            IndustryType.ELECTRICAL_EQUIPMENT: [
                "test_pass_rate", "rework_percentage", "compliance_score", 
                "traceability_completeness", "supplier_quality_rating"
            ],
            IndustryType.TEXTILE: [
                "loom_efficiency", "yarn_utilization", "color_matching_accuracy",
                "fabric_quality_grade", "order_fulfillment_rate"
            ]
        }
        return kpi_mapping.get(industry, ["efficiency", "quality", "delivery", "cost"])
    
    def _get_industry_dashboards(self, industry: IndustryType) -> List[str]:
        """Get industry-specific dashboard configurations"""
        return [
            "executive_summary",
            "operations_overview", 
            "quality_metrics",
            "financial_performance",
            f"{industry.value.lower()}_specific_metrics"
        ]
    
    def _get_compliance_reports(self, industry: IndustryType) -> List[str]:
        """Get compliance reports for industry"""
        compliance_mapping = {
            IndustryType.FURNITURE: ["carb_compliance", "lacey_act_reporting"],
            IndustryType.INJECTION_MOLDING: ["fda_device_history", "iso_audit_preparation"],
            IndustryType.ELECTRICAL_EQUIPMENT: ["ul_certification_status", "rohs_compliance"],
            IndustryType.TEXTILE: ["oeko_tex_certification", "gots_compliance"]
        }
        return compliance_mapping.get(industry, ["general_audit", "quality_system"])
    
    def _get_predictive_models(self, industry: IndustryType) -> List[str]:
        """Get predictive models for industry"""
        return [
            "demand_forecasting",
            "maintenance_prediction", 
            "quality_prediction",
            "capacity_optimization"
        ]
    
    def _get_optimization_algorithms(self, industry: IndustryType) -> List[str]:
        """Get optimization algorithms for industry"""
        return [
            "production_scheduling",
            "inventory_optimization",
            "resource_allocation",
            "cost_optimization"
        ]
    
    def _get_industry_compliance_standards(self, industry: IndustryType) -> Dict[str, Any]:
        """Get compliance standards for industry"""
        standards_mapping = {
            IndustryType.FURNITURE: {
                "carb_phase_2": "formaldehyde_emission_limits",
                "lacey_act": "wood_species_documentation",
                "cpsc_regulations": "consumer_product_safety"
            },
            IndustryType.INJECTION_MOLDING: {
                "iso_13485": "medical_device_quality_system",
                "fda_qsr": "quality_system_regulation",
                "iso_14971": "risk_management_medical_devices"
            },
            IndustryType.ELECTRICAL_EQUIPMENT: {
                "ul_standards": "safety_certification",
                "fcc_part_15": "electromagnetic_compatibility",
                "rohs_directive": "restricted_substance_compliance"
            },
            IndustryType.TEXTILE: {
                "oeko_tex": "textile_chemical_safety",
                "gots": "organic_textile_standard",
                "cpsia": "consumer_product_safety_improvement"
            }
        }
        return standards_mapping.get(industry, {"iso_9001": "quality_management_system"})
    
    def _build_template_index(self):
        """Build index of templates by industry for fast lookup"""
        for template_id, template in self.templates.items():
            industry = template.industry_type
            if industry not in self.template_index:
                self.template_index[industry] = []
            self.template_index[industry].append(template_id)
    
    async def get_templates_for_industry(
        self, 
        industry: IndustryType, 
        template_type: Optional[TemplateType] = None,
        complexity: Optional[ManufacturingComplexity] = None
    ) -> List[ConfigurationTemplate]:
        """Get available templates for specific industry"""
        
        if industry not in self.template_index:
            return []
        
        templates = []
        for template_id in self.template_index[industry]:
            template = self.templates[template_id]
            
            # Apply filters
            if template_type and template.template_type != template_type:
                continue
            if complexity and template.complexity_level != complexity:
                continue
            
            templates.append(template)
        
        # Sort by confidence score (highest first)
        templates.sort(key=lambda t: t.confidence_score, reverse=True)
        
        return templates
    
    async def generate_custom_configuration(
        self,
        industry: IndustryType,
        customer_requirements: Dict[str, Any],
        complexity: ManufacturingComplexity = ManufacturingComplexity.MODERATE
    ) -> Dict[str, Any]:
        """Generate customized configuration based on customer requirements"""
        
        # Select base template
        templates = await self.get_templates_for_industry(
            industry, TemplateType.COMPLETE_SOLUTION, complexity
        )
        
        if not templates:
            raise ValueError(f"No templates available for {industry.value} at {complexity.value} complexity")
        
        base_template = templates[0]
        configuration = copy.deepcopy(base_template.base_configuration)
        
        # Apply customizations based on requirements
        applied_customizations = []
        
        # Company size customizations
        if "company_size" in customer_requirements:
            size_customizations = self._apply_size_customizations(
                configuration, customer_requirements["company_size"]
            )
            applied_customizations.extend(size_customizations)
        
        # Regulatory customizations
        if "regulatory_requirements" in customer_requirements:
            reg_customizations = self._apply_regulatory_customizations(
                configuration, customer_requirements["regulatory_requirements"]
            )
            applied_customizations.extend(reg_customizations)
        
        # Integration customizations
        if "integration_requirements" in customer_requirements:
            int_customizations = self._apply_integration_customizations(
                configuration, customer_requirements["integration_requirements"]
            )
            applied_customizations.extend(int_customizations)
        
        # Performance customizations
        if "performance_requirements" in customer_requirements:
            perf_customizations = self._apply_performance_customizations(
                configuration, customer_requirements["performance_requirements"]
            )
            applied_customizations.extend(perf_customizations)
        
        return {
            "base_template_id": base_template.template_id,
            "generated_configuration": configuration,
            "applied_customizations": applied_customizations,
            "estimated_implementation_hours": base_template.estimated_implementation_hours,
            "confidence_score": self._calculate_customization_confidence(
                base_template, applied_customizations
            )
        }
    
    def _apply_size_customizations(self, config: Dict[str, Any], company_size: str) -> List[str]:
        """Apply company size customizations"""
        customizations = []
        
        if company_size.lower() in self.customization_patterns["company_size_patterns"]:
            size_config = self.customization_patterns["company_size_patterns"][company_size.lower()]
            
            # Update system configuration
            if "system_core" in config:
                config["system_core"].setdefault("sizing", {}).update(size_config)
                customizations.append(f"Applied {company_size} company sizing")
        
        return customizations
    
    def _apply_regulatory_customizations(self, config: Dict[str, Any], requirements: List[str]) -> List[str]:
        """Apply regulatory requirement customizations"""
        customizations = []
        
        for requirement in requirements:
            req_lower = requirement.lower()
            
            for pattern_name, pattern_config in self.customization_patterns["regulatory_patterns"].items():
                if pattern_name.replace("_", "") in req_lower.replace("_", ""):
                    config.setdefault("regulatory_compliance", {}).update(pattern_config)
                    customizations.append(f"Applied {requirement} compliance controls")
        
        return customizations
    
    def _apply_integration_customizations(self, config: Dict[str, Any], requirements: List[str]) -> List[str]:
        """Apply integration requirement customizations"""
        customizations = []
        
        for requirement in requirements:
            req_lower = requirement.lower()
            
            for pattern_name, pattern_config in self.customization_patterns["integration_patterns"].items():
                if pattern_name in req_lower or any(word in req_lower for word in pattern_name.split("_")):
                    config.setdefault("integration", {}).setdefault(pattern_name, {}).update(pattern_config)
                    customizations.append(f"Configured {requirement} integration")
        
        return customizations
    
    def _apply_performance_customizations(self, config: Dict[str, Any], requirements: Dict[str, Any]) -> List[str]:
        """Apply performance requirement customizations"""
        customizations = []
        
        if "response_time" in requirements:
            config.setdefault("performance", {})["response_time_target"] = requirements["response_time"]
            customizations.append("Applied response time requirements")
        
        if "concurrent_users" in requirements:
            config.setdefault("performance", {})["concurrent_users"] = requirements["concurrent_users"]
            customizations.append("Applied concurrent user requirements")
        
        if "data_volume" in requirements:
            config.setdefault("performance", {})["data_volume_capacity"] = requirements["data_volume"]
            customizations.append("Applied data volume requirements")
        
        return customizations
    
    def _calculate_customization_confidence(
        self, 
        base_template: ConfigurationTemplate, 
        customizations: List[str]
    ) -> float:
        """Calculate confidence score after customizations"""
        
        base_confidence = base_template.confidence_score
        
        # Slight reduction in confidence for each customization (complexity increase)
        customization_penalty = len(customizations) * 0.01
        
        # But reward successful pattern matching (shown by applied customizations)
        pattern_bonus = min(0.05, len(customizations) * 0.005)
        
        final_confidence = base_confidence - customization_penalty + pattern_bonus
        
        return min(1.0, max(0.0, final_confidence))
    
    async def validate_configuration(self, configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration against best practices"""
        
        validation_results = {
            "is_valid": True,
            "confidence_score": 0.0,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Validate core components
        core_checks = self._validate_core_components(configuration)
        validation_results["warnings"].extend(core_checks["warnings"])
        validation_results["errors"].extend(core_checks["errors"])
        
        # Validate security configuration
        security_checks = self._validate_security_configuration(configuration)
        validation_results["warnings"].extend(security_checks["warnings"])
        validation_results["errors"].extend(security_checks["errors"])
        
        # Validate performance configuration
        performance_checks = self._validate_performance_configuration(configuration)
        validation_results["warnings"].extend(performance_checks["warnings"])
        
        # Calculate overall validation score
        total_issues = len(validation_results["errors"]) + len(validation_results["warnings"])
        if total_issues == 0:
            validation_results["confidence_score"] = 1.0
        else:
            # Penalize errors more than warnings
            error_penalty = len(validation_results["errors"]) * 0.2
            warning_penalty = len(validation_results["warnings"]) * 0.05
            validation_results["confidence_score"] = max(0.0, 1.0 - error_penalty - warning_penalty)
        
        validation_results["is_valid"] = len(validation_results["errors"]) == 0
        
        return validation_results
    
    def _validate_core_components(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate core system components"""
        warnings = []
        errors = []
        
        if "system_core" not in config:
            errors.append("Missing core system configuration")
        elif "modules" not in config["system_core"]:
            errors.append("Core modules not specified")
        
        if "user_management" not in config:
            warnings.append("User management configuration not found")
        
        if "audit_logging" not in config:
            warnings.append("Audit logging configuration not found")
        
        return {"warnings": warnings, "errors": errors}
    
    def _validate_security_configuration(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate security configuration"""
        warnings = []
        errors = []
        
        if "system_core" in config and "security" in config["system_core"]:
            security_config = config["system_core"]["security"]
            
            if security_config.get("authentication") != "multi_factor":
                warnings.append("Multi-factor authentication not configured")
            
            if "aes_256" not in security_config.get("encryption", ""):
                warnings.append("Strong encryption not configured")
        else:
            errors.append("Security configuration missing")
        
        return {"warnings": warnings, "errors": errors}
    
    def _validate_performance_configuration(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate performance configuration"""
        warnings = []
        errors = []
        
        if "system_core" in config and "caching" not in config["system_core"]:
            warnings.append("Caching configuration not found")
        
        if "performance" in config:
            perf_config = config["performance"]
            if "response_time_target" in perf_config and perf_config["response_time_target"] > 5000:
                warnings.append("Response time target may be too high (>5 seconds)")
        
        return {"warnings": warnings, "errors": errors}
    
    def get_template_analytics(self) -> Dict[str, Any]:
        """Get template usage and performance analytics"""
        return {
            "total_templates": len(self.templates),
            "templates_by_industry": {
                industry.value: len(template_ids)
                for industry, template_ids in self.template_index.items()
            },
            "templates_by_type": {
                template_type.value: len([
                    t for t in self.templates.values() 
                    if t.template_type == template_type
                ])
                for template_type in TemplateType
            },
            "average_confidence_score": sum(
                t.confidence_score for t in self.templates.values()
            ) / len(self.templates) if self.templates else 0.0,
            "usage_analytics": self.usage_analytics
        }


# Export key components
__all__ = [
    "TemplateType",
    "ConfigurationTemplate",
    "TemplateCustomization", 
    "IndustryTemplateEngine"
]