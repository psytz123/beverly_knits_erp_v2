#!/usr/bin/env python3
"""
Agent Role Definitions for Beverly Knits ERP
Defines specific roles, responsibilities, and capabilities for each agent type
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class DecisionAuthority(Enum):
    """Levels of decision-making authority"""
    READ_ONLY = "read_only"           # Can only query and analyze
    RECOMMEND = "recommend"           # Can make recommendations
    SUPERVISED = "supervised"         # Can execute with approval
    AUTONOMOUS = "autonomous"         # Can execute independently


class RiskLevel(Enum):
    """Risk levels for agent actions"""
    LOW = "low"           # Minimal business impact
    MEDIUM = "medium"     # Moderate business impact
    HIGH = "high"         # Significant business impact
    CRITICAL = "critical" # Critical business impact


@dataclass
class AgentCapability:
    """Defines a specific capability of an agent"""
    name: str
    description: str
    api_endpoints: List[str] = field(default_factory=list)
    required_data: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    authority_required: DecisionAuthority = DecisionAuthority.READ_ONLY


@dataclass
class AgentResponsibility:
    """Defines a responsibility area for an agent"""
    area: str
    description: str
    kpi_targets: Dict[str, Any] = field(default_factory=dict)
    sla_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRole:
    """Complete role definition for an agent"""
    role_id: str
    name: str
    description: str
    primary_objective: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    responsibilities: List[AgentResponsibility] = field(default_factory=list)
    required_knowledge: List[str] = field(default_factory=list)
    api_permissions: List[str] = field(default_factory=list)
    data_access: List[str] = field(default_factory=list)
    max_authority: DecisionAuthority = DecisionAuthority.READ_ONLY
    escalation_triggers: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# INVENTORY INTELLIGENCE AGENT
# ============================================================================

INVENTORY_AGENT_ROLE = AgentRole(
    role_id="inventory_intelligence",
    name="Inventory Intelligence Agent",
    description="Manages real-time inventory analysis, balance calculations, and shortage detection",
    primary_objective="Maintain optimal inventory levels while minimizing stockouts and excess inventory",
    
    capabilities=[
        AgentCapability(
            name="planning_balance_calculation",
            description="Calculate Planning Balance (Physical - Allocated + On Order)",
            api_endpoints=["/api/inventory-intelligence-enhanced"],
            required_data=["yarn_inventory.xlsx", "BOM_updated.csv"],
            risk_level=RiskLevel.LOW,
            authority_required=DecisionAuthority.AUTONOMOUS
        ),
        AgentCapability(
            name="shortage_detection",
            description="Identify yarn shortages and predict stockout risks",
            api_endpoints=["/api/yarn-intelligence"],
            required_data=["yarn_inventory.xlsx", "demand_forecast"],
            risk_level=RiskLevel.MEDIUM,
            authority_required=DecisionAuthority.RECOMMEND
        ),
        AgentCapability(
            name="multi_level_netting",
            description="Perform multi-level BOM netting calculations",
            api_endpoints=["/api/inventory-netting"],
            required_data=["BOM_updated.csv", "production_orders"],
            risk_level=RiskLevel.LOW,
            authority_required=DecisionAuthority.AUTONOMOUS
        ),
        AgentCapability(
            name="reorder_recommendation",
            description="Generate purchase order recommendations",
            api_endpoints=["/api/po-risk-analysis"],
            required_data=["supplier_data", "lead_times", "pricing"],
            risk_level=RiskLevel.HIGH,
            authority_required=DecisionAuthority.SUPERVISED
        )
    ],
    
    responsibilities=[
        AgentResponsibility(
            area="Inventory Accuracy",
            description="Maintain accurate inventory records and calculations",
            kpi_targets={
                "calculation_accuracy": 0.995,  # 99.5% accuracy
                "data_freshness": 300,          # 5 minutes max age
                "reconciliation_rate": 0.98     # 98% reconciliation success
            },
            sla_requirements={
                "query_response_time": 100,     # 100ms
                "batch_processing": 3600,        # 1 hour for bulk updates
                "alert_latency": 60              # 1 minute for critical alerts
            }
        ),
        AgentResponsibility(
            area="Shortage Prevention",
            description="Proactively identify and prevent stockouts",
            kpi_targets={
                "stockout_prevention_rate": 0.95,  # 95% prevention
                "false_positive_rate": 0.10,       # <10% false alarms
                "lead_time_accuracy": 0.90         # 90% lead time prediction
            },
            sla_requirements={
                "shortage_alert_time": 48,         # 48 hours advance warning
                "critical_alert_time": 2           # 2 hours for critical items
            }
        )
    ],
    
    required_knowledge=[
        "Planning Balance formula and calculations",
        "BOM structure and multi-level netting",
        "Yarn specifications and characteristics",
        "Historical consumption patterns",
        "Supplier lead times and reliability",
        "Safety stock calculations",
        "Economic order quantity (EOQ)"
    ],
    
    api_permissions=[
        "/api/inventory-intelligence-enhanced",
        "/api/inventory-netting",
        "/api/yarn-intelligence",
        "/api/po-risk-analysis"
    ],
    
    data_access=[
        "yarn_inventory",
        "bom_data",
        "production_orders",
        "supplier_information",
        "historical_consumption"
    ],
    
    max_authority=DecisionAuthority.SUPERVISED,
    
    escalation_triggers={
        "stockout_imminent": "Alert production manager within 2 hours",
        "data_discrepancy": "Alert data team if >5% variance",
        "system_failure": "Alert IT team immediately",
        "high_value_order": "Require approval for orders >$50,000"
    }
)


# ============================================================================
# FORECAST INTELLIGENCE AGENT
# ============================================================================

FORECAST_AGENT_ROLE = AgentRole(
    role_id="forecast_intelligence",
    name="Forecast Intelligence Agent",
    description="Manages ML-based demand forecasting and predictive analytics",
    primary_objective="Provide accurate demand forecasts to optimize inventory and production planning",
    
    capabilities=[
        AgentCapability(
            name="demand_forecasting",
            description="Generate short and long-term demand forecasts",
            api_endpoints=["/api/ml-forecast-detailed"],
            required_data=["sales_history", "seasonal_patterns", "market_trends"],
            risk_level=RiskLevel.MEDIUM,
            authority_required=DecisionAuthority.AUTONOMOUS
        ),
        AgentCapability(
            name="model_training",
            description="Train and retrain ML models (ARIMA, Prophet, LSTM, XGBoost)",
            api_endpoints=["/api/ml-training"],
            required_data=["historical_data", "validation_sets"],
            risk_level=RiskLevel.LOW,
            authority_required=DecisionAuthority.AUTONOMOUS
        ),
        AgentCapability(
            name="accuracy_monitoring",
            description="Monitor forecast accuracy and trigger retraining",
            api_endpoints=["/api/forecast-accuracy"],
            required_data=["predictions", "actuals"],
            risk_level=RiskLevel.LOW,
            authority_required=DecisionAuthority.AUTONOMOUS
        ),
        AgentCapability(
            name="ensemble_optimization",
            description="Optimize ensemble model weights",
            api_endpoints=["/api/ensemble-optimization"],
            required_data=["model_performances"],
            risk_level=RiskLevel.MEDIUM,
            authority_required=DecisionAuthority.RECOMMEND
        )
    ],
    
    responsibilities=[
        AgentResponsibility(
            area="Forecast Accuracy",
            description="Maintain high accuracy in demand predictions",
            kpi_targets={
                "mape_30_day": 0.15,           # <15% MAPE for 30-day
                "mape_90_day": 0.20,           # <20% MAPE for 90-day
                "bias": 0.05,                  # <5% systematic bias
                "forecast_value_add": 1.20     # 20% better than naive
            },
            sla_requirements={
                "forecast_generation": 300,     # 5 minutes
                "model_training": 3600,         # 1 hour max
                "daily_update": 21600          # 6 AM daily update
            }
        ),
        AgentResponsibility(
            area="Model Management",
            description="Maintain and optimize forecasting models",
            kpi_targets={
                "model_freshness": 7,          # Retrain weekly
                "model_diversity": 4,          # Maintain 4+ models
                "ensemble_weight_optimization": 30  # Optimize monthly
            },
            sla_requirements={
                "retraining_time": 3600,       # 1 hour
                "validation_time": 600,        # 10 minutes
                "deployment_time": 300         # 5 minutes
            }
        )
    ],
    
    required_knowledge=[
        "Time series analysis and forecasting",
        "Machine learning algorithms (ARIMA, Prophet, LSTM, XGBoost)",
        "Seasonal decomposition",
        "Trend analysis",
        "Feature engineering",
        "Cross-validation techniques",
        "Ensemble methods",
        "Statistical metrics (MAPE, RMSE, MAE)"
    ],
    
    api_permissions=[
        "/api/ml-forecast-detailed",
        "/api/ml-training",
        "/api/forecast-accuracy",
        "/api/ensemble-optimization"
    ],
    
    data_access=[
        "sales_activity_report",
        "historical_demand",
        "seasonal_patterns",
        "market_indicators",
        "model_registry"
    ],
    
    max_authority=DecisionAuthority.AUTONOMOUS,
    
    escalation_triggers={
        "accuracy_degradation": "Alert if MAPE increases >10%",
        "model_failure": "Alert data science team immediately",
        "data_quality_issue": "Alert if >5% missing data",
        "unusual_pattern": "Flag anomalies for human review"
    }
)


# ============================================================================
# PRODUCTION PLANNING AGENT
# ============================================================================

PRODUCTION_AGENT_ROLE = AgentRole(
    role_id="production_planning",
    name="Production Planning Agent",
    description="Optimizes production scheduling and capacity planning across all machines",
    primary_objective="Maximize production efficiency while meeting delivery commitments",
    
    capabilities=[
        AgentCapability(
            name="production_scheduling",
            description="Create and optimize production schedules",
            api_endpoints=["/api/production-planning"],
            required_data=["production_orders", "machine_capacity", "work_centers"],
            risk_level=RiskLevel.HIGH,
            authority_required=DecisionAuthority.SUPERVISED
        ),
        AgentCapability(
            name="capacity_planning",
            description="Analyze and optimize machine capacity utilization",
            api_endpoints=["/api/capacity-planning"],
            required_data=["machine_availability", "maintenance_schedule"],
            risk_level=RiskLevel.MEDIUM,
            authority_required=DecisionAuthority.RECOMMEND
        ),
        AgentCapability(
            name="machine_assignment",
            description="Assign orders to optimal machines",
            api_endpoints=["/api/machine-assignment-suggestions"],
            required_data=["machine_capabilities", "order_requirements"],
            risk_level=RiskLevel.MEDIUM,
            authority_required=DecisionAuthority.SUPERVISED
        ),
        AgentCapability(
            name="six_phase_optimization",
            description="Optimize 6-phase production planning",
            api_endpoints=["/api/six-phase-planning"],
            required_data=["phase_transitions", "bottlenecks"],
            risk_level=RiskLevel.HIGH,
            authority_required=DecisionAuthority.RECOMMEND
        )
    ],
    
    responsibilities=[
        AgentResponsibility(
            area="Schedule Optimization",
            description="Create efficient production schedules",
            kpi_targets={
                "on_time_delivery": 0.95,      # 95% on-time
                "machine_utilization": 0.85,    # 85% utilization
                "setup_time_reduction": 0.20,   # 20% reduction
                "schedule_adherence": 0.90      # 90% adherence
            },
            sla_requirements={
                "schedule_generation": 600,     # 10 minutes
                "rescheduling": 300,           # 5 minutes
                "urgent_order": 60             # 1 minute response
            }
        ),
        AgentResponsibility(
            area="Capacity Management",
            description="Optimize machine and work center capacity",
            kpi_targets={
                "capacity_utilization": 0.85,   # 85% target
                "bottleneck_reduction": 0.30,   # 30% reduction
                "load_balancing": 0.15          # <15% variance
            },
            sla_requirements={
                "capacity_analysis": 300,       # 5 minutes
                "bottleneck_alert": 120,       # 2 minutes
                "rebalancing": 600             # 10 minutes
            }
        )
    ],
    
    required_knowledge=[
        "Production planning and scheduling algorithms",
        "Machine capabilities and constraints",
        "Work center configurations (x.xx.xx.X pattern)",
        "6-phase production flow",
        "Capacity planning techniques",
        "Bottleneck analysis",
        "Setup time optimization",
        "Lean manufacturing principles"
    ],
    
    api_permissions=[
        "/api/production-planning",
        "/api/capacity-planning",
        "/api/machine-assignment-suggestions",
        "/api/six-phase-planning",
        "/api/production-pipeline"
    ],
    
    data_access=[
        "efab_knit_orders",
        "machine_report",
        "work_center_mapping",
        "production_calendar",
        "maintenance_schedule"
    ],
    
    max_authority=DecisionAuthority.SUPERVISED,
    
    escalation_triggers={
        "capacity_overload": "Alert if >95% capacity",
        "delivery_risk": "Alert if on-time risk >10%",
        "machine_failure": "Immediate rescheduling required",
        "quality_issue": "Stop production and alert QA"
    }
)


# ============================================================================
# YARN SUBSTITUTION AGENT
# ============================================================================

YARN_AGENT_ROLE = AgentRole(
    role_id="yarn_substitution",
    name="Yarn Substitution Agent",
    description="Identifies compatible yarn substitutes and manages yarn procurement optimization",
    primary_objective="Ensure yarn availability through intelligent substitution and procurement",
    
    capabilities=[
        AgentCapability(
            name="substitution_analysis",
            description="Identify compatible yarn substitutes",
            api_endpoints=["/api/yarn-substitution-intelligent"],
            required_data=["yarn_specifications", "compatibility_matrix"],
            risk_level=RiskLevel.MEDIUM,
            authority_required=DecisionAuthority.RECOMMEND
        ),
        AgentCapability(
            name="quality_impact_assessment",
            description="Assess quality impact of substitutions",
            api_endpoints=["/api/substitution-quality"],
            required_data=["quality_standards", "historical_substitutions"],
            risk_level=RiskLevel.HIGH,
            authority_required=DecisionAuthority.SUPERVISED
        ),
        AgentCapability(
            name="supplier_optimization",
            description="Optimize supplier selection and procurement",
            api_endpoints=["/api/supplier-optimization"],
            required_data=["supplier_performance", "pricing", "lead_times"],
            risk_level=RiskLevel.HIGH,
            authority_required=DecisionAuthority.RECOMMEND
        ),
        AgentCapability(
            name="interchangeability_learning",
            description="Learn yarn interchangeability patterns",
            api_endpoints=["/api/interchangeability-ml"],
            required_data=["substitution_history", "quality_outcomes"],
            risk_level=RiskLevel.LOW,
            authority_required=DecisionAuthority.AUTONOMOUS
        )
    ],
    
    responsibilities=[
        AgentResponsibility(
            area="Substitution Management",
            description="Manage yarn substitution recommendations",
            kpi_targets={
                "substitution_success_rate": 0.90,  # 90% success
                "quality_maintenance": 0.95,        # 95% quality maintained
                "cost_savings": 0.15,               # 15% cost savings
                "availability_improvement": 0.20    # 20% improvement
            },
            sla_requirements={
                "substitution_search": 60,          # 1 minute
                "quality_assessment": 300,          # 5 minutes
                "recommendation": 120               # 2 minutes
            }
        ),
        AgentResponsibility(
            area="Procurement Optimization",
            description="Optimize yarn procurement and supplier management",
            kpi_targets={
                "supplier_performance": 0.85,       # 85% on-time
                "cost_optimization": 0.10,          # 10% cost reduction
                "lead_time_reduction": 0.20,        # 20% faster
                "quality_compliance": 0.98          # 98% quality
            },
            sla_requirements={
                "supplier_evaluation": 600,         # 10 minutes
                "order_generation": 300,            # 5 minutes
                "urgent_procurement": 1800          # 30 minutes
            }
        )
    ],
    
    required_knowledge=[
        "Yarn specifications and properties",
        "Color matching and tolerances",
        "Material compatibility",
        "Quality standards and testing",
        "Supplier capabilities and performance",
        "Procurement best practices",
        "Cost-benefit analysis",
        "ML for pattern recognition"
    ],
    
    api_permissions=[
        "/api/yarn-substitution-intelligent",
        "/api/substitution-quality",
        "/api/supplier-optimization",
        "/api/interchangeability-ml"
    ],
    
    data_access=[
        "yarn_id_master",
        "supplier_database",
        "quality_specifications",
        "substitution_history",
        "procurement_records"
    ],
    
    max_authority=DecisionAuthority.SUPERVISED,
    
    escalation_triggers={
        "quality_risk": "Alert QA if quality impact >5%",
        "cost_increase": "Alert if substitution costs >10% more",
        "supplier_issue": "Alert procurement if supplier fails",
        "no_substitute": "Alert production planning immediately"
    }
)


# ============================================================================
# QUALITY ASSURANCE AGENT
# ============================================================================

QUALITY_AGENT_ROLE = AgentRole(
    role_id="quality_assurance",
    name="Quality Assurance Agent",
    description="Monitors data quality, system performance, and operational excellence",
    primary_objective="Ensure data integrity and system reliability across all operations",
    
    capabilities=[
        AgentCapability(
            name="data_validation",
            description="Validate data integrity and consistency",
            api_endpoints=["/api/data-validation"],
            required_data=["all_data_sources"],
            risk_level=RiskLevel.LOW,
            authority_required=DecisionAuthority.AUTONOMOUS
        ),
        AgentCapability(
            name="anomaly_detection",
            description="Detect anomalies in data and processes",
            api_endpoints=["/api/anomaly-detection"],
            required_data=["historical_patterns", "thresholds"],
            risk_level=RiskLevel.MEDIUM,
            authority_required=DecisionAuthority.RECOMMEND
        ),
        AgentCapability(
            name="performance_monitoring",
            description="Monitor system and agent performance",
            api_endpoints=["/api/performance-metrics"],
            required_data=["system_logs", "kpi_metrics"],
            risk_level=RiskLevel.LOW,
            authority_required=DecisionAuthority.AUTONOMOUS
        ),
        AgentCapability(
            name="compliance_checking",
            description="Ensure compliance with business rules",
            api_endpoints=["/api/compliance-check"],
            required_data=["business_rules", "audit_logs"],
            risk_level=RiskLevel.HIGH,
            authority_required=DecisionAuthority.RECOMMEND
        )
    ],
    
    responsibilities=[
        AgentResponsibility(
            area="Data Quality",
            description="Maintain high data quality standards",
            kpi_targets={
                "data_accuracy": 0.99,          # 99% accuracy
                "completeness": 0.95,           # 95% complete
                "consistency": 0.98,            # 98% consistent
                "timeliness": 0.90              # 90% on-time
            },
            sla_requirements={
                "validation_time": 60,          # 1 minute
                "anomaly_detection": 120,       # 2 minutes
                "quality_report": 3600          # 1 hour
            }
        ),
        AgentResponsibility(
            area="System Performance",
            description="Monitor and optimize system performance",
            kpi_targets={
                "system_uptime": 0.999,         # 99.9% uptime
                "response_time": 200,           # <200ms average
                "error_rate": 0.01,            # <1% errors
                "alert_accuracy": 0.90          # 90% accurate alerts
            },
            sla_requirements={
                "performance_check": 60,        # 1 minute intervals
                "alert_generation": 30,         # 30 seconds
                "incident_response": 300        # 5 minutes
            }
        )
    ],
    
    required_knowledge=[
        "Data quality dimensions",
        "Statistical process control",
        "Anomaly detection algorithms",
        "Performance monitoring techniques",
        "Compliance requirements",
        "Audit procedures",
        "Root cause analysis",
        "Continuous improvement methodologies"
    ],
    
    api_permissions=[
        "/api/data-validation",
        "/api/anomaly-detection",
        "/api/performance-metrics",
        "/api/compliance-check",
        "/api/comprehensive-kpis"
    ],
    
    data_access=[
        "system_logs",
        "audit_trails",
        "performance_metrics",
        "data_lineage",
        "quality_standards"
    ],
    
    max_authority=DecisionAuthority.RECOMMEND,
    
    escalation_triggers={
        "data_corruption": "Alert IT team immediately",
        "compliance_violation": "Alert compliance officer",
        "system_degradation": "Alert if performance drops >20%",
        "security_breach": "Alert security team immediately"
    }
)


# ============================================================================
# ROLE REGISTRY
# ============================================================================

AGENT_ROLES_REGISTRY = {
    "inventory_intelligence": INVENTORY_AGENT_ROLE,
    "forecast_intelligence": FORECAST_AGENT_ROLE,
    "production_planning": PRODUCTION_AGENT_ROLE,
    "yarn_substitution": YARN_AGENT_ROLE,
    "quality_assurance": QUALITY_AGENT_ROLE
}


def get_agent_role(role_id: str) -> Optional[AgentRole]:
    """Get agent role definition by ID"""
    return AGENT_ROLES_REGISTRY.get(role_id)


def get_all_roles() -> Dict[str, AgentRole]:
    """Get all agent role definitions"""
    return AGENT_ROLES_REGISTRY.copy()


def get_role_capabilities(role_id: str) -> List[AgentCapability]:
    """Get capabilities for a specific role"""
    role = get_agent_role(role_id)
    return role.capabilities if role else []


def get_role_responsibilities(role_id: str) -> List[AgentResponsibility]:
    """Get responsibilities for a specific role"""
    role = get_agent_role(role_id)
    return role.responsibilities if role else []


def validate_agent_authority(role_id: str, action: str, authority_level: DecisionAuthority) -> bool:
    """Validate if an agent has authority for an action"""
    role = get_agent_role(role_id)
    if not role:
        return False
    
    # Check if the agent's max authority meets the required level
    authority_hierarchy = [
        DecisionAuthority.READ_ONLY,
        DecisionAuthority.RECOMMEND,
        DecisionAuthority.SUPERVISED,
        DecisionAuthority.AUTONOMOUS
    ]
    
    role_level = authority_hierarchy.index(role.max_authority)
    required_level = authority_hierarchy.index(authority_level)
    
    return role_level >= required_level


def get_escalation_procedure(role_id: str, trigger: str) -> Optional[str]:
    """Get escalation procedure for a specific trigger"""
    role = get_agent_role(role_id)
    if role and trigger in role.escalation_triggers:
        return role.escalation_triggers[trigger]
    return None


# Example usage
if __name__ == "__main__":
    print("Agent Role Definitions Loaded")
    print("=" * 60)
    
    for role_id, role in AGENT_ROLES_REGISTRY.items():
        print(f"\n{role.name}")
        print(f"  Role ID: {role.role_id}")
        print(f"  Objective: {role.primary_objective}")
        print(f"  Capabilities: {len(role.capabilities)}")
        print(f"  Responsibilities: {len(role.responsibilities)}")
        print(f"  Max Authority: {role.max_authority.value}")
        
        # Show KPI targets for first responsibility
        if role.responsibilities:
            first_resp = role.responsibilities[0]
            print(f"  Primary KPIs: {list(first_resp.kpi_targets.keys())}")