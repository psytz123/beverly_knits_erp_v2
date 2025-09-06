#!/usr/bin/env python3
"""
State Management for eFab AI Agent System
Manages shared state, customer context, and system-wide information
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid
import asyncio
from threading import Lock

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImplementationPhase(Enum):
    """Implementation phases for customer onboarding"""
    PRE_ASSESSMENT = "PRE_ASSESSMENT"
    DISCOVERY = "DISCOVERY" 
    CONFIGURATION = "CONFIGURATION"
    DATA_MIGRATION = "DATA_MIGRATION"
    TRAINING = "TRAINING"
    TESTING = "TESTING"
    GO_LIVE = "GO_LIVE"
    STABILIZATION = "STABILIZATION"
    OPTIMIZATION = "OPTIMIZATION"
    COMPLETED = "COMPLETED"
    
    @property
    def week_number(self) -> int:
        """Get week number for this phase"""
        phases = {
            "PRE_ASSESSMENT": 0,
            "DISCOVERY": 1,
            "CONFIGURATION": 2, 
            "DATA_MIGRATION": 3,
            "TRAINING": 4,
            "TESTING": 5,
            "GO_LIVE": 6,
            "STABILIZATION": 6,
            "OPTIMIZATION": 7,
            "COMPLETED": 8
        }
        return phases.get(self.value, 0)


class IndustryType(Enum):
    """Supported manufacturing industry types"""
    FURNITURE = "FURNITURE"
    INJECTION_MOLDING = "INJECTION_MOLDING"
    ELECTRICAL_EQUIPMENT = "ELECTRICAL_EQUIPMENT"
    GENERIC_MANUFACTURING = "GENERIC_MANUFACTURING"


class CompanySize(Enum):
    """Company size categories"""
    SMALL = "SMALL"          # 10-50 employees
    MEDIUM = "MEDIUM"        # 50-200 employees  
    LARGE = "LARGE"          # 200-500 employees
    ENTERPRISE = "ENTERPRISE" # 500+ employees


class RiskLevel(Enum):
    """Implementation risk levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    
    @property
    def score(self) -> int:
        """Numeric risk score"""
        scores = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return scores.get(self.value, 1)


@dataclass
class CustomerProfile:
    """Comprehensive customer profile for implementation planning"""
    
    # Basic Information
    customer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    company_name: str = ""
    industry: IndustryType = IndustryType.GENERIC_MANUFACTURING
    company_size: CompanySize = CompanySize.MEDIUM
    annual_revenue: float = 0.0
    employee_count: int = 0
    
    # Geographic and Contact Info  
    country: str = "US"
    timezone: str = "UTC"
    primary_contact: str = ""
    technical_contact: str = ""
    
    # Current System Information
    current_erp_system: str = ""
    current_erp_version: str = ""
    data_volume_estimate: str = ""  # Small, Medium, Large, Enterprise
    integration_complexity: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    
    # Manufacturing Specifics
    manufacturing_processes: List[str] = field(default_factory=list)
    product_types: List[str] = field(default_factory=list)
    production_volume_annual: float = 0.0
    quality_certifications: List[str] = field(default_factory=list)
    
    # Implementation Context
    implementation_start_date: Optional[datetime] = None
    target_go_live_date: Optional[datetime] = None
    budget_range: str = ""  # Starter, Professional, Enterprise
    critical_requirements: List[str] = field(default_factory=list)
    
    # Risk Assessment
    change_management_readiness: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    technical_team_capability: str = "MEDIUM"   # LOW, MEDIUM, HIGH
    executive_sponsorship: str = "MEDIUM"       # LOW, MEDIUM, HIGH
    previous_erp_experience: str = "SOME"       # NONE, SOME, EXTENSIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = asdict(self)
        result['industry'] = self.industry.value
        result['company_size'] = self.company_size.value
        if self.implementation_start_date:
            result['implementation_start_date'] = self.implementation_start_date.isoformat()
        if self.target_go_live_date:
            result['target_go_live_date'] = self.target_go_live_date.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomerProfile':
        """Create from dictionary"""
        # Convert enum fields
        if 'industry' in data and isinstance(data['industry'], str):
            data['industry'] = IndustryType(data['industry'])
        if 'company_size' in data and isinstance(data['company_size'], str):
            data['company_size'] = CompanySize(data['company_size'])
        
        # Convert datetime fields
        if 'implementation_start_date' in data and isinstance(data['implementation_start_date'], str):
            data['implementation_start_date'] = datetime.fromisoformat(data['implementation_start_date'])
        if 'target_go_live_date' in data and isinstance(data['target_go_live_date'], str):
            data['target_go_live_date'] = datetime.fromisoformat(data['target_go_live_date'])
            
        return cls(**data)
    
    def calculate_complexity_score(self) -> int:
        """Calculate implementation complexity score (1-10)"""
        score = 5  # Base score
        
        # Company size impact
        size_impact = {
            CompanySize.SMALL: -1,
            CompanySize.MEDIUM: 0,
            CompanySize.LARGE: 1,
            CompanySize.ENTERPRISE: 2
        }
        score += size_impact.get(self.company_size, 0)
        
        # Integration complexity
        integration_impact = {"LOW": -1, "MEDIUM": 0, "HIGH": 2}
        score += integration_impact.get(self.integration_complexity, 0)
        
        # Data volume impact
        data_impact = {"Small": -1, "Medium": 0, "Large": 1, "Enterprise": 2}
        score += data_impact.get(self.data_volume_estimate, 0)
        
        # Change management readiness (inverse impact)
        change_impact = {"HIGH": -1, "MEDIUM": 0, "LOW": 1}
        score += change_impact.get(self.change_management_readiness, 0)
        
        return max(1, min(10, score))  # Clamp between 1-10


@dataclass 
class ImplementationPlan:
    """Implementation plan with timeline and milestones"""
    customer_id: str
    estimated_duration_weeks: int = 6
    confidence_interval: float = 0.85
    risk_level: RiskLevel = RiskLevel.MEDIUM
    phases: List[Dict[str, Any]] = field(default_factory=list)
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['risk_level'] = self.risk_level.value
        return result


@dataclass
class SystemMetrics:
    """System-wide performance and operational metrics"""
    total_customers: int = 0
    active_implementations: int = 0
    successful_implementations: int = 0
    failed_implementations: int = 0
    average_implementation_duration_weeks: float = 0.0
    customer_satisfaction_score: float = 0.0
    system_uptime_percentage: float = 99.9
    average_response_time_ms: float = 0.0
    error_rate_percentage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate implementation success rate"""
        total = self.successful_implementations + self.failed_implementations
        if total == 0:
            return 0.0
        return (self.successful_implementations / total) * 100


class SystemState:
    """
    Central system state manager for eFab AI Agent System
    Manages shared state, customer contexts, and system-wide information
    """
    
    def __init__(self):
        """Initialize system state manager"""
        self.logger = logging.getLogger("SystemState")
        self.lock = Lock()
        
        # Customer and implementation tracking
        self.customers: Dict[str, CustomerProfile] = {}
        self.implementation_plans: Dict[str, ImplementationPlan] = {}
        self.implementation_phases: Dict[str, ImplementationPhase] = {}
        
        # Agent state tracking
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_assignments: Dict[str, List[str]] = {}  # customer_id -> agent_ids
        
        # Shared knowledge and configuration
        self.shared_knowledge: Dict[str, Any] = {}
        self.system_config: Dict[str, Any] = {}
        self.feature_flags: Dict[str, bool] = {}
        
        # Performance and operational metrics
        self.metrics = SystemMetrics()
        self.performance_history: List[Dict[str, Any]] = []
        
        # Event tracking
        self.events: List[Dict[str, Any]] = []
        
        self.logger.info("SystemState manager initialized")
    
    def register_customer(self, customer_profile: CustomerProfile) -> str:
        """Register new customer and return customer ID"""
        with self.lock:
            customer_id = customer_profile.customer_id
            self.customers[customer_id] = customer_profile
            self.implementation_phases[customer_id] = ImplementationPhase.PRE_ASSESSMENT
            
            # Generate initial implementation plan
            plan = self._generate_implementation_plan(customer_profile)
            self.implementation_plans[customer_id] = plan
            
            self.metrics.total_customers += 1
            
            self._log_event("CUSTOMER_REGISTERED", {
                "customer_id": customer_id,
                "company_name": customer_profile.company_name,
                "industry": customer_profile.industry.value
            })
            
            self.logger.info(f"Registered customer: {customer_profile.company_name} ({customer_id})")
            return customer_id
    
    def _generate_implementation_plan(self, customer_profile: CustomerProfile) -> ImplementationPlan:
        """Generate implementation plan based on customer profile"""
        complexity_score = customer_profile.calculate_complexity_score()
        
        # Determine duration and risk based on complexity
        if complexity_score <= 3:
            duration_weeks = 6
            risk_level = RiskLevel.LOW
            confidence = 0.95
        elif complexity_score <= 6:
            duration_weeks = 7
            risk_level = RiskLevel.MEDIUM
            confidence = 0.85
        elif complexity_score <= 8:
            duration_weeks = 8
            risk_level = RiskLevel.HIGH
            confidence = 0.75
        else:
            duration_weeks = 9
            risk_level = RiskLevel.CRITICAL
            confidence = 0.65
        
        # Generate phase plan
        phases = [
            {"name": "Discovery & Assessment", "week": 1, "duration_days": 5},
            {"name": "System Configuration", "week": 2, "duration_days": 7},
            {"name": "Data Migration", "week": 3, "duration_days": 5},
            {"name": "User Training", "week": 4, "duration_days": 5},
            {"name": "Testing & Validation", "week": 5, "duration_days": 5},
            {"name": "Go-Live & Stabilization", "week": 6, "duration_days": 7}
        ]
        
        if duration_weeks > 6:
            phases.append({"name": "Optimization", "week": 7, "duration_days": 7})
        if duration_weeks > 7:
            phases.append({"name": "Advanced Features", "week": 8, "duration_days": 7})
        if duration_weeks > 8:
            phases.append({"name": "Enterprise Integration", "week": 9, "duration_days": 7})
        
        # Identify risk factors
        risk_factors = []
        if complexity_score > 7:
            risk_factors.append("High system complexity")
        if customer_profile.integration_complexity == "HIGH":
            risk_factors.append("Complex integrations required")
        if customer_profile.change_management_readiness == "LOW":
            risk_factors.append("Low change management readiness")
        if customer_profile.data_volume_estimate == "Enterprise":
            risk_factors.append("Large data migration volume")
        
        return ImplementationPlan(
            customer_id=customer_profile.customer_id,
            estimated_duration_weeks=duration_weeks,
            confidence_interval=confidence,
            risk_level=risk_level,
            phases=phases,
            risk_factors=risk_factors
        )
    
    def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        """Get customer profile by ID"""
        return self.customers.get(customer_id)
    
    def get_implementation_plan(self, customer_id: str) -> Optional[ImplementationPlan]:
        """Get implementation plan for customer"""
        return self.implementation_plans.get(customer_id)
    
    def update_implementation_phase(self, customer_id: str, phase: ImplementationPhase):
        """Update implementation phase for customer"""
        with self.lock:
            if customer_id in self.customers:
                old_phase = self.implementation_phases.get(customer_id)
                self.implementation_phases[customer_id] = phase
                
                self._log_event("PHASE_TRANSITION", {
                    "customer_id": customer_id,
                    "old_phase": old_phase.value if old_phase else None,
                    "new_phase": phase.value
                })
                
                # Update metrics if implementation completed
                if phase == ImplementationPhase.COMPLETED:
                    self.metrics.successful_implementations += 1
                    self.metrics.active_implementations -= 1
                elif old_phase == ImplementationPhase.PRE_ASSESSMENT and phase == ImplementationPhase.DISCOVERY:
                    self.metrics.active_implementations += 1
                
                self.logger.info(f"Customer {customer_id} moved to phase: {phase.value}")
    
    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """Register active agent"""
        with self.lock:
            self.active_agents[agent_id] = {
                **agent_info,
                "registered_at": datetime.now(),
                "last_heartbeat": datetime.now(),
                "status": "ACTIVE"
            }
            
            self.logger.info(f"Registered agent: {agent_id}")
    
    def update_agent_heartbeat(self, agent_id: str, status_info: Dict[str, Any] = None):
        """Update agent heartbeat and status"""
        with self.lock:
            if agent_id in self.active_agents:
                self.active_agents[agent_id]["last_heartbeat"] = datetime.now()
                if status_info:
                    self.active_agents[agent_id].update(status_info)
    
    def assign_agents_to_customer(self, customer_id: str, agent_ids: List[str]):
        """Assign agents to customer implementation"""
        with self.lock:
            self.agent_assignments[customer_id] = agent_ids
            
            self._log_event("AGENTS_ASSIGNED", {
                "customer_id": customer_id,
                "agent_ids": agent_ids
            })
    
    def get_customer_agents(self, customer_id: str) -> List[str]:
        """Get agents assigned to customer"""
        return self.agent_assignments.get(customer_id, [])
    
    def set_shared_knowledge(self, key: str, value: Any):
        """Set shared knowledge item"""
        with self.lock:
            self.shared_knowledge[key] = {
                "value": value,
                "updated_at": datetime.now()
            }
    
    def get_shared_knowledge(self, key: str) -> Any:
        """Get shared knowledge item"""
        knowledge = self.shared_knowledge.get(key)
        return knowledge["value"] if knowledge else None
    
    def set_feature_flag(self, flag_name: str, enabled: bool):
        """Set feature flag"""
        with self.lock:
            self.feature_flags[flag_name] = enabled
            self.logger.info(f"Feature flag {flag_name} set to {enabled}")
    
    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get feature flag value"""
        return self.feature_flags.get(flag_name, default)
    
    def update_metrics(self, metric_updates: Dict[str, Any]):
        """Update system metrics"""
        with self.lock:
            for key, value in metric_updates.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
            
            self.metrics.last_updated = datetime.now()
    
    def _log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log system event"""
        event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "data": event_data
        }
        
        self.events.append(event)
        
        # Keep only last 10000 events
        if len(self.events) > 10000:
            self.events = self.events[-5000:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self.lock:
            # Calculate active agent count
            active_agent_count = len([
                agent for agent in self.active_agents.values()
                if (datetime.now() - agent["last_heartbeat"]).total_seconds() < 300
            ])
            
            return {
                "timestamp": datetime.now().isoformat(),
                "customers": {
                    "total": self.metrics.total_customers,
                    "active_implementations": self.metrics.active_implementations,
                    "success_rate": self.metrics.success_rate
                },
                "agents": {
                    "total_registered": len(self.active_agents),
                    "currently_active": active_agent_count
                },
                "performance": {
                    "uptime_percentage": self.metrics.system_uptime_percentage,
                    "average_response_time_ms": self.metrics.average_response_time_ms,
                    "error_rate_percentage": self.metrics.error_rate_percentage
                },
                "feature_flags": self.feature_flags.copy()
            }
    
    def get_customer_dashboard(self, customer_id: str) -> Dict[str, Any]:
        """Get customer-specific dashboard data"""
        customer = self.customers.get(customer_id)
        if not customer:
            return {}
        
        plan = self.implementation_plans.get(customer_id)
        phase = self.implementation_phases.get(customer_id)
        assigned_agents = self.agent_assignments.get(customer_id, [])
        
        return {
            "customer": customer.to_dict(),
            "implementation_plan": plan.to_dict() if plan else None,
            "current_phase": phase.value if phase else None,
            "assigned_agents": assigned_agents,
            "progress_percentage": phase.week_number * 12.5 if phase else 0  # 8 weeks = 100%
        }
    
    def save_state(self, file_path: str):
        """Save system state to file"""
        try:
            state_data = {
                "customers": {cid: customer.to_dict() for cid, customer in self.customers.items()},
                "implementation_plans": {cid: plan.to_dict() for cid, plan in self.implementation_plans.items()},
                "implementation_phases": {cid: phase.value for cid, phase in self.implementation_phases.items()},
                "shared_knowledge": self.shared_knowledge,
                "system_config": self.system_config,
                "feature_flags": self.feature_flags,
                "metrics": asdict(self.metrics),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
                
            self.logger.info(f"System state saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save system state: {str(e)}")
    
    def load_state(self, file_path: str):
        """Load system state from file"""
        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore customers
            for cid, customer_data in state_data.get("customers", {}).items():
                self.customers[cid] = CustomerProfile.from_dict(customer_data)
            
            # Restore implementation plans
            for cid, plan_data in state_data.get("implementation_plans", {}).items():
                plan = ImplementationPlan(customer_id=cid, **plan_data)
                self.implementation_plans[cid] = plan
            
            # Restore phases
            for cid, phase_value in state_data.get("implementation_phases", {}).items():
                self.implementation_phases[cid] = ImplementationPhase(phase_value)
            
            # Restore other data
            self.shared_knowledge = state_data.get("shared_knowledge", {})
            self.system_config = state_data.get("system_config", {})
            self.feature_flags = state_data.get("feature_flags", {})
            
            # Restore metrics
            metrics_data = state_data.get("metrics", {})
            if metrics_data:
                self.metrics = SystemMetrics(**metrics_data)
            
            self.logger.info(f"System state loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load system state: {str(e)}")


# Global system state instance
system_state = SystemState()

# Export key components
__all__ = [
    "SystemState",
    "CustomerProfile", 
    "ImplementationPlan",
    "SystemMetrics",
    "ImplementationPhase",
    "IndustryType",
    "CompanySize",
    "RiskLevel",
    "system_state"
]