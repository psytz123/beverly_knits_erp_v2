#!/usr/bin/env python3
"""
Implementation Project Manager Agent for eFab AI Agent System
Autonomous project management for 6-9 week ERP implementations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import math

from ..core.agent_base import BaseAgent, AgentMessage, AgentCapability, MessageType, Priority
from ..core.state_manager import SystemState, CustomerProfile, ImplementationPhase, RiskLevel, system_state

# Setup logging
logger = logging.getLogger(__name__)


class ProjectStatus(Enum):
    """Project management status"""
    PLANNING = "PLANNING"
    IN_PROGRESS = "IN_PROGRESS"
    ON_TRACK = "ON_TRACK"
    AT_RISK = "AT_RISK"
    DELAYED = "DELAYED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class MilestoneStatus(Enum):
    """Milestone completion status"""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    OVERDUE = "OVERDUE"
    BLOCKED = "BLOCKED"


@dataclass
class ProjectMilestone:
    """Implementation project milestone"""
    milestone_id: str
    name: str
    description: str
    planned_start_date: datetime
    planned_end_date: datetime
    actual_start_date: Optional[datetime] = None
    actual_end_date: Optional[datetime] = None
    status: MilestoneStatus = MilestoneStatus.NOT_STARTED
    progress_percentage: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    assigned_agents: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "milestone_id": self.milestone_id,
            "name": self.name,
            "description": self.description,
            "planned_start_date": self.planned_start_date.isoformat(),
            "planned_end_date": self.planned_end_date.isoformat(),
            "actual_start_date": self.actual_start_date.isoformat() if self.actual_start_date else None,
            "actual_end_date": self.actual_end_date.isoformat() if self.actual_end_date else None,
            "status": self.status.value,
            "progress_percentage": self.progress_percentage,
            "dependencies": self.dependencies,
            "assigned_agents": self.assigned_agents,
            "deliverables": self.deliverables,
            "risks": self.risks
        }
    
    def is_overdue(self) -> bool:
        """Check if milestone is overdue"""
        if self.status == MilestoneStatus.COMPLETED:
            return False
        return datetime.now() > self.planned_end_date
    
    def calculate_delay_days(self) -> float:
        """Calculate delay in days"""
        if self.status == MilestoneStatus.COMPLETED and self.actual_end_date:
            return max(0, (self.actual_end_date - self.planned_end_date).days)
        elif datetime.now() > self.planned_end_date:
            return (datetime.now() - self.planned_end_date).days
        return 0


@dataclass
class RiskAssessment:
    """Implementation risk assessment"""
    risk_id: str
    category: str  # TECHNICAL, BUSINESS, RESOURCE, TIMELINE, EXTERNAL
    description: str
    probability: float  # 0.0 to 1.0
    impact: float      # 0.0 to 1.0  
    risk_score: float = field(init=False)
    mitigation_strategy: str = ""
    responsible_agent: str = ""
    status: str = "IDENTIFIED"  # IDENTIFIED, MITIGATING, MITIGATED, REALIZED
    
    def __post_init__(self):
        """Calculate risk score"""
        self.risk_score = self.probability * self.impact
    
    @property
    def risk_level(self) -> str:
        """Get risk level based on score"""
        if self.risk_score >= 0.8:
            return "CRITICAL"
        elif self.risk_score >= 0.6:
            return "HIGH"
        elif self.risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "risk_id": self.risk_id,
            "category": self.category,
            "description": self.description,
            "probability": self.probability,
            "impact": self.impact,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "mitigation_strategy": self.mitigation_strategy,
            "responsible_agent": self.responsible_agent,
            "status": self.status
        }


@dataclass
class ProjectMetrics:
    """Project performance metrics"""
    customer_id: str
    planned_duration_weeks: float
    actual_duration_weeks: float = 0.0
    progress_percentage: float = 0.0
    milestones_completed: int = 0
    total_milestones: int = 0
    risks_identified: int = 0
    risks_mitigated: int = 0
    budget_planned: float = 0.0
    budget_actual: float = 0.0
    team_velocity: float = 0.0  # Milestones per week
    customer_satisfaction: float = 0.0
    
    @property
    def completion_rate(self) -> float:
        """Calculate milestone completion rate"""
        if self.total_milestones == 0:
            return 0.0
        return (self.milestones_completed / self.total_milestones) * 100
    
    @property
    def schedule_variance(self) -> float:
        """Calculate schedule variance (positive = ahead, negative = behind)"""
        if self.planned_duration_weeks == 0:
            return 0.0
        return ((self.planned_duration_weeks - self.actual_duration_weeks) / self.planned_duration_weeks) * 100
    
    @property
    def budget_variance(self) -> float:
        """Calculate budget variance (positive = under budget, negative = over budget)"""
        if self.budget_planned == 0:
            return 0.0
        return ((self.budget_planned - self.budget_actual) / self.budget_planned) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "customer_id": self.customer_id,
            "planned_duration_weeks": self.planned_duration_weeks,
            "actual_duration_weeks": self.actual_duration_weeks,
            "progress_percentage": self.progress_percentage,
            "milestones_completed": self.milestones_completed,
            "total_milestones": self.total_milestones,
            "completion_rate": self.completion_rate,
            "schedule_variance": self.schedule_variance,
            "budget_variance": self.budget_variance,
            "risks_identified": self.risks_identified,
            "risks_mitigated": self.risks_mitigated,
            "team_velocity": self.team_velocity,
            "customer_satisfaction": self.customer_satisfaction
        }


class ImplementationProjectManagerAgent(BaseAgent):
    """
    Implementation Project Manager Agent
    
    Responsibilities:
    - Autonomous project planning and timeline management
    - Customer assessment and complexity analysis
    - Risk identification, assessment, and mitigation
    - Progress monitoring and timeline prediction
    - Stakeholder communication and reporting
    - Resource allocation and team coordination
    """
    
    def __init__(self):
        """Initialize Implementation Project Manager Agent"""
        super().__init__(
            agent_id="implementation_pm",
            agent_name="Implementation Project Manager",
            agent_description="Autonomous project management for ERP implementations"
        )
        
        # Project management data
        self.active_projects: Dict[str, Dict[str, Any]] = {}
        self.project_milestones: Dict[str, List[ProjectMilestone]] = {}
        self.risk_assessments: Dict[str, List[RiskAssessment]] = {}
        self.project_metrics: Dict[str, ProjectMetrics] = {}
        
        # Predictive models for timeline and risk assessment
        self.complexity_factors = {
            "company_size": {"SMALL": 1.0, "MEDIUM": 1.2, "LARGE": 1.5, "ENTERPRISE": 2.0},
            "data_volume": {"Small": 1.0, "Medium": 1.2, "Large": 1.5, "Enterprise": 2.0},
            "integration_complexity": {"LOW": 1.0, "MEDIUM": 1.3, "HIGH": 1.8},
            "change_readiness": {"HIGH": 0.9, "MEDIUM": 1.0, "LOW": 1.4},
            "erp_experience": {"EXTENSIVE": 0.8, "SOME": 1.0, "NONE": 1.3}
        }
        
        # Historical data for predictions
        self.implementation_history: List[Dict[str, Any]] = []
        
        # Communication templates
        self.communication_templates = self._load_communication_templates()
        
        self.logger.info("Implementation Project Manager Agent initialized")
    
    def _initialize(self):
        """Initialize agent-specific components"""
        # Register capabilities
        self.register_capability(AgentCapability(
            name="customer_assessment",
            description="Assess customer complexity and generate implementation plan",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "customer_profile": {"type": "object"}
                },
                "required": ["customer_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "complexity_score": {"type": "number"},
                    "implementation_plan": {"type": "object"},
                    "risk_assessment": {"type": "object"},
                    "timeline_estimate": {"type": "object"}
                }
            },
            estimated_duration_seconds=3600,  # 1 hour
            risk_level="LOW"
        ))
        
        self.register_capability(AgentCapability(
            name="project_monitoring",
            description="Monitor implementation progress and identify risks",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "milestone_updates": {"type": "array"}
                },
                "required": ["customer_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "project_status": {"type": "object"},
                    "risks_identified": {"type": "array"},
                    "recommendations": {"type": "array"}
                }
            },
            estimated_duration_seconds=1800,  # 30 minutes
            risk_level="LOW"
        ))
        
        self.register_capability(AgentCapability(
            name="timeline_prediction",
            description="Predict implementation timeline and completion dates",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "current_progress": {"type": "object"}
                },
                "required": ["customer_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "predicted_completion": {"type": "string"},
                    "confidence_interval": {"type": "number"},
                    "risk_factors": {"type": "array"}
                }
            },
            estimated_duration_seconds=900,  # 15 minutes
            risk_level="LOW"
        ))
        
        # Register message handlers
        self.register_message_handler(MessageType.REQUEST, self._handle_request)
        self.register_message_handler(MessageType.NOTIFICATION, self._handle_notification)
        
        # Start background monitoring
        asyncio.create_task(self._project_monitoring_loop())
        asyncio.create_task(self._risk_assessment_loop())
    
    async def assess_customer_complexity(self, customer_id: str) -> Dict[str, Any]:
        """
        Assess customer implementation complexity and generate plan
        
        Args:
            customer_id: Customer to assess
            
        Returns:
            Assessment results with complexity score and implementation plan
        """
        try:
            # Get customer profile
            customer_profile = system_state.get_customer_profile(customer_id)
            if not customer_profile:
                return {"error": "Customer profile not found"}
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(customer_profile)
            
            # Generate implementation plan
            implementation_plan = self._generate_implementation_plan(customer_profile, complexity_score)
            
            # Perform risk assessment
            risk_assessment = self._perform_risk_assessment(customer_profile, complexity_score)
            
            # Estimate timeline
            timeline_estimate = self._estimate_timeline(customer_profile, complexity_score)
            
            # Store project data
            self.active_projects[customer_id] = {
                "customer_profile": customer_profile.to_dict(),
                "complexity_score": complexity_score,
                "implementation_plan": implementation_plan,
                "risk_assessment": risk_assessment,
                "timeline_estimate": timeline_estimate,
                "created_at": datetime.now().isoformat(),
                "status": ProjectStatus.PLANNING.value
            }
            
            # Initialize project metrics
            self.project_metrics[customer_id] = ProjectMetrics(
                customer_id=customer_id,
                planned_duration_weeks=timeline_estimate["duration_weeks"],
                total_milestones=len(implementation_plan["milestones"])
            )
            
            self.logger.info(f"Customer assessment completed for {customer_id}: complexity={complexity_score:.2f}")
            
            return {
                "customer_id": customer_id,
                "complexity_score": complexity_score,
                "implementation_plan": implementation_plan,
                "risk_assessment": risk_assessment,
                "timeline_estimate": timeline_estimate
            }
            
        except Exception as e:
            self.logger.error(f"Failed to assess customer {customer_id}: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_complexity_score(self, customer_profile: CustomerProfile) -> float:
        """Calculate implementation complexity score"""
        base_score = 5.0  # Base complexity (1-10 scale)
        
        # Apply complexity factors
        factors = {
            "company_size": self.complexity_factors["company_size"].get(customer_profile.company_size.value, 1.0),
            "data_volume": self.complexity_factors["data_volume"].get(customer_profile.data_volume_estimate, 1.0),
            "integration": self.complexity_factors["integration_complexity"].get(customer_profile.integration_complexity, 1.0),
            "change_readiness": self.complexity_factors["change_readiness"].get(customer_profile.change_management_readiness, 1.0),
            "erp_experience": self.complexity_factors["erp_experience"].get(customer_profile.previous_erp_experience, 1.0)
        }
        
        # Calculate weighted score
        complexity_score = base_score
        for factor_name, factor_value in factors.items():
            complexity_score *= factor_value
        
        # Normalize to 1-10 scale
        complexity_score = max(1.0, min(10.0, complexity_score / 2.0))
        
        return complexity_score
    
    def _generate_implementation_plan(self, customer_profile: CustomerProfile, complexity_score: float) -> Dict[str, Any]:
        """Generate detailed implementation plan"""
        # Base timeline (6-9 weeks based on complexity)
        base_duration = 6
        if complexity_score > 7.5:
            duration_weeks = 9
        elif complexity_score > 6.0:
            duration_weeks = 8
        elif complexity_score > 4.5:
            duration_weeks = 7
        else:
            duration_weeks = 6
        
        # Generate milestones
        milestones = self._generate_milestones(customer_profile, duration_weeks)
        
        # Store milestones
        self.project_milestones[customer_profile.customer_id] = milestones
        
        return {
            "customer_id": customer_profile.customer_id,
            "duration_weeks": duration_weeks,
            "complexity_score": complexity_score,
            "milestones": [milestone.to_dict() for milestone in milestones],
            "resource_requirements": self._calculate_resource_requirements(complexity_score),
            "success_factors": self._identify_success_factors(customer_profile),
            "potential_challenges": self._identify_challenges(customer_profile, complexity_score)
        }
    
    def _generate_milestones(self, customer_profile: CustomerProfile, duration_weeks: int) -> List[ProjectMilestone]:
        """Generate implementation milestones"""
        start_date = datetime.now()
        milestones = []
        
        # Phase 1: Discovery & Assessment (Week 1)
        milestone1 = ProjectMilestone(
            milestone_id=f"{customer_profile.customer_id}_m1",
            name="Discovery & Assessment Complete",
            description="Legacy system analysis, data assessment, and process mapping completed",
            planned_start_date=start_date,
            planned_end_date=start_date + timedelta(weeks=1),
            deliverables=[
                "Legacy system analysis report",
                "Data quality assessment", 
                "Business process mapping",
                "Gap analysis document"
            ]
        )
        milestones.append(milestone1)
        
        # Phase 2: System Configuration (Week 2)
        milestone2 = ProjectMilestone(
            milestone_id=f"{customer_profile.customer_id}_m2",
            name="System Configuration Complete",
            description="ERP system configured for customer requirements",
            planned_start_date=start_date + timedelta(weeks=1),
            planned_end_date=start_date + timedelta(weeks=2),
            dependencies=[milestone1.milestone_id],
            deliverables=[
                "System configuration document",
                "Business rules implementation",
                "User interface customization",
                "Integration setup"
            ]
        )
        milestones.append(milestone2)
        
        # Phase 3: Data Migration (Week 3)
        milestone3 = ProjectMilestone(
            milestone_id=f"{customer_profile.customer_id}_m3",
            name="Data Migration Complete",
            description="All data migrated and validated in new system",
            planned_start_date=start_date + timedelta(weeks=2),
            planned_end_date=start_date + timedelta(weeks=3),
            dependencies=[milestone2.milestone_id],
            deliverables=[
                "Data migration scripts",
                "Data validation reports",
                "Migration test results",
                "Rollback procedures"
            ]
        )
        milestones.append(milestone3)
        
        # Phase 4: User Training (Week 4)
        milestone4 = ProjectMilestone(
            milestone_id=f"{customer_profile.customer_id}_m4",
            name="User Training Complete",
            description="All users trained and competency validated",
            planned_start_date=start_date + timedelta(weeks=3),
            planned_end_date=start_date + timedelta(weeks=4),
            dependencies=[milestone3.milestone_id],
            deliverables=[
                "Training materials",
                "User competency assessments",
                "Training completion certificates",
                "Support documentation"
            ]
        )
        milestones.append(milestone4)
        
        # Phase 5: Testing & Validation (Week 5)
        milestone5 = ProjectMilestone(
            milestone_id=f"{customer_profile.customer_id}_m5",
            name="Testing & Validation Complete",
            description="System tested and validated for production use",
            planned_start_date=start_date + timedelta(weeks=4),
            planned_end_date=start_date + timedelta(weeks=5),
            dependencies=[milestone4.milestone_id],
            deliverables=[
                "Test execution reports",
                "User acceptance test results",
                "Performance benchmarks",
                "Go-live readiness checklist"
            ]
        )
        milestones.append(milestone5)
        
        # Phase 6: Go-Live & Stabilization (Week 6)
        milestone6 = ProjectMilestone(
            milestone_id=f"{customer_profile.customer_id}_m6",
            name="Go-Live & Stabilization Complete",
            description="System live in production and stabilized",
            planned_start_date=start_date + timedelta(weeks=5),
            planned_end_date=start_date + timedelta(weeks=6),
            dependencies=[milestone5.milestone_id],
            deliverables=[
                "Production deployment report",
                "System monitoring setup",
                "Support procedures",
                "Performance metrics baseline"
            ]
        )
        milestones.append(milestone6)
        
        # Additional phases for complex implementations
        if duration_weeks > 6:
            milestone7 = ProjectMilestone(
                milestone_id=f"{customer_profile.customer_id}_m7",
                name="Optimization & Enhancement",
                description="System optimized and additional features implemented",
                planned_start_date=start_date + timedelta(weeks=6),
                planned_end_date=start_date + timedelta(weeks=7),
                dependencies=[milestone6.milestone_id],
                deliverables=[
                    "Performance optimization report",
                    "Advanced feature implementation",
                    "User feedback integration",
                    "ROI measurement"
                ]
            )
            milestones.append(milestone7)
        
        if duration_weeks > 7:
            milestone8 = ProjectMilestone(
                milestone_id=f"{customer_profile.customer_id}_m8",
                name="Advanced Integration",
                description="Complex integrations and advanced workflows implemented",
                planned_start_date=start_date + timedelta(weeks=7),
                planned_end_date=start_date + timedelta(weeks=8),
                dependencies=[milestone7.milestone_id],
                deliverables=[
                    "Integration test results",
                    "Advanced workflow documentation",
                    "Performance benchmarks",
                    "Security validation"
                ]
            )
            milestones.append(milestone8)
        
        if duration_weeks > 8:
            milestone9 = ProjectMilestone(
                milestone_id=f"{customer_profile.customer_id}_m9",
                name="Enterprise Features",
                description="Enterprise-grade features and scalability implemented",
                planned_start_date=start_date + timedelta(weeks=8),
                planned_end_date=start_date + timedelta(weeks=9),
                dependencies=[milestone8.milestone_id],
                deliverables=[
                    "Enterprise feature documentation",
                    "Scalability test results",
                    "Security audit report",
                    "Final implementation report"
                ]
            )
            milestones.append(milestone9)
        
        return milestones
    
    def _perform_risk_assessment(self, customer_profile: CustomerProfile, complexity_score: float) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        risks = []
        
        # Technical risks
        if complexity_score > 7:
            risks.append(RiskAssessment(
                risk_id=f"{customer_profile.customer_id}_r1",
                category="TECHNICAL",
                description="High system complexity may lead to integration challenges",
                probability=0.6,
                impact=0.7,
                mitigation_strategy="Implement phased integration approach with comprehensive testing"
            ))
        
        if customer_profile.integration_complexity == "HIGH":
            risks.append(RiskAssessment(
                risk_id=f"{customer_profile.customer_id}_r2",
                category="TECHNICAL", 
                description="Complex integrations may cause implementation delays",
                probability=0.5,
                impact=0.8,
                mitigation_strategy="Early integration testing and fallback procedures"
            ))
        
        # Business risks
        if customer_profile.change_management_readiness == "LOW":
            risks.append(RiskAssessment(
                risk_id=f"{customer_profile.customer_id}_r3",
                category="BUSINESS",
                description="Low change readiness may result in poor user adoption",
                probability=0.7,
                impact=0.6,
                mitigation_strategy="Enhanced change management program and executive sponsorship"
            ))
        
        if customer_profile.previous_erp_experience == "NONE":
            risks.append(RiskAssessment(
                risk_id=f"{customer_profile.customer_id}_r4",
                category="RESOURCE",
                description="Limited ERP experience may slow implementation progress",
                probability=0.6,
                impact=0.5,
                mitigation_strategy="Additional training and hands-on support"
            ))
        
        # Timeline risks
        if complexity_score > 8:
            risks.append(RiskAssessment(
                risk_id=f"{customer_profile.customer_id}_r5",
                category="TIMELINE",
                description="High complexity may extend implementation timeline",
                probability=0.7,
                impact=0.6,
                mitigation_strategy="Buffer time allocation and parallel work streams"
            ))
        
        # Store risks
        self.risk_assessments[customer_profile.customer_id] = risks
        
        return {
            "total_risks": len(risks),
            "risk_distribution": {
                "CRITICAL": len([r for r in risks if r.risk_level == "CRITICAL"]),
                "HIGH": len([r for r in risks if r.risk_level == "HIGH"]),
                "MEDIUM": len([r for r in risks if r.risk_level == "MEDIUM"]),
                "LOW": len([r for r in risks if r.risk_level == "LOW"])
            },
            "overall_risk_score": sum(r.risk_score for r in risks) / len(risks) if risks else 0,
            "risks": [risk.to_dict() for risk in risks]
        }
    
    def _estimate_timeline(self, customer_profile: CustomerProfile, complexity_score: float) -> Dict[str, Any]:
        """Estimate implementation timeline with confidence intervals"""
        # Base timeline calculation
        if complexity_score <= 3:
            duration_weeks = 6
            confidence = 0.95
        elif complexity_score <= 5:
            duration_weeks = 7
            confidence = 0.85
        elif complexity_score <= 7:
            duration_weeks = 8
            confidence = 0.75
        else:
            duration_weeks = 9
            confidence = 0.65
        
        # Calculate confidence intervals
        lower_bound = duration_weeks * 0.85
        upper_bound = duration_weeks * 1.25
        
        return {
            "duration_weeks": duration_weeks,
            "confidence_interval": confidence,
            "range": {
                "optimistic": lower_bound,
                "expected": duration_weeks,
                "pessimistic": upper_bound
            },
            "critical_path": [
                "Discovery & Assessment",
                "System Configuration", 
                "Data Migration",
                "Testing & Validation",
                "Go-Live & Stabilization"
            ]
        }
    
    def _calculate_resource_requirements(self, complexity_score: float) -> Dict[str, Any]:
        """Calculate resource requirements based on complexity"""
        # Base resource allocation
        base_hours = 240  # 6 weeks * 40 hours
        
        if complexity_score > 7:
            multiplier = 1.5
        elif complexity_score > 5:
            multiplier = 1.25
        else:
            multiplier = 1.0
        
        total_hours = base_hours * multiplier
        
        return {
            "total_hours": total_hours,
            "team_composition": {
                "project_manager": 0.25,      # 25% allocation
                "implementation_consultant": 0.5,  # 50% allocation
                "data_migration_specialist": 0.3,  # 30% allocation
                "training_specialist": 0.2,        # 20% allocation
                "technical_support": 0.1           # 10% allocation
            },
            "peak_utilization_weeks": [3, 4, 5],  # Data migration, training, testing
            "critical_skills": [
                "ERP implementation",
                "Data migration",
                "Change management",
                "Industry expertise"
            ]
        }
    
    def _identify_success_factors(self, customer_profile: CustomerProfile) -> List[str]:
        """Identify critical success factors"""
        success_factors = [
            "Executive sponsorship and leadership commitment",
            "Clear project scope and requirements definition",
            "Dedicated customer project team",
            "Effective change management and communication"
        ]
        
        if customer_profile.previous_erp_experience == "NONE":
            success_factors.append("Comprehensive training and user adoption program")
        
        if customer_profile.integration_complexity == "HIGH":
            success_factors.append("Early integration testing and validation")
        
        if customer_profile.company_size.value in ["LARGE", "ENTERPRISE"]:
            success_factors.append("Phased rollout and stakeholder management")
        
        return success_factors
    
    def _identify_challenges(self, customer_profile: CustomerProfile, complexity_score: float) -> List[str]:
        """Identify potential implementation challenges"""
        challenges = []
        
        if complexity_score > 7:
            challenges.append("Managing high system complexity and integration requirements")
        
        if customer_profile.change_management_readiness == "LOW":
            challenges.append("Overcoming resistance to change and ensuring user adoption")
        
        if customer_profile.data_volume_estimate == "Enterprise":
            challenges.append("Managing large-scale data migration with minimal downtime")
        
        if customer_profile.integration_complexity == "HIGH":
            challenges.append("Ensuring seamless integration with existing systems")
        
        if customer_profile.previous_erp_experience == "NONE":
            challenges.append("Building ERP expertise and operational capabilities")
        
        return challenges
    
    def _load_communication_templates(self) -> Dict[str, str]:
        """Load communication templates"""
        return {
            "project_kickoff": """
Dear {customer_name},

Welcome to your eFab ERP implementation! I'm your AI Project Manager, and I'll be orchestrating your {duration_weeks}-week implementation journey.

Based on our assessment:
- Implementation Complexity: {complexity_score}/10
- Estimated Duration: {duration_weeks} weeks
- Success Probability: {confidence}%

Key Milestones:
{milestones}

I'll be monitoring progress continuously and will keep you informed of any adjustments needed.

Best regards,
eFab AI Project Manager
            """,
            
            "milestone_update": """
Project Update: {milestone_name}

Status: {status}
Progress: {progress}%
Expected Completion: {expected_completion}

{update_details}

Next Steps:
{next_steps}

Regards,
eFab AI Project Manager
            """,
            
            "risk_alert": """
Risk Alert: {risk_description}

Risk Level: {risk_level}
Probability: {probability}%
Potential Impact: {impact}

Mitigation Strategy:
{mitigation_strategy}

Action Required: {action_required}

Regards,
eFab AI Project Manager
            """
        }
    
    async def monitor_project_progress(self, customer_id: str) -> Dict[str, Any]:
        """Monitor and analyze project progress"""
        try:
            if customer_id not in self.active_projects:
                return {"error": "Project not found"}
            
            # Get current milestones
            milestones = self.project_milestones.get(customer_id, [])
            if not milestones:
                return {"error": "No milestones found"}
            
            # Calculate progress metrics
            completed_milestones = [m for m in milestones if m.status == MilestoneStatus.COMPLETED]
            in_progress_milestones = [m for m in milestones if m.status == MilestoneStatus.IN_PROGRESS]
            overdue_milestones = [m for m in milestones if m.is_overdue()]
            
            overall_progress = (len(completed_milestones) / len(milestones)) * 100
            
            # Update project metrics
            metrics = self.project_metrics[customer_id]
            metrics.progress_percentage = overall_progress
            metrics.milestones_completed = len(completed_milestones)
            
            # Determine project status
            if len(overdue_milestones) > 0:
                project_status = ProjectStatus.DELAYED
            elif len([m for m in milestones if m.status == MilestoneStatus.AT_RISK]) > 0:
                project_status = ProjectStatus.AT_RISK
            elif overall_progress == 100:
                project_status = ProjectStatus.COMPLETED
            elif overall_progress > 0:
                project_status = ProjectStatus.ON_TRACK
            else:
                project_status = ProjectStatus.IN_PROGRESS
            
            # Update project status
            self.active_projects[customer_id]["status"] = project_status.value
            
            # Generate recommendations
            recommendations = self._generate_recommendations(customer_id, milestones, project_status)
            
            return {
                "customer_id": customer_id,
                "project_status": project_status.value,
                "overall_progress": overall_progress,
                "milestones": {
                    "total": len(milestones),
                    "completed": len(completed_milestones),
                    "in_progress": len(in_progress_milestones),
                    "overdue": len(overdue_milestones)
                },
                "metrics": metrics.to_dict(),
                "recommendations": recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Failed to monitor project progress for {customer_id}: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(
        self, 
        customer_id: str, 
        milestones: List[ProjectMilestone],
        project_status: ProjectStatus
    ) -> List[Dict[str, Any]]:
        """Generate project recommendations based on current status"""
        recommendations = []
        
        if project_status == ProjectStatus.DELAYED:
            overdue_milestones = [m for m in milestones if m.is_overdue()]
            recommendations.append({
                "type": "URGENT_ACTION",
                "title": "Address Overdue Milestones",
                "description": f"{len(overdue_milestones)} milestones are overdue. Immediate action required.",
                "actions": [
                    "Review resource allocation",
                    "Identify bottlenecks",
                    "Consider parallel execution",
                    "Escalate to customer leadership"
                ]
            })
        
        elif project_status == ProjectStatus.AT_RISK:
            recommendations.append({
                "type": "RISK_MITIGATION",
                "title": "Implement Risk Mitigation",
                "description": "Project shows risk indicators. Proactive measures recommended.",
                "actions": [
                    "Review and update risk register",
                    "Implement mitigation strategies",
                    "Increase monitoring frequency",
                    "Prepare contingency plans"
                ]
            })
        
        # Check for resource optimization opportunities
        active_milestones = [m for m in milestones if m.status == MilestoneStatus.IN_PROGRESS]
        if len(active_milestones) > 3:
            recommendations.append({
                "type": "OPTIMIZATION",
                "title": "Optimize Resource Allocation",
                "description": "Multiple concurrent milestones detected. Consider resource reallocation.",
                "actions": [
                    "Prioritize critical path activities",
                    "Reallocate resources to bottlenecks",
                    "Consider additional resources",
                    "Sequence dependencies optimally"
                ]
            })
        
        return recommendations
    
    async def predict_timeline_completion(self, customer_id: str) -> Dict[str, Any]:
        """Predict project completion timeline using current progress"""
        try:
            if customer_id not in self.active_projects:
                return {"error": "Project not found"}
            
            milestones = self.project_milestones.get(customer_id, [])
            if not milestones:
                return {"error": "No milestones found"}
            
            # Calculate velocity (milestones per week)
            completed_milestones = [m for m in milestones if m.status == MilestoneStatus.COMPLETED]
            
            if not completed_milestones:
                # No completed milestones - use planned timeline
                return {
                    "predicted_completion": milestones[-1].planned_end_date.isoformat(),
                    "confidence": 0.6,
                    "method": "PLANNED_TIMELINE"
                }
            
            # Calculate actual velocity
            project_start = min(m.actual_start_date for m in completed_milestones if m.actual_start_date)
            project_elapsed_weeks = (datetime.now() - project_start).days / 7
            
            if project_elapsed_weeks == 0:
                velocity = 0
            else:
                velocity = len(completed_milestones) / project_elapsed_weeks
            
            # Predict completion
            remaining_milestones = len(milestones) - len(completed_milestones)
            
            if velocity > 0:
                weeks_remaining = remaining_milestones / velocity
                predicted_completion = datetime.now() + timedelta(weeks=weeks_remaining)
                confidence = min(0.95, max(0.5, 1.0 - (abs(weeks_remaining - 3) / 10)))  # Higher confidence for 3-week remaining
            else:
                # Fall back to planned timeline
                predicted_completion = milestones[-1].planned_end_date
                confidence = 0.5
            
            # Calculate delay vs original plan
            original_completion = milestones[-1].planned_end_date
            delay_days = (predicted_completion - original_completion).days
            
            return {
                "predicted_completion": predicted_completion.isoformat(),
                "confidence": confidence,
                "velocity": velocity,
                "weeks_remaining": weeks_remaining if velocity > 0 else "unknown",
                "delay_days": delay_days,
                "method": "VELOCITY_BASED",
                "milestones_completed": len(completed_milestones),
                "milestones_remaining": remaining_milestones
            }
            
        except Exception as e:
            self.logger.error(f"Failed to predict timeline for {customer_id}: {str(e)}")
            return {"error": str(e)}
    
    async def _handle_request(self, message: AgentMessage) -> AgentMessage:
        """Handle incoming requests"""
        action = message.payload.get("action")
        
        if action == "CUSTOMER_ASSESSMENT":
            customer_id = message.payload.get("customer_id")
            result = await self.assess_customer_complexity(customer_id)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload=result,
                correlation_id=message.correlation_id
            )
        
        elif action == "MONITOR_PROGRESS":
            customer_id = message.payload.get("customer_id")
            result = await self.monitor_project_progress(customer_id)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload=result,
                correlation_id=message.correlation_id
            )
        
        elif action == "PREDICT_TIMELINE":
            customer_id = message.payload.get("customer_id")
            result = await self.predict_timeline_completion(customer_id)
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload=result,
                correlation_id=message.correlation_id
            )
        
        else:
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.ERROR,
                payload={"error": "UNSUPPORTED_ACTION", "action": action},
                correlation_id=message.correlation_id
            )
    
    async def _handle_notification(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming notifications"""
        notification_type = message.payload.get("notification_type")
        
        if notification_type == "MILESTONE_UPDATE":
            customer_id = message.payload.get("customer_id")
            milestone_id = message.payload.get("milestone_id")
            status = message.payload.get("status")
            progress = message.payload.get("progress", 0)
            
            # Update milestone status
            if customer_id in self.project_milestones:
                for milestone in self.project_milestones[customer_id]:
                    if milestone.milestone_id == milestone_id:
                        milestone.status = MilestoneStatus(status)
                        milestone.progress_percentage = progress
                        
                        if status == "COMPLETED":
                            milestone.actual_end_date = datetime.now()
                        elif status == "IN_PROGRESS" and not milestone.actual_start_date:
                            milestone.actual_start_date = datetime.now()
                        
                        self.logger.info(f"Milestone {milestone_id} updated: {status} ({progress}%)")
                        break
        
        elif notification_type == "PHASE_TRANSITION":
            customer_id = message.payload.get("customer_id")
            new_phase = message.payload.get("new_phase")
            
            # Update project phase
            if customer_id in self.active_projects:
                self.active_projects[customer_id]["current_phase"] = new_phase
                self.logger.info(f"Customer {customer_id} transitioned to phase: {new_phase}")
        
        return None
    
    async def _project_monitoring_loop(self):
        """Background project monitoring loop"""
        while self.status.value != "SHUTDOWN":
            try:
                for customer_id in self.active_projects.keys():
                    await self.monitor_project_progress(customer_id)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in project monitoring loop: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _risk_assessment_loop(self):
        """Background risk assessment loop"""
        while self.status.value != "SHUTDOWN":
            try:
                for customer_id, risks in self.risk_assessments.items():
                    # Re-evaluate risks based on current progress
                    for risk in risks:
                        if risk.status == "IDENTIFIED":
                            # Check if mitigation is needed
                            if risk.risk_score > 0.7:
                                self.logger.warning(f"High risk detected for {customer_id}: {risk.description}")
                                # TODO: Trigger mitigation actions
                
                await asyncio.sleep(7200)  # Check every 2 hours
                
            except Exception as e:
                self.logger.error(f"Error in risk assessment loop: {str(e)}")
                await asyncio.sleep(7200)


# Export main component
__all__ = ["ImplementationProjectManagerAgent", "ProjectMilestone", "RiskAssessment", "ProjectMetrics"]