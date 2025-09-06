"""
Competency Assessment Framework
===============================

Comprehensive assessment system for evaluating agent competencies across
all training phases and determining certification levels.
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from ...core.base_agent import BaseAgent


class CompetencyLevel(Enum):
    """Competency achievement levels"""
    NOVICE = "novice"           # 0-40%
    DEVELOPING = "developing"    # 40-65%  
    PROFICIENT = "proficient"   # 65-85%
    ADVANCED = "advanced"       # 85-95%
    EXPERT = "expert"           # 95%+


class CertificationLevel(Enum):
    """Agent certification levels"""
    NONE = "none"
    FOUNDATION = "foundation"
    SPECIALIZED = "specialized"
    INTEGRATED = "integrated" 
    PRODUCTION_READY = "production_ready"


@dataclass
class CompetencyScore:
    """Individual competency scoring"""
    competency_name: str
    score: float
    level: CompetencyLevel
    evidence_count: int
    last_assessment: datetime
    improvement_trend: float  # Positive = improving
    
    
@dataclass
class AssessmentResult:
    """Complete assessment result for an agent"""
    agent_id: str
    agent_type: str
    assessment_date: datetime
    overall_score: float
    certification_level: CertificationLevel
    competency_scores: Dict[str, CompetencyScore]
    phase_scores: Dict[str, float]
    recommendations: List[str]
    readiness_indicators: Dict[str, bool]
    next_assessment_due: datetime


class CompetencyAssessor:
    """
    Comprehensive competency assessment system
    
    Evaluates agent performance across:
    - Core competencies (communication, task management, error handling)
    - Role-specific competencies (customer service, project management, etc.)
    - Integration competencies (collaboration, crisis management)
    - Advanced competencies (innovation, adaptability)
    """
    
    def __init__(self):
        self.assessor_id = str(uuid.uuid4())
        self.logger = logging.getLogger(f"eFab.Assessment.{self.assessor_id}")
        
        # Assessment framework
        self.competency_definitions: Dict[str, Dict[str, Any]] = {}
        self.assessment_criteria: Dict[str, Dict[str, Any]] = {}
        self.certification_requirements: Dict[str, Dict[str, Any]] = {}
        
        # Assessment data
        self.assessment_history: Dict[str, List[AssessmentResult]] = {}
        self.performance_trends: Dict[str, Dict[str, List[float]]] = {}
        
        # Initialize assessment framework
        self._initialize_competency_definitions()
        self._initialize_assessment_criteria()
        self._initialize_certification_requirements()
        
        self.logger.info("Competency Assessor initialized")
    
    # =============================================================================
    # Core Assessment Methods
    # =============================================================================
    
    async def conduct_comprehensive_assessment(self, agent_id: str, 
                                             training_history: List[Dict[str, Any]],
                                             agent_metrics: Dict[str, Any]) -> AssessmentResult:
        """Conduct comprehensive competency assessment"""
        self.logger.info(f"Conducting comprehensive assessment for agent {agent_id}")
        
        # Get agent information
        agent_type = agent_metrics.get("agent_type", "unknown")
        
        # Assess core competencies
        core_scores = await self._assess_core_competencies(
            agent_id, training_history, agent_metrics
        )
        
        # Assess role-specific competencies
        role_scores = await self._assess_role_competencies(
            agent_id, agent_type, training_history, agent_metrics
        )
        
        # Assess integration competencies
        integration_scores = await self._assess_integration_competencies(
            agent_id, training_history, agent_metrics
        )
        
        # Combine all competency scores
        all_competency_scores = {**core_scores, **role_scores, **integration_scores}
        
        # Calculate phase scores
        phase_scores = await self._calculate_phase_scores(
            agent_id, training_history, all_competency_scores
        )
        
        # Calculate overall score
        overall_score = await self._calculate_overall_score(
            all_competency_scores, phase_scores
        )
        
        # Determine certification level
        certification_level = await self._determine_certification_level(
            overall_score, all_competency_scores, phase_scores
        )
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            agent_id, agent_type, all_competency_scores, certification_level
        )
        
        # Assess readiness indicators
        readiness_indicators = await self._assess_readiness_indicators(
            all_competency_scores, phase_scores, agent_metrics
        )
        
        # Create assessment result
        assessment_result = AssessmentResult(
            agent_id=agent_id,
            agent_type=agent_type,
            assessment_date=datetime.now(),
            overall_score=overall_score,
            certification_level=certification_level,
            competency_scores=all_competency_scores,
            phase_scores=phase_scores,
            recommendations=recommendations,
            readiness_indicators=readiness_indicators,
            next_assessment_due=datetime.now() + timedelta(weeks=2)
        )
        
        # Store assessment result
        if agent_id not in self.assessment_history:
            self.assessment_history[agent_id] = []
        self.assessment_history[agent_id].append(assessment_result)
        
        # Update performance trends
        await self._update_performance_trends(agent_id, all_competency_scores)
        
        self.logger.info(f"Assessment completed for {agent_id}: "
                        f"Overall score {overall_score:.1f}, "
                        f"Certification {certification_level.value}")
        
        return assessment_result
    
    # =============================================================================
    # Core Competency Assessment
    # =============================================================================
    
    async def _assess_core_competencies(self, agent_id: str,
                                       training_history: List[Dict[str, Any]],
                                       agent_metrics: Dict[str, Any]) -> Dict[str, CompetencyScore]:
        """Assess fundamental agent competencies"""
        core_competencies = [
            "communication_protocol",
            "task_coordination", 
            "knowledge_management",
            "error_handling",
            "performance_optimization"
        ]
        
        competency_scores = {}
        
        for competency in core_competencies:
            score = await self._evaluate_competency(
                competency, agent_id, training_history, agent_metrics
            )
            competency_scores[competency] = score
        
        return competency_scores
    
    async def _assess_role_competencies(self, agent_id: str, agent_type: str,
                                       training_history: List[Dict[str, Any]],
                                       agent_metrics: Dict[str, Any]) -> Dict[str, CompetencyScore]:
        """Assess role-specific competencies"""
        role_competency_map = {
            "lead_agent": [
                "customer_interaction",
                "implementation_orchestration", 
                "escalation_handling",
                "satisfaction_management"
            ],
            "customer_manager_agent": [
                "document_processing",
                "workflow_automation",
                "agent_coordination",
                "resource_optimization"
            ],
            "project_manager_agent": [
                "project_planning",
                "timeline_management",
                "risk_management", 
                "stakeholder_communication"
            ],
            "data_migration_agent": [
                "data_extraction",
                "data_transformation",
                "data_validation",
                "migration_optimization"
            ]
        }
        
        role_competencies = role_competency_map.get(agent_type, [])
        competency_scores = {}
        
        for competency in role_competencies:
            score = await self._evaluate_competency(
                competency, agent_id, training_history, agent_metrics
            )
            competency_scores[competency] = score
        
        return competency_scores
    
    async def _assess_integration_competencies(self, agent_id: str,
                                             training_history: List[Dict[str, Any]],
                                             agent_metrics: Dict[str, Any]) -> Dict[str, CompetencyScore]:
        """Assess integration and collaboration competencies"""
        integration_competencies = [
            "multi_agent_collaboration",
            "crisis_management",
            "adaptability",
            "continuous_learning",
            "innovation"
        ]
        
        competency_scores = {}
        
        for competency in integration_competencies:
            score = await self._evaluate_competency(
                competency, agent_id, training_history, agent_metrics
            )
            competency_scores[competency] = score
        
        return competency_scores
    
    # =============================================================================
    # Competency Evaluation Engine
    # =============================================================================
    
    async def _evaluate_competency(self, competency_name: str, agent_id: str,
                                  training_history: List[Dict[str, Any]],
                                  agent_metrics: Dict[str, Any]) -> CompetencyScore:
        """Evaluate individual competency using multiple evidence sources"""
        
        # Get competency definition and criteria
        competency_def = self.competency_definitions.get(competency_name, {})
        assessment_criteria = self.assessment_criteria.get(competency_name, {})
        
        # Collect evidence from different sources
        evidence_sources = {
            "training_performance": await self._evaluate_training_performance(
                competency_name, training_history
            ),
            "operational_metrics": await self._evaluate_operational_metrics(
                competency_name, agent_metrics
            ),
            "behavioral_indicators": await self._evaluate_behavioral_indicators(
                competency_name, training_history, agent_metrics
            ),
            "peer_assessment": await self._evaluate_peer_performance(
                competency_name, agent_id, training_history
            )
        }
        
        # Calculate weighted score
        weighted_score = await self._calculate_weighted_competency_score(
            evidence_sources, assessment_criteria
        )
        
        # Determine competency level
        competency_level = self._determine_competency_level(weighted_score)
        
        # Calculate improvement trend
        improvement_trend = await self._calculate_improvement_trend(
            agent_id, competency_name, weighted_score
        )
        
        # Count evidence items
        evidence_count = sum(
            len(evidence) if isinstance(evidence, list) else 1
            for evidence in evidence_sources.values()
            if evidence is not None
        )
        
        return CompetencyScore(
            competency_name=competency_name,
            score=weighted_score,
            level=competency_level,
            evidence_count=evidence_count,
            last_assessment=datetime.now(),
            improvement_trend=improvement_trend
        )
    
    async def _evaluate_training_performance(self, competency_name: str,
                                           training_history: List[Dict[str, Any]]) -> float:
        """Evaluate competency based on training scenario performance"""
        relevant_sessions = []
        
        # Find training sessions relevant to this competency
        competency_keywords = {
            "communication_protocol": ["communication", "message", "protocol"],
            "task_coordination": ["task", "coordination", "assignment"],
            "customer_interaction": ["customer", "interaction", "satisfaction"],
            "multi_agent_collaboration": ["collaboration", "multi_agent", "coordination"]
        }
        
        keywords = competency_keywords.get(competency_name, [competency_name])
        
        for session in training_history:
            scenario_name = session.get("scenario_name", "").lower()
            if any(keyword in scenario_name for keyword in keywords):
                relevant_sessions.append(session)
        
        if not relevant_sessions:
            return 0.7  # Default score if no relevant training data
        
        # Calculate average performance score
        scores = [session.get("score", 0.0) for session in relevant_sessions]
        return sum(scores) / len(scores) / 100.0  # Convert to 0-1 scale
    
    async def _evaluate_operational_metrics(self, competency_name: str,
                                          agent_metrics: Dict[str, Any]) -> float:
        """Evaluate competency based on operational performance metrics"""
        metric_mappings = {
            "communication_protocol": {
                "message_success_rate": 0.4,
                "average_response_time": 0.3,
                "protocol_violations": 0.3
            },
            "task_coordination": {
                "task_completion_rate": 0.5,
                "escalation_accuracy": 0.3,
                "dependency_management": 0.2
            },
            "error_handling": {
                "error_recovery_rate": 0.6,
                "graceful_degradation": 0.4
            },
            "customer_interaction": {
                "customer_satisfaction": 0.7,
                "issue_resolution_rate": 0.3
            }
        }
        
        mapping = metric_mappings.get(competency_name, {})
        if not mapping:
            return 0.7  # Default score
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in mapping.items():
            metric_value = agent_metrics.get(metric_name, 0.0)
            
            # Normalize different metric types
            if "rate" in metric_name or "accuracy" in metric_name:
                normalized_value = metric_value / 100.0  # Assume percentage
            elif "time" in metric_name:
                # Lower is better for time metrics - convert to score
                normalized_value = max(0.0, 1.0 - (metric_value / 1000.0))  # Assume ms
            elif "violations" in metric_name:
                # Lower is better for violations
                normalized_value = max(0.0, 1.0 - metric_value / 10.0)
            else:
                normalized_value = min(metric_value, 1.0)
            
            weighted_score += normalized_value * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.7
    
    async def _evaluate_behavioral_indicators(self, competency_name: str,
                                            training_history: List[Dict[str, Any]],
                                            agent_metrics: Dict[str, Any]) -> float:
        """Evaluate competency based on behavioral indicators"""
        behavior_indicators = {
            "adaptability": ["scenario_variety", "performance_consistency"],
            "continuous_learning": ["improvement_rate", "feedback_incorporation"],
            "innovation": ["creative_solutions", "process_improvements"],
            "crisis_management": ["stress_performance", "recovery_speed"]
        }
        
        indicators = behavior_indicators.get(competency_name, [])
        if not indicators:
            return 0.7  # Default score
        
        # Analyze behavioral patterns
        behavior_scores = []
        
        for indicator in indicators:
            if indicator == "scenario_variety":
                # Check performance across different scenario types
                scenario_types = set(s.get("scenario_type", "") for s in training_history)
                variety_score = min(len(scenario_types) / 10.0, 1.0)
                behavior_scores.append(variety_score)
                
            elif indicator == "performance_consistency":
                # Check consistency of performance scores
                scores = [s.get("score", 0.0) for s in training_history if s.get("score")]
                if scores:
                    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
                    consistency_score = max(0.0, 1.0 - std_dev / 100.0)
                    behavior_scores.append(consistency_score)
                    
            elif indicator == "improvement_rate":
                # Check if performance improves over time
                scores = [s.get("score", 0.0) for s in training_history if s.get("score")]
                if len(scores) >= 3:
                    early_avg = sum(scores[:len(scores)//3]) / (len(scores)//3)
                    late_avg = sum(scores[-len(scores)//3:]) / (len(scores)//3)
                    improvement = (late_avg - early_avg) / 100.0
                    improvement_score = min(max(improvement + 0.5, 0.0), 1.0)
                    behavior_scores.append(improvement_score)
        
        return sum(behavior_scores) / len(behavior_scores) if behavior_scores else 0.7
    
    async def _evaluate_peer_performance(self, competency_name: str, agent_id: str,
                                       training_history: List[Dict[str, Any]]) -> float:
        """Evaluate competency based on peer collaboration performance"""
        collaboration_sessions = [
            s for s in training_history 
            if "collaboration" in s.get("scenario_name", "").lower()
        ]
        
        if not collaboration_sessions:
            return 0.7  # Default score
        
        # Evaluate collaboration effectiveness
        collaboration_scores = []
        
        for session in collaboration_sessions:
            session_results = session.get("results", {})
            
            # Look for collaboration-specific metrics
            coordination_success = session_results.get("coordination_success", True)
            role_performance = session_results.get("role_performance", 0.8)
            peer_rating = session_results.get("peer_rating", 0.75)
            
            session_score = (
                (1.0 if coordination_success else 0.0) * 0.4 +
                role_performance * 0.4 +
                peer_rating * 0.2
            )
            collaboration_scores.append(session_score)
        
        return sum(collaboration_scores) / len(collaboration_scores)
    
    # =============================================================================
    # Scoring and Evaluation
    # =============================================================================
    
    async def _calculate_weighted_competency_score(self, evidence_sources: Dict[str, float],
                                                  assessment_criteria: Dict[str, Any]) -> float:
        """Calculate weighted competency score from evidence sources"""
        default_weights = {
            "training_performance": 0.4,
            "operational_metrics": 0.3, 
            "behavioral_indicators": 0.2,
            "peer_assessment": 0.1
        }
        
        weights = assessment_criteria.get("evidence_weights", default_weights)
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for source, score in evidence_sources.items():
            if score is not None:
                weight = weights.get(source, 0.1)
                weighted_score += score * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_competency_level(self, score: float) -> CompetencyLevel:
        """Determine competency level based on score"""
        if score >= 0.95:
            return CompetencyLevel.EXPERT
        elif score >= 0.85:
            return CompetencyLevel.ADVANCED
        elif score >= 0.65:
            return CompetencyLevel.PROFICIENT
        elif score >= 0.40:
            return CompetencyLevel.DEVELOPING
        else:
            return CompetencyLevel.NOVICE
    
    async def _calculate_improvement_trend(self, agent_id: str, competency_name: str,
                                         current_score: float) -> float:
        """Calculate improvement trend for a competency"""
        if agent_id not in self.performance_trends:
            self.performance_trends[agent_id] = {}
        
        if competency_name not in self.performance_trends[agent_id]:
            self.performance_trends[agent_id][competency_name] = []
        
        # Add current score to trend
        trend_data = self.performance_trends[agent_id][competency_name]
        trend_data.append(current_score)
        
        # Keep only recent data points (last 10)
        if len(trend_data) > 10:
            trend_data = trend_data[-10:]
            self.performance_trends[agent_id][competency_name] = trend_data
        
        # Calculate trend
        if len(trend_data) < 3:
            return 0.0  # Not enough data for trend
        
        # Simple linear trend calculation
        x_values = list(range(len(trend_data)))
        y_values = trend_data
        
        # Calculate slope using least squares
        n = len(trend_data)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope * 10  # Scale for readability
    
    # =============================================================================
    # Phase and Overall Scoring
    # =============================================================================
    
    async def _calculate_phase_scores(self, agent_id: str,
                                    training_history: List[Dict[str, Any]],
                                    competency_scores: Dict[str, CompetencyScore]) -> Dict[str, float]:
        """Calculate scores for each training phase"""
        phase_competency_mapping = {
            "foundation": [
                "communication_protocol",
                "task_coordination", 
                "knowledge_management",
                "error_handling"
            ],
            "specialization": [
                "customer_interaction",
                "document_processing",
                "project_planning",
                "workflow_automation"
            ],
            "integration": [
                "multi_agent_collaboration",
                "crisis_management",
                "resource_optimization"
            ],
            "advanced": [
                "adaptability",
                "continuous_learning",
                "innovation",
                "performance_optimization"
            ]
        }
        
        phase_scores = {}
        
        for phase, competencies in phase_competency_mapping.items():
            relevant_scores = []
            
            for competency in competencies:
                if competency in competency_scores:
                    relevant_scores.append(competency_scores[competency].score)
            
            if relevant_scores:
                phase_scores[phase] = sum(relevant_scores) / len(relevant_scores)
            else:
                phase_scores[phase] = 0.0
        
        return phase_scores
    
    async def _calculate_overall_score(self, competency_scores: Dict[str, CompetencyScore],
                                     phase_scores: Dict[str, float]) -> float:
        """Calculate overall competency score"""
        # Weight different score sources
        weights = {
            "competency_average": 0.6,
            "phase_average": 0.4
        }
        
        # Calculate competency average
        competency_values = [score.score for score in competency_scores.values()]
        competency_avg = sum(competency_values) / len(competency_values) if competency_values else 0.0
        
        # Calculate phase average
        phase_values = list(phase_scores.values())
        phase_avg = sum(phase_values) / len(phase_values) if phase_values else 0.0
        
        # Calculate weighted overall score
        overall_score = (
            competency_avg * weights["competency_average"] +
            phase_avg * weights["phase_average"]
        )
        
        return overall_score
    
    # =============================================================================
    # Certification and Recommendations
    # =============================================================================
    
    async def _determine_certification_level(self, overall_score: float,
                                           competency_scores: Dict[str, CompetencyScore],
                                           phase_scores: Dict[str, float]) -> CertificationLevel:
        """Determine agent certification level"""
        certification_criteria = self.certification_requirements
        
        # Check production ready criteria
        prod_criteria = certification_criteria.get("production_ready", {})
        if (overall_score >= prod_criteria.get("min_overall_score", 0.95) and
            all(score >= prod_criteria.get("min_competency_score", 0.90) 
                for score in [cs.score for cs in competency_scores.values()]) and
            all(score >= prod_criteria.get("min_phase_score", 0.90) 
                for score in phase_scores.values())):
            return CertificationLevel.PRODUCTION_READY
        
        # Check integrated criteria
        int_criteria = certification_criteria.get("integrated", {})
        if (overall_score >= int_criteria.get("min_overall_score", 0.85) and
            phase_scores.get("integration", 0.0) >= int_criteria.get("min_integration_score", 0.80)):
            return CertificationLevel.INTEGRATED
        
        # Check specialized criteria
        spec_criteria = certification_criteria.get("specialized", {})
        if (overall_score >= spec_criteria.get("min_overall_score", 0.75) and
            phase_scores.get("specialization", 0.0) >= spec_criteria.get("min_specialization_score", 0.70)):
            return CertificationLevel.SPECIALIZED
        
        # Check foundation criteria
        found_criteria = certification_criteria.get("foundation", {})
        if (overall_score >= found_criteria.get("min_overall_score", 0.65) and
            phase_scores.get("foundation", 0.0) >= found_criteria.get("min_foundation_score", 0.60)):
            return CertificationLevel.FOUNDATION
        
        return CertificationLevel.NONE
    
    async def _generate_recommendations(self, agent_id: str, agent_type: str,
                                      competency_scores: Dict[str, CompetencyScore],
                                      certification_level: CertificationLevel) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Identify weak competencies
        weak_competencies = [
            name for name, score in competency_scores.items()
            if score.level in [CompetencyLevel.NOVICE, CompetencyLevel.DEVELOPING]
        ]
        
        if weak_competencies:
            recommendations.append(
                f"Focus on improving weak competencies: {', '.join(weak_competencies)}"
            )
        
        # Role-specific recommendations
        if agent_type == "lead_agent":
            customer_score = competency_scores.get("customer_interaction", CompetencyScore("", 0.0, CompetencyLevel.NOVICE, 0, datetime.now(), 0.0))
            if customer_score.level != CompetencyLevel.EXPERT:
                recommendations.append("Continue developing customer interaction skills through practice scenarios")
        
        # Certification progression recommendations
        if certification_level == CertificationLevel.NONE:
            recommendations.append("Focus on foundation competencies to achieve basic certification")
        elif certification_level == CertificationLevel.FOUNDATION:
            recommendations.append("Develop role-specific skills to achieve specialized certification")
        elif certification_level == CertificationLevel.SPECIALIZED:
            recommendations.append("Practice multi-agent scenarios to achieve integrated certification")
        elif certification_level == CertificationLevel.INTEGRATED:
            recommendations.append("Demonstrate consistent excellence to achieve production readiness")
        
        # Trend-based recommendations
        declining_competencies = [
            name for name, score in competency_scores.items()
            if score.improvement_trend < -0.1
        ]
        
        if declining_competencies:
            recommendations.append(
                f"Address declining performance in: {', '.join(declining_competencies)}"
            )
        
        return recommendations
    
    async def _assess_readiness_indicators(self, competency_scores: Dict[str, CompetencyScore],
                                         phase_scores: Dict[str, float],
                                         agent_metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Assess production readiness indicators"""
        indicators = {
            "technical_competency": all(
                score.level in [CompetencyLevel.ADVANCED, CompetencyLevel.EXPERT]
                for name, score in competency_scores.items()
                if "technical" in name or "protocol" in name
            ),
            "customer_readiness": competency_scores.get("customer_interaction", CompetencyScore("", 0.0, CompetencyLevel.NOVICE, 0, datetime.now(), 0.0)).level in [CompetencyLevel.ADVANCED, CompetencyLevel.EXPERT],
            "collaboration_readiness": competency_scores.get("multi_agent_collaboration", CompetencyScore("", 0.0, CompetencyLevel.NOVICE, 0, datetime.now(), 0.0)).level in [CompetencyLevel.PROFICIENT, CompetencyLevel.ADVANCED, CompetencyLevel.EXPERT],
            "crisis_handling": competency_scores.get("crisis_management", CompetencyScore("", 0.0, CompetencyLevel.NOVICE, 0, datetime.now(), 0.0)).level in [CompetencyLevel.PROFICIENT, CompetencyLevel.ADVANCED, CompetencyLevel.EXPERT],
            "performance_stability": agent_metrics.get("error_rate", 100.0) < 5.0,
            "response_time": agent_metrics.get("average_response_time", 1000.0) < 100.0
        }
        
        return indicators
    
    # =============================================================================
    # Initialization Methods
    # =============================================================================
    
    def _initialize_competency_definitions(self):
        """Initialize competency definitions and requirements"""
        self.competency_definitions = {
            "communication_protocol": {
                "name": "Communication Protocol Mastery",
                "description": "Ability to handle inter-agent communication effectively",
                "key_indicators": [
                    "Message success rate > 99%",
                    "Response time < 50ms",
                    "Zero protocol violations"
                ]
            },
            "task_coordination": {
                "name": "Task Coordination",
                "description": "Effective task management and coordination",
                "key_indicators": [
                    "Task completion rate > 95%",
                    "Proper escalation handling",
                    "Dependency management"
                ]
            },
            "customer_interaction": {
                "name": "Customer Interaction Excellence",
                "description": "Superior customer service and communication",
                "key_indicators": [
                    "Customer satisfaction > 4.5/5.0",
                    "First-contact resolution > 85%",
                    "Appropriate escalation decisions"
                ]
            },
            "multi_agent_collaboration": {
                "name": "Multi-Agent Collaboration",
                "description": "Effective collaboration with other agents",
                "key_indicators": [
                    "Coordination success rate > 90%",
                    "Effective role execution",
                    "Knowledge sharing"
                ]
            },
            "crisis_management": {
                "name": "Crisis Management",
                "description": "Effective handling of crisis situations",
                "key_indicators": [
                    "Response time < 5 minutes for critical issues",
                    "Resolution success rate > 80%",
                    "Proper stakeholder communication"
                ]
            }
        }
    
    def _initialize_assessment_criteria(self):
        """Initialize assessment criteria for each competency"""
        self.assessment_criteria = {
            "communication_protocol": {
                "evidence_weights": {
                    "training_performance": 0.3,
                    "operational_metrics": 0.5,
                    "behavioral_indicators": 0.1,
                    "peer_assessment": 0.1
                },
                "minimum_evidence": 5
            },
            "customer_interaction": {
                "evidence_weights": {
                    "training_performance": 0.4,
                    "operational_metrics": 0.4,
                    "behavioral_indicators": 0.2,
                    "peer_assessment": 0.0
                },
                "minimum_evidence": 3
            },
            "multi_agent_collaboration": {
                "evidence_weights": {
                    "training_performance": 0.3,
                    "operational_metrics": 0.2,
                    "behavioral_indicators": 0.2,
                    "peer_assessment": 0.3
                },
                "minimum_evidence": 4
            }
        }
    
    def _initialize_certification_requirements(self):
        """Initialize certification level requirements"""
        self.certification_requirements = {
            "foundation": {
                "min_overall_score": 0.65,
                "min_foundation_score": 0.60,
                "required_competencies": [
                    "communication_protocol",
                    "task_coordination"
                ]
            },
            "specialized": {
                "min_overall_score": 0.75,
                "min_specialization_score": 0.70,
                "required_competencies": [
                    "communication_protocol",
                    "task_coordination",
                    "role_specific_primary"
                ]
            },
            "integrated": {
                "min_overall_score": 0.85,
                "min_integration_score": 0.80,
                "required_competencies": [
                    "multi_agent_collaboration",
                    "crisis_management"
                ]
            },
            "production_ready": {
                "min_overall_score": 0.95,
                "min_competency_score": 0.90,
                "min_phase_score": 0.90,
                "required_indicators": [
                    "technical_competency",
                    "customer_readiness", 
                    "performance_stability"
                ]
            }
        }
    
    # =============================================================================
    # Reporting and Analytics
    # =============================================================================
    
    def generate_assessment_report(self, assessment_result: AssessmentResult) -> Dict[str, Any]:
        """Generate comprehensive assessment report"""
        return {
            "agent_summary": {
                "agent_id": assessment_result.agent_id,
                "agent_type": assessment_result.agent_type,
                "assessment_date": assessment_result.assessment_date.isoformat(),
                "overall_score": assessment_result.overall_score,
                "certification_level": assessment_result.certification_level.value
            },
            "competency_breakdown": {
                name: {
                    "score": score.score,
                    "level": score.level.value,
                    "evidence_count": score.evidence_count,
                    "trend": score.improvement_trend
                }
                for name, score in assessment_result.competency_scores.items()
            },
            "phase_performance": assessment_result.phase_scores,
            "readiness_indicators": assessment_result.readiness_indicators,
            "recommendations": assessment_result.recommendations,
            "next_steps": {
                "next_assessment_due": assessment_result.next_assessment_due.isoformat(),
                "priority_improvements": assessment_result.recommendations[:3],
                "certification_path": self._determine_certification_path(assessment_result)
            }
        }
    
    def _determine_certification_path(self, assessment_result: AssessmentResult) -> List[str]:
        """Determine path to next certification level"""
        current_level = assessment_result.certification_level
        
        paths = {
            CertificationLevel.NONE: [
                "Achieve 65% overall score",
                "Master foundation competencies", 
                "Complete foundation training phase"
            ],
            CertificationLevel.FOUNDATION: [
                "Achieve 75% overall score",
                "Develop role-specific competencies",
                "Complete specialization training"
            ],
            CertificationLevel.SPECIALIZED: [
                "Achieve 85% overall score",
                "Master collaboration competencies",
                "Complete integration training"
            ],
            CertificationLevel.INTEGRATED: [
                "Achieve 95% overall score",
                "Excel in all competency areas",
                "Demonstrate production readiness"
            ],
            CertificationLevel.PRODUCTION_READY: [
                "Maintain excellence",
                "Mentor other agents",
                "Drive continuous improvement"
            ]
        }
        
        return paths.get(current_level, [])
    
    def get_system_assessment_summary(self) -> Dict[str, Any]:
        """Get summary of all agent assessments"""
        all_assessments = []
        for agent_history in self.assessment_history.values():
            if agent_history:
                all_assessments.append(agent_history[-1])  # Most recent assessment
        
        if not all_assessments:
            return {"error": "No assessments available"}
        
        # Calculate system-wide statistics
        overall_scores = [a.overall_score for a in all_assessments]
        certification_counts = {}
        
        for assessment in all_assessments:
            cert_level = assessment.certification_level.value
            certification_counts[cert_level] = certification_counts.get(cert_level, 0) + 1
        
        return {
            "total_agents_assessed": len(all_assessments),
            "average_overall_score": sum(overall_scores) / len(overall_scores),
            "score_distribution": {
                "min": min(overall_scores),
                "max": max(overall_scores),
                "median": statistics.median(overall_scores),
                "std_dev": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0
            },
            "certification_distribution": certification_counts,
            "production_ready_count": certification_counts.get("production_ready", 0),
            "production_ready_percentage": certification_counts.get("production_ready", 0) / len(all_assessments) * 100
        }