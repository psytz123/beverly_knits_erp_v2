"""
eFab AI Agent Training System
=============================

Main entry point for the comprehensive 12-week AI agent training program.
Integrates all training components and provides a unified interface for
training execution and management.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

# Import training components
from .core.agent_orchestrator import AgentOrchestrator
from .training.training_orchestrator import TrainingOrchestrator, TrainingPhase, TrainingWeek
from .training.assessment.competency_assessor import CompetencyAssessor, CertificationLevel
from .training.continuous_learning import ContinuousLearningSystem
from .specialized.lead_agent import LeadAgent
from .specialized.customer_manager_agent import CustomerManagerAgent


class TrainingSystemManager:
    """
    Main training system manager
    
    Coordinates all aspects of the comprehensive AI agent training program:
    - Agent lifecycle management
    - Training program execution
    - Assessment and certification
    - Continuous learning
    - Performance monitoring and reporting
    """
    
    def __init__(self):
        self.manager_id = "training_system_manager"
        self.logger = logging.getLogger("eFab.Training.Manager")
        
        # Core components
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.training_orchestrator: Optional[TrainingOrchestrator] = None
        self.competency_assessor: Optional[CompetencyAssessor] = None
        self.continuous_learning: Optional[ContinuousLearningSystem] = None
        
        # System state
        self.is_initialized = False
        self.training_active = False
        self.agents: Dict[str, Any] = {}
        
        # Configuration
        self.config = {
            "training_duration_weeks": 12,
            "assessment_frequency_days": 14,
            "certification_threshold": 0.85,
            "production_readiness_threshold": 0.95,
            "max_concurrent_agents": 50,
            "training_data_path": "./training_data",
            "reports_output_path": "./training_reports"
        }
        
        self.logger.info("Training System Manager initialized")
    
    # =============================================================================
    # System Initialization
    # =============================================================================
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the complete training system"""
        try:
            if config:
                self.config.update(config)
            
            self.logger.info("Initializing eFab AI Agent Training System")
            
            # Initialize core orchestrator
            self.orchestrator = AgentOrchestrator("training_orchestrator")
            await self.orchestrator.start()
            
            # Initialize training orchestrator
            self.training_orchestrator = TrainingOrchestrator(self.orchestrator)
            
            # Initialize competency assessor
            self.competency_assessor = CompetencyAssessor()
            
            # Initialize continuous learning system
            self.continuous_learning = ContinuousLearningSystem(self.orchestrator)
            
            # Create output directories
            await self._create_output_directories()
            
            # Set up logging
            await self._setup_training_logging()
            
            self.is_initialized = True
            self.logger.info("Training system initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize training system: {str(e)}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the training system gracefully"""
        try:
            self.logger.info("Shutting down training system")
            
            # Stop training if active
            if self.training_active:
                await self.stop_training_program()
            
            # Stop continuous learning
            if self.continuous_learning and self.continuous_learning.is_active:
                await self.continuous_learning.stop_continuous_learning()
            
            # Shutdown orchestrator
            if self.orchestrator:
                await self.orchestrator.stop()
            
            self.is_initialized = False
            self.logger.info("Training system shutdown complete")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            return False
    
    # =============================================================================
    # Agent Management
    # =============================================================================
    
    async def create_training_agents(self, agent_configs: List[Dict[str, Any]]) -> List[str]:
        """Create and register agents for training"""
        if not self.is_initialized:
            raise RuntimeError("Training system not initialized")
        
        created_agents = []
        
        for config in agent_configs:
            agent_type = config.get("type", "lead_agent")
            agent_id = config.get("id", f"{agent_type}_{len(created_agents)+1:03d}")
            name = config.get("name", f"{agent_type.title()} {len(created_agents)+1}")
            
            # Create agent based on type
            if agent_type == "lead_agent":
                agent = LeadAgent(agent_id, name)
            elif agent_type == "customer_manager_agent":
                agent = CustomerManagerAgent(agent_id, name)
            else:
                # Create base agent for other types
                from .core.base_agent import BaseAgent
                agent = BaseAgent(agent_id, agent_type, name)
            
            # Register with orchestrator
            success = await self.orchestrator.register_agent(agent)
            
            if success:
                self.agents[agent_id] = {
                    "agent": agent,
                    "config": config,
                    "created_at": datetime.now(),
                    "training_status": "registered"
                }
                created_agents.append(agent_id)
                self.logger.info(f"Created and registered agent: {name} ({agent_type})")
            else:
                self.logger.error(f"Failed to register agent: {agent_id}")
        
        self.logger.info(f"Created {len(created_agents)} training agents")
        return created_agents
    
    async def create_default_training_cohort(self) -> List[str]:
        """Create a default cohort of agents for training"""
        default_configs = [
            # Lead Agents
            {"type": "lead_agent", "id": "lead_001", "name": "Lead Agent Alpha"},
            {"type": "lead_agent", "id": "lead_002", "name": "Lead Agent Beta"},
            
            # Customer Manager Agents
            {"type": "customer_manager_agent", "id": "cm_001", "name": "Customer Manager Alpha"},
            {"type": "customer_manager_agent", "id": "cm_002", "name": "Customer Manager Beta"},
            
            # Implementation Agents (simulated)
            {"type": "project_manager_agent", "id": "pm_001", "name": "Project Manager Alpha"},
            {"type": "data_migration_agent", "id": "dm_001", "name": "Data Migration Alpha"},
            {"type": "configuration_agent", "id": "config_001", "name": "Configuration Alpha"},
            
            # Industry Specialists (simulated)
            {"type": "manufacturing_specialist", "id": "mfg_001", "name": "Manufacturing Specialist"},
            {"type": "textile_specialist", "id": "textile_001", "name": "Textile Specialist"}
        ]
        
        return await self.create_training_agents(default_configs)
    
    # =============================================================================
    # Training Program Execution
    # =============================================================================
    
    async def start_training_program(self, agent_ids: Optional[List[str]] = None,
                                   start_phase: TrainingPhase = TrainingPhase.FOUNDATION) -> str:
        """Start the comprehensive 12-week training program"""
        if not self.is_initialized:
            raise RuntimeError("Training system not initialized")
        
        if self.training_active:
            raise RuntimeError("Training program already active")
        
        # Use all registered agents if none specified
        if agent_ids is None:
            agent_ids = list(self.agents.keys())
        
        # Validate agents exist
        valid_agents = [aid for aid in agent_ids if aid in self.agents]
        if not valid_agents:
            raise ValueError("No valid agents found for training")
        
        self.logger.info(f"Starting comprehensive training program for {len(valid_agents)} agents")
        
        # Start training program
        program_id = await self.training_orchestrator.start_training_program(valid_agents)
        
        # Start continuous learning
        await self.continuous_learning.start_continuous_learning(valid_agents)
        
        # Update agent statuses
        for agent_id in valid_agents:
            if agent_id in self.agents:
                self.agents[agent_id]["training_status"] = "in_training"
        
        self.training_active = True
        
        # Schedule periodic assessments
        asyncio.create_task(self._periodic_assessment_loop(valid_agents))
        
        self.logger.info(f"Training program started: {program_id}")
        return program_id
    
    async def stop_training_program(self) -> Dict[str, Any]:
        """Stop the training program and generate final reports"""
        if not self.training_active:
            return {"error": "No active training program"}
        
        self.logger.info("Stopping training program")
        
        # Stop training orchestrator
        training_result = await self.training_orchestrator.stop_training_program()
        
        # Stop continuous learning
        learning_result = await self.continuous_learning.stop_continuous_learning()
        
        # Generate final assessments
        final_assessments = {}
        for agent_id in self.agents:
            assessment = await self._conduct_final_assessment(agent_id)
            if assessment:
                final_assessments[agent_id] = assessment
        
        # Generate comprehensive report
        final_report = await self._generate_final_training_report(
            training_result, learning_result, final_assessments
        )
        
        # Update agent statuses
        for agent_id in self.agents:
            assessment = final_assessments.get(agent_id)
            if assessment:
                cert_level = assessment.certification_level.value
                self.agents[agent_id]["training_status"] = f"completed_{cert_level}"
            else:
                self.agents[agent_id]["training_status"] = "completed_unassessed"
        
        self.training_active = False
        
        # Save final report
        await self._save_training_report(final_report)
        
        self.logger.info("Training program stopped and final report generated")
        
        return final_report
    
    # =============================================================================
    # Assessment and Certification
    # =============================================================================
    
    async def conduct_agent_assessment(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Conduct comprehensive assessment for specific agent"""
        if agent_id not in self.agents:
            return None
        
        # Collect training history
        training_sessions = self.training_orchestrator.completed_sessions
        agent_sessions = [s for s in training_sessions if s.agent_id == agent_id]
        
        # Convert sessions to training history format
        training_history = []
        for session in agent_sessions:
            training_history.append({
                "session_id": session.session_id,
                "scenario_id": session.scenario_id,
                "scenario_name": f"Training Scenario {session.scenario_id}",
                "scenario_type": "training",
                "score": session.score,
                "results": session.results,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "status": session.status
            })
        
        # Get agent metrics
        agent_metrics = {
            "agent_type": self.agents[agent_id]["agent"].agent_type,
            "tasks_completed": self.agents[agent_id]["agent"].metrics.tasks_completed,
            "tasks_failed": self.agents[agent_id]["agent"].metrics.tasks_failed,
            "average_response_time": self.agents[agent_id]["agent"].metrics.average_response_time,
            "error_rate": self.agents[agent_id]["agent"].metrics.error_rate,
            "uptime": self.agents[agent_id]["agent"].metrics.uptime
        }
        
        # Conduct assessment
        assessment_result = await self.competency_assessor.conduct_comprehensive_assessment(
            agent_id, training_history, agent_metrics
        )
        
        # Generate assessment report
        report = self.competency_assessor.generate_assessment_report(assessment_result)
        
        self.logger.info(f"Assessment completed for {agent_id}: "
                        f"{assessment_result.certification_level.value} certification")
        
        return report
    
    async def _conduct_final_assessment(self, agent_id: str) -> Optional[Any]:
        """Conduct final comprehensive assessment"""
        try:
            assessment_report = await self.conduct_agent_assessment(agent_id)
            
            if assessment_report:
                # Extract assessment result for internal use
                agent_summary = assessment_report["agent_summary"]
                
                # Create simplified assessment result
                from .training.assessment.competency_assessor import AssessmentResult, CertificationLevel
                
                cert_level = CertificationLevel(agent_summary["certification_level"])
                
                return type('AssessmentResult', (), {
                    'agent_id': agent_id,
                    'certification_level': cert_level,
                    'overall_score': agent_summary["overall_score"],
                    'assessment_date': datetime.fromisoformat(agent_summary["assessment_date"])
                })()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Final assessment failed for {agent_id}: {str(e)}")
            return None
    
    # =============================================================================
    # Monitoring and Reporting
    # =============================================================================
    
    async def _periodic_assessment_loop(self, agent_ids: List[str]):
        """Periodic assessment of training progress"""
        assessment_interval = self.config.get("assessment_frequency_days", 14) * 24 * 3600  # Convert to seconds
        
        while self.training_active:
            try:
                await asyncio.sleep(assessment_interval)
                
                if not self.training_active:
                    break
                
                # Conduct assessments
                self.logger.info("Conducting periodic assessments")
                
                for agent_id in agent_ids:
                    if agent_id in self.agents:
                        assessment = await self.conduct_agent_assessment(agent_id)
                        
                        if assessment:
                            # Log assessment results
                            agent_summary = assessment["agent_summary"]
                            self.logger.info(f"Agent {agent_id} assessment: "
                                           f"Score {agent_summary['overall_score']:.2f}, "
                                           f"Level {agent_summary['certification_level']}")
                        
                        # Small delay between assessments
                        await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in periodic assessment loop: {str(e)}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get comprehensive training status"""
        status = {
            "system_initialized": self.is_initialized,
            "training_active": self.training_active,
            "total_agents": len(self.agents),
            "agents_by_status": {},
            "training_progress": {},
            "system_health": {}
        }
        
        # Count agents by training status
        for agent_data in self.agents.values():
            training_status = agent_data.get("training_status", "unknown")
            status["agents_by_status"][training_status] = \
                status["agents_by_status"].get(training_status, 0) + 1
        
        # Get training progress if active
        if self.training_active and self.training_orchestrator:
            training_status = self.training_orchestrator.get_training_status()
            status["training_progress"] = training_status
        
        # Get system health
        if self.orchestrator:
            system_status = self.orchestrator.get_system_status()
            status["system_health"] = system_status
        
        return status
    
    async def generate_progress_report(self) -> Dict[str, Any]:
        """Generate detailed progress report"""
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "training_status": await self.get_training_status(),
            "agent_assessments": {},
            "system_metrics": {},
            "recommendations": []
        }
        
        # Generate assessments for all agents
        for agent_id in self.agents:
            assessment = await self.conduct_agent_assessment(agent_id)
            if assessment:
                report["agent_assessments"][agent_id] = assessment
        
        # System-wide metrics
        if self.competency_assessor:
            system_summary = self.competency_assessor.get_system_assessment_summary()
            report["system_metrics"] = system_summary
        
        # Generate recommendations
        recommendations = await self._generate_system_recommendations(report)
        report["recommendations"] = recommendations
        
        return report
    
    async def _generate_system_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate system-wide recommendations based on training progress"""
        recommendations = []
        
        # Analyze agent assessments
        assessments = report.get("agent_assessments", {})
        
        if assessments:
            # Find agents needing additional training
            low_performers = [
                agent_id for agent_id, assessment in assessments.items()
                if assessment["agent_summary"]["overall_score"] < 0.7
            ]
            
            if low_performers:
                recommendations.append(
                    f"Consider additional training for {len(low_performers)} low-performing agents: "
                    f"{', '.join(low_performers[:3])}"
                )
            
            # Check certification distribution
            cert_levels = {}
            for assessment in assessments.values():
                level = assessment["agent_summary"]["certification_level"]
                cert_levels[level] = cert_levels.get(level, 0) + 1
            
            production_ready = cert_levels.get("production_ready", 0)
            total_agents = len(assessments)
            
            if production_ready / total_agents < 0.5:
                recommendations.append(
                    f"Only {production_ready}/{total_agents} agents are production-ready. "
                    f"Consider extending training program."
                )
        
        # System health recommendations
        training_status = report.get("training_status", {})
        system_health = training_status.get("system_health", {})
        
        if "system_health" in system_health and system_health["system_health"] < 90:
            recommendations.append("System health below 90% - investigate agent connectivity issues")
        
        return recommendations
    
    # =============================================================================
    # Reporting and Data Management
    # =============================================================================
    
    async def _generate_final_training_report(self, training_result: Dict[str, Any],
                                            learning_result: Dict[str, Any],
                                            assessments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final training report"""
        return {
            "final_training_report": {
                "report_generated_at": datetime.now().isoformat(),
                "training_program_duration": self.config["training_duration_weeks"],
                "total_agents_trained": len(self.agents)
            },
            "training_results": training_result,
            "continuous_learning_results": learning_result,
            "final_assessments": assessments,
            "certification_summary": self._summarize_certifications(assessments),
            "system_performance": await self.get_training_status(),
            "recommendations": await self._generate_deployment_recommendations(assessments),
            "next_steps": self._generate_next_steps(assessments)
        }
    
    def _summarize_certifications(self, assessments: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize certification levels achieved"""
        cert_summary = {
            "total_assessed": len(assessments),
            "certification_distribution": {},
            "production_ready_agents": [],
            "agents_needing_additional_training": []
        }
        
        for agent_id, assessment in assessments.items():
            cert_level = assessment.certification_level.value
            
            # Count by certification level
            cert_summary["certification_distribution"][cert_level] = \
                cert_summary["certification_distribution"].get(cert_level, 0) + 1
            
            # Track production ready agents
            if cert_level == "production_ready":
                cert_summary["production_ready_agents"].append(agent_id)
            elif assessment.overall_score < 0.7:
                cert_summary["agents_needing_additional_training"].append(agent_id)
        
        return cert_summary
    
    async def _generate_deployment_recommendations(self, assessments: Dict[str, Any]) -> List[str]:
        """Generate recommendations for production deployment"""
        recommendations = []
        
        production_ready = [
            agent_id for agent_id, assessment in assessments.items()
            if assessment.certification_level.value == "production_ready"
        ]
        
        if production_ready:
            recommendations.append(
                f"Deploy {len(production_ready)} production-ready agents to initial customer implementations"
            )
        
        integrated_agents = [
            agent_id for agent_id, assessment in assessments.items()
            if assessment.certification_level.value == "integrated"
        ]
        
        if integrated_agents:
            recommendations.append(
                f"Consider pilot deployment for {len(integrated_agents)} integrated-level agents with supervision"
            )
        
        # Additional training recommendations
        needs_training = [
            agent_id for agent_id, assessment in assessments.items()
            if assessment.overall_score < 0.8
        ]
        
        if needs_training:
            recommendations.append(
                f"Provide additional specialized training for {len(needs_training)} agents before deployment"
            )
        
        return recommendations
    
    def _generate_next_steps(self, assessments: Dict[str, Any]) -> List[str]:
        """Generate next steps for post-training phase"""
        return [
            "Conduct pilot customer implementations with production-ready agents",
            "Implement continuous learning system in production environment", 
            "Establish performance monitoring and feedback loops",
            "Schedule regular competency assessments (monthly)",
            "Plan advanced specialization training for high-performers",
            "Document lessons learned for future training cohorts"
        ]
    
    async def _save_training_report(self, report: Dict[str, Any]):
        """Save training report to file"""
        try:
            reports_path = Path(self.config["reports_output_path"])
            reports_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_path / f"final_training_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Training report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save training report: {str(e)}")
    
    # =============================================================================
    # Utility Methods
    # =============================================================================
    
    async def _create_output_directories(self):
        """Create necessary output directories"""
        directories = [
            self.config["reports_output_path"],
            self.config["training_data_path"]
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def _setup_training_logging(self):
        """Set up comprehensive logging for training system"""
        # Configure logging for training system
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    Path(self.config["reports_output_path"]) / "training_system.log"
                )
            ]
        )


# =============================================================================
# Command Line Interface
# =============================================================================

async def main():
    """Main entry point for training system"""
    parser = argparse.ArgumentParser(description="eFab AI Agent Training System")
    parser.add_argument("--action", choices=["start", "stop", "status", "assess", "report"],
                       default="start", help="Action to perform")
    parser.add_argument("--agents", type=int, default=9,
                       help="Number of agents to create for training")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--duration", type=int, default=12,
                       help="Training duration in weeks")
    
    args = parser.parse_args()
    
    # Create and initialize training system
    training_system = TrainingSystemManager()
    
    try:
        # Load configuration if provided
        config = {}
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        config["training_duration_weeks"] = args.duration
        
        # Initialize system
        success = await training_system.initialize(config)
        if not success:
            print("Failed to initialize training system")
            return 1
        
        if args.action == "start":
            # Create training cohort
            agent_ids = await training_system.create_default_training_cohort()
            print(f"Created {len(agent_ids)} training agents")
            
            # Start training program
            program_id = await training_system.start_training_program(agent_ids)
            print(f"Started training program: {program_id}")
            print(f"Training will run for {args.duration} weeks")
            print("Press Ctrl+C to stop training and generate final report")
            
            # Wait for interrupt or completion
            try:
                while True:
                    await asyncio.sleep(60)
                    status = await training_system.get_training_status()
                    print(f"Training status: {status['training_progress'].get('current_phase', 'N/A')}")
                    
            except KeyboardInterrupt:
                print("\nStopping training program...")
                final_report = await training_system.stop_training_program()
                print("Training completed. Final report generated.")
                
                # Print summary
                cert_summary = final_report.get("certification_summary", {})
                print(f"\nCertification Summary:")
                print(f"  Total agents assessed: {cert_summary.get('total_assessed', 0)}")
                print(f"  Production ready: {len(cert_summary.get('production_ready_agents', []))}")
                
        elif args.action == "status":
            status = await training_system.get_training_status()
            print(json.dumps(status, indent=2, default=str))
            
        elif args.action == "report":
            report = await training_system.generate_progress_report()
            print(json.dumps(report, indent=2, default=str))
            
        else:
            print(f"Action '{args.action}' not yet implemented")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    finally:
        await training_system.shutdown()


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))