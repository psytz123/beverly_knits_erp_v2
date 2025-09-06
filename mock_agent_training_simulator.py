#!/usr/bin/env python3
"""
Mock Agent Training Simulator for eFab ERP AI Agents
Simulates customer implementations for training AI agents before deployment
"""

import random
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Industry(Enum):
    FURNITURE = "furniture"
    INJECTION_MOLDING = "injection_molding"
    ELECTRICAL = "electrical_equipment"
    TEXTILE = "textile_manufacturing"
    AUTOMOTIVE = "automotive_parts"


class ComplexityLevel(Enum):
    LOW = (1, 3)
    MEDIUM = (4, 6)
    HIGH = (7, 8)
    VERY_HIGH = (9, 10)


@dataclass
class CustomerProfile:
    """Represents a mock customer for training"""
    company_name: str
    industry: Industry
    employees: int
    revenue: float
    locations: int
    complexity_score: int
    data_quality: float
    existing_systems: List[str]
    custom_requirements: List[str]
    timeline_weeks: int
    challenges: List[str]


@dataclass
class ImplementationTask:
    """Represents a task in the implementation"""
    task_id: str
    name: str
    week: int
    duration_hours: float
    dependencies: List[str]
    required_agents: List[str]
    complexity: int
    status: str = "pending"
    completion_percentage: float = 0.0
    issues_encountered: List[str] = None


@dataclass
class AgentAction:
    """Records an agent's action during training"""
    agent_id: str
    action_type: str
    timestamp: datetime
    task_id: str
    success: bool
    confidence: float
    time_taken: float
    errors: List[str]
    learning_points: List[str]


class MockCustomerGenerator:
    """Generates realistic mock customers for training"""
    
    def __init__(self):
        self.furniture_companies = [
            "Custom Craft Furniture", "Heritage Woods", "Modern Living Co",
            "Office Solutions Pro", "Comfort Home Furnishings"
        ]
        self.molding_companies = [
            "TechMold Industries", "Precision Plastics", "InjectionPro Corp",
            "PolyForm Manufacturing", "Advanced Molding Systems"
        ]
        self.electrical_companies = [
            "PowerTech Solutions", "Global Electric", "Circuit Masters",
            "Industrial Controls Inc", "Smart Grid Technologies"
        ]
        
    def generate_customer(self, industry: Industry = None) -> CustomerProfile:
        """Generate a realistic customer profile"""
        if industry is None:
            industry = random.choice(list(Industry))
            
        company_name = self._get_company_name(industry)
        complexity = random.randint(1, 10)
        
        return CustomerProfile(
            company_name=company_name,
            industry=industry,
            employees=self._get_employee_count(complexity),
            revenue=self._get_revenue(complexity),
            locations=self._get_locations(complexity),
            complexity_score=complexity,
            data_quality=random.uniform(0.3, 1.0),
            existing_systems=self._get_existing_systems(complexity),
            custom_requirements=self._get_requirements(industry, complexity),
            timeline_weeks=self._get_timeline(complexity),
            challenges=self._get_challenges(complexity, industry)
        )
    
    def _get_company_name(self, industry: Industry) -> str:
        if industry == Industry.FURNITURE:
            return random.choice(self.furniture_companies)
        elif industry == Industry.INJECTION_MOLDING:
            return random.choice(self.molding_companies)
        elif industry == Industry.ELECTRICAL:
            return random.choice(self.electrical_companies)
        else:
            return f"{industry.value.title()} Corp"
    
    def _get_employee_count(self, complexity: int) -> int:
        if complexity <= 3:
            return random.randint(10, 50)
        elif complexity <= 6:
            return random.randint(50, 200)
        elif complexity <= 8:
            return random.randint(200, 500)
        else:
            return random.randint(500, 2000)
    
    def _get_revenue(self, complexity: int) -> float:
        if complexity <= 3:
            return random.uniform(1, 10) * 1e6
        elif complexity <= 6:
            return random.uniform(10, 50) * 1e6
        elif complexity <= 8:
            return random.uniform(50, 200) * 1e6
        else:
            return random.uniform(200, 1000) * 1e6
    
    def _get_locations(self, complexity: int) -> int:
        if complexity <= 4:
            return 1
        elif complexity <= 7:
            return random.randint(2, 4)
        else:
            return random.randint(5, 15)
    
    def _get_existing_systems(self, complexity: int) -> List[str]:
        basic_systems = ["Excel", "QuickBooks", "Google Sheets"]
        mid_systems = ["Sage", "NetSuite", "Dynamics 365"]
        enterprise_systems = ["SAP", "Oracle", "Infor"]
        
        if complexity <= 3:
            return random.sample(basic_systems, k=random.randint(1, 2))
        elif complexity <= 6:
            return random.sample(basic_systems + mid_systems, k=random.randint(2, 3))
        else:
            return random.sample(mid_systems + enterprise_systems, k=random.randint(2, 4))
    
    def _get_requirements(self, industry: Industry, complexity: int) -> List[str]:
        base_requirements = [
            "Inventory management", "Production planning", "Quality control",
            "Financial reporting", "Customer management"
        ]
        
        industry_specific = {
            Industry.FURNITURE: [
                "Custom configurator", "Wood optimization", "Finish tracking",
                "Assembly instructions", "Delivery scheduling"
            ],
            Industry.INJECTION_MOLDING: [
                "Mold management", "Recipe control", "Cycle time optimization",
                "Cavity tracking", "Material traceability"
            ],
            Industry.ELECTRICAL: [
                "Serial number tracking", "Compliance management", "Test data",
                "Warranty tracking", "Component sourcing"
            ]
        }
        
        num_requirements = min(complexity + 2, 10)
        requirements = random.sample(base_requirements, k=min(len(base_requirements), num_requirements // 2))
        
        if industry in industry_specific:
            specific = industry_specific[industry]
            requirements.extend(random.sample(specific, k=min(len(specific), num_requirements // 2)))
        
        return requirements
    
    def _get_timeline(self, complexity: int) -> int:
        if complexity <= 3:
            return random.randint(4, 6)
        elif complexity <= 6:
            return random.randint(6, 9)
        else:
            return random.randint(9, 12)
    
    def _get_challenges(self, complexity: int, industry: Industry) -> List[str]:
        challenges = []
        
        # Data challenges
        if random.random() < 0.7:
            challenges.append(f"Data quality issues ({int((1 - random.random()) * 100)}% dirty data)")
        
        # Integration challenges
        if complexity > 5 and random.random() < 0.6:
            challenges.append(f"Complex integrations with {random.randint(2, 8)} systems")
        
        # Process challenges
        if random.random() < 0.5:
            challenges.append("Undefined or inconsistent business processes")
        
        # User challenges
        if random.random() < 0.4:
            challenges.append("User resistance to change")
        
        # Technical challenges
        if complexity > 7 and random.random() < 0.6:
            challenges.append("Performance requirements for high-volume operations")
        
        # Industry-specific challenges
        industry_challenges = {
            Industry.FURNITURE: "Complex product customization rules",
            Industry.INJECTION_MOLDING: "Real-time machine monitoring requirements",
            Industry.ELECTRICAL: "Regulatory compliance tracking"
        }
        
        if industry in industry_challenges and random.random() < 0.5:
            challenges.append(industry_challenges[industry])
        
        return challenges


class ImplementationSimulator:
    """Simulates an ERP implementation for agent training"""
    
    def __init__(self, customer: CustomerProfile):
        self.customer = customer
        self.tasks = []
        self.agent_actions = []
        self.current_week = 1
        self.implementation_score = 100  # Start at 100, deduct for issues
        
    def generate_implementation_plan(self) -> List[ImplementationTask]:
        """Generate a realistic implementation plan based on customer profile"""
        tasks = []
        task_counter = 1
        
        # Week 1: Discovery and Assessment
        tasks.extend([
            self._create_task(f"T{task_counter:03d}", "Initial assessment", 1, 8, [], 
                            ["orchestrator", "project_manager"]),
            self._create_task(f"T{task_counter+1:03d}", "Data quality analysis", 1, 16, [], 
                            ["data_migration"]),
            self._create_task(f"T{task_counter+2:03d}", "Requirements gathering", 1, 24, [], 
                            ["project_manager", "industry_expert"]),
        ])
        task_counter += 3
        
        # Week 2-3: Data Migration Planning
        for week in range(2, 4):
            tasks.extend([
                self._create_task(f"T{task_counter:03d}", f"Data mapping week {week}", week, 20, 
                                [f"T{task_counter-3:03d}"], ["data_migration"]),
                self._create_task(f"T{task_counter+1:03d}", f"Data cleansing week {week}", week, 16, 
                                [f"T{task_counter:03d}"], ["data_migration"]),
                self._create_task(f"T{task_counter+2:03d}", f"Migration scripts week {week}", week, 12, 
                                [f"T{task_counter+1:03d}"], ["data_migration", "configuration"]),
            ])
            task_counter += 3
        
        # Middle weeks: Configuration and Customization
        mid_week_start = 4
        mid_week_end = self.customer.timeline_weeks - 2
        
        for week in range(mid_week_start, mid_week_end + 1):
            # Add configuration tasks based on requirements
            for i, req in enumerate(self.customer.custom_requirements[:3]):  # Top 3 requirements
                tasks.append(
                    self._create_task(f"T{task_counter:03d}", f"Configure: {req}", week, 
                                    random.randint(8, 24), [], ["configuration", "industry_expert"])
                )
                task_counter += 1
        
        # Final weeks: Testing and Deployment
        final_week = self.customer.timeline_weeks
        tasks.extend([
            self._create_task(f"T{task_counter:03d}", "System integration testing", 
                            final_week - 1, 16, [], ["orchestrator", "configuration"]),
            self._create_task(f"T{task_counter+1:03d}", "User acceptance testing", 
                            final_week - 1, 20, [f"T{task_counter:03d}"], ["project_manager"]),
            self._create_task(f"T{task_counter+2:03d}", "Performance testing", 
                            final_week - 1, 12, [f"T{task_counter:03d}"], ["performance"]),
            self._create_task(f"T{task_counter+3:03d}", "Go-live preparation", 
                            final_week, 24, [f"T{task_counter+1:03d}", f"T{task_counter+2:03d}"], 
                            ["orchestrator", "project_manager"]),
            self._create_task(f"T{task_counter+4:03d}", "Deployment", 
                            final_week, 8, [f"T{task_counter+3:03d}"], ["orchestrator"]),
        ])
        
        self.tasks = tasks
        return tasks
    
    def _create_task(self, task_id: str, name: str, week: int, duration: float, 
                    dependencies: List[str], agents: List[str]) -> ImplementationTask:
        """Helper to create a task"""
        complexity = self.customer.complexity_score + random.randint(-2, 2)
        complexity = max(1, min(10, complexity))  # Keep between 1-10
        
        return ImplementationTask(
            task_id=task_id,
            name=name,
            week=week,
            duration_hours=duration,
            dependencies=dependencies,
            required_agents=agents,
            complexity=complexity
        )
    
    def simulate_week(self, week: int) -> Dict[str, Any]:
        """Simulate a week of implementation"""
        week_tasks = [t for t in self.tasks if t.week == week]
        week_results = {
            "week": week,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "issues": [],
            "agent_performance": {},
            "overall_progress": 0.0
        }
        
        for task in week_tasks:
            result = self._simulate_task(task)
            
            if result["success"]:
                week_results["tasks_completed"] += 1
                task.status = "completed"
                task.completion_percentage = 100.0
            else:
                week_results["tasks_failed"] += 1
                task.status = "failed"
                week_results["issues"].extend(result["issues"])
                self.implementation_score -= 5  # Deduct for failure
            
            # Record agent actions
            for agent in task.required_agents:
                action = AgentAction(
                    agent_id=agent,
                    action_type="task_execution",
                    timestamp=datetime.now(),
                    task_id=task.task_id,
                    success=result["success"],
                    confidence=result["confidence"],
                    time_taken=result["time_taken"],
                    errors=result.get("errors", []),
                    learning_points=result.get("learning_points", [])
                )
                self.agent_actions.append(action)
        
        # Calculate overall progress
        completed_tasks = len([t for t in self.tasks if t.status == "completed"])
        week_results["overall_progress"] = (completed_tasks / len(self.tasks)) * 100
        
        return week_results
    
    def _simulate_task(self, task: ImplementationTask) -> Dict[str, Any]:
        """Simulate execution of a single task"""
        # Base success probability depends on complexity and data quality
        base_success_prob = 0.9 - (task.complexity * 0.05) + (self.customer.data_quality * 0.2)
        
        # Add randomness
        success_prob = max(0.3, min(0.95, base_success_prob + random.uniform(-0.1, 0.1)))
        
        success = random.random() < success_prob
        
        result = {
            "success": success,
            "confidence": random.uniform(0.6, 0.95) if success else random.uniform(0.3, 0.6),
            "time_taken": task.duration_hours * random.uniform(0.8, 1.2),
            "issues": [],
            "errors": [],
            "learning_points": []
        }
        
        if not success:
            # Generate realistic issues
            issue_types = [
                "Data format mismatch",
                "Missing dependencies",
                "Performance bottleneck",
                "Integration failure",
                "Business logic conflict",
                "User requirement change"
            ]
            result["issues"] = random.sample(issue_types, k=random.randint(1, 3))
            result["errors"] = [f"Error in {task.name}: {issue}" for issue in result["issues"]]
        
        # Generate learning points
        result["learning_points"] = self._generate_learning_points(task, success)
        
        return result
    
    def _generate_learning_points(self, task: ImplementationTask, success: bool) -> List[str]:
        """Generate learning points from task execution"""
        learning_points = []
        
        if success:
            learning_points.append(f"Successfully completed {task.name} using {', '.join(task.required_agents)}")
            if task.complexity > 7:
                learning_points.append(f"Handled high-complexity task (level {task.complexity}) effectively")
        else:
            learning_points.append(f"Failed {task.name} - requires additional training for {', '.join(task.required_agents)}")
            learning_points.append(f"Complexity level {task.complexity} proved challenging")
        
        # Industry-specific learnings
        if self.customer.industry == Industry.FURNITURE and "configurator" in task.name.lower():
            learning_points.append("Furniture configurator patterns identified")
        elif self.customer.industry == Industry.INJECTION_MOLDING and "mold" in task.name.lower():
            learning_points.append("Injection molding specific requirements cataloged")
        
        return learning_points
    
    def run_complete_simulation(self) -> Dict[str, Any]:
        """Run the complete implementation simulation"""
        logger.info(f"Starting simulation for {self.customer.company_name}")
        logger.info(f"Industry: {self.customer.industry.value}, Complexity: {self.customer.complexity_score}/10")
        
        # Generate implementation plan
        self.generate_implementation_plan()
        logger.info(f"Generated {len(self.tasks)} tasks over {self.customer.timeline_weeks} weeks")
        
        # Convert customer to dict properly, handling enum values
        customer_dict = asdict(self.customer)
        customer_dict['industry'] = self.customer.industry.value
        
        # Run week by week simulation
        simulation_results = {
            "customer": customer_dict,
            "weeks": [],
            "final_score": 0,
            "total_issues": [],
            "agent_performance_summary": {},
            "recommendations": []
        }
        
        for week in range(1, self.customer.timeline_weeks + 1):
            logger.info(f"Simulating week {week}...")
            week_result = self.simulate_week(week)
            simulation_results["weeks"].append(week_result)
            simulation_results["total_issues"].extend(week_result["issues"])
        
        # Calculate final metrics
        simulation_results["final_score"] = self.implementation_score
        simulation_results["success"] = self.implementation_score >= 70
        
        # Analyze agent performance
        agent_performance = {}
        for action in self.agent_actions:
            if action.agent_id not in agent_performance:
                agent_performance[action.agent_id] = {
                    "tasks": 0,
                    "successes": 0,
                    "avg_confidence": 0,
                    "total_time": 0
                }
            
            agent_performance[action.agent_id]["tasks"] += 1
            if action.success:
                agent_performance[action.agent_id]["successes"] += 1
            agent_performance[action.agent_id]["avg_confidence"] += action.confidence
            agent_performance[action.agent_id]["total_time"] += action.time_taken
        
        # Calculate averages
        for agent_id, perf in agent_performance.items():
            perf["success_rate"] = (perf["successes"] / perf["tasks"]) * 100 if perf["tasks"] > 0 else 0
            perf["avg_confidence"] = perf["avg_confidence"] / perf["tasks"] if perf["tasks"] > 0 else 0
        
        simulation_results["agent_performance_summary"] = agent_performance
        
        # Generate recommendations
        simulation_results["recommendations"] = self._generate_recommendations(simulation_results)
        
        return simulation_results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate training recommendations based on simulation results"""
        recommendations = []
        
        # Check overall success
        if results["final_score"] < 70:
            recommendations.append("Additional training required for complex implementations")
        
        # Check agent performance
        for agent_id, perf in results["agent_performance_summary"].items():
            if perf["success_rate"] < 80:
                recommendations.append(f"Agent '{agent_id}' needs improvement (success rate: {perf['success_rate']:.1f}%)")
            if perf["avg_confidence"] < 0.7:
                recommendations.append(f"Agent '{agent_id}' shows low confidence (avg: {perf['avg_confidence']:.2f})")
        
        # Check common issues
        issue_counts = {}
        for issue in results["total_issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        for issue, count in issue_counts.items():
            if count >= 3:
                recommendations.append(f"Recurring issue detected: '{issue}' ({count} occurrences)")
        
        # Industry-specific recommendations
        if self.customer.industry == Industry.INJECTION_MOLDING:
            recommendations.append("Focus on mold management and cycle time optimization training")
        elif self.customer.industry == Industry.FURNITURE:
            recommendations.append("Enhance custom configurator and wood optimization capabilities")
        
        return recommendations


class TrainingOrchestrator:
    """Orchestrates multiple training simulations"""
    
    def __init__(self):
        self.generator = MockCustomerGenerator()
        self.simulations_run = 0
        self.aggregate_results = {
            "total_simulations": 0,
            "successful_implementations": 0,
            "failed_implementations": 0,
            "agent_scores": {},
            "industry_performance": {},
            "complexity_performance": {},
            "common_issues": {},
            "learning_catalog": []
        }
    
    def run_training_batch(self, num_simulations: int = 10, save_results: bool = True) -> Dict[str, Any]:
        """Run a batch of training simulations"""
        logger.info(f"Starting training batch with {num_simulations} simulations")
        
        for i in range(num_simulations):
            logger.info(f"\n{'='*50}")
            logger.info(f"Simulation {i+1}/{num_simulations}")
            logger.info(f"{'='*50}")
            
            # Generate random customer
            customer = self.generator.generate_customer()
            
            # Run simulation
            simulator = ImplementationSimulator(customer)
            results = simulator.run_complete_simulation()
            
            # Update aggregate results
            self._update_aggregate_results(results)
            
            # Save individual result if requested
            if save_results:
                self._save_simulation_result(results, i+1)
            
            self.simulations_run += 1
            
            # Brief pause between simulations
            time.sleep(0.5)
        
        # Generate final report
        report = self._generate_training_report()
        
        if save_results:
            self._save_training_report(report)
        
        return report
    
    def _update_aggregate_results(self, simulation: Dict[str, Any]):
        """Update aggregate training results"""
        self.aggregate_results["total_simulations"] += 1
        
        if simulation["success"]:
            self.aggregate_results["successful_implementations"] += 1
        else:
            self.aggregate_results["failed_implementations"] += 1
        
        # Update agent scores
        for agent_id, perf in simulation["agent_performance_summary"].items():
            if agent_id not in self.aggregate_results["agent_scores"]:
                self.aggregate_results["agent_scores"][agent_id] = {
                    "total_tasks": 0,
                    "total_successes": 0,
                    "confidence_sum": 0,
                    "appearances": 0
                }
            
            scores = self.aggregate_results["agent_scores"][agent_id]
            scores["total_tasks"] += perf["tasks"]
            scores["total_successes"] += perf["successes"]
            scores["confidence_sum"] += perf["avg_confidence"]
            scores["appearances"] += 1
        
        # Update industry performance
        industry = simulation["customer"]["industry"]
        if industry not in self.aggregate_results["industry_performance"]:
            self.aggregate_results["industry_performance"][industry] = {
                "simulations": 0,
                "successes": 0
            }
        
        self.aggregate_results["industry_performance"][industry]["simulations"] += 1
        if simulation["success"]:
            self.aggregate_results["industry_performance"][industry]["successes"] += 1
        
        # Update complexity performance
        complexity = simulation["customer"]["complexity_score"]
        complexity_bucket = "LOW" if complexity <= 3 else "MEDIUM" if complexity <= 6 else "HIGH" if complexity <= 8 else "VERY_HIGH"
        
        if complexity_bucket not in self.aggregate_results["complexity_performance"]:
            self.aggregate_results["complexity_performance"][complexity_bucket] = {
                "simulations": 0,
                "successes": 0
            }
        
        self.aggregate_results["complexity_performance"][complexity_bucket]["simulations"] += 1
        if simulation["success"]:
            self.aggregate_results["complexity_performance"][complexity_bucket]["successes"] += 1
        
        # Track common issues
        for issue in simulation["total_issues"]:
            self.aggregate_results["common_issues"][issue] = \
                self.aggregate_results["common_issues"].get(issue, 0) + 1
        
        # Collect learning points
        self.aggregate_results["learning_catalog"].extend(simulation["recommendations"])
    
    def _generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            "summary": {
                "total_simulations": self.aggregate_results["total_simulations"],
                "success_rate": (self.aggregate_results["successful_implementations"] / 
                               self.aggregate_results["total_simulations"] * 100) 
                               if self.aggregate_results["total_simulations"] > 0 else 0,
                "failed_implementations": self.aggregate_results["failed_implementations"]
            },
            "agent_performance": {},
            "industry_analysis": {},
            "complexity_analysis": {},
            "top_issues": [],
            "key_learnings": [],
            "recommendations": []
        }
        
        # Calculate agent performance
        for agent_id, scores in self.aggregate_results["agent_scores"].items():
            report["agent_performance"][agent_id] = {
                "success_rate": (scores["total_successes"] / scores["total_tasks"] * 100) 
                               if scores["total_tasks"] > 0 else 0,
                "avg_confidence": scores["confidence_sum"] / scores["appearances"] 
                                 if scores["appearances"] > 0 else 0,
                "total_tasks": scores["total_tasks"]
            }
        
        # Industry analysis
        for industry, perf in self.aggregate_results["industry_performance"].items():
            report["industry_analysis"][industry] = {
                "success_rate": (perf["successes"] / perf["simulations"] * 100) 
                               if perf["simulations"] > 0 else 0,
                "simulations": perf["simulations"]
            }
        
        # Complexity analysis
        for complexity, perf in self.aggregate_results["complexity_performance"].items():
            report["complexity_analysis"][complexity] = {
                "success_rate": (perf["successes"] / perf["simulations"] * 100) 
                               if perf["simulations"] > 0 else 0,
                "simulations": perf["simulations"]
            }
        
        # Top issues
        sorted_issues = sorted(self.aggregate_results["common_issues"].items(), 
                              key=lambda x: x[1], reverse=True)
        report["top_issues"] = [{"issue": issue, "count": count} 
                                for issue, count in sorted_issues[:10]]
        
        # Key learnings (unique)
        unique_learnings = list(set(self.aggregate_results["learning_catalog"]))
        report["key_learnings"] = unique_learnings[:20]  # Top 20 unique learnings
        
        # Generate recommendations
        report["recommendations"] = self._generate_overall_recommendations(report)
        
        return report
    
    def _generate_overall_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate overall training recommendations"""
        recommendations = []
        
        # Overall success rate
        if report["summary"]["success_rate"] < 80:
            recommendations.append(
                f"Overall success rate ({report['summary']['success_rate']:.1f}%) below target. "
                "Increase training intensity."
            )
        
        # Agent-specific recommendations
        for agent_id, perf in report["agent_performance"].items():
            if perf["success_rate"] < 75:
                recommendations.append(
                    f"Agent '{agent_id}' requires additional training "
                    f"(current success rate: {perf['success_rate']:.1f}%)"
                )
        
        # Industry-specific recommendations
        for industry, perf in report["industry_analysis"].items():
            if perf["success_rate"] < 70:
                recommendations.append(
                    f"{industry} implementations need improvement "
                    f"(success rate: {perf['success_rate']:.1f}%)"
                )
        
        # Complexity recommendations
        if "VERY_HIGH" in report["complexity_analysis"]:
            if report["complexity_analysis"]["VERY_HIGH"]["success_rate"] < 60:
                recommendations.append(
                    "Focus on high-complexity scenarios - current performance inadequate"
                )
        
        # Issue-based recommendations
        if report["top_issues"] and report["top_issues"][0]["count"] > 5:
            recommendations.append(
                f"Address recurring issue: '{report['top_issues'][0]['issue']}' "
                f"(occurred {report['top_issues'][0]['count']} times)"
            )
        
        return recommendations
    
    def _save_simulation_result(self, result: Dict[str, Any], simulation_num: int):
        """Save individual simulation result"""
        filename = f"training_simulation_{simulation_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            # Convert any datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            json.dump(result, f, indent=2, default=convert_datetime)
        logger.info(f"Saved simulation result to {filename}")
    
    def _save_training_report(self, report: Dict[str, Any]):
        """Save training report"""
        filename = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved training report to {filename}")
        
        # Also save a human-readable version
        txt_filename = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(txt_filename, 'w') as f:
            f.write("AI AGENT TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Simulations: {report['summary']['total_simulations']}\n")
            f.write(f"Success Rate: {report['summary']['success_rate']:.1f}%\n")
            f.write(f"Failed Implementations: {report['summary']['failed_implementations']}\n\n")
            
            f.write("AGENT PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            for agent_id, perf in report["agent_performance"].items():
                f.write(f"{agent_id}:\n")
                f.write(f"  Success Rate: {perf['success_rate']:.1f}%\n")
                f.write(f"  Avg Confidence: {perf['avg_confidence']:.2f}\n")
                f.write(f"  Total Tasks: {perf['total_tasks']}\n")
            
            f.write("\nTOP ISSUES\n")
            f.write("-" * 30 + "\n")
            for issue in report["top_issues"][:5]:
                f.write(f"- {issue['issue']}: {issue['count']} occurrences\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for rec in report["recommendations"]:
                f.write(f"- {rec}\n")
        
        logger.info(f"Saved human-readable report to {txt_filename}")


def main():
    """Main function to run the training simulator"""
    print("\n" + "="*60)
    print("eFab ERP AI AGENT TRAINING SIMULATOR")
    print("="*60 + "\n")
    
    # Create training orchestrator
    orchestrator = TrainingOrchestrator()
    
    # Run training batch
    print("Running training simulations...")
    print("-" * 40)
    
    # You can adjust the number of simulations here
    report = orchestrator.run_training_batch(num_simulations=5, save_results=True)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nSuccess Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Total Simulations: {report['summary']['total_simulations']}")
    
    print("\nAgent Performance:")
    for agent_id, perf in report["agent_performance"].items():
        print(f"  {agent_id}: {perf['success_rate']:.1f}% success rate")
    
    print("\nTop Recommendations:")
    for i, rec in enumerate(report["recommendations"][:3], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*60)
    print("Training reports saved to disk")
    print("Review the JSON and TXT files for detailed analysis")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()