#!/usr/bin/env python3
"""
Agent Training Framework for Beverly Knits ERP
Orchestrates training, evaluation, and deployment of intelligent agents
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Enumeration of agent roles"""
    INVENTORY = "inventory_intelligence"
    FORECAST = "forecast_intelligence"
    PRODUCTION = "production_planning"
    YARN = "yarn_substitution"
    QUALITY = "quality_assurance"


class TrainingPhase(Enum):
    """Training phases for agents"""
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    SKILL_DEVELOPMENT = "skill_development"
    INTEGRATION_TESTING = "integration_testing"
    CERTIFICATION = "certification"
    SHADOW_MODE = "shadow_mode"
    ADVISORY_MODE = "advisory_mode"
    SUPERVISED_AUTONOMY = "supervised_autonomy"
    FULL_AUTONOMY = "full_autonomy"


@dataclass
class AgentConfig:
    """Configuration for an individual agent"""
    role: AgentRole
    name: str
    version: str = "1.0.0"
    min_accuracy: float = 0.85
    max_response_time_ms: int = 200
    max_error_rate: float = 0.05
    training_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    capabilities: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = asdict(self)
        config_dict['role'] = self.role.value
        return config_dict


@dataclass
class TrainingMetrics:
    """Metrics for tracking training progress"""
    phase: TrainingPhase
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    training_samples: int = 0
    validation_samples: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def meets_requirements(self, config: AgentConfig) -> bool:
        """Check if metrics meet minimum requirements"""
        return (
            self.accuracy >= config.min_accuracy and
            self.response_time_ms <= config.max_response_time_ms and
            self.error_rate <= config.max_error_rate
        )


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.knowledge_base = {}
        self.trained_models = {}
        self.training_history = []
        self.current_phase = TrainingPhase.KNOWLEDGE_ACQUISITION
        self.is_certified = False
        self.performance_metrics = TrainingMetrics(phase=self.current_phase)
        
        # Create agent-specific directories
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / "data" / "agent_training"
        self.model_path = self.base_path / "models" / "agents" / config.role.value
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {config.name} agent (role: {config.role.value})")
    
    @abstractmethod
    def load_knowledge_base(self, knowledge_files: List[str]) -> bool:
        """Load domain knowledge from documentation"""
        pass
    
    @abstractmethod
    def train(self, training_data: pd.DataFrame, **kwargs) -> TrainingMetrics:
        """Train the agent on historical data"""
        pass
    
    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame) -> TrainingMetrics:
        """Evaluate agent performance"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions or recommendations"""
        pass
    
    def practice_scenario(self, scenario_name: str) -> TrainingMetrics:
        """Practice a specific scenario"""
        scenario_path = self.data_path / "scenarios" / f"{scenario_name}.json"
        
        if not scenario_path.exists():
            logger.warning(f"Scenario {scenario_name} not found")
            return self.performance_metrics
        
        with open(scenario_path, 'r') as f:
            scenario_data = json.load(f)
        
        # Run through scenario
        results = []
        start_time = datetime.now()
        
        for case in scenario_data['cases']:
            try:
                prediction = self.predict(case['input'])
                expected = case['expected']
                
                # Calculate accuracy
                if isinstance(expected, dict):
                    accuracy = self._calculate_dict_similarity(prediction, expected)
                else:
                    accuracy = 1.0 if prediction == expected else 0.0
                
                results.append(accuracy)
            except Exception as e:
                logger.error(f"Error in scenario {scenario_name}: {e}")
                results.append(0.0)
        
        # Calculate metrics
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds() * 1000 / len(scenario_data['cases'])
        
        metrics = TrainingMetrics(
            phase=self.current_phase,
            accuracy=np.mean(results),
            response_time_ms=response_time,
            error_rate=1.0 - np.mean(results),
            training_samples=len(scenario_data['cases'])
        )
        
        self.training_history.append(metrics)
        return metrics
    
    def _calculate_dict_similarity(self, dict1: Dict, dict2: Dict) -> float:
        """Calculate similarity between two dictionaries"""
        if not dict1 or not dict2:
            return 0.0
        
        common_keys = set(dict1.keys()) & set(dict2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            if dict1[key] == dict2[key]:
                similarities.append(1.0)
            elif isinstance(dict1[key], (int, float)) and isinstance(dict2[key], (int, float)):
                # For numeric values, calculate relative similarity
                max_val = max(abs(dict1[key]), abs(dict2[key]))
                if max_val > 0:
                    similarity = 1.0 - abs(dict1[key] - dict2[key]) / max_val
                    similarities.append(similarity)
                else:
                    similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities)
    
    def save_model(self, model_name: str = None) -> str:
        """Save trained model to disk"""
        if model_name is None:
            model_name = f"{self.config.role.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        model_path = self.model_path / model_name
        
        model_data = {
            'config': self.config.to_dict(),
            'knowledge_base': self.knowledge_base,
            'trained_models': self.trained_models,
            'training_history': self.training_history,
            'is_certified': self.is_certified,
            'performance_metrics': asdict(self.performance_metrics)
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model from disk"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.knowledge_base = model_data['knowledge_base']
            self.trained_models = model_data['trained_models']
            self.training_history = model_data['training_history']
            self.is_certified = model_data['is_certified']
            self.performance_metrics = TrainingMetrics(**model_data['performance_metrics'])
            
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def certify(self) -> bool:
        """Run certification tests"""
        if not self.performance_metrics.meets_requirements(self.config):
            logger.warning(f"Agent does not meet minimum requirements for certification")
            return False
        
        # Run certification suite
        certification_path = self.data_path / "scenarios" / "certification.json"
        
        if certification_path.exists():
            metrics = self.practice_scenario("certification")
            
            if metrics.accuracy >= 0.9:  # Higher bar for certification
                self.is_certified = True
                self.current_phase = TrainingPhase.SHADOW_MODE
                logger.info(f"Agent {self.config.name} certified successfully!")
                return True
        
        logger.warning(f"Certification failed for {self.config.name}")
        return False
    
    def advance_phase(self) -> bool:
        """Advance to the next training phase"""
        phase_order = [
            TrainingPhase.KNOWLEDGE_ACQUISITION,
            TrainingPhase.SKILL_DEVELOPMENT,
            TrainingPhase.INTEGRATION_TESTING,
            TrainingPhase.CERTIFICATION,
            TrainingPhase.SHADOW_MODE,
            TrainingPhase.ADVISORY_MODE,
            TrainingPhase.SUPERVISED_AUTONOMY,
            TrainingPhase.FULL_AUTONOMY
        ]
        
        current_index = phase_order.index(self.current_phase)
        
        if current_index < len(phase_order) - 1:
            # Check if ready to advance
            if self.performance_metrics.meets_requirements(self.config):
                self.current_phase = phase_order[current_index + 1]
                logger.info(f"Advanced to phase: {self.current_phase.value}")
                return True
            else:
                logger.warning(f"Not ready to advance from {self.current_phase.value}")
                return False
        
        logger.info(f"Already at final phase: {self.current_phase.value}")
        return False


class AgentTrainingOrchestrator:
    """Orchestrates training of multiple agents"""
    
    def __init__(self):
        self.agents: Dict[AgentRole, BaseAgent] = {}
        self.training_schedule = {}
        self.performance_history = []
        self.base_path = Path(__file__).parent.parent.parent
        
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for training"""
        self.agents[agent.config.role] = agent
        logger.info(f"Registered agent: {agent.config.name}")
    
    def load_training_data(self, data_type: str = "all") -> pd.DataFrame:
        """Load training data for agents"""
        data_path = self.base_path / "data" / "production" / "5"
        
        if data_type == "inventory":
            # Load inventory data
            file_path = data_path / "ERP Data" / "yarn_inventory.xlsx"
            if file_path.exists():
                return pd.read_excel(file_path)
        
        elif data_type == "production":
            # Load production orders
            file_path = data_path / "ERP Data" / "8-28-2025" / "eFab_Knit_Orders.csv"
            if file_path.exists():
                return pd.read_csv(file_path)
        
        elif data_type == "sales":
            # Load sales data
            file_path = data_path / "ERP Data" / "Sales Activity Report.csv"
            if file_path.exists():
                return pd.read_csv(file_path)
        
        elif data_type == "bom":
            # Load BOM data
            file_path = data_path / "BOM_updated.csv"
            if file_path.exists():
                return pd.read_csv(file_path)
        
        # Return empty DataFrame if file not found
        logger.warning(f"Training data not found for type: {data_type}")
        return pd.DataFrame()
    
    def train_all_agents(self, epochs: int = 100) -> Dict[AgentRole, TrainingMetrics]:
        """Train all registered agents"""
        results = {}
        
        for role, agent in self.agents.items():
            logger.info(f"Training {role.value} agent...")
            
            # Load appropriate training data
            if role == AgentRole.INVENTORY:
                training_data = self.load_training_data("inventory")
            elif role == AgentRole.PRODUCTION:
                training_data = self.load_training_data("production")
            elif role == AgentRole.FORECAST:
                training_data = self.load_training_data("sales")
            elif role == AgentRole.YARN:
                training_data = self.load_training_data("bom")
            else:
                training_data = pd.DataFrame()
            
            if not training_data.empty:
                metrics = agent.train(training_data, epochs=epochs)
                results[role] = metrics
                logger.info(f"Training complete for {role.value}: accuracy={metrics.accuracy:.2%}")
            else:
                logger.warning(f"No training data available for {role.value}")
        
        return results
    
    def evaluate_all_agents(self) -> Dict[AgentRole, TrainingMetrics]:
        """Evaluate all agents"""
        results = {}
        
        for role, agent in self.agents.items():
            # Load test data (using same data for now, should be separate)
            if role == AgentRole.INVENTORY:
                test_data = self.load_training_data("inventory")
            elif role == AgentRole.PRODUCTION:
                test_data = self.load_training_data("production")
            elif role == AgentRole.FORECAST:
                test_data = self.load_training_data("sales")
            elif role == AgentRole.YARN:
                test_data = self.load_training_data("bom")
            else:
                test_data = pd.DataFrame()
            
            if not test_data.empty:
                metrics = agent.evaluate(test_data)
                results[role] = metrics
                logger.info(f"Evaluation complete for {role.value}: accuracy={metrics.accuracy:.2%}")
        
        return results
    
    def run_certification_suite(self) -> Dict[AgentRole, bool]:
        """Run certification for all agents"""
        results = {}
        
        for role, agent in self.agents.items():
            logger.info(f"Running certification for {role.value}...")
            certified = agent.certify()
            results[role] = certified
            
            if certified:
                logger.info(f"{role.value} agent PASSED certification")
            else:
                logger.warning(f"{role.value} agent FAILED certification")
        
        return results
    
    def deploy_agents(self, mode: str = "shadow") -> bool:
        """Deploy certified agents in specified mode"""
        deployment_modes = {
            "shadow": TrainingPhase.SHADOW_MODE,
            "advisory": TrainingPhase.ADVISORY_MODE,
            "supervised": TrainingPhase.SUPERVISED_AUTONOMY,
            "full": TrainingPhase.FULL_AUTONOMY
        }
        
        if mode not in deployment_modes:
            logger.error(f"Invalid deployment mode: {mode}")
            return False
        
        target_phase = deployment_modes[mode]
        deployed = []
        
        for role, agent in self.agents.items():
            if agent.is_certified:
                agent.current_phase = target_phase
                deployed.append(role.value)
                logger.info(f"Deployed {role.value} in {mode} mode")
            else:
                logger.warning(f"Cannot deploy {role.value} - not certified")
        
        if deployed:
            logger.info(f"Successfully deployed {len(deployed)} agents: {deployed}")
            return True
        else:
            logger.warning("No agents deployed - none are certified")
            return False
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "agents": {},
            "summary": {
                "total_agents": len(self.agents),
                "certified_agents": 0,
                "average_accuracy": 0,
                "average_response_time": 0
            }
        }
        
        accuracies = []
        response_times = []
        
        for role, agent in self.agents.items():
            agent_info = {
                "name": agent.config.name,
                "version": agent.config.version,
                "is_certified": agent.is_certified,
                "current_phase": agent.current_phase.value,
                "performance": {
                    "accuracy": agent.performance_metrics.accuracy,
                    "response_time_ms": agent.performance_metrics.response_time_ms,
                    "error_rate": agent.performance_metrics.error_rate
                },
                "training_history_count": len(agent.training_history)
            }
            
            report["agents"][role.value] = agent_info
            
            if agent.is_certified:
                report["summary"]["certified_agents"] += 1
            
            accuracies.append(agent.performance_metrics.accuracy)
            response_times.append(agent.performance_metrics.response_time_ms)
        
        report["summary"]["average_accuracy"] = np.mean(accuracies) if accuracies else 0
        report["summary"]["average_response_time"] = np.mean(response_times) if response_times else 0
        
        return report
    
    def save_training_state(self, filepath: str = None) -> str:
        """Save the current training state"""
        if filepath is None:
            filepath = self.base_path / "models" / "training_state.json"
        
        state = {
            "timestamp": datetime.now().isoformat(),
            "agents": {},
            "performance_history": self.performance_history
        }
        
        for role, agent in self.agents.items():
            # Save each agent's model
            model_path = agent.save_model()
            state["agents"][role.value] = {
                "model_path": model_path,
                "is_certified": agent.is_certified,
                "current_phase": agent.current_phase.value
            }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Training state saved to {filepath}")
        return str(filepath)
    
    def load_training_state(self, filepath: str) -> bool:
        """Load a saved training state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.performance_history = state.get("performance_history", [])
            
            # Note: Agent loading would require the agent instances to be created first
            logger.info(f"Training state loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load training state: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = AgentTrainingOrchestrator()
    
    # Create sample agent configs
    inventory_config = AgentConfig(
        role=AgentRole.INVENTORY,
        name="InventoryBot-v1",
        capabilities=["balance_calculation", "shortage_detection", "netting"],
        min_accuracy=0.95
    )
    
    forecast_config = AgentConfig(
        role=AgentRole.FORECAST,
        name="ForecastBot-v1",
        capabilities=["demand_prediction", "seasonal_analysis", "model_training"],
        min_accuracy=0.85
    )
    
    # Note: Actual agent implementations would be in separate files
    # This is just the framework
    
    logger.info("Agent Training Framework initialized successfully")
    print("\nAgent Training Framework is ready for use")
    print("Import specific agent implementations to begin training")