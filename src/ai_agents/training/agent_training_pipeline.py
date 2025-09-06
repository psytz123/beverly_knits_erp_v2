#!/usr/bin/env python3
"""
AI Agent Training Pipeline for eFab ERP System
Machine learning pipeline for training agents using Beverly Knits patterns and continuous improvement
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

from .beverly_pattern_extractor import BeverlyPatternExtractor, ExtractedPattern, TrainingDataset, PatternType
from ..core.agent_base import BaseAgent, AgentCapability
from ..core.state_manager import system_state

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingObjective(Enum):
    """Training objectives for different agent capabilities"""
    PATTERN_RECOGNITION = "PATTERN_RECOGNITION"             # Recognize successful patterns
    COMPLEXITY_ASSESSMENT = "COMPLEXITY_ASSESSMENT"         # Assess implementation complexity
    RISK_PREDICTION = "RISK_PREDICTION"                    # Predict implementation risks
    TIMELINE_ESTIMATION = "TIMELINE_ESTIMATION"            # Estimate implementation timelines
    OPTIMIZATION_SELECTION = "OPTIMIZATION_SELECTION"      # Select optimization strategies
    ERROR_CLASSIFICATION = "ERROR_CLASSIFICATION"          # Classify and handle errors
    INDUSTRY_ADAPTATION = "INDUSTRY_ADAPTATION"            # Adapt patterns to industries
    PERFORMANCE_PREDICTION = "PERFORMANCE_PREDICTION"      # Predict performance outcomes


class ModelType(Enum):
    """Types of ML models used for training"""
    CLASSIFICATION = "CLASSIFICATION"                       # Decision classification
    REGRESSION = "REGRESSION"                              # Continuous value prediction
    CLUSTERING = "CLUSTERING"                              # Pattern grouping
    TRANSFORMER = "TRANSFORMER"                            # Language understanding
    NEURAL_NETWORK = "NEURAL_NETWORK"                      # Deep learning
    ENSEMBLE = "ENSEMBLE"                                  # Multiple model combination


@dataclass
class TrainingConfiguration:
    """Configuration for agent training"""
    objective: TrainingObjective
    model_type: ModelType
    features: List[str]
    target_variable: str
    training_parameters: Dict[str, Any] = field(default_factory=dict)
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    performance_threshold: float = 0.8
    model_save_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "objective": self.objective.value,
            "model_type": self.model_type.value,
            "features": self.features,
            "target_variable": self.target_variable,
            "training_parameters": self.training_parameters,
            "validation_split": self.validation_split,
            "cross_validation_folds": self.cross_validation_folds,
            "performance_threshold": self.performance_threshold,
            "model_save_path": self.model_save_path
        }


@dataclass
class TrainingResult:
    """Results from model training"""
    configuration: TrainingConfiguration
    model_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    validation_metrics: Dict[str, float]
    model_path: str
    training_duration_seconds: float
    samples_trained: int
    training_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "configuration": self.configuration.to_dict(),
            "model_performance": self.model_performance,
            "feature_importance": self.feature_importance,
            "validation_metrics": self.validation_metrics,
            "model_path": self.model_path,
            "training_duration_seconds": self.training_duration_seconds,
            "samples_trained": self.samples_trained,
            "training_timestamp": self.training_timestamp.isoformat()
        }


@dataclass
class AgentTrainingProfile:
    """Training profile for specific agent types"""
    agent_type: str
    capabilities: List[str]
    training_objectives: List[TrainingObjective]
    required_patterns: List[PatternType]
    performance_targets: Dict[str, float]
    training_priority: int = 1  # 1 = highest priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "training_objectives": [obj.value for obj in self.training_objectives],
            "required_patterns": [pattern.value for pattern in self.required_patterns],
            "performance_targets": self.performance_targets,
            "training_priority": self.training_priority
        }


class AgentTrainingPipeline:
    """
    Comprehensive training pipeline for eFab AI agents
    
    Features:
    - Pattern-based learning from Beverly Knits success cases
    - Multi-objective training for different agent capabilities
    - Continuous improvement through implementation feedback
    - Industry adaptation and transfer learning
    - Performance monitoring and model optimization
    """
    
    def __init__(self, training_data_path: str = None):
        """Initialize training pipeline"""
        self.training_data_path = training_data_path or "src/ai_agents/training/data"
        self.models_path = Path(self.training_data_path) / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Training dataset
        self.training_dataset: Optional[TrainingDataset] = None
        self.feature_matrices: Dict[TrainingObjective, np.ndarray] = {}
        self.target_vectors: Dict[TrainingObjective, np.ndarray] = {}
        
        # Trained models
        self.trained_models: Dict[TrainingObjective, Any] = {}
        self.model_performance: Dict[TrainingObjective, Dict[str, float]] = {}
        
        # Agent training profiles
        self.agent_profiles = self._initialize_agent_profiles()
        
        # Feature extractors
        self.feature_extractors = self._initialize_feature_extractors()
        
        logger.info("AI Agent Training Pipeline initialized")
    
    def _initialize_agent_profiles(self) -> Dict[str, AgentTrainingProfile]:
        """Initialize training profiles for different agent types"""
        profiles = {
            "implementation_pm": AgentTrainingProfile(
                agent_type="ImplementationProjectManagerAgent",
                capabilities=["customer_assessment", "timeline_prediction", "risk_assessment"],
                training_objectives=[
                    TrainingObjective.COMPLEXITY_ASSESSMENT,
                    TrainingObjective.TIMELINE_ESTIMATION,
                    TrainingObjective.RISK_PREDICTION
                ],
                required_patterns=[PatternType.BUSINESS_LOGIC, PatternType.WORKFLOW_PATTERN],
                performance_targets={"accuracy": 0.85, "timeline_precision": 0.8},
                training_priority=1
            ),
            
            "data_migration": AgentTrainingProfile(
                agent_type="DataMigrationIntelligenceAgent",
                capabilities=["schema_mapping", "data_transformation", "quality_validation"],
                training_objectives=[
                    TrainingObjective.PATTERN_RECOGNITION,
                    TrainingObjective.ERROR_CLASSIFICATION,
                    TrainingObjective.PERFORMANCE_PREDICTION
                ],
                required_patterns=[PatternType.DATA_TRANSFORMATION, PatternType.ERROR_HANDLING],
                performance_targets={"mapping_accuracy": 0.95, "error_detection": 0.9},
                training_priority=1
            ),
            
            "configuration_generator": AgentTrainingProfile(
                agent_type="ConfigurationGenerationAgent",
                capabilities=["system_configuration", "business_rule_adaptation"],
                training_objectives=[
                    TrainingObjective.INDUSTRY_ADAPTATION,
                    TrainingObjective.OPTIMIZATION_SELECTION
                ],
                required_patterns=[PatternType.BUSINESS_LOGIC, PatternType.OPTIMIZATION_STRATEGY],
                performance_targets={"configuration_accuracy": 0.9, "optimization_improvement": 0.3},
                training_priority=2
            ),
            
            "furniture_specialist": AgentTrainingProfile(
                agent_type="FurnitureManufacturingAgent",
                capabilities=["custom_configuration", "material_optimization", "waste_reduction"],
                training_objectives=[
                    TrainingObjective.INDUSTRY_ADAPTATION,
                    TrainingObjective.OPTIMIZATION_SELECTION
                ],
                required_patterns=[PatternType.BUSINESS_LOGIC, PatternType.OPTIMIZATION_STRATEGY],
                performance_targets={"industry_accuracy": 0.85, "waste_reduction": 0.2},
                training_priority=3
            ),
            
            "performance_optimizer": AgentTrainingProfile(
                agent_type="PerformanceOptimizationAgent",
                capabilities=["performance_monitoring", "optimization_recommendation"],
                training_objectives=[
                    TrainingObjective.PERFORMANCE_PREDICTION,
                    TrainingObjective.OPTIMIZATION_SELECTION
                ],
                required_patterns=[PatternType.OPTIMIZATION_STRATEGY, PatternType.PERFORMANCE_METRIC],
                performance_targets={"prediction_accuracy": 0.8, "optimization_effectiveness": 0.25},
                training_priority=2
            )
        }
        
        return profiles
    
    def _initialize_feature_extractors(self) -> Dict[TrainingObjective, callable]:
        """Initialize feature extraction functions for different objectives"""
        return {
            TrainingObjective.COMPLEXITY_ASSESSMENT: self._extract_complexity_features,
            TrainingObjective.TIMELINE_ESTIMATION: self._extract_timeline_features,
            TrainingObjective.RISK_PREDICTION: self._extract_risk_features,
            TrainingObjective.PATTERN_RECOGNITION: self._extract_pattern_features,
            TrainingObjective.OPTIMIZATION_SELECTION: self._extract_optimization_features,
            TrainingObjective.ERROR_CLASSIFICATION: self._extract_error_features,
            TrainingObjective.INDUSTRY_ADAPTATION: self._extract_industry_features,
            TrainingObjective.PERFORMANCE_PREDICTION: self._extract_performance_features
        }
    
    async def load_training_data(self, dataset_path: str = None) -> bool:
        """Load training dataset from Beverly Knits patterns"""
        try:
            if dataset_path:
                extractor = BeverlyPatternExtractor()
                self.training_dataset = extractor.load_training_dataset(dataset_path)
            else:
                # Extract fresh data from Beverly Knits
                logger.info("🔄 Extracting fresh training data from Beverly Knits...")
                extractor = BeverlyPatternExtractor()
                self.training_dataset = await extractor.extract_all_patterns()
                
                # Save the dataset for future use
                dataset_path = Path(self.training_data_path) / "beverly_training_dataset.json"
                extractor.save_training_dataset(self.training_dataset, str(dataset_path))
            
            logger.info(f"✅ Training data loaded: {len(self.training_dataset.patterns)} patterns")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            return False
    
    async def prepare_training_features(self):
        """Prepare feature matrices and target vectors for all training objectives"""
        if not self.training_dataset:
            raise ValueError("Training dataset not loaded")
        
        logger.info("🔧 Preparing training features for all objectives...")
        
        for objective in TrainingObjective:
            logger.info(f"Preparing features for: {objective.value}")
            
            # Extract features using appropriate extractor
            extractor = self.feature_extractors.get(objective)
            if extractor:
                features, targets = extractor(self.training_dataset)
                
                if len(features) > 0 and len(targets) > 0:
                    self.feature_matrices[objective] = np.array(features)
                    self.target_vectors[objective] = np.array(targets)
                    logger.info(f"  ✅ {objective.value}: {len(features)} samples, {len(features[0])} features")
                else:
                    logger.warning(f"  ⚠️  {objective.value}: No valid samples found")
            else:
                logger.warning(f"  ❌ {objective.value}: No feature extractor found")
    
    def _extract_complexity_features(self, dataset: TrainingDataset) -> Tuple[List[List[float]], List[float]]:
        """Extract features for complexity assessment"""
        features = []
        targets = []
        
        for pattern in dataset.patterns:
            if pattern.pattern_type == PatternType.BUSINESS_LOGIC:
                feature_vector = [
                    pattern.complexity_score,
                    pattern.applicability_score,
                    len(pattern.pattern_data.get("business_concepts", [])),
                    len(pattern.pattern_data.get("calculation_patterns", [])),
                    len(pattern.pattern_data.get("data_dependencies", [])),
                    pattern.pattern_data.get("complexity_metrics", {}).get("line_count", 0) / 100,
                    1.0 if pattern.pattern_data.get("complexity_metrics", {}).get("has_loops") else 0.0,
                    1.0 if pattern.pattern_data.get("complexity_metrics", {}).get("has_conditions") else 0.0,
                    1.0 if pattern.pattern_data.get("complexity_metrics", {}).get("uses_pandas") else 0.0
                ]
                
                features.append(feature_vector)
                targets.append(pattern.complexity_score)
        
        return features, targets
    
    def _extract_timeline_features(self, dataset: TrainingDataset) -> Tuple[List[List[float]], List[float]]:
        """Extract features for timeline estimation"""
        features = []
        targets = []
        
        # Use workflow patterns to estimate implementation timelines
        for pattern in dataset.patterns:
            if pattern.pattern_type == PatternType.WORKFLOW_PATTERN:
                feature_vector = [
                    pattern.complexity_score,
                    pattern.pattern_data.get("method_count", 0),
                    len(pattern.pattern_data.get("orchestration_patterns", [])),
                    1.0 if "sequential" in pattern.pattern_data.get("orchestration_patterns", []) else 0.0,
                    1.0 if "parallel" in pattern.pattern_data.get("orchestration_patterns", []) else 0.0,
                    1.0 if "error_handling" in pattern.pattern_data.get("orchestration_patterns", []) else 0.0,
                    pattern.applicability_score
                ]
                
                # Estimate timeline based on complexity (simplified)
                estimated_weeks = 6 + (pattern.complexity_score * 3)  # 6-9 weeks based on complexity
                
                features.append(feature_vector)
                targets.append(estimated_weeks)
        
        return features, targets
    
    def _extract_risk_features(self, dataset: TrainingDataset) -> Tuple[List[List[float]], List[int]]:
        """Extract features for risk prediction"""
        features = []
        targets = []
        
        # Use failure cases to train risk prediction
        for failure in dataset.failure_cases:
            feature_vector = [
                1.0 if "integration" in failure.get("failure_type", "") else 0.0,
                1.0 if "data" in failure.get("failure_type", "") else 0.0,
                1.0 if "performance" in failure.get("failure_type", "") else 0.0,
                len(failure.get("lessons_learned", [])),
                np.random.uniform(0.3, 0.8)  # Simulated complexity score
            ]
            
            # Risk level: 0 = low, 1 = medium, 2 = high
            risk_level = 2  # High risk for known failure cases
            
            features.append(feature_vector)
            targets.append(risk_level)
        
        # Add some low-risk samples from success outcomes
        for success in dataset.success_outcomes:
            feature_vector = [
                0.0, 0.0, 0.0,  # No specific failure indicators
                1.0,  # One success metric
                np.random.uniform(0.1, 0.4)  # Low complexity
            ]
            
            risk_level = 0  # Low risk for success cases
            
            features.append(feature_vector)
            targets.append(risk_level)
        
        return features, targets
    
    def _extract_pattern_features(self, dataset: TrainingDataset) -> Tuple[List[List[float]], List[int]]:
        """Extract features for pattern recognition"""
        features = []
        targets = []
        
        pattern_type_mapping = {
            PatternType.BUSINESS_LOGIC: 0,
            PatternType.WORKFLOW_PATTERN: 1,
            PatternType.OPTIMIZATION_STRATEGY: 2,
            PatternType.DATA_TRANSFORMATION: 3,
            PatternType.ERROR_HANDLING: 4
        }
        
        for pattern in dataset.patterns:
            if pattern.pattern_type in pattern_type_mapping:
                feature_vector = [
                    pattern.complexity_score,
                    pattern.applicability_score,
                    pattern.usage_frequency,
                    len(pattern.industry_tags),
                    len(pattern.pattern_data)
                ]
                
                features.append(feature_vector)
                targets.append(pattern_type_mapping[pattern.pattern_type])
        
        return features, targets
    
    def _extract_optimization_features(self, dataset: TrainingDataset) -> Tuple[List[List[float]], List[float]]:
        """Extract features for optimization selection"""
        features = []
        targets = []
        
        for pattern in dataset.patterns:
            if pattern.pattern_type == PatternType.OPTIMIZATION_STRATEGY:
                techniques = pattern.pattern_data.get("optimization_techniques", [])
                feature_vector = [
                    1.0 if "caching" in techniques else 0.0,
                    1.0 if "parallel" in techniques else 0.0,
                    1.0 if "vectorization" in techniques else 0.0,
                    1.0 if "indexing" in techniques else 0.0,
                    1.0 if "streaming" in techniques else 0.0,
                    len(techniques),
                    pattern.complexity_score,
                    pattern.applicability_score
                ]
                
                # Estimate performance improvement (simplified)
                performance_improvement = len(techniques) * 0.1  # 10% per technique
                
                features.append(feature_vector)
                targets.append(performance_improvement)
        
        return features, targets
    
    def _extract_error_features(self, dataset: TrainingDataset) -> Tuple[List[List[float]], List[int]]:
        """Extract features for error classification"""
        features = []
        targets = []
        
        error_type_mapping = {
            "integration": 0,
            "data": 1,
            "performance": 2,
            "configuration": 3,
            "user": 4
        }
        
        for failure in dataset.failure_cases:
            failure_type = failure.get("failure_type", "")
            for error_category, error_id in error_type_mapping.items():
                if error_category in failure_type:
                    feature_vector = [
                        len(failure.get("description", "")),
                        len(failure.get("lessons_learned", [])),
                        1.0 if "timeout" in failure.get("description", "") else 0.0,
                        1.0 if "connection" in failure.get("description", "") else 0.0,
                        1.0 if "validation" in failure.get("description", "") else 0.0
                    ]
                    
                    features.append(feature_vector)
                    targets.append(error_id)
                    break
        
        return features, targets
    
    def _extract_industry_features(self, dataset: TrainingDataset) -> Tuple[List[List[float]], List[int]]:
        """Extract features for industry adaptation"""
        features = []
        targets = []
        
        industry_mapping = {"furniture": 0, "injection_molding": 1, "electrical": 2}
        
        for industry, patterns in dataset.industry_mappings.items():
            if "to_" in industry:  # Skip mapping entries
                continue
                
            industry_id = industry_mapping.get(industry, 0)
            
            for pattern_name in patterns:
                feature_vector = [
                    len(pattern_name),
                    1.0 if "material" in pattern_name else 0.0,
                    1.0 if "production" in pattern_name else 0.0,
                    1.0 if "quality" in pattern_name else 0.0,
                    1.0 if "planning" in pattern_name else 0.0
                ]
                
                features.append(feature_vector)
                targets.append(industry_id)
        
        return features, targets
    
    def _extract_performance_features(self, dataset: TrainingDataset) -> Tuple[List[List[float]], List[float]]:
        """Extract features for performance prediction"""
        features = []
        targets = []
        
        # Use performance baselines to train prediction models
        baseline_metrics = [
            ("api_response_time_ms", [200, 150, 180, 220, 160]),
            ("system_uptime_percentage", [99.7, 99.9, 99.5, 99.8, 99.6]),
            ("user_adoption_rate", [94.5, 96.2, 92.8, 95.1, 93.7])
        ]
        
        for metric_name, values in baseline_metrics:
            for i, value in enumerate(values):
                feature_vector = [
                    i + 1,  # Implementation phase
                    np.random.uniform(0.3, 0.8),  # System complexity
                    np.random.uniform(0.5, 1.0),  # Team experience
                    np.random.uniform(0.6, 0.9),  # Process maturity
                    len(dataset.patterns) / 100  # Pattern richness
                ]
                
                features.append(feature_vector)
                targets.append(value)
        
        return features, targets
    
    async def train_agent_models(self, agent_type: str = None) -> Dict[str, TrainingResult]:
        """Train models for specific agent type or all agents"""
        if not self.feature_matrices:
            await self.prepare_training_features()
        
        results = {}
        
        # Determine which agents to train
        agents_to_train = [agent_type] if agent_type else list(self.agent_profiles.keys())
        
        for agent_name in agents_to_train:
            if agent_name not in self.agent_profiles:
                logger.warning(f"Unknown agent type: {agent_name}")
                continue
            
            profile = self.agent_profiles[agent_name]
            logger.info(f"🎯 Training models for: {profile.agent_type}")
            
            agent_results = {}
            
            for objective in profile.training_objectives:
                if objective not in self.feature_matrices:
                    logger.warning(f"No training data for objective: {objective.value}")
                    continue
                
                logger.info(f"  🔧 Training {objective.value} model...")
                
                start_time = datetime.now()
                result = await self._train_objective_model(objective, profile)
                
                if result:
                    agent_results[objective.value] = result
                    logger.info(f"  ✅ {objective.value}: {result.model_performance}")
                else:
                    logger.error(f"  ❌ {objective.value}: Training failed")
            
            results[agent_name] = agent_results
        
        return results
    
    async def _train_objective_model(
        self, 
        objective: TrainingObjective, 
        profile: AgentTrainingProfile
    ) -> Optional[TrainingResult]:
        """Train model for specific objective"""
        try:
            features = self.feature_matrices[objective]
            targets = self.target_vectors[objective]
            
            if len(features) < 10:  # Need minimum samples
                logger.warning(f"Insufficient samples for {objective.value}: {len(features)}")
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Select model type based on objective
            model = self._select_model(objective)
            
            # Train model
            start_time = datetime.now()
            
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            
            training_duration = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                if objective in [TrainingObjective.RISK_PREDICTION, TrainingObjective.ERROR_CLASSIFICATION, 
                               TrainingObjective.PATTERN_RECOGNITION, TrainingObjective.INDUSTRY_ADAPTATION]:
                    # Classification metrics
                    performance = {"accuracy": accuracy_score(y_test, y_pred)}
                else:
                    # Regression metrics
                    performance = {"mse": mean_squared_error(y_test, y_pred)}
            else:
                performance = {"trained": 1.0}
            
            # Feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = [f"feature_{i}" for i in range(len(features[0]))]
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Save model
            model_filename = f"{profile.agent_type}_{objective.value}.joblib"
            model_path = self.models_path / model_filename
            joblib.dump(model, model_path)
            
            # Store trained model
            self.trained_models[objective] = model
            self.model_performance[objective] = performance
            
            return TrainingResult(
                configuration=TrainingConfiguration(
                    objective=objective,
                    model_type=ModelType.CLASSIFICATION if objective in [
                        TrainingObjective.RISK_PREDICTION, TrainingObjective.ERROR_CLASSIFICATION,
                        TrainingObjective.PATTERN_RECOGNITION, TrainingObjective.INDUSTRY_ADAPTATION
                    ] else ModelType.REGRESSION,
                    features=[f"feature_{i}" for i in range(len(features[0]))],
                    target_variable=objective.value
                ),
                model_performance=performance,
                feature_importance=feature_importance,
                validation_metrics=performance,  # Simplified
                model_path=str(model_path),
                training_duration_seconds=training_duration,
                samples_trained=len(features)
            )
            
        except Exception as e:
            logger.error(f"Failed to train model for {objective.value}: {str(e)}")
            return None
    
    def _select_model(self, objective: TrainingObjective):
        """Select appropriate model for training objective"""
        if objective in [
            TrainingObjective.RISK_PREDICTION, 
            TrainingObjective.ERROR_CLASSIFICATION,
            TrainingObjective.PATTERN_RECOGNITION,
            TrainingObjective.INDUSTRY_ADAPTATION
        ]:
            # Classification models
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            # Regression models
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
    
    def predict_with_trained_model(
        self, 
        objective: TrainingObjective, 
        features: List[float]
    ) -> Optional[Union[float, int]]:
        """Make prediction using trained model"""
        if objective not in self.trained_models:
            logger.warning(f"No trained model for {objective.value}")
            return None
        
        try:
            model = self.trained_models[objective]
            prediction = model.predict([features])[0]
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed for {objective.value}: {str(e)}")
            return None
    
    async def evaluate_agent_performance(self, agent_type: str) -> Dict[str, Any]:
        """Evaluate trained agent performance against targets"""
        if agent_type not in self.agent_profiles:
            return {"error": "Unknown agent type"}
        
        profile = self.agent_profiles[agent_type]
        evaluation = {
            "agent_type": agent_type,
            "training_objectives": [obj.value for obj in profile.training_objectives],
            "performance_evaluation": {},
            "meets_targets": True,
            "recommendations": []
        }
        
        for objective in profile.training_objectives:
            if objective in self.model_performance:
                performance = self.model_performance[objective]
                evaluation["performance_evaluation"][objective.value] = performance
                
                # Check against targets (simplified)
                target_key = objective.value.lower() + "_accuracy"
                if target_key in profile.performance_targets:
                    target = profile.performance_targets[target_key]
                    actual = performance.get("accuracy", 0.0)
                    
                    if actual < target:
                        evaluation["meets_targets"] = False
                        evaluation["recommendations"].append(
                            f"Improve {objective.value} performance: {actual:.3f} < {target:.3f}"
                        )
        
        return evaluation
    
    def save_training_results(self, results: Dict[str, Any], output_path: str):
        """Save training results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        for agent_name, agent_results in results.items():
            serializable_results[agent_name] = {}
            for objective_name, result in agent_results.items():
                if isinstance(result, TrainingResult):
                    serializable_results[agent_name][objective_name] = result.to_dict()
                else:
                    serializable_results[agent_name][objective_name] = result
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"💾 Training results saved to: {output_file}")


async def main():
    """Main training pipeline execution"""
    try:
        # Initialize training pipeline
        pipeline = AgentTrainingPipeline()
        
        # Load training data
        logger.info("📂 Loading training data...")
        await pipeline.load_training_data()
        
        # Train all agent models
        logger.info("🎯 Starting agent training...")
        results = await pipeline.train_agent_models()
        
        # Save results
        results_path = pipeline.training_data_path + "/training_results.json"
        pipeline.save_training_results(results, results_path)
        
        # Evaluate performance
        logger.info("📊 Evaluating agent performance...")
        for agent_type in pipeline.agent_profiles.keys():
            evaluation = await pipeline.evaluate_agent_performance(agent_type)
            logger.info(f"Agent {agent_type}: {'✅ PASS' if evaluation['meets_targets'] else '⚠️  NEEDS IMPROVEMENT'}")
        
        print("✅ Agent training pipeline complete!")
        print(f"📊 Trained {len(results)} agent types")
        print(f"💾 Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())