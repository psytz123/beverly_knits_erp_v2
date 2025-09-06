#!/usr/bin/env python3
"""
Training Data Generation Script for Agent Training
Creates synthetic and historical training datasets for all agent types
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import argparse

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)


class TrainingDataGenerator:
    """Generates training data for agent learning"""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            self.base_path = Path(__file__).parent.parent.parent
        else:
            self.base_path = Path(base_path)
        
        self.data_path = self.base_path / "data" / "production" / "5"
        self.training_path = self.base_path / "data" / "agent_training"
        self.scenarios_path = self.training_path / "scenarios"
        
        # Create directories if they don't exist
        self.scenarios_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Training data generator initialized")
        print(f"Base path: {self.base_path}")
        print(f"Scenarios path: {self.scenarios_path}")
    
    def generate_inventory_scenarios(self) -> Dict[str, Any]:
        """Generate training scenarios for Inventory Intelligence Agent"""
        
        scenarios = {
            "name": "inventory_training",
            "description": "Training scenarios for inventory management",
            "cases": []
        }
        
        # Scenario 1: Normal Planning Balance Calculation
        for i in range(50):
            physical = random.randint(1000, 10000)
            allocated = random.randint(500, min(physical, 5000))
            on_order = random.randint(0, 3000)
            planning_balance = physical - allocated + on_order
            
            scenarios["cases"].append({
                "scenario": "normal_balance",
                "input": {
                    "yarn_id": f"Y{1000 + i:04d}",
                    "physical_inventory": physical,
                    "allocated": allocated,
                    "on_order": on_order
                },
                "expected": {
                    "planning_balance": planning_balance,
                    "status": "normal" if planning_balance > 500 else "low"
                }
            })
        
        # Scenario 2: Shortage Detection
        for i in range(30):
            physical = random.randint(0, 500)
            allocated = random.randint(1000, 5000)
            on_order = random.randint(0, 500)
            planning_balance = physical - allocated + on_order
            
            scenarios["cases"].append({
                "scenario": "shortage",
                "input": {
                    "yarn_id": f"Y{2000 + i:04d}",
                    "physical_inventory": physical,
                    "allocated": allocated,
                    "on_order": on_order,
                    "weekly_demand": random.randint(200, 500)
                },
                "expected": {
                    "planning_balance": planning_balance,
                    "status": "critical_shortage",
                    "action": "urgent_reorder"
                }
            })
        
        # Scenario 3: Multi-level BOM Netting
        for i in range(20):
            style_demand = random.randint(100, 1000)
            yarn_per_style = random.uniform(0.5, 2.0)
            total_yarn_needed = style_demand * yarn_per_style
            
            scenarios["cases"].append({
                "scenario": "bom_netting",
                "input": {
                    "style_id": f"ST{3000 + i:04d}",
                    "style_demand": style_demand,
                    "bom_entries": [
                        {
                            "yarn_id": f"Y{3000 + j:04d}",
                            "quantity_per_style": yarn_per_style,
                            "available_inventory": random.randint(500, 2000)
                        } for j in range(3)
                    ]
                },
                "expected": {
                    "total_yarn_requirement": total_yarn_needed,
                    "can_fulfill": random.choice([True, False]),
                    "shortage_yarns": []
                }
            })
        
        return scenarios
    
    def generate_forecast_scenarios(self) -> Dict[str, Any]:
        """Generate training scenarios for Forecast Intelligence Agent"""
        
        scenarios = {
            "name": "forecast_training",
            "description": "Training scenarios for demand forecasting",
            "cases": []
        }
        
        # Generate time series data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        # Scenario 1: Trend Forecasting
        for i in range(30):
            base_demand = random.randint(100, 500)
            trend = random.uniform(-5, 10)
            
            historical_data = []
            for j in range(90):  # 90 days of history
                demand = base_demand + (trend * j) + np.random.normal(0, 20)
                historical_data.append(max(0, demand))
            
            # Expected forecast (simplified)
            next_30_days = [base_demand + (trend * (90 + k)) for k in range(30)]
            
            scenarios["cases"].append({
                "scenario": "trend_forecast",
                "input": {
                    "product_id": f"P{4000 + i:04d}",
                    "historical_demand": historical_data[-30:],  # Last 30 days
                    "forecast_horizon": 30
                },
                "expected": {
                    "forecast": [max(0, int(v + np.random.normal(0, 10))) for v in next_30_days[:10]],
                    "confidence": 0.85,
                    "trend_direction": "increasing" if trend > 0 else "decreasing"
                }
            })
        
        # Scenario 2: Seasonal Pattern Recognition
        for i in range(20):
            base_demand = random.randint(200, 400)
            seasonal_amplitude = random.randint(50, 150)
            
            historical_data = []
            for j in range(365):  # Full year
                seasonal_factor = np.sin(2 * np.pi * j / 365) * seasonal_amplitude
                demand = base_demand + seasonal_factor + np.random.normal(0, 30)
                historical_data.append(max(0, demand))
            
            scenarios["cases"].append({
                "scenario": "seasonal_forecast",
                "input": {
                    "product_id": f"P{5000 + i:04d}",
                    "historical_demand": historical_data[-90:],  # Last quarter
                    "forecast_horizon": 90,
                    "include_seasonality": True
                },
                "expected": {
                    "has_seasonality": True,
                    "seasonal_period": 365,
                    "peak_season": "summer" if i % 2 == 0 else "winter"
                }
            })
        
        # Scenario 3: Anomaly Handling
        for i in range(20):
            normal_demand = random.randint(100, 300)
            historical_data = [normal_demand + np.random.normal(0, 20) for _ in range(60)]
            
            # Insert anomalies
            anomaly_indices = random.sample(range(60), 5)
            for idx in anomaly_indices:
                historical_data[idx] *= random.choice([0.1, 3.0])  # Outliers
            
            scenarios["cases"].append({
                "scenario": "anomaly_detection",
                "input": {
                    "product_id": f"P{6000 + i:04d}",
                    "historical_demand": historical_data,
                    "detect_anomalies": True
                },
                "expected": {
                    "anomalies_detected": len(anomaly_indices),
                    "cleaned_forecast": normal_demand,
                    "confidence_after_cleaning": 0.90
                }
            })
        
        return scenarios
    
    def generate_production_scenarios(self) -> Dict[str, Any]:
        """Generate training scenarios for Production Planning Agent"""
        
        scenarios = {
            "name": "production_training",
            "description": "Training scenarios for production planning",
            "cases": []
        }
        
        # Define work centers and machines
        work_centers = [f"{i}.{j:02d}.{k:02d}.{l}" 
                       for i in range(1, 10) 
                       for j in [30, 32, 34, 36, 38] 
                       for k in [18, 20, 22, 24]
                       for l in ['F', 'M', 'C', 'V']][:91]
        
        machines = list(range(100, 385))
        
        # Scenario 1: Machine Assignment
        for i in range(40):
            order_quantity = random.randint(1000, 10000)
            required_hours = order_quantity / random.uniform(50, 200)
            
            scenarios["cases"].append({
                "scenario": "machine_assignment",
                "input": {
                    "order_id": f"ORD{7000 + i:04d}",
                    "style_id": f"ST{7000 + i:04d}",
                    "quantity": order_quantity,
                    "work_center_pattern": random.choice(work_centers),
                    "due_date": (datetime.now() + timedelta(days=random.randint(7, 30))).isoformat()
                },
                "expected": {
                    "assigned_machine": random.choice(machines),
                    "estimated_hours": round(required_hours, 2),
                    "can_meet_deadline": random.choice([True, False]),
                    "utilization_after": random.uniform(0.7, 0.95)
                }
            })
        
        # Scenario 2: Capacity Planning
        for i in range(30):
            total_orders = random.randint(5, 20)
            total_capacity = random.randint(1000, 2000)
            total_demand = random.randint(800, 2200)
            
            scenarios["cases"].append({
                "scenario": "capacity_planning",
                "input": {
                    "planning_period": f"Week{i + 1}",
                    "orders_count": total_orders,
                    "total_demand_hours": total_demand,
                    "available_capacity_hours": total_capacity,
                    "machines_available": random.randint(200, 285)
                },
                "expected": {
                    "capacity_utilization": min(1.0, total_demand / total_capacity),
                    "is_overloaded": total_demand > total_capacity,
                    "recommended_action": "add_shifts" if total_demand > total_capacity else "normal",
                    "bottleneck_work_centers": []
                }
            })
        
        # Scenario 3: Schedule Optimization
        for i in range(20):
            num_orders = random.randint(10, 30)
            orders = []
            
            for j in range(num_orders):
                orders.append({
                    "order_id": f"ORD{8000 + i * 30 + j:04d}",
                    "quantity": random.randint(500, 5000),
                    "priority": random.choice(["high", "medium", "low"]),
                    "setup_time": random.uniform(0.5, 2.0),
                    "run_time": random.uniform(10, 50)
                })
            
            scenarios["cases"].append({
                "scenario": "schedule_optimization",
                "input": {
                    "schedule_id": f"SCH{9000 + i:04d}",
                    "orders": orders,
                    "optimization_goal": random.choice(["minimize_setup", "maximize_throughput", "balance_load"])
                },
                "expected": {
                    "optimized_sequence": list(range(num_orders)),
                    "total_setup_time": sum(o["setup_time"] for o in orders),
                    "total_run_time": sum(o["run_time"] for o in orders),
                    "efficiency_gain": random.uniform(0.05, 0.25)
                }
            })
        
        return scenarios
    
    def generate_yarn_substitution_scenarios(self) -> Dict[str, Any]:
        """Generate training scenarios for Yarn Substitution Agent"""
        
        scenarios = {
            "name": "yarn_substitution_training",
            "description": "Training scenarios for yarn substitution",
            "cases": []
        }
        
        # Define yarn properties
        colors = ["White", "Black", "Navy", "Red", "Blue", "Green", "Gray", "Brown"]
        materials = ["Cotton", "Polyester", "Wool", "Nylon", "Acrylic", "Blend"]
        weights = ["Light", "Medium", "Heavy"]
        
        # Scenario 1: Direct Substitution
        for i in range(30):
            original_yarn = {
                "id": f"Y{10000 + i:04d}",
                "color": random.choice(colors),
                "material": random.choice(materials),
                "weight": random.choice(weights),
                "price": random.uniform(5, 50)
            }
            
            # Generate similar substitute
            substitute_yarn = {
                "id": f"Y{11000 + i:04d}",
                "color": original_yarn["color"] if i % 3 != 0 else random.choice(colors),
                "material": original_yarn["material"] if i % 2 == 0 else random.choice(materials),
                "weight": original_yarn["weight"],
                "price": original_yarn["price"] * random.uniform(0.8, 1.2)
            }
            
            scenarios["cases"].append({
                "scenario": "direct_substitution",
                "input": {
                    "original_yarn": original_yarn,
                    "candidate_yarn": substitute_yarn,
                    "quality_tolerance": 0.05
                },
                "expected": {
                    "is_compatible": original_yarn["material"] == substitute_yarn["material"],
                    "compatibility_score": random.uniform(0.7, 1.0),
                    "quality_impact": random.uniform(0, 0.1),
                    "cost_impact": (substitute_yarn["price"] - original_yarn["price"]) / original_yarn["price"]
                }
            })
        
        # Scenario 2: Multi-criteria Substitution
        for i in range(20):
            original_yarn_id = f"Y{12000 + i:04d}"
            num_candidates = random.randint(3, 8)
            
            candidates = []
            for j in range(num_candidates):
                candidates.append({
                    "id": f"Y{13000 + i * 10 + j:04d}",
                    "similarity_score": random.uniform(0.5, 1.0),
                    "availability": random.randint(0, 5000),
                    "lead_time": random.randint(1, 30),
                    "price_ratio": random.uniform(0.7, 1.3)
                })
            
            scenarios["cases"].append({
                "scenario": "multi_criteria_selection",
                "input": {
                    "original_yarn_id": original_yarn_id,
                    "required_quantity": random.randint(1000, 3000),
                    "candidates": candidates,
                    "deadline_days": random.randint(7, 21)
                },
                "expected": {
                    "best_substitute": max(candidates, key=lambda x: x["similarity_score"])["id"],
                    "selection_confidence": random.uniform(0.7, 0.95),
                    "risk_assessment": random.choice(["low", "medium", "high"])
                }
            })
        
        # Scenario 3: Emergency Substitution
        for i in range(15):
            scenarios["cases"].append({
                "scenario": "emergency_substitution",
                "input": {
                    "yarn_id": f"Y{14000 + i:04d}",
                    "shortage_quantity": random.randint(500, 2000),
                    "production_deadline": (datetime.now() + timedelta(hours=random.randint(24, 72))).isoformat(),
                    "quality_priority": random.choice(["critical", "important", "flexible"])
                },
                "expected": {
                    "immediate_options": random.randint(1, 5),
                    "recommended_action": random.choice(["substitute", "partial_substitute", "delay_production"]),
                    "quality_risk": random.choice(["minimal", "moderate", "significant"]),
                    "delivery_impact": random.choice(["none", "minor_delay", "major_delay"])
                }
            })
        
        return scenarios
    
    def generate_quality_scenarios(self) -> Dict[str, Any]:
        """Generate training scenarios for Quality Assurance Agent"""
        
        scenarios = {
            "name": "quality_training",
            "description": "Training scenarios for quality assurance",
            "cases": []
        }
        
        # Scenario 1: Data Validation
        for i in range(30):
            data_records = random.randint(1000, 10000)
            invalid_records = random.randint(0, int(data_records * 0.1))
            
            scenarios["cases"].append({
                "scenario": "data_validation",
                "input": {
                    "dataset_id": f"DS{15000 + i:04d}",
                    "total_records": data_records,
                    "validation_rules": [
                        "non_null_check",
                        "range_check",
                        "format_check",
                        "referential_integrity"
                    ]
                },
                "expected": {
                    "valid_records": data_records - invalid_records,
                    "invalid_records": invalid_records,
                    "validation_rate": (data_records - invalid_records) / data_records,
                    "critical_issues": random.randint(0, 5),
                    "data_quality_score": random.uniform(0.85, 1.0)
                }
            })
        
        # Scenario 2: Anomaly Detection
        for i in range(25):
            metric_value = random.uniform(50, 150)
            is_anomaly = random.choice([True, False])
            
            if is_anomaly:
                metric_value *= random.choice([0.3, 3.0])  # Make it anomalous
            
            scenarios["cases"].append({
                "scenario": "anomaly_detection",
                "input": {
                    "metric_id": f"KPI{16000 + i:04d}",
                    "current_value": metric_value,
                    "historical_mean": 100,
                    "historical_std": 20,
                    "detection_threshold": 2.5  # Z-score threshold
                },
                "expected": {
                    "is_anomaly": is_anomaly,
                    "z_score": abs((metric_value - 100) / 20),
                    "severity": "high" if is_anomaly else "none",
                    "recommended_action": "investigate" if is_anomaly else "monitor"
                }
            })
        
        # Scenario 3: Performance Monitoring
        for i in range(20):
            response_times = [random.uniform(50, 500) for _ in range(100)]
            error_count = random.randint(0, 10)
            
            scenarios["cases"].append({
                "scenario": "performance_monitoring",
                "input": {
                    "system_id": f"SYS{17000 + i:04d}",
                    "monitoring_period": "last_hour",
                    "response_times_ms": response_times[:10],  # Sample
                    "total_requests": 1000,
                    "failed_requests": error_count
                },
                "expected": {
                    "avg_response_time": np.mean(response_times),
                    "p95_response_time": np.percentile(response_times, 95),
                    "error_rate": error_count / 1000,
                    "performance_status": "healthy" if np.mean(response_times) < 200 else "degraded",
                    "alert_required": error_count > 5 or np.mean(response_times) > 300
                }
            })
        
        return scenarios
    
    def generate_certification_scenarios(self) -> Dict[str, Any]:
        """Generate comprehensive certification test scenarios"""
        
        scenarios = {
            "name": "certification",
            "description": "Certification test suite for all agents",
            "cases": []
        }
        
        # Mix of all scenario types for comprehensive testing
        test_types = [
            "inventory_critical",
            "forecast_accuracy",
            "production_optimization",
            "yarn_emergency",
            "quality_compliance"
        ]
        
        for i in range(100):  # 100 certification tests
            test_type = test_types[i % len(test_types)]
            
            if test_type == "inventory_critical":
                scenarios["cases"].append({
                    "scenario": "inventory_critical_test",
                    "test_id": f"CERT{20000 + i:04d}",
                    "input": {
                        "test_type": "critical_shortage",
                        "yarn_id": f"Y{20000 + i:04d}",
                        "current_stock": random.randint(0, 100),
                        "committed_orders": random.randint(500, 1000),
                        "production_impact": "high"
                    },
                    "expected": {
                        "action": "emergency_procurement",
                        "response_time_ms": 50,
                        "accuracy_required": 0.99
                    }
                })
            
            elif test_type == "forecast_accuracy":
                scenarios["cases"].append({
                    "scenario": "forecast_accuracy_test",
                    "test_id": f"CERT{20000 + i:04d}",
                    "input": {
                        "test_type": "forecast_validation",
                        "historical_actuals": [random.randint(100, 500) for _ in range(30)],
                        "previous_forecast": [random.randint(100, 500) for _ in range(30)],
                        "mape_threshold": 0.15
                    },
                    "expected": {
                        "mape": random.uniform(0.08, 0.14),
                        "bias": random.uniform(-0.05, 0.05),
                        "pass_criteria": True
                    }
                })
            
            elif test_type == "production_optimization":
                scenarios["cases"].append({
                    "scenario": "production_optimization_test",
                    "test_id": f"CERT{20000 + i:04d}",
                    "input": {
                        "test_type": "schedule_optimization",
                        "orders_count": random.randint(20, 50),
                        "machines_available": random.randint(200, 285),
                        "optimization_window": "24_hours"
                    },
                    "expected": {
                        "utilization_improvement": random.uniform(0.1, 0.3),
                        "on_time_delivery": 0.95,
                        "setup_time_reduction": 0.2
                    }
                })
            
            elif test_type == "yarn_emergency":
                scenarios["cases"].append({
                    "scenario": "yarn_emergency_test",
                    "test_id": f"CERT{20000 + i:04d}",
                    "input": {
                        "test_type": "emergency_substitution",
                        "original_yarn": f"Y{20000 + i:04d}",
                        "shortage_quantity": random.randint(1000, 5000),
                        "time_constraint": "48_hours"
                    },
                    "expected": {
                        "solution_found": True,
                        "quality_maintained": 0.95,
                        "cost_increase": random.uniform(0, 0.15)
                    }
                })
            
            else:  # quality_compliance
                scenarios["cases"].append({
                    "scenario": "quality_compliance_test",
                    "test_id": f"CERT{20000 + i:04d}",
                    "input": {
                        "test_type": "compliance_check",
                        "audit_scope": "full_system",
                        "compliance_standards": ["ISO9001", "data_integrity", "security"]
                    },
                    "expected": {
                        "compliance_score": random.uniform(0.95, 1.0),
                        "violations": random.randint(0, 2),
                        "certification_ready": True
                    }
                })
        
        return scenarios
    
    def save_scenarios(self, scenarios: Dict[str, Any], filename: str) -> str:
        """Save scenarios to JSON file"""
        filepath = self.scenarios_path / f"{filename}.json"
        
        with open(filepath, 'w') as f:
            json.dump(scenarios, f, indent=2, default=str)
        
        print(f"Saved {len(scenarios['cases'])} scenarios to {filepath}")
        return str(filepath)
    
    def generate_all_training_data(self) -> Dict[str, str]:
        """Generate all training data for all agent types"""
        generated_files = {}
        
        # Generate and save inventory scenarios
        inventory_scenarios = self.generate_inventory_scenarios()
        generated_files["inventory"] = self.save_scenarios(inventory_scenarios, "inventory_training")
        
        # Generate and save forecast scenarios
        forecast_scenarios = self.generate_forecast_scenarios()
        generated_files["forecast"] = self.save_scenarios(forecast_scenarios, "forecast_training")
        
        # Generate and save production scenarios
        production_scenarios = self.generate_production_scenarios()
        generated_files["production"] = self.save_scenarios(production_scenarios, "production_training")
        
        # Generate and save yarn substitution scenarios
        yarn_scenarios = self.generate_yarn_substitution_scenarios()
        generated_files["yarn"] = self.save_scenarios(yarn_scenarios, "yarn_substitution_training")
        
        # Generate and save quality scenarios
        quality_scenarios = self.generate_quality_scenarios()
        generated_files["quality"] = self.save_scenarios(quality_scenarios, "quality_training")
        
        # Generate and save certification scenarios
        cert_scenarios = self.generate_certification_scenarios()
        generated_files["certification"] = self.save_scenarios(cert_scenarios, "certification")
        
        return generated_files
    
    def generate_benchmark_data(self) -> Dict[str, Any]:
        """Generate benchmark data for performance comparison"""
        benchmarks = {
            "timestamp": datetime.now().isoformat(),
            "agents": {
                "inventory_intelligence": {
                    "accuracy_target": 0.95,
                    "response_time_target": 100,
                    "test_cases": 100,
                    "baseline_performance": {
                        "accuracy": 0.85,
                        "response_time_ms": 150,
                        "error_rate": 0.05
                    }
                },
                "forecast_intelligence": {
                    "accuracy_target": 0.85,
                    "mape_target": 0.15,
                    "response_time_target": 200,
                    "test_cases": 70,
                    "baseline_performance": {
                        "mape": 0.20,
                        "response_time_ms": 250,
                        "model_count": 4
                    }
                },
                "production_planning": {
                    "utilization_target": 0.85,
                    "on_time_target": 0.95,
                    "response_time_target": 200,
                    "test_cases": 90,
                    "baseline_performance": {
                        "utilization": 0.75,
                        "on_time_rate": 0.90,
                        "response_time_ms": 300
                    }
                },
                "yarn_substitution": {
                    "accuracy_target": 0.90,
                    "quality_maintenance": 0.95,
                    "response_time_target": 100,
                    "test_cases": 70,
                    "baseline_performance": {
                        "substitution_success": 0.80,
                        "quality_impact": 0.05,
                        "cost_savings": 0.10
                    }
                },
                "quality_assurance": {
                    "accuracy_target": 0.99,
                    "anomaly_detection": 0.85,
                    "response_time_target": 50,
                    "test_cases": 80,
                    "baseline_performance": {
                        "data_accuracy": 0.95,
                        "anomaly_detection_rate": 0.75,
                        "false_positive_rate": 0.15
                    }
                }
            }
        }
        
        # Save benchmarks
        benchmark_file = self.training_path / "benchmarks" / "performance_benchmarks.json"
        benchmark_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(benchmark_file, 'w') as f:
            json.dump(benchmarks, f, indent=2)
        
        print(f"Saved benchmark data to {benchmark_file}")
        return benchmarks


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate training data for agents')
    parser.add_argument('--agent', type=str, help='Generate data for specific agent')
    parser.add_argument('--all', action='store_true', help='Generate all training data')
    parser.add_argument('--benchmarks', action='store_true', help='Generate benchmark data')
    
    args = parser.parse_args()
    
    generator = TrainingDataGenerator()
    
    if args.all:
        print("\nGenerating all training data...")
        print("=" * 60)
        
        files = generator.generate_all_training_data()
        
        print("\nTraining data generation complete!")
        print("Generated files:")
        for agent, filepath in files.items():
            print(f"  - {agent}: {filepath}")
        
        if args.benchmarks:
            print("\nGenerating benchmark data...")
            benchmarks = generator.generate_benchmark_data()
            print(f"Generated benchmarks for {len(benchmarks['agents'])} agents")
    
    elif args.agent:
        print(f"\nGenerating training data for {args.agent} agent...")
        
        if args.agent == "inventory":
            scenarios = generator.generate_inventory_scenarios()
            filepath = generator.save_scenarios(scenarios, "inventory_training")
        elif args.agent == "forecast":
            scenarios = generator.generate_forecast_scenarios()
            filepath = generator.save_scenarios(scenarios, "forecast_training")
        elif args.agent == "production":
            scenarios = generator.generate_production_scenarios()
            filepath = generator.save_scenarios(scenarios, "production_training")
        elif args.agent == "yarn":
            scenarios = generator.generate_yarn_substitution_scenarios()
            filepath = generator.save_scenarios(scenarios, "yarn_substitution_training")
        elif args.agent == "quality":
            scenarios = generator.generate_quality_scenarios()
            filepath = generator.save_scenarios(scenarios, "quality_training")
        else:
            print(f"Unknown agent type: {args.agent}")
            return
        
        print(f"Generated training data: {filepath}")
    
    else:
        # Default: generate all
        print("\nNo specific option provided. Generating all training data...")
        files = generator.generate_all_training_data()
        benchmarks = generator.generate_benchmark_data()
        
        print("\nTraining data generation complete!")
        print(f"Generated {len(files)} training files and benchmark data")


if __name__ == "__main__":
    main()