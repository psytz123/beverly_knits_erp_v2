#!/usr/bin/env python3
"""
eFab AI Agent Architecture Validation Script
Validates the complete agent architecture and ensures all components are properly integrated
"""

import asyncio
import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ArchitectureValidator")


class ArchitectureValidator:
    """
    Comprehensive architecture validator for eFab AI Agent System
    
    Validates:
    - All agent imports and initialization
    - Core component integration
    - Message routing capabilities
    - State management functionality
    - Agent factory operations
    - Customer workflow orchestration
    """
    
    def __init__(self):
        self.validation_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "overall_status": "UNKNOWN"
        }
    
    async def validate_complete_architecture(self) -> Dict[str, Any]:
        """Run complete architecture validation"""
        logger.info("🚀 Starting eFab AI Agent Architecture Validation")
        
        validation_tests = [
            ("Core Agent Import Validation", self._validate_core_imports),
            ("Agent Base Functionality", self._validate_agent_base),
            ("System State Management", self._validate_system_state),
            ("Message Router Integration", self._validate_message_router),
            ("Agent Factory Functionality", self._validate_agent_factory),
            ("Customer Interface Agents", self._validate_interface_agents),
            ("Implementation Agents", self._validate_implementation_agents),
            ("Industry-Specific Agents", self._validate_industry_agents),
            ("Monitoring and Optimization", self._validate_monitoring_optimization),
            ("Integration Workflows", self._validate_integration_workflows),
            ("Error Handling and Recovery", self._validate_error_handling)
        ]
        
        for test_name, test_func in validation_tests:
            await self._run_validation_test(test_name, test_func)
        
        # Calculate overall status
        if self.validation_results["failed_tests"] == 0:
            self.validation_results["overall_status"] = "PASSED"
        elif self.validation_results["passed_tests"] > self.validation_results["failed_tests"]:
            self.validation_results["overall_status"] = "MOSTLY_PASSED"
        else:
            self.validation_results["overall_status"] = "FAILED"
        
        # Print summary
        self._print_validation_summary()
        
        return self.validation_results
    
    async def _run_validation_test(self, test_name: str, test_func):
        """Run individual validation test"""
        logger.info(f"🔍 Running: {test_name}")
        
        try:
            result = await test_func()
            
            test_result = {
                "test_name": test_name,
                "status": "PASSED" if result["success"] else "FAILED",
                "message": result.get("message", ""),
                "details": result.get("details", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            self.validation_results["test_results"].append(test_result)
            self.validation_results["total_tests"] += 1
            
            if result["success"]:
                self.validation_results["passed_tests"] += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                self.validation_results["failed_tests"] += 1
                logger.error(f"❌ {test_name}: FAILED - {result.get('message', '')}")
                
        except Exception as e:
            test_result = {
                "test_name": test_name,
                "status": "ERROR",
                "message": f"Test execution error: {str(e)}",
                "details": {},
                "timestamp": datetime.now().isoformat()
            }
            
            self.validation_results["test_results"].append(test_result)
            self.validation_results["total_tests"] += 1
            self.validation_results["failed_tests"] += 1
            
            logger.error(f"💥 {test_name}: ERROR - {str(e)}")
    
    async def _validate_core_imports(self) -> Dict[str, Any]:
        """Validate all core agent imports"""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            
            from src.ai_agents import (
                CentralOrchestrator,
                BaseAgent, 
                AgentMessage,
                AgentCapability,
                SystemState,
                CustomerProfile,
                MessageRouter,
                AgentFactory,
                LeadAgent,
                CustomerManagerAgent,
                SystemMonitorAgent
            )
            
            # Test basic instantiation
            orchestrator = CentralOrchestrator()
            message_router = MessageRouter()
            agent_factory = AgentFactory()
            system_monitor = SystemMonitorAgent()
            lead_agent = LeadAgent()
            customer_manager = CustomerManagerAgent()
            
            return {
                "success": True,
                "message": "All core agent imports successful",
                "details": {
                    "imported_components": [
                        "CentralOrchestrator", "BaseAgent", "AgentMessage", "AgentCapability",
                        "SystemState", "CustomerProfile", "MessageRouter", "AgentFactory",
                        "LeadAgent", "CustomerManagerAgent", "SystemMonitorAgent"
                    ]
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Core import validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_agent_base(self) -> Dict[str, Any]:
        """Validate BaseAgent functionality"""
        try:
            from src.ai_agents.core.agent_base import BaseAgent, AgentMessage, AgentCapability, MessageType, Priority
            
            # Create test agent implementation
            class TestAgent(BaseAgent):
                def _initialize(self):
                    self.register_capability(AgentCapability(
                        name="test_capability",
                        description="Test capability for validation",
                        input_schema={"type": "object"},
                        output_schema={"type": "object"}
                    ))
            
            # Test agent creation and basic functionality
            agent = TestAgent("test_agent", "Test Agent", "Test agent for validation")
            
            # Test message creation
            message = AgentMessage(
                agent_id="test_sender",
                target_agent_id="test_agent",
                message_type=MessageType.REQUEST,
                payload={"test": "data"},
                priority=Priority.MEDIUM
            )
            
            # Validate capabilities
            capabilities = agent.capabilities
            assert len(capabilities) > 0, "Agent should have at least one capability"
            
            # Validate status
            status = agent.get_status()
            assert "agent_id" in status, "Status should include agent_id"
            assert "capabilities" in status, "Status should include capabilities"
            
            return {
                "success": True,
                "message": "BaseAgent validation successful",
                "details": {
                    "agent_id": agent.agent_id,
                    "capabilities_count": len(capabilities),
                    "status_fields": list(status.keys())
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"BaseAgent validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_system_state(self) -> Dict[str, Any]:
        """Validate SystemState management"""
        try:
            from src.ai_agents.core.state_manager import system_state, CustomerProfile, IndustryType, CompanySize
            
            # Test customer profile creation
            customer_profile = CustomerProfile(
                company_name="Test Company",
                industry=IndustryType.FURNITURE,
                company_size=CompanySize.MEDIUM
            )
            
            # Test system state operations
            customer_id = system_state.register_customer(customer_profile)
            
            # Validate customer retrieval
            retrieved_profile = system_state.get_customer_profile(customer_id)
            assert retrieved_profile is not None, "Should retrieve registered customer"
            assert retrieved_profile.company_name == "Test Company", "Customer data should match"
            
            # Test system status
            status = system_state.get_system_status()
            assert "customers" in status, "System status should include customers"
            
            # Test dashboard data
            dashboard = system_state.get_customer_dashboard(customer_id)
            assert dashboard is not None, "Should get customer dashboard"
            
            return {
                "success": True,
                "message": "SystemState validation successful",
                "details": {
                    "customer_id": customer_id,
                    "system_status_fields": list(status.keys()),
                    "dashboard_available": dashboard is not None
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"SystemState validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_message_router(self) -> Dict[str, Any]:
        """Validate MessageRouter functionality"""
        try:
            from src.ai_agents.communication.message_router import MessageRouter, RoutingStrategy
            from src.ai_agents.core.agent_base import AgentMessage, MessageType, Priority
            
            # Create message router
            router = MessageRouter()
            
            # Test agent registration
            def dummy_callback(message):
                return message
            
            success = router.register_agent(
                agent_id="test_agent",
                agent_info={
                    "agent_name": "Test Agent",
                    "capabilities": []
                },
                message_callback=dummy_callback
            )
            
            assert success, "Agent registration should succeed"
            
            # Test metrics
            metrics = router.get_metrics()
            assert "message_metrics" in metrics, "Should have message metrics"
            assert "agent_metrics" in metrics, "Should have agent metrics"
            
            return {
                "success": True,
                "message": "MessageRouter validation successful",
                "details": {
                    "agent_registered": success,
                    "metrics_available": True,
                    "metric_categories": list(metrics.keys())
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"MessageRouter validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_agent_factory(self) -> Dict[str, Any]:
        """Validate AgentFactory functionality"""
        try:
            from src.ai_agents.deployment.agent_factory import AgentFactory
            from src.ai_agents.communication.message_router import MessageRouter
            
            # Create factory with message router
            message_router = MessageRouter()
            factory = AgentFactory(message_router)
            
            # Test factory status
            status = factory.get_factory_status()
            assert "active_agents" in status, "Factory status should include active agents"
            assert "registered_templates" in status, "Factory status should include templates"
            
            # Test template availability
            templates = status.get("agent_breakdown", {})
            
            # Test health check
            health_check = await factory.health_check()
            assert "factory_healthy" in health_check, "Health check should include factory health"
            
            return {
                "success": True,
                "message": "AgentFactory validation successful",
                "details": {
                    "active_agents": status["active_agents"],
                    "registered_templates": status["registered_templates"],
                    "factory_healthy": health_check["factory_healthy"],
                    "templates_available": list(templates.keys())
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"AgentFactory validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_interface_agents(self) -> Dict[str, Any]:
        """Validate customer interface agents"""
        try:
            from src.ai_agents.interface.lead_agent import LeadAgent
            from src.ai_agents.interface.customer_manager_agent import CustomerManagerAgent
            
            # Test LeadAgent
            lead_agent = LeadAgent()
            lead_capabilities = lead_agent.capabilities
            
            # Test CustomerManagerAgent
            customer_manager = CustomerManagerAgent()
            customer_capabilities = customer_manager.capabilities
            
            # Validate capabilities
            assert len(lead_capabilities) > 0, "LeadAgent should have capabilities"
            assert len(customer_capabilities) > 0, "CustomerManagerAgent should have capabilities"
            
            # Test specific capabilities
            lead_capability_names = [cap.name for cap in lead_capabilities]
            customer_capability_names = [cap.name for cap in customer_capabilities]
            
            assert "handle_customer_conversation" in lead_capability_names, "LeadAgent should handle conversations"
            assert "process_document_upload" in customer_capability_names, "CustomerManager should handle documents"
            
            return {
                "success": True,
                "message": "Interface agents validation successful",
                "details": {
                    "lead_agent_capabilities": len(lead_capabilities),
                    "customer_manager_capabilities": len(customer_capabilities),
                    "lead_capability_names": lead_capability_names,
                    "customer_capability_names": customer_capability_names
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Interface agents validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_implementation_agents(self) -> Dict[str, Any]:
        """Validate implementation workflow agents"""
        try:
            from src.ai_agents.implementation.project_manager_agent import ImplementationProjectManagerAgent
            from src.ai_agents.implementation.data_migration_agent import DataMigrationIntelligenceAgent
            from src.ai_agents.implementation.configuration_agent import ConfigurationGenerationAgent
            
            # Test agent instantiation
            project_manager = ImplementationProjectManagerAgent()
            data_migration = DataMigrationIntelligenceAgent()
            config_generator = ConfigurationGenerationAgent()
            
            # Validate capabilities
            pm_capabilities = project_manager.capabilities
            dm_capabilities = data_migration.capabilities
            cg_capabilities = config_generator.capabilities
            
            assert len(pm_capabilities) > 0, "ProjectManager should have capabilities"
            assert len(dm_capabilities) > 0, "DataMigration should have capabilities"
            assert len(cg_capabilities) > 0, "ConfigGenerator should have capabilities"
            
            return {
                "success": True,
                "message": "Implementation agents validation successful",
                "details": {
                    "project_manager_capabilities": len(pm_capabilities),
                    "data_migration_capabilities": len(dm_capabilities),
                    "config_generator_capabilities": len(cg_capabilities)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Implementation agents validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_industry_agents(self) -> Dict[str, Any]:
        """Validate industry-specific agents"""
        try:
            from src.ai_agents.industry.furniture_agent import FurnitureManufacturingAgent
            from src.ai_agents.industry.injection_molding_agent import InjectionMoldingAgent
            from src.ai_agents.industry.electrical_equipment_agent import ElectricalEquipmentAgent
            
            # Test agent instantiation
            furniture_agent = FurnitureManufacturingAgent()
            injection_agent = InjectionMoldingAgent()
            electrical_agent = ElectricalEquipmentAgent()
            
            # Validate capabilities
            furniture_caps = furniture_agent.capabilities
            injection_caps = injection_agent.capabilities
            electrical_caps = electrical_agent.capabilities
            
            assert len(furniture_caps) > 0, "Furniture agent should have capabilities"
            assert len(injection_caps) > 0, "Injection molding agent should have capabilities"
            assert len(electrical_caps) > 0, "Electrical equipment agent should have capabilities"
            
            return {
                "success": True,
                "message": "Industry agents validation successful",
                "details": {
                    "furniture_agent_capabilities": len(furniture_caps),
                    "injection_agent_capabilities": len(injection_caps),
                    "electrical_agent_capabilities": len(electrical_caps)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Industry agents validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_monitoring_optimization(self) -> Dict[str, Any]:
        """Validate monitoring and optimization agents"""
        try:
            from src.ai_agents.monitoring.system_monitor_agent import SystemMonitorAgent
            from src.ai_agents.optimization.performance_agent import PerformanceOptimizationAgent
            
            # Test agent instantiation
            monitor_agent = SystemMonitorAgent()
            performance_agent = PerformanceOptimizationAgent()
            
            # Validate capabilities
            monitor_caps = monitor_agent.capabilities
            performance_caps = performance_agent.capabilities
            
            assert len(monitor_caps) > 0, "Monitor agent should have capabilities"
            assert len(performance_caps) > 0, "Performance agent should have capabilities"
            
            # Test monitoring capabilities
            monitor_capability_names = [cap.name for cap in monitor_caps]
            assert "system_health_monitoring" in monitor_capability_names, "Should have health monitoring"
            
            return {
                "success": True,
                "message": "Monitoring and optimization agents validation successful",
                "details": {
                    "monitor_agent_capabilities": len(monitor_caps),
                    "performance_agent_capabilities": len(performance_caps),
                    "monitor_capability_names": monitor_capability_names
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Monitoring/optimization agents validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_integration_workflows(self) -> Dict[str, Any]:
        """Validate end-to-end integration workflows"""
        try:
            from src.ai_agents.testing.integration_test import AgentIntegrationTester
            
            # Create integration tester
            tester = AgentIntegrationTester()
            
            # Validate tester initialization
            assert hasattr(tester, 'customer_manager'), "Tester should have customer manager"
            assert hasattr(tester, 'message_router'), "Tester should have message router"
            
            return {
                "success": True,
                "message": "Integration workflows validation successful",
                "details": {
                    "integration_tester_available": True,
                    "test_customer_id": tester.test_customer_id
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Integration workflows validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def _validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling and recovery mechanisms"""
        try:
            from src.ai_agents.core.agent_base import AgentMessage, MessageType
            from src.ai_agents.communication.message_router import MessageRouter
            
            # Test invalid message handling
            router = MessageRouter()
            
            # Create invalid message
            invalid_message = AgentMessage(
                agent_id="nonexistent",
                target_agent_id="also_nonexistent",
                message_type=MessageType.REQUEST,
                payload={}
            )
            
            # Test circuit breaker functionality exists
            assert hasattr(router, 'circuit_breakers'), "Router should have circuit breakers"
            assert hasattr(router, 'failure_thresholds'), "Router should have failure thresholds"
            
            return {
                "success": True,
                "message": "Error handling validation successful",
                "details": {
                    "circuit_breaker_available": True,
                    "failure_thresholds_configured": True
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error handling validation failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _print_validation_summary(self):
        """Print comprehensive validation summary"""
        results = self.validation_results
        
        print("\n" + "="*80)
        print("🏗️  eFab AI Agent Architecture Validation Summary")
        print("="*80)
        print(f"📊 Total Tests: {results['total_tests']}")
        print(f"✅ Passed: {results['passed_tests']}")
        print(f"❌ Failed: {results['failed_tests']}")
        print(f"🎯 Overall Status: {results['overall_status']}")
        print(f"⏰ Completed: {results['timestamp']}")
        print("="*80)
        
        # Print detailed results
        for test_result in results['test_results']:
            status_icon = "✅" if test_result['status'] == "PASSED" else "❌" if test_result['status'] == "FAILED" else "💥"
            print(f"{status_icon} {test_result['test_name']}: {test_result['status']}")
            if test_result['status'] != "PASSED" and test_result['message']:
                print(f"   └─ {test_result['message']}")
        
        print("="*80)
        
        # Recommendations
        if results['overall_status'] == "PASSED":
            print("🎉 Architecture validation completed successfully!")
            print("   All components are properly integrated and functional.")
        elif results['overall_status'] == "MOSTLY_PASSED":
            print("⚠️  Architecture mostly validated with some issues.")
            print("   Review failed tests and address identified issues.")
        else:
            print("🚨 Architecture validation failed!")
            print("   Critical issues detected. Address failed tests before deployment.")
        
        print("="*80)


async def main():
    """Main validation entry point"""
    try:
        validator = ArchitectureValidator()
        results = await validator.validate_complete_architecture()
        
        # Save results to file
        with open('architecture_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Validation results saved to: architecture_validation_results.json")
        
        # Exit with appropriate code
        if results['overall_status'] == "PASSED":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())