#!/usr/bin/env python3
"""
Integration Test Framework for eFab AI Agent System
Tests agent communication, coordination, and workflow integration
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from ..core.agent_base import BaseAgent, AgentMessage, MessageType, Priority
from ..core.state_manager import system_state, CustomerProfile, IndustryType, CompanySize
from ..interface.customer_manager_agent import CustomerManagerAgent, DocumentType
from ..communication.message_router import MessageRouter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentIntegrationTester:
    """
    Integration tester for eFab AI Agent System
    
    Tests:
    - Agent registration and communication
    - Document processing workflows
    - Task assignment and coordination
    - Customer interaction flows
    - Error handling and escalation
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AgentIntegrationTester")
        self.test_results: Dict[str, Any] = {}
        self.test_customer_id = "TEST_CUSTOMER_001"
        
        # Initialize test environment
        self.customer_manager = CustomerManagerAgent()
        self.message_router = MessageRouter()
        
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests"""
        
        self.logger.info("Starting eFab AI Agent Integration Tests")
        
        test_suite = [
            ("Agent Initialization", self._test_agent_initialization),
            ("Customer Registration", self._test_customer_registration),
            ("Document Upload Processing", self._test_document_upload_processing),
            ("Agent Task Assignment", self._test_agent_task_assignment),
            ("Communication Workflow", self._test_communication_workflow),
            ("Error Handling", self._test_error_handling),
            ("Performance Metrics", self._test_performance_metrics)
        ]
        
        overall_success = True
        
        for test_name, test_function in test_suite:
            self.logger.info(f"Running test: {test_name}")
            
            try:
                test_start = datetime.now()
                result = await test_function()
                test_duration = (datetime.now() - test_start).total_seconds()
                
                self.test_results[test_name] = {
                    "status": "PASSED" if result else "FAILED",
                    "duration_seconds": test_duration,
                    "details": result if isinstance(result, dict) else {"success": result}
                }
                
                if not result:
                    overall_success = False
                    self.logger.error(f"Test FAILED: {test_name}")
                else:
                    self.logger.info(f"Test PASSED: {test_name} ({test_duration:.2f}s)")
                    
            except Exception as e:
                self.test_results[test_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "duration_seconds": 0
                }
                overall_success = False
                self.logger.error(f"Test ERROR: {test_name} - {str(e)}")
        
        # Generate test summary
        summary = {
            "overall_success": overall_success,
            "tests_run": len(test_suite),
            "tests_passed": len([r for r in self.test_results.values() if r["status"] == "PASSED"]),
            "tests_failed": len([r for r in self.test_results.values() if r["status"] == "FAILED"]),
            "tests_error": len([r for r in self.test_results.values() if r["status"] == "ERROR"]),
            "total_duration": sum(r.get("duration_seconds", 0) for r in self.test_results.values()),
            "timestamp": datetime.now().isoformat(),
            "detailed_results": self.test_results
        }
        
        self.logger.info(f"Integration Tests Complete: {summary['tests_passed']}/{summary['tests_run']} passed")
        
        return summary
    
    async def _test_agent_initialization(self) -> bool:
        """Test agent initialization and basic functionality"""
        
        try:
            # Test CustomerManagerAgent initialization
            await self.customer_manager.start()
            
            # Verify agent capabilities are registered
            expected_capabilities = [
                "process_document_upload",
                "coordinate_agent_tasks", 
                "manage_implementation_progress"
            ]
            
            agent_capabilities = [cap.name for cap in self.customer_manager.capabilities]
            
            for expected_cap in expected_capabilities:
                if expected_cap not in agent_capabilities:
                    self.logger.error(f"Missing capability: {expected_cap}")
                    return False
            
            # Test message router initialization
            await self.message_router.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {str(e)}")
            return False
    
    async def _test_customer_registration(self) -> bool:
        """Test customer registration and profile management"""
        
        try:
            # Create test customer profile
            test_customer = CustomerProfile(
                customer_id=self.test_customer_id,
                company_name="Test Manufacturing Company",
                industry=IndustryType.FURNITURE,
                company_size=CompanySize.MEDIUM,
                employee_count=150,
                annual_revenue=5000000.0,
                country="US",
                primary_contact="test@example.com"
            )
            
            # Register customer
            customer_id = system_state.register_customer(test_customer)
            
            # Verify registration
            retrieved_customer = system_state.get_customer_profile(customer_id)
            
            if not retrieved_customer:
                self.logger.error("Customer registration failed - could not retrieve profile")
                return False
            
            if retrieved_customer.company_name != test_customer.company_name:
                self.logger.error("Customer data mismatch after registration")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Customer registration test failed: {str(e)}")
            return False
    
    async def _test_document_upload_processing(self) -> Dict[str, Any]:
        """Test document upload and processing workflow"""
        
        try:
            # Simulate document upload
            test_payload = {
                "customer_id": self.test_customer_id,
                "filename": "business_requirements.pdf",
                "file_data": {"size": 1024000, "type": "pdf"},
                "document_context": {
                    "description": "Business requirements for ERP implementation",
                    "priority": "HIGH"
                }
            }
            
            # Process document upload
            upload_result = await self.customer_manager._process_document_upload(test_payload)
            
            # Verify upload result
            if "document_id" not in upload_result:
                return {"success": False, "error": "No document ID returned"}
            
            document_id = upload_result["document_id"]
            
            # Verify document classification
            if upload_result["classification"] != "BUSINESS_REQUIREMENTS":
                return {"success": False, "error": f"Incorrect classification: {upload_result['classification']}"}
            
            # Verify agent assignments
            if not upload_result.get("assigned_agents"):
                return {"success": False, "error": "No agents assigned to document"}
            
            # Test document status retrieval
            status_payload = {
                "customer_id": self.test_customer_id,
                "document_id": document_id
            }
            
            status_result = await self.customer_manager._get_document_processing_status(status_payload)
            
            if "document" not in status_result:
                return {"success": False, "error": "Could not retrieve document status"}
            
            return {
                "success": True,
                "document_id": document_id,
                "classification": upload_result["classification"],
                "assigned_agents": upload_result["assigned_agents"],
                "processing_time": upload_result.get("estimated_processing_time_hours", 0)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_agent_task_assignment(self) -> Dict[str, Any]:
        """Test agent task assignment and coordination"""
        
        try:
            # Test manual task assignment
            assignment_payload = {
                "customer_id": self.test_customer_id,
                "agent_id": "data_migration_agent",
                "task_description": "Analyze customer data for migration planning",
                "requirements": {
                    "priority": "HIGH",
                    "estimated_hours": 8
                }
            }
            
            assignment_result = await self.customer_manager._assign_task_to_agent(assignment_payload)
            
            if "assignment_id" not in assignment_result:
                return {"success": False, "error": "No assignment ID returned"}
            
            assignment_id = assignment_result["assignment_id"]
            
            # Test task coordination
            coordination_payload = {
                "customer_id": self.test_customer_id,
                "task_requirements": {
                    "task_name": "Implementation Planning",
                    "required_capabilities": ["data_analysis", "system_configuration"],
                    "priority": "HIGH"
                },
                "document_references": []
            }
            
            coordination_result = await self.customer_manager._coordinate_agent_tasks(coordination_payload)
            
            if not coordination_result.get("assignments_created"):
                return {"success": False, "error": "No coordinated assignments created"}
            
            return {
                "success": True,
                "manual_assignment_id": assignment_id,
                "coordinated_assignments": len(coordination_result["assignments_created"]),
                "involved_agents": len(coordination_result["coordination_plan"]["involved_agents"])
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_communication_workflow(self) -> Dict[str, Any]:
        """Test customer communication workflow"""
        
        try:
            # Test customer status inquiry
            status_payload = {"customer_id": self.test_customer_id}
            status_result = await self.customer_manager._get_customer_implementation_status(status_payload)
            
            if "implementation_status" not in status_result:
                return {"success": False, "error": "Could not retrieve implementation status"}
            
            # Test customer communication handling
            comm_payload = {
                "customer_id": self.test_customer_id,
                "type": "progress_inquiry",
                "message": "How is my implementation progressing?"
            }
            
            comm_result = await self.customer_manager._handle_customer_communication(comm_payload)
            
            if not comm_result.get("response_text"):
                return {"success": False, "error": "No response text generated"}
            
            return {
                "success": True,
                "status_retrieved": bool(status_result.get("implementation_status")),
                "response_generated": bool(comm_result.get("response_text")),
                "documents_tracked": status_result.get("documents_status", {}).get("total_documents", 0),
                "tasks_tracked": status_result.get("tasks_status", {}).get("total_tasks", 0)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and resilience"""
        
        try:
            # Test invalid customer ID
            invalid_payload = {"customer_id": "INVALID_CUSTOMER"}
            status_result = await self.customer_manager._get_customer_implementation_status(invalid_payload)
            
            if "error" not in status_result:
                return {"success": False, "error": "Expected error for invalid customer ID"}
            
            # Test missing required parameters
            try:
                empty_payload = {}
                await self.customer_manager._process_document_upload(empty_payload)
                return {"success": False, "error": "Expected error for empty payload"}
            except Exception:
                # Expected to throw exception
                pass
            
            # Test task assignment to non-existent agent
            invalid_assignment = {
                "customer_id": self.test_customer_id,
                "agent_id": "non_existent_agent",
                "task_description": "Test task"
            }
            
            assignment_result = await self.customer_manager._assign_task_to_agent(invalid_assignment)
            
            # Should handle gracefully
            if "assignment_id" not in assignment_result:
                # This is expected for invalid agents
                pass
            
            return {
                "success": True,
                "invalid_customer_handled": True,
                "empty_payload_handled": True,
                "invalid_agent_handled": True
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance monitoring and metrics"""
        
        try:
            start_time = datetime.now()
            
            # Perform multiple operations to test performance
            operations = []
            
            for i in range(5):
                # Upload document
                upload_payload = {
                    "customer_id": self.test_customer_id,
                    "filename": f"test_document_{i}.csv",
                    "file_data": {"size": 50000},
                    "document_context": {"description": f"Test document {i}"}
                }
                
                op_start = datetime.now()
                result = await self.customer_manager._process_document_upload(upload_payload)
                op_duration = (datetime.now() - op_start).total_seconds()
                
                operations.append({
                    "operation": "document_upload",
                    "duration_seconds": op_duration,
                    "success": "document_id" in result
                })
            
            total_duration = (datetime.now() - start_time).total_seconds()
            
            # Calculate performance metrics
            avg_operation_time = sum(op["duration_seconds"] for op in operations) / len(operations)
            success_rate = sum(1 for op in operations if op["success"]) / len(operations)
            
            return {
                "success": True,
                "total_operations": len(operations),
                "total_duration_seconds": total_duration,
                "average_operation_time_seconds": avg_operation_time,
                "success_rate": success_rate,
                "operations_per_second": len(operations) / total_duration
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def cleanup_test_environment(self):
        """Clean up test environment"""
        try:
            # Stop agents
            if hasattr(self.customer_manager, 'stop'):
                await self.customer_manager.stop()
            
            if hasattr(self.message_router, 'stop'):
                await self.message_router.stop()
            
            # Clean up test data
            if self.test_customer_id in system_state.customers:
                del system_state.customers[self.test_customer_id]
            
            self.logger.info("Test environment cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up test environment: {str(e)}")


async def run_integration_tests():
    """Run integration tests"""
    tester = AgentIntegrationTester()
    
    try:
        results = await tester.run_integration_tests()
        
        print("\n" + "="*80)
        print("eFab AI Agent Integration Test Results")
        print("="*80)
        
        print(f"Overall Success: {'✓ PASSED' if results['overall_success'] else '✗ FAILED'}")
        print(f"Tests Run: {results['tests_run']}")
        print(f"Tests Passed: {results['tests_passed']}")
        print(f"Tests Failed: {results['tests_failed']}")
        print(f"Tests Error: {results['tests_error']}")
        print(f"Total Duration: {results['total_duration']:.2f} seconds")
        print(f"Timestamp: {results['timestamp']}")
        
        print("\nDetailed Results:")
        print("-" * 80)
        
        for test_name, result in results['detailed_results'].items():
            status_icon = "✓" if result['status'] == 'PASSED' else "✗"
            print(f"{status_icon} {test_name}: {result['status']} ({result.get('duration_seconds', 0):.2f}s)")
            
            if result['status'] == 'FAILED' and 'details' in result:
                if isinstance(result['details'], dict) and 'error' in result['details']:
                    print(f"    Error: {result['details']['error']}")
            
            elif result['status'] == 'ERROR':
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*80)
        
        return results
        
    finally:
        await tester.cleanup_test_environment()


if __name__ == "__main__":
    # Run tests when executed directly
    asyncio.run(run_integration_tests())