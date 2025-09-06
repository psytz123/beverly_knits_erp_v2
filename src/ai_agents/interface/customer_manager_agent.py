#!/usr/bin/env python3
"""
Customer Manager Agent - Enhanced Lead Agent
Primary customer manager, document coordinator, and agent orchestrator
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import os
import hashlib
from pathlib import Path

from ..core.agent_base import BaseAgent, AgentMessage, AgentCapability, MessageType, Priority
from ..core.state_manager import system_state, CustomerProfile, ImplementationPhase, IndustryType
from .lead_agent import ConversationState, MessageIntent, ResponseType, CustomerConversation, ChatbotResponse

# Setup logging
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Types of documents customers can upload"""
    BUSINESS_REQUIREMENTS = "BUSINESS_REQUIREMENTS"     # Requirements docs
    CURRENT_SYSTEM_DATA = "CURRENT_SYSTEM_DATA"         # Existing system exports
    FINANCIAL_DATA = "FINANCIAL_DATA"                   # Financial reports/data
    INVENTORY_DATA = "INVENTORY_DATA"                   # Current inventory
    CUSTOMER_DATA = "CUSTOMER_DATA"                     # Customer lists/data
    VENDOR_DATA = "VENDOR_DATA"                         # Supplier/vendor data
    PRODUCTION_DATA = "PRODUCTION_DATA"                 # Manufacturing data
    ORGANIZATIONAL_CHART = "ORGANIZATIONAL_CHART"       # Company structure
    PROCESS_DOCUMENTATION = "PROCESS_DOCUMENTATION"     # Current processes
    TECHNICAL_SPECS = "TECHNICAL_SPECS"                 # Technical specifications
    COMPLIANCE_DOCS = "COMPLIANCE_DOCS"                 # Regulatory documents
    INTEGRATION_REQUIREMENTS = "INTEGRATION_REQUIREMENTS" # System integration needs
    TRAINING_MATERIALS = "TRAINING_MATERIALS"           # Existing training docs
    UNKNOWN = "UNKNOWN"                                 # Unclassified documents


class AgentAssignmentType(Enum):
    """Types of agent assignments"""
    PRIMARY = "PRIMARY"                 # Primary agent responsible
    SUPPORTING = "SUPPORTING"           # Supporting role
    CONSULTATIVE = "CONSULTATIVE"       # Consultation only
    REVIEW = "REVIEW"                   # Review and feedback


@dataclass
class DocumentUpload:
    """Customer document upload"""
    document_id: str
    customer_id: str
    original_filename: str
    file_path: str
    file_size_bytes: int
    document_type: DocumentType
    content_summary: str = ""
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    assigned_agents: List[str] = field(default_factory=list)
    processing_status: str = "UPLOADED"  # UPLOADED, ANALYZING, PROCESSED, DISTRIBUTED
    uploaded_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    priority: Priority = Priority.MEDIUM
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "document_id": self.document_id,
            "customer_id": self.customer_id,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "document_type": self.document_type.value,
            "content_summary": self.content_summary,
            "extracted_data": self.extracted_data,
            "assigned_agents": self.assigned_agents,
            "processing_status": self.processing_status,
            "uploaded_at": self.uploaded_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "priority": self.priority.value,
            "tags": self.tags
        }


@dataclass
class AgentTaskAssignment:
    """Task assignment to specialized agent"""
    assignment_id: str
    customer_id: str
    agent_id: str
    agent_type: str
    task_description: str
    assignment_type: AgentAssignmentType
    documents: List[str]  # Document IDs
    requirements: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    estimated_duration_hours: float = 4.0
    status: str = "ASSIGNED"  # ASSIGNED, IN_PROGRESS, COMPLETED, BLOCKED, CANCELLED
    assigned_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_data: Dict[str, Any] = field(default_factory=dict)
    feedback: str = ""


class CustomerManagerAgent(BaseAgent):
    """
    Customer Manager Agent - Enhanced Lead Agent
    
    Primary customer relationship manager providing:
    - Document intake and intelligent classification
    - Agent task coordination and assignment
    - Implementation project management
    - Customer communication and status updates
    - Document processing workflow management
    - Progress tracking and reporting
    - Issue escalation and resolution
    - Customer satisfaction monitoring
    """
    
    def __init__(self, agent_id: str = "customer_manager_agent"):
        """Initialize Customer Manager Agent"""
        super().__init__(
            agent_id=agent_id,
            agent_name="Customer Manager",
            agent_description="Primary customer manager and document coordinator"
        )
        
        # Document management
        self.customer_documents: Dict[str, List[DocumentUpload]] = {}  # customer_id -> documents
        self.document_processing_queue: List[DocumentUpload] = []
        self.document_upload_path = "customer_uploads"
        
        # Agent coordination
        self.agent_assignments: Dict[str, List[AgentTaskAssignment]] = {}  # customer_id -> assignments
        self.active_tasks: Dict[str, AgentTaskAssignment] = {}  # assignment_id -> task
        
        # Available specialized agents
        self.available_agents = {
            "data_migration_agent": {
                "specialties": ["CURRENT_SYSTEM_DATA", "INVENTORY_DATA", "CUSTOMER_DATA", "VENDOR_DATA"],
                "capabilities": ["data_analysis", "etl_planning", "data_validation"]
            },
            "configuration_agent": {
                "specialties": ["BUSINESS_REQUIREMENTS", "TECHNICAL_SPECS", "INTEGRATION_REQUIREMENTS"],
                "capabilities": ["system_configuration", "workflow_setup", "integration_planning"]
            },
            "project_manager_agent": {
                "specialties": ["ORGANIZATIONAL_CHART", "PROCESS_DOCUMENTATION", "TRAINING_MATERIALS"],
                "capabilities": ["project_planning", "timeline_management", "resource_coordination"]
            },
            "furniture_agent": {
                "specialties": ["PRODUCTION_DATA", "INVENTORY_DATA"],
                "capabilities": ["furniture_specific_analysis", "wood_optimization", "bom_management"],
                "industry": "FURNITURE"
            },
            "injection_molding_agent": {
                "specialties": ["PRODUCTION_DATA", "TECHNICAL_SPECS"],
                "capabilities": ["molding_process_optimization", "defect_analysis", "material_planning"],
                "industry": "INJECTION_MOLDING"
            },
            "electrical_equipment_agent": {
                "specialties": ["TECHNICAL_SPECS", "COMPLIANCE_DOCS"],
                "capabilities": ["electrical_design_validation", "certification_tracking", "test_procedures"],
                "industry": "ELECTRICAL_EQUIPMENT"
            }
        }
        
        # Document classification patterns
        self.document_classifiers = {
            "business_requirements": ["requirements", "business", "needs", "specs", "specification"],
            "current_data": ["export", "data", "current", "existing", "legacy", "database"],
            "financial": ["financial", "accounting", "budget", "cost", "expense", "revenue"],
            "inventory": ["inventory", "stock", "warehouse", "parts", "materials", "items"],
            "production": ["production", "manufacturing", "schedule", "orders", "bom", "bill of materials"],
            "organizational": ["org chart", "organization", "structure", "hierarchy", "roles"],
            "process": ["process", "procedure", "workflow", "sop", "standard operating"],
            "technical": ["technical", "architecture", "system", "network", "infrastructure"],
            "compliance": ["compliance", "regulatory", "certification", "standards", "audit"]
        }
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.document_upload_path, exist_ok=True)
    
    def _initialize(self):
        """Initialize Customer Manager capabilities"""
        # Document management capabilities
        self.register_capability(AgentCapability(
            name="process_document_upload",
            description="Process customer document uploads and classify them",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "file_data": {"type": "object"},
                    "filename": {"type": "string"},
                    "document_context": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "classification": {"type": "string"},
                    "assigned_agents": {"type": "array"},
                    "processing_status": {"type": "string"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="coordinate_agent_tasks",
            description="Coordinate task assignments across specialized agents",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "task_requirements": {"type": "object"},
                    "document_references": {"type": "array"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "assignments_created": {"type": "array"},
                    "coordination_plan": {"type": "object"},
                    "timeline_estimate": {"type": "object"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="manage_implementation_progress",
            description="Manage overall implementation progress and customer communication",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "progress_update_type": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "progress_summary": {"type": "object"},
                    "next_actions": {"type": "array"},
                    "customer_communication": {"type": "object"}
                }
            }
        ))
        
        # Register message handlers
        self.register_message_handler(MessageType.REQUEST, self._handle_customer_manager_request)
        self.register_message_handler(MessageType.NOTIFICATION, self._handle_agent_notification)
        
        # Start background processing
        asyncio.create_task(self._document_processing_loop())
        asyncio.create_task(self._task_monitoring_loop())
        asyncio.create_task(self._customer_communication_loop())
    
    async def _handle_customer_manager_request(self, message: AgentMessage) -> AgentMessage:
        """Handle customer manager requests"""
        action = message.payload.get("action")
        
        try:
            if action == "upload_document":
                result = await self._process_document_upload(message.payload)
            elif action == "coordinate_agents":
                result = await self._coordinate_agent_tasks(message.payload)
            elif action == "get_customer_status":
                result = await self._get_customer_implementation_status(message.payload)
            elif action == "assign_task_to_agent":
                result = await self._assign_task_to_agent(message.payload)
            elif action == "get_document_status":
                result = await self._get_document_processing_status(message.payload)
            elif action == "customer_communication":
                result = await self._handle_customer_communication(message.payload)
            else:
                result = {"error": "Unsupported action", "action": action}
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload=result,
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            self.logger.error(f"Error handling customer manager request: {str(e)}")
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                correlation_id=message.correlation_id
            )
    
    async def _handle_agent_notification(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle notifications from other agents"""
        notification_type = message.payload.get("notification_type")
        
        if notification_type == "TASK_COMPLETED":
            await self._handle_task_completion(message.payload)
        elif notification_type == "TASK_BLOCKED":
            await self._handle_task_blocked(message.payload)
        elif notification_type == "AGENT_QUESTION":
            await self._handle_agent_question(message.payload)
        elif notification_type == "PROGRESS_UPDATE":
            await self._handle_progress_update(message.payload)
        
        return None
    
    async def _process_document_upload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process customer document upload"""
        customer_id = payload.get("customer_id")
        file_data = payload.get("file_data", {})
        filename = payload.get("filename", "unknown_file")
        document_context = payload.get("document_context", {})
        
        # Generate document ID
        document_id = f"DOC_{customer_id}_{int(datetime.now().timestamp())}_{hashlib.md5(filename.encode()).hexdigest()[:8]}"
        
        # Simulate file storage (in real implementation, would handle actual file upload)
        file_path = os.path.join(self.document_upload_path, f"{document_id}_{filename}")
        file_size = file_data.get("size", 0)
        
        # Classify document based on filename and context
        document_type = self._classify_document(filename, document_context)
        
        # Create document upload record
        document_upload = DocumentUpload(
            document_id=document_id,
            customer_id=customer_id,
            original_filename=filename,
            file_path=file_path,
            file_size_bytes=file_size,
            document_type=document_type,
            content_summary=document_context.get("description", ""),
            priority=Priority(document_context.get("priority", "MEDIUM"))
        )
        
        # Store document record
        if customer_id not in self.customer_documents:
            self.customer_documents[customer_id] = []
        self.customer_documents[customer_id].append(document_upload)
        
        # Add to processing queue
        self.document_processing_queue.append(document_upload)
        
        # Determine which agents should process this document
        assigned_agents = await self._determine_agent_assignments(document_upload)
        document_upload.assigned_agents = assigned_agents
        
        # Create agent task assignments
        assignments_created = await self._create_agent_assignments(document_upload)
        
        self.logger.info(f"Document {document_id} uploaded and classified as {document_type.value}")
        
        return {
            "document_id": document_id,
            "classification": document_type.value,
            "assigned_agents": assigned_agents,
            "processing_status": "QUEUED_FOR_PROCESSING",
            "assignments_created": len(assignments_created),
            "estimated_processing_time_hours": self._estimate_processing_time(document_type)
        }
    
    def _classify_document(self, filename: str, context: Dict[str, Any]) -> DocumentType:
        """Classify document based on filename and context"""
        filename_lower = filename.lower()
        description = context.get("description", "").lower()
        combined_text = f"{filename_lower} {description}"
        
        # Score each document type
        type_scores = {}
        
        for doc_type, keywords in self.document_classifiers.items():
            score = 0
            for keyword in keywords:
                if keyword in combined_text:
                    score += 1
            
            if score > 0:
                type_scores[doc_type] = score
        
        # Map classifier keys to DocumentType enum
        classifier_to_enum = {
            "business_requirements": DocumentType.BUSINESS_REQUIREMENTS,
            "current_data": DocumentType.CURRENT_SYSTEM_DATA,
            "financial": DocumentType.FINANCIAL_DATA,
            "inventory": DocumentType.INVENTORY_DATA,
            "production": DocumentType.PRODUCTION_DATA,
            "organizational": DocumentType.ORGANIZATIONAL_CHART,
            "process": DocumentType.PROCESS_DOCUMENTATION,
            "technical": DocumentType.TECHNICAL_SPECS,
            "compliance": DocumentType.COMPLIANCE_DOCS
        }
        
        if type_scores:
            best_match = max(type_scores.keys(), key=lambda k: type_scores[k])
            return classifier_to_enum.get(best_match, DocumentType.UNKNOWN)
        
        # File extension-based fallback
        if filename_lower.endswith(('.xls', '.xlsx', '.csv')):
            if any(word in combined_text for word in ["inventory", "stock", "parts"]):
                return DocumentType.INVENTORY_DATA
            elif any(word in combined_text for word in ["financial", "accounting"]):
                return DocumentType.FINANCIAL_DATA
            else:
                return DocumentType.CURRENT_SYSTEM_DATA
        
        elif filename_lower.endswith(('.doc', '.docx', '.pdf')):
            return DocumentType.BUSINESS_REQUIREMENTS
        
        return DocumentType.UNKNOWN
    
    async def _determine_agent_assignments(self, document: DocumentUpload) -> List[str]:
        """Determine which agents should process this document"""
        assigned_agents = []
        
        # Get customer profile to determine industry-specific needs
        customer_profile = system_state.get_customer_profile(document.customer_id)
        customer_industry = customer_profile.industry.value if customer_profile else "GENERIC"
        
        # Find agents that specialize in this document type
        for agent_id, agent_info in self.available_agents.items():
            specialties = agent_info.get("specialties", [])
            agent_industry = agent_info.get("industry")
            
            # Check if agent handles this document type
            if document.document_type.value in specialties:
                # If agent is industry-specific, check industry match
                if agent_industry and agent_industry != customer_industry:
                    continue
                
                assigned_agents.append(agent_id)
        
        # Always assign project manager for coordination
        if "project_manager_agent" not in assigned_agents:
            assigned_agents.append("project_manager_agent")
        
        # Add data migration agent for any data files
        if document.document_type in [DocumentType.CURRENT_SYSTEM_DATA, DocumentType.INVENTORY_DATA, 
                                     DocumentType.CUSTOMER_DATA, DocumentType.VENDOR_DATA]:
            if "data_migration_agent" not in assigned_agents:
                assigned_agents.append("data_migration_agent")
        
        return assigned_agents
    
    async def _create_agent_assignments(self, document: DocumentUpload) -> List[str]:
        """Create task assignments for agents"""
        assignments_created = []
        
        for agent_id in document.assigned_agents:
            assignment_id = f"TASK_{document.customer_id}_{agent_id}_{int(datetime.now().timestamp())}"
            
            # Determine task description and type based on agent and document
            task_description, assignment_type = self._generate_task_description(agent_id, document)
            
            assignment = AgentTaskAssignment(
                assignment_id=assignment_id,
                customer_id=document.customer_id,
                agent_id=agent_id,
                agent_type=self._get_agent_type(agent_id),
                task_description=task_description,
                assignment_type=assignment_type,
                documents=[document.document_id],
                priority=document.priority,
                due_date=datetime.now() + timedelta(hours=48)  # 48 hour default
            )
            
            # Store assignment
            if document.customer_id not in self.agent_assignments:
                self.agent_assignments[document.customer_id] = []
            
            self.agent_assignments[document.customer_id].append(assignment)
            self.active_tasks[assignment_id] = assignment
            
            # Send task to agent
            await self._send_task_to_agent(assignment)
            
            assignments_created.append(assignment_id)
        
        return assignments_created
    
    def _generate_task_description(self, agent_id: str, document: DocumentUpload) -> Tuple[str, AgentAssignmentType]:
        """Generate task description for agent assignment"""
        
        agent_type = self._get_agent_type(agent_id)
        doc_type = document.document_type
        
        if agent_id == "data_migration_agent":
            if doc_type == DocumentType.CURRENT_SYSTEM_DATA:
                return ("Analyze current system data export and create migration plan", AgentAssignmentType.PRIMARY)
            elif doc_type == DocumentType.INVENTORY_DATA:
                return ("Review inventory data structure and validate for migration", AgentAssignmentType.PRIMARY)
            else:
                return ("Review document for data migration considerations", AgentAssignmentType.SUPPORTING)
        
        elif agent_id == "configuration_agent":
            if doc_type == DocumentType.BUSINESS_REQUIREMENTS:
                return ("Analyze business requirements and create system configuration plan", AgentAssignmentType.PRIMARY)
            elif doc_type == DocumentType.TECHNICAL_SPECS:
                return ("Review technical specifications for system configuration", AgentAssignmentType.PRIMARY)
            else:
                return ("Provide configuration input based on document content", AgentAssignmentType.CONSULTATIVE)
        
        elif agent_id == "project_manager_agent":
            return ("Coordinate project activities related to this document", AgentAssignmentType.SUPPORTING)
        
        elif "furniture" in agent_id:
            return ("Analyze document for furniture manufacturing-specific requirements", AgentAssignmentType.PRIMARY)
        
        elif "injection_molding" in agent_id:
            return ("Review document for injection molding process considerations", AgentAssignmentType.PRIMARY)
        
        elif "electrical" in agent_id:
            return ("Evaluate document for electrical equipment manufacturing needs", AgentAssignmentType.PRIMARY)
        
        else:
            return (f"Review and process document: {document.original_filename}", AgentAssignmentType.SUPPORTING)
    
    def _get_agent_type(self, agent_id: str) -> str:
        """Get agent type from agent ID"""
        if "data_migration" in agent_id:
            return "DATA_MIGRATION"
        elif "configuration" in agent_id:
            return "CONFIGURATION"
        elif "project_manager" in agent_id:
            return "PROJECT_MANAGEMENT"
        elif "furniture" in agent_id:
            return "INDUSTRY_FURNITURE"
        elif "injection_molding" in agent_id:
            return "INDUSTRY_INJECTION_MOLDING"
        elif "electrical" in agent_id:
            return "INDUSTRY_ELECTRICAL"
        else:
            return "GENERAL"
    
    async def _send_task_to_agent(self, assignment: AgentTaskAssignment):
        """Send task assignment to specialized agent"""
        task_message = AgentMessage(
            agent_id=self.agent_id,
            target_agent_id=assignment.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "action": "process_customer_document",
                "assignment_id": assignment.assignment_id,
                "customer_id": assignment.customer_id,
                "task_description": assignment.task_description,
                "documents": assignment.documents,
                "requirements": assignment.requirements,
                "priority": assignment.priority.value,
                "due_date": assignment.due_date.isoformat() if assignment.due_date else None
            },
            priority=assignment.priority
        )
        
        # In real implementation, this would route through the message router
        self.logger.info(f"Sent task {assignment.assignment_id} to agent {assignment.agent_id}")
    
    def _estimate_processing_time(self, document_type: DocumentType) -> float:
        """Estimate processing time in hours for document type"""
        processing_times = {
            DocumentType.BUSINESS_REQUIREMENTS: 4.0,
            DocumentType.CURRENT_SYSTEM_DATA: 6.0,
            DocumentType.FINANCIAL_DATA: 3.0,
            DocumentType.INVENTORY_DATA: 4.0,
            DocumentType.PRODUCTION_DATA: 5.0,
            DocumentType.TECHNICAL_SPECS: 3.0,
            DocumentType.PROCESS_DOCUMENTATION: 2.0,
            DocumentType.COMPLIANCE_DOCS: 2.0,
            DocumentType.UNKNOWN: 2.0
        }
        
        return processing_times.get(document_type, 2.0)
    
    async def _coordinate_agent_tasks(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate tasks across multiple agents"""
        customer_id = payload.get("customer_id")
        task_requirements = payload.get("task_requirements", {})
        document_references = payload.get("document_references", [])
        
        # Create coordination plan
        coordination_plan = {
            "coordination_id": f"COORD_{customer_id}_{int(datetime.now().timestamp())}",
            "customer_id": customer_id,
            "involved_agents": [],
            "task_dependencies": [],
            "parallel_tasks": [],
            "sequential_tasks": []
        }
        
        assignments_created = []
        
        # Determine which agents need to be involved
        required_capabilities = task_requirements.get("required_capabilities", [])
        for capability in required_capabilities:
            suitable_agents = self._find_agents_by_capability(capability)
            for agent_id in suitable_agents:
                if agent_id not in coordination_plan["involved_agents"]:
                    coordination_plan["involved_agents"].append(agent_id)
        
        # Create task assignments with dependencies
        for agent_id in coordination_plan["involved_agents"]:
            assignment_id = f"COORD_TASK_{customer_id}_{agent_id}_{int(datetime.now().timestamp())}"
            
            assignment = AgentTaskAssignment(
                assignment_id=assignment_id,
                customer_id=customer_id,
                agent_id=agent_id,
                agent_type=self._get_agent_type(agent_id),
                task_description=f"Coordinate on {task_requirements.get('task_name', 'customer requirements')}",
                assignment_type=AgentAssignmentType.PRIMARY,
                documents=document_references,
                requirements=task_requirements,
                priority=Priority(task_requirements.get("priority", "MEDIUM"))
            )
            
            assignments_created.append(assignment_id)
            self.active_tasks[assignment_id] = assignment
            
            await self._send_task_to_agent(assignment)
        
        # Estimate timeline
        timeline_estimate = {
            "total_estimated_hours": len(coordination_plan["involved_agents"]) * 4.0,
            "parallel_processing_hours": 6.0,
            "estimated_completion": (datetime.now() + timedelta(hours=48)).isoformat()
        }
        
        return {
            "assignments_created": assignments_created,
            "coordination_plan": coordination_plan,
            "timeline_estimate": timeline_estimate
        }
    
    def _find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents with specific capability"""
        suitable_agents = []
        
        for agent_id, agent_info in self.available_agents.items():
            agent_capabilities = agent_info.get("capabilities", [])
            if capability in agent_capabilities:
                suitable_agents.append(agent_id)
        
        return suitable_agents
    
    async def _get_customer_implementation_status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive customer implementation status"""
        customer_id = payload.get("customer_id")
        
        if not customer_id:
            return {"error": "Customer ID required"}
        
        # Get basic implementation status
        dashboard_data = system_state.get_customer_dashboard(customer_id)
        
        # Get document processing status
        customer_docs = self.customer_documents.get(customer_id, [])
        documents_status = {
            "total_documents": len(customer_docs),
            "processed_documents": len([d for d in customer_docs if d.processing_status == "PROCESSED"]),
            "pending_documents": len([d for d in customer_docs if d.processing_status in ["UPLOADED", "ANALYZING"]]),
            "recent_uploads": [
                d.to_dict() for d in customer_docs 
                if (datetime.now() - d.uploaded_at).days <= 7
            ]
        }
        
        # Get agent task status
        customer_assignments = self.agent_assignments.get(customer_id, [])
        tasks_status = {
            "total_tasks": len(customer_assignments),
            "completed_tasks": len([t for t in customer_assignments if t.status == "COMPLETED"]),
            "active_tasks": len([t for t in customer_assignments if t.status in ["ASSIGNED", "IN_PROGRESS"]]),
            "blocked_tasks": len([t for t in customer_assignments if t.status == "BLOCKED"])
        }
        
        # Generate next recommended actions
        next_actions = []
        if documents_status["pending_documents"] > 0:
            next_actions.append("Complete document processing and analysis")
        
        if tasks_status["blocked_tasks"] > 0:
            next_actions.append("Resolve blocked tasks to continue progress")
        
        if documents_status["total_documents"] < 3:
            next_actions.append("Upload additional business requirements or current system data")
        
        return {
            "implementation_status": dashboard_data,
            "documents_status": documents_status,
            "tasks_status": tasks_status,
            "next_actions": next_actions,
            "overall_progress": {
                "documents_processed": (documents_status["processed_documents"] / max(1, documents_status["total_documents"])) * 100,
                "tasks_completed": (tasks_status["completed_tasks"] / max(1, tasks_status["total_tasks"])) * 100
            }
        }
    
    async def _assign_task_to_agent(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Manually assign task to specific agent"""
        customer_id = payload.get("customer_id")
        agent_id = payload.get("agent_id")
        task_description = payload.get("task_description")
        requirements = payload.get("requirements", {})
        
        assignment_id = f"MANUAL_TASK_{customer_id}_{agent_id}_{int(datetime.now().timestamp())}"
        
        assignment = AgentTaskAssignment(
            assignment_id=assignment_id,
            customer_id=customer_id,
            agent_id=agent_id,
            agent_type=self._get_agent_type(agent_id),
            task_description=task_description,
            assignment_type=AgentAssignmentType.PRIMARY,
            requirements=requirements,
            priority=Priority(requirements.get("priority", "MEDIUM"))
        )
        
        # Store assignment
        if customer_id not in self.agent_assignments:
            self.agent_assignments[customer_id] = []
        
        self.agent_assignments[customer_id].append(assignment)
        self.active_tasks[assignment_id] = assignment
        
        # Send to agent
        await self._send_task_to_agent(assignment)
        
        return {
            "assignment_id": assignment_id,
            "status": "ASSIGNED",
            "agent_id": agent_id,
            "estimated_completion": (datetime.now() + timedelta(hours=24)).isoformat()
        }
    
    async def _get_document_processing_status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of document processing"""
        customer_id = payload.get("customer_id")
        document_id = payload.get("document_id")
        
        if document_id:
            # Get specific document status
            document = None
            for doc in self.customer_documents.get(customer_id, []):
                if doc.document_id == document_id:
                    document = doc
                    break
            
            if not document:
                return {"error": "Document not found"}
            
            return {
                "document": document.to_dict(),
                "processing_details": {
                    "assigned_agents": document.assigned_agents,
                    "processing_status": document.processing_status,
                    "content_summary": document.content_summary
                }
            }
        
        else:
            # Get all documents status for customer
            customer_docs = self.customer_documents.get(customer_id, [])
            
            return {
                "total_documents": len(customer_docs),
                "documents": [doc.to_dict() for doc in customer_docs],
                "processing_summary": {
                    "uploaded": len([d for d in customer_docs if d.processing_status == "UPLOADED"]),
                    "analyzing": len([d for d in customer_docs if d.processing_status == "ANALYZING"]),
                    "processed": len([d for d in customer_docs if d.processing_status == "PROCESSED"]),
                    "distributed": len([d for d in customer_docs if d.processing_status == "DISTRIBUTED"])
                }
            }
    
    async def _handle_customer_communication(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer communication requests"""
        customer_id = payload.get("customer_id")
        communication_type = payload.get("type", "general")
        message = payload.get("message", "")
        
        # This integrates with the original Lead Agent conversation handling
        response = {
            "response_text": "",
            "suggested_actions": [],
            "follow_up_required": False
        }
        
        if communication_type == "document_question":
            response["response_text"] = "I can help you with document-related questions. What specific information do you need about your uploaded documents?"
            response["suggested_actions"] = [
                {"action": "check_document_status", "label": "Check Document Processing Status"},
                {"action": "upload_additional", "label": "Upload Additional Documents"},
                {"action": "speak_to_specialist", "label": "Speak to Document Specialist"}
            ]
        
        elif communication_type == "progress_inquiry":
            status = await self._get_customer_implementation_status({"customer_id": customer_id})
            response["response_text"] = f"Your implementation is progressing well. You have {status['documents_status']['processed_documents']} documents processed and {status['tasks_status']['completed_tasks']} tasks completed."
            
        else:
            response["response_text"] = "I'm here to help manage your implementation. How can I assist you today?"
            response["suggested_actions"] = [
                {"action": "upload_document", "label": "Upload Document"},
                {"action": "check_progress", "label": "Check Progress"},
                {"action": "ask_question", "label": "Ask Question"}
            ]
        
        return response
    
    async def _handle_task_completion(self, payload: Dict[str, Any]):
        """Handle task completion notification"""
        assignment_id = payload.get("assignment_id")
        result_data = payload.get("result_data", {})
        
        if assignment_id in self.active_tasks:
            task = self.active_tasks[assignment_id]
            task.status = "COMPLETED"
            task.completed_at = datetime.now()
            task.result_data = result_data
            
            # Update document processing status if applicable
            for doc_id in task.documents:
                for customer_docs in self.customer_documents.values():
                    for doc in customer_docs:
                        if doc.document_id == doc_id:
                            doc.processing_status = "PROCESSED"
                            doc.processed_at = datetime.now()
            
            self.logger.info(f"Task {assignment_id} completed by {task.agent_id}")
    
    async def _handle_task_blocked(self, payload: Dict[str, Any]):
        """Handle task blocked notification"""
        assignment_id = payload.get("assignment_id")
        blocking_reason = payload.get("reason", "Unknown")
        
        if assignment_id in self.active_tasks:
            task = self.active_tasks[assignment_id]
            task.status = "BLOCKED"
            task.feedback = blocking_reason
            
            self.logger.warning(f"Task {assignment_id} blocked: {blocking_reason}")
    
    async def _handle_agent_question(self, payload: Dict[str, Any]):
        """Handle questions from agents"""
        # Implementation for handling agent questions
        pass
    
    async def _handle_progress_update(self, payload: Dict[str, Any]):
        """Handle progress updates from agents"""
        # Implementation for handling progress updates
        pass
    
    async def _document_processing_loop(self):
        """Background document processing loop"""
        while self.status != "SHUTDOWN":
            try:
                if self.document_processing_queue:
                    document = self.document_processing_queue.pop(0)
                    
                    # Process document
                    document.processing_status = "ANALYZING"
                    
                    # Simulate document analysis
                    await self._analyze_document_content(document)
                    
                    document.processing_status = "PROCESSED"
                    document.processed_at = datetime.now()
                    
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in document processing loop: {str(e)}")
                await asyncio.sleep(10)
    
    async def _analyze_document_content(self, document: DocumentUpload):
        """Analyze document content and extract key information"""
        # Simulate document analysis
        if document.document_type == DocumentType.BUSINESS_REQUIREMENTS:
            document.extracted_data = {
                "requirements_count": 15,
                "priority_requirements": ["inventory management", "production scheduling"],
                "integration_needs": ["accounting system", "CRM"]
            }
            document.tags = ["requirements", "integration", "priority"]
        
        elif document.document_type == DocumentType.CURRENT_SYSTEM_DATA:
            document.extracted_data = {
                "record_count": 50000,
                "data_quality_score": 0.85,
                "missing_fields": ["updated_date", "category"],
                "data_types": ["inventory", "transactions", "customers"]
            }
            document.tags = ["data", "migration", "quality-check"]
        
        elif document.document_type == DocumentType.INVENTORY_DATA:
            document.extracted_data = {
                "item_count": 1200,
                "categories": ["raw_materials", "finished_goods", "supplies"],
                "data_completeness": 0.92,
                "value_analysis": {"total_value": 250000, "top_category": "raw_materials"}
            }
            document.tags = ["inventory", "analysis", "valuation"]
    
    async def _task_monitoring_loop(self):
        """Background task monitoring loop"""
        while self.status != "SHUTDOWN":
            try:
                current_time = datetime.now()
                
                # Check for overdue tasks
                for assignment_id, task in self.active_tasks.items():
                    if task.due_date and current_time > task.due_date and task.status not in ["COMPLETED", "BLOCKED"]:
                        # Mark as overdue and potentially escalate
                        task.feedback = "Task overdue - may require escalation"
                        self.logger.warning(f"Task {assignment_id} is overdue")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in task monitoring loop: {str(e)}")
                await asyncio.sleep(300)
    
    async def _customer_communication_loop(self):
        """Background customer communication loop"""
        while self.status != "SHUTDOWN":
            try:
                # Check for customers who might need proactive updates
                # This would integrate with implementation milestones and progress tracking
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in customer communication loop: {str(e)}")
                await asyncio.sleep(1800)


# Export main component
__all__ = ["CustomerManagerAgent", "DocumentType", "AgentAssignmentType", "DocumentUpload", "AgentTaskAssignment"]