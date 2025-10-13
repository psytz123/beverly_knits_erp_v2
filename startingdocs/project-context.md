# Project Context
**Created:** 2025-10-03T02:06:00Z
**Last Updated:** 2025-10-03T02:06:00Z

## Overview
Complete Enterprise Resource Planning (ERP) system for Beverly Knits manufacturing operations. The system implements a modern microservices architecture with event-driven communication, workflow orchestration, and comprehensive business process automation.

## Objectives

### Primary
- Build a production-ready ERP system with 10 specialized microservices
- Implement event-driven architecture for real-time data synchronization
- Establish workflow orchestration for complex business processes
- Create unified API Gateway for external access
- Ensure high availability, scalability, and maintainability

### Secondary
- Implement AI-driven forecasting and analytics capabilities
- Enable real-time inventory tracking and management
- Automate procurement and shipping workflows
- Integrate quality control processes
- Support maintenance scheduling and equipment tracking

## Current State
- Workspace initialized: 2025-10-03T02:06:00Z
- Next agent: orchestrator-agent
- Phase: initialization-complete
- Status: ready-for-planning

## Project Type
Enterprise ERP System with Microservices Architecture

## Success Criteria
1. All 10 microservices operational with FastAPI
2. API Gateway (Kong) routing requests correctly
3. Event bus (Kafka/RabbitMQ) handling async communication
4. Temporal workflows orchestrating business processes
5. Test coverage >= 85%, type coverage = 100%
6. Code quality metrics within defined gates
7. Documentation complete for all services
8. Deployment ready with Docker/Kubernetes configurations

## Constraints
- Python 3.10+ required
- Must follow code quality standards (cyclomatic complexity <= 10, <3% duplication)
- Files limited to 500 LOC, functions to 50 LOC
- Type hints required for all public functions
- Test-Driven Development (TDD) approach mandatory
- All architectural decisions must be documented (ADR)

## Next Steps
1. Orchestrator agent to analyze requirements and create detailed architecture
2. Design agent to create API specifications and data models
3. Implementation agents for each microservice
4. Testing and integration verification
5. Documentation and deployment preparation
