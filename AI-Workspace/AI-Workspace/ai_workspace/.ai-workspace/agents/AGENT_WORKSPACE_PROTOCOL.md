---
name: agent-workspace-protocol
---

# Agent Workspace Protocol

This document defines how agents share outputs and provide context through a structured project workspace.

## Workspace Structure

Every multi-agent project should initialize with this structure:

```
project-root/
├── .agent-workspace/           # Agent collaboration directory
│   ├── manifest.json          # Project metadata and agent registry
│   ├── context/               # Shared context and state
│   │   ├── project-context.md # Overall project understanding
│   │   ├── technical-stack.md # Technology decisions
│   │   ├── constraints.md     # Limitations and requirements
│   │   └── decisions.md       # Architectural decisions log
│   ├── handoffs/              # Agent-to-agent communication
│   │   ├── {timestamp}-{from}-{to}.json
│   │   └── active/            # Current active handoffs
│   ├── outputs/               # Agent work products
│   │   ├── analysis/          # Research and analysis outputs
│   │   ├── design/            # Architecture and design docs
│   │   ├── implementation/    # Code and config outputs
│   │   ├── testing/           # Test results and reports
│   │   └── documentation/     # Generated documentation
│   ├── artifacts/             # Reusable components
│   │   ├── schemas/           # API schemas, DB schemas
│   │   ├── templates/         # Code templates
│   │   ├── configs/           # Configuration files
│   │   └── scripts/           # Utility scripts
│   └── logs/                  # Agent activity logs
│       └── {date}-{agent}.log
├── src/                       # Actual project source
├── tests/                     # Project tests
└── docs/                      # Project documentation
```

## Workspace Initialization

### 1. Workspace Creator Agent

Create a new agent specifically for workspace initialization:

```yaml
---
name: workspace-initializer
description: Initializes and maintains the agent workspace structure for multi-agent collaboration
tools: Read, Write, MultiEdit, Bash, Glob
---

You are a workspace initialization specialist that sets up and maintains the structured environment for multi-agent collaboration.

When invoked:
1. Create the .agent-workspace directory structure
2. Initialize manifest.json with project metadata
3. Set up initial context files
4. Establish handoff protocols
5. Configure logging system
```

### 2. Manifest File Structure

```json
{
  "project": {
    "name": "project-name",
    "description": "Project description",
    "created": "2024-01-15T10:00:00Z",
    "type": "web-app|api|library|etc"
  },
  "workspace": {
    "version": "1.0",
    "initialized": "2024-01-15T10:00:00Z"
  },
  "agents": {
    "active": [],
    "completed": [],
    "registry": {
      "agent-name": {
        "invoked": "timestamp",
        "status": "active|completed",
        "outputs": ["file1", "file2"]
      }
    }
  },
  "context": {
    "last_updated": "timestamp",
    "checksum": "hash"
  }
}
```

## Agent Handoff Protocol

### 1. Standard Handoff Format

Every agent should create a handoff file when passing work:

```json
{
  "handoff": {
    "id": "unique-id",
    "timestamp": "2024-01-15T10:30:00Z",
    "from": "backend-developer",
    "to": "frontend-developer",
    "status": "pending|accepted|completed"
  },
  "context": {
    "task": "Implement user authentication",
    "phase": "api-complete",
    "summary": "Authentication API endpoints completed"
  },
  "outputs": {
    "created": [
      "src/api/auth.js",
      "src/models/user.js"
    ],
    "modified": [
      "src/routes/index.js"
    ],
    "documentation": [
      ".agent-workspace/outputs/design/auth-api.md"
    ]
  },
  "next_steps": {
    "required": [
      "Create login form component",
      "Implement token storage",
      "Add auth state management"
    ],
    "optional": [
      "Add remember me functionality",
      "Implement social auth"
    ]
  },
  "technical_details": {
    "api_endpoints": [
      "POST /api/auth/login",
      "POST /api/auth/logout",
      "GET /api/auth/status"
    ],
    "dependencies": [
      "jsonwebtoken@9.0.0",
      "bcrypt@5.1.1"
    ],
    "environment": {
      "JWT_SECRET": "required in .env"
    }
  },
  "warnings": [
    "CORS must be configured for frontend domain"
  ]
}
```

### 2. Context Preservation

Agents should read and update shared context:

```markdown
# .agent-workspace/context/project-context.md

## Project Overview
E-commerce platform with microservices architecture

## Current State
- Authentication service: Complete
- Product catalog: In progress
- Order management: Planned

## Key Decisions
- Using JWT for authentication
- PostgreSQL for data persistence
- React for frontend
```

## Context Management and Compression

### Preventing Agent Overwhelm

As projects progress, context can grow exponentially. To prevent agents from being overwhelmed:

1. **Context Budget**: Each agent type has a maximum context size
   - Orchestration agents: 50KB
   - Implementation agents: 20KB  
   - Review agents: 30KB
   - Testing agents: 15KB

2. **Progressive Summarization**: Use context-compressor agent
   ```
   Raw outputs (100%) → Technical Summary (30%) → Executive Summary (10%) → Key Points (5%)
   ```

3. **Layered Context Structure**:
   ```
   context/
   ├── active/        # Current working context (compressed)
   │   ├── summary.md
   │   ├── current-task.md
   │   └── key-decisions.md
   ├── detailed/      # Full historical context (archived)
   └── indices/       # Quick lookups without full context
   ```

### Context Compression Workflow

```markdown
# Automatic compression triggers:
- Before invoking agents with >20KB context
- After 5+ agents have contributed
- When handoffs exceed 5KB

# Compression chain:
tech-lead-orchestrator → context-compressor → next-agent
```

### Agent-Specific Context Packages

Each agent receives a tailored context package:

```json
{
  "context_package": {
    "brief": "5KB executive summary",
    "interfaces": "API/component contracts only",
    "decisions": "Relevant decisions only", 
    "tasks": "Specific assigned tasks",
    "references": "Links to detailed archives"
  }
}
```

## Integration with Orchestration Agents

### 1. Update tech-lead-orchestrator

Add workspace initialization and context management:

```markdown
### Task Analysis
- Initialize workspace with @workspace-initializer
- Set up context compression with @context-compressor
- [Continue with existing analysis...]

### Agent Assignments
Task 0: Initialize workspace → AGENT: @workspace-initializer
Task 1: Compress initial context → AGENT: @context-compressor
Task 2: [Existing tasks...]

### Context Management Strategy
- Run context-compressor after every 3-5 agents
- Create agent-specific briefings before each invocation
- Archive detailed outputs immediately after use
```

### 2. Agent Execution Pattern

Each agent should follow this pattern:

```yaml
# In agent definition
When invoked:
1. Read .agent-workspace/manifest.json
2. Check for pending handoffs in .agent-workspace/handoffs/active/
3. Read relevant context from .agent-workspace/context/
4. Perform assigned work
5. Write outputs to .agent-workspace/outputs/{category}/
6. Create handoff for next agent
7. Update manifest.json
8. Log activity to .agent-workspace/logs/
```

## Best Practices

### 1. Output Organization

```
outputs/
├── analysis/
│   ├── 2024-01-15-code-archaeologist-legacy-analysis.md
│   └── 2024-01-15-security-auditor-vulnerabilities.json
├── design/
│   ├── api-specification-v1.yaml
│   └── database-schema.sql
├── implementation/
│   ├── backend-api-complete.md
│   └── frontend-components-list.md
└── testing/
    ├── test-coverage-report.html
    └── performance-benchmarks.json
```

### 2. Context Updates

Agents should append to context files, not overwrite:

```markdown
## 2024-01-15 14:30 - backend-developer
Added authentication endpoints:
- POST /api/auth/login
- POST /api/auth/logout
- JWT token expiry: 24 hours
```

### 3. Artifact Sharing

Reusable components go in artifacts:

```
artifacts/
├── schemas/
│   ├── user.schema.json
│   └── openapi.yaml
├── templates/
│   ├── react-component.template
│   └── api-endpoint.template
└── configs/
    ├── eslint.config.js
    └── prettier.config.js
```

## Implementation Example

### Complete Workflow with Workspace

```bash
# 1. User initiates project
User: "Build a task management API with React frontend"

# 2. Tech-lead-orchestrator responds
"I'll coordinate this project. First, let me set up the workspace..."

# 3. Workspace initialization
workspace-initializer creates:
- .agent-workspace/ structure
- manifest.json
- Initial context files

# 4. Sequential agent execution
code-archaeologist → writes to outputs/analysis/
api-designer → reads analysis, writes to outputs/design/
backend-developer → reads design, writes to outputs/implementation/
frontend-developer → reads API docs, writes to outputs/implementation/

# 5. Each agent:
- Reads from shared context
- Checks previous handoffs
- Does work
- Creates handoff for next agent
- Updates shared context
```

## Monitoring and Visualization

### 1. Workspace Status Command

Create a workspace-monitor agent that can show:
- Active agents
- Pending handoffs
- Recent outputs
- Context changes
- Progress visualization

### 2. Handoff Queue

```json
{
  "queue": [
    {
      "priority": 1,
      "from": "api-designer",
      "to": "backend-developer",
      "status": "pending",
      "created": "2024-01-15T11:00:00Z"
    }
  ]
}
```

## Benefits

1. **Persistent Context**: No information lost between agents
2. **Clear Handoffs**: Explicit work transfer with requirements
3. **Artifact Reuse**: Shared schemas, templates, configs
4. **Progress Tracking**: Clear view of what's done and pending
5. **Debugging**: Complete audit trail of agent activities
6. **Parallel Work**: Multiple agents can work on different aspects
7. **Resume Capability**: Can pause and resume complex workflows

This workspace protocol ensures smooth collaboration between agents while maintaining context and enabling complex, multi-phase projects.