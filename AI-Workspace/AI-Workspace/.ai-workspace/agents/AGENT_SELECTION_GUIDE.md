---
name: agent-selection-guide
---

# Claude Agent Selection Guide

This guide helps you choose the right agent for your specific needs and avoid confusion between similar agents.

## Quick Selection Matrix

### When You Need General vs. Specialized Agents

| If you need... | Use this agent | Instead of... | Why |
|----------------|----------------|---------------|-----|
| Frontend work (any framework) | `frontend-developer` | Framework-specific agents | Start here for general frontend tasks |
| React-specific work | `react-specialist` | `frontend-developer` | When you need deep React expertise |
| Vue-specific work | `vue-expert` | `frontend-developer` | When you need deep Vue expertise |
| Angular-specific work | `angular-architect` | `frontend-developer` | When you need deep Angular expertise |
| API design (general) | `api-designer` | `api-architect`, `api-documenter` | For initial API design and architecture |
| API documentation | `api-documenter` | `documentation-specialist` | Specialized in OpenAPI/Swagger docs |
| General documentation | `documentation-specialist` | `technical-writer` | For code and technical documentation |
| Technical writing | `technical-writer` | `documentation-specialist` | For user-facing documentation |
| ML engineering | `ml-engineer` | N/A (merged agent) | Complete ML lifecycle coverage |
| DevOps incidents | `devops-incident-responder` | `security-incident-responder` | For operational incidents |
| Security incidents | `security-incident-responder` | `devops-incident-responder` | For security breaches |
| Code quality | `code-reviewer` | N/A | Located in quality-security folder |
| WordPress development | `wordpress-master` | N/A | Located in business-product folder |

## Decision Trees

### Frontend Development Decision Tree

```
Need frontend work?
├── Multiple frameworks or general UI? → frontend-developer
├── React specific?
│   ├── Component architecture? → react-component-architect
│   ├── Next.js? → react-nextjs-expert
│   └── General React? → react-specialist
├── Vue specific?
│   ├── Component architecture? → vue-component-architect
│   ├── Nuxt? → vue-nuxt-expert
│   └── State management? → vue-state-manager
└── Angular specific? → angular-architect
```

### API Development Decision Tree

```
Need API work?
├── Initial design/architecture? → api-designer
├── Universal API patterns? → api-architect
├── API documentation? → api-documenter
├── GraphQL specific? → graphql-architect
└── REST implementation? → backend-developer
```

### Documentation Decision Tree

```
Need documentation?
├── API documentation? → api-documenter
├── Code/technical docs? → documentation-specialist
├── User-facing content? → technical-writer
├── Developer experience? → documentation-engineer
└── Marketing content? → content-marketer
```

### Incident Response Decision Tree

```
Have an incident?
├── Security breach? → security-incident-responder
├── System outage? → devops-incident-responder
├── Performance issue? → performance-engineer
└── General debugging? → debugger
```

## Agent Hierarchies

### 1. General → Specialized

Always consider starting with general agents before moving to specialized ones:

- **General**: `frontend-developer` → **Specialized**: `react-specialist`, `vue-expert`
- **General**: `backend-developer` → **Specialized**: `django-developer`, `rails-expert`
- **General**: `mobile-developer` → **Specialized**: `flutter-expert`, `swift-expert`

### 2. Planning → Implementation

For complex projects, use this flow:

1. **Planning**: `tech-lead-orchestrator` or `project-analyst`
2. **Architecture**: `architect-reviewer` or domain architects
3. **Implementation**: Specific development agents
4. **Quality**: `code-reviewer`, `qa-expert`
5. **Documentation**: `documentation-specialist`

### 3. Complementary Pairs

These agents work well together:

- `backend-developer` + `frontend-developer` = Full-stack development
- `api-designer` + `api-documenter` = Complete API lifecycle
- `security-engineer` + `security-auditor` = Comprehensive security
- `data-engineer` + `data-scientist` = End-to-end data solutions

## Common Confusions Resolved

### ML/AI Agents

- **ml-engineer**: Complete ML engineering (training + deployment)
- **ai-engineer**: Broader AI systems and integration
- **data-scientist**: Analysis and modeling focus
- **llm-architect**: Specific to large language models

### DevOps/Infrastructure Agents

- **devops-engineer**: General DevOps practices
- **platform-engineer**: Platform building and tooling
- **sre-engineer**: Reliability and monitoring focus
- **deployment-engineer**: Deployment specialization

### Performance Agents

- **performance-optimizer**: Code-level optimization
- **performance-engineer**: System-wide performance
- **performance-monitor**: Monitoring and metrics focus

## Directory Structure and Organization

1. **Categorical Organization**
   - Agents are organized in numbered category folders (e.g., `01-development`, `06-data-ai`)
   - Each agent must be in its appropriate category folder
   - Never place agents in the global/root agents directory

2. **Agent Directory Names**
   - Use kebab-case for directory names (e.g., `ai-engineer`, not `ai` or `AI`)
   - Names should be specific and descriptive
   - Follow existing naming patterns in the category

3. **Required Structure**
   - Each agent should have its own directory
   - Main agent file should match directory name (e.g., `ai-engineer/ai-engineer.md`)
   - No nested global directories (e.g., avoid `/ai/` containing multiple agents)

## Best Practices

1. **Start with the tech-lead-orchestrator** for complex projects - it will recommend the right agents
2. **Use general agents first**, then bring in specialists as needed
3. **Combine complementary agents** for comprehensive solutions
4. **Check agent descriptions** when unsure - they clearly state their expertise
5. **Look at the folder structure** - agents are organized by primary function
6. **Follow directory structure rules** - proper organization ensures correct initialization

## Quick Reference

### By Technology Stack

**JavaScript/TypeScript**: `javascript-pro`, `typescript-pro`, `nodejs` variants
**Python**: `python-pro`, `django-developer`, `data-scientist`
**Mobile**: `mobile-developer`, `flutter-expert`, `swift-expert`, `kotlin-specialist`
**Cloud**: `cloud-architect`, `aws`/`azure`/`gcp` specialists

### By Task Type

**New Project**: `tech-lead-orchestrator` → relevant specialists
**Bug Fixing**: `debugger` → `code-reviewer`
**Performance**: `performance-engineer` → `performance-optimizer`
**Security**: `security-auditor` → `security-engineer`
**Documentation**: Choose based on decision tree above

## Agent Chaining for Complex Workflows

For complex tasks, chain multiple agents in sequence to leverage their specialized expertise:

### Common Agent Chains

#### Performance Optimization Chain
```
performance-monitor → performance-engineer → performance-optimizer → code-reviewer
```
1. **Monitor** identifies bottlenecks
2. **Engineer** analyzes system-wide issues
3. **Optimizer** implements code-level fixes
4. **Reviewer** validates the changes

#### Security Hardening Chain
```
security-auditor → security-engineer → penetration-tester → security-incident-responder
```
1. **Auditor** identifies vulnerabilities
2. **Engineer** implements security controls
3. **Tester** validates security measures
4. **Responder** prepares incident procedures

#### Legacy Modernization Chain
```
code-archaeologist → architect-reviewer → refactoring-specialist → test-automator
```
1. **Archaeologist** maps legacy code
2. **Architect** designs modernization plan
3. **Refactoring** specialist implements changes
4. **Automator** adds test coverage

#### Full Feature Development Chain
```
project-analyst → api-designer → backend-developer → frontend-developer → qa-expert
```
1. **Analyst** breaks down requirements
2. **Designer** creates API specifications
3. **Backend** implements server logic
4. **Frontend** builds user interface
5. **QA** ensures quality

### Advanced Chaining Patterns

#### Parallel-Sequential Pattern
```
                    ┌─→ frontend-developer ─┐
tech-lead-orchestrator                      → qa-expert → code-reviewer
                    └─→ backend-developer ──┘
```

#### Iterative Pattern
```
debugger → error-detective → code-reviewer → debugger (repeat until resolved)
```

#### Conditional Pattern
```
code-reviewer → {
    if security issues → security-engineer
    if performance issues → performance-optimizer
    if architecture issues → architect-reviewer
}
```

### Chaining Best Practices

1. **Information Handoff**: Each agent should produce clear outputs for the next
2. **Context Preservation**: Maintain relevant context between agents
3. **Error Handling**: Plan for failures in the chain
4. **Progress Tracking**: Monitor completion of each stage
5. **Parallel Execution**: Run independent agents simultaneously when possible

### Example: Complete API Redesign

```
Step 1: code-archaeologist
- Analyze existing API structure
- Document current usage patterns
- Identify technical debt

Step 2: api-designer
- Design new API structure
- Create OpenAPI specifications
- Plan migration strategy

Step 3: backend-developer + frontend-developer (parallel)
- Implement new endpoints
- Update client code
- Maintain backward compatibility

Step 4: api-documenter
- Generate comprehensive docs
- Create migration guides
- Update examples

Step 5: qa-expert
- Test all endpoints
- Verify backward compatibility
- Performance benchmarking

Step 6: deployment-engineer
- Plan rollout strategy
- Implement feature flags
- Monitor deployment
```

### Creating Custom Chains

When designing your own chains:

1. **Identify the phases** of your workflow
2. **Match agents to phases** based on expertise
3. **Define handoff points** between agents
4. **Plan parallel opportunities** for efficiency
5. **Include validation steps** with review agents

Remember: When in doubt, start with the `tech-lead-orchestrator` - it's designed to analyze your needs and recommend the right team of agents, including optimal chaining strategies!