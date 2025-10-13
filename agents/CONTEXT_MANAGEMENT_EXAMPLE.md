---
name: context-management-example
---

# Context Management Example: Preventing Agent Overwhelm

This example demonstrates how the workspace protocol prevents agents from being overwhelmed with context as work progresses through multiple agents.

## Scenario: E-commerce Platform Development

### Without Context Management (Traditional Approach)
```
Agent 1 (Analyzer): 50KB analysis
    ↓ (passes all 50KB)
Agent 2 (Designer): +30KB design docs = 80KB total
    ↓ (passes all 80KB)
Agent 3 (Backend): +100KB implementation = 180KB total
    ↓ (passes all 180KB)
Agent 4 (Frontend): +80KB UI code = 260KB total
    ↓ (passes all 260KB)
Agent 5 (Tester): +120KB test results = 380KB total
    ❌ OVERWHELMED - Too much context to process effectively
```

### With Context Management (Workspace Protocol)
```
Agent 1 (Analyzer): 50KB analysis
    ↓
Context Compressor: 
    - Archives detailed analysis → detailed/analyzer-full-50KB.tar.gz
    - Creates summary → active/summary-5KB.md
    - Extracts decisions → indices/decisions.json (2KB)
    ↓ (passes 7KB focused context)
    
Agent 2 (Designer): Receives 7KB brief + creates 30KB design
    ↓
Context Compressor:
    - Archives design docs → detailed/designer-full-30KB.tar.gz  
    - Updates summary → active/summary-6KB.md
    - Adds API contracts → indices/interfaces.json (3KB)
    ↓ (passes 9KB focused context)
    
Agent 3 (Backend): Receives 9KB brief + creates 100KB implementation
    ↓
Context Compressor:
    - Archives code details → detailed/backend-full-100KB.tar.gz
    - Creates implementation summary → 4KB
    - Preserves only interfaces & endpoints
    ↓ (passes 8KB focused context)
    
Agent 4 (Frontend): Receives 8KB API-focused brief
    ✅ OPTIMAL - Only sees what's needed (API contracts, not backend internals)
    
Agent 5 (Tester): Receives 10KB test-focused brief
    ✅ EFFICIENT - Gets test targets, not implementation details
```

## Real Example: Authentication System

### Traditional Context Explosion
```markdown
## Full Context Passed to Frontend Developer (180KB)

### From Analyzer (50KB)
- Market research on auth methods...
- Competitive analysis of 20 platforms...
- User survey results...
[45KB of analysis details that frontend doesn't need]

### From Designer (30KB)  
- Detailed user flow diagrams...
- Color psychology research...
- Accessibility compliance docs...
[25KB of design rationale that frontend already incorporated]

### From Backend (100KB)
- Database schema migrations...
- Password hashing implementation...
- Session management internals...
- Rate limiting algorithms...
[90KB of backend details that frontend shouldn't know]

### What Frontend Actually Needs
❌ Buried in 180KB of context
```

### Workspace Protocol Compression
```markdown
## Compressed Context for Frontend Developer (8KB)

### Executive Summary (2KB)
- Building: JWT-based authentication system
- Status: Backend complete, APIs tested
- Your task: Create login/register UI components

### API Interfaces (3KB)
```json
{
  "endpoints": {
    "login": "POST /api/auth/login",
    "register": "POST /api/auth/register", 
    "logout": "POST /api/auth/logout"
  },
  "request_format": { "email": "string", "password": "string" },
  "response_format": { "token": "string", "user": "object" }
}
```

### Key Decisions (2KB)
- Use React Hook Form for validation
- Store JWT in httpOnly cookie
- Show password strength indicator
- Email verification required

### Your Tasks (1KB)
1. Create LoginForm component
2. Create RegisterForm component  
3. Implement form validation
4. Handle API responses
5. Update auth context

### Full Details Available
- Design mockups: artifacts/designs/auth-ui.fig
- Backend details: archive/2024-01-15/backend-auth.tar.gz
- Test cases: outputs/testing/auth-test-cases.md
```

## Context Compression Strategies

### 1. Role-Based Filtering
```python
def create_agent_context(agent_type, full_context):
    if agent_type == "frontend":
        return {
            "apis": extract_api_contracts(full_context),
            "ui_requirements": extract_ui_specs(full_context),
            "decisions": filter_decisions(full_context, "frontend")
        }
    elif agent_type == "tester":
        return {
            "test_targets": extract_testable_components(full_context),
            "requirements": extract_requirements(full_context),
            "known_issues": extract_issues(full_context)
        }
```

### 2. Progressive Detail Reduction
```
Level 1: Full Implementation (100KB)
- Complete source code with comments
- Detailed error handling
- All edge cases

Level 2: Technical Summary (30KB)
- Function signatures
- Key algorithms
- Important decisions

Level 3: Interface Summary (10KB)
- Public APIs only
- Input/output formats
- Usage examples

Level 4: Executive Brief (3KB)
- What it does
- How to use it
- Where to find details
```

### 3. Smart Archiving
```
Active Context (Latest Only):
.agent-workspace/context/active/
├── current-sprint.md (5KB)
├── blocking-issues.md (2KB)
└── next-tasks.md (3KB)

Archived Context (Compressed):
.agent-workspace/archive/2024-01-15/
├── sprint-1-complete.tar.gz (500KB→50KB)
├── resolved-issues.tar.gz (200KB→20KB)
└── completed-tasks.tar.gz (300KB→30KB)
```

## Benefits Achieved

1. **Performance**: Agents process 95% less context
2. **Accuracy**: Reduced confusion from irrelevant information
3. **Speed**: Faster agent response times
4. **Quality**: More focused, relevant outputs
5. **Scalability**: Can handle projects with 50+ agent handoffs
6. **Debugging**: Full history preserved in archives

## Implementation Tips

1. **Compress Early**: Don't wait for context to grow
2. **Preserve Interfaces**: Always keep API contracts accessible
3. **Index Everything**: Quick lookups without loading full context
4. **Tailor Content**: Each agent type needs different information
5. **Archive Aggressively**: Move detailed content to archives immediately
6. **Measure Impact**: Track context sizes and agent performance

This approach ensures agents remain efficient and focused throughout the entire development lifecycle, regardless of project complexity or team size.