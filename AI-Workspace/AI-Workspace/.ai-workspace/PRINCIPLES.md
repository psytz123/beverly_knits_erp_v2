# The 5 Core Principles of AI Workspace

**Version:** 1.1.0
**Last Updated:** 2025-10-07
**Status:** Production

---

## Overview

AI Workspace enforces 5 core principles through automation and tooling to ensure code quality, maintainability, and sustainable development practices.

---

## Principle 1: Less is More

**Philosophy:** Maximize reuse, minimize complexity

### What It Means
- Reuse existing code before writing new code
- Keep functions simple and focused (≤50 LOC per function)
- Maintain low cyclomatic complexity (≤10 per function)
- Keep files manageable (≤500 LOC per file)
- Minimize code duplication (<3% threshold)

### How It's Enforced
- **Pre-commit hooks** check complexity and duplication
- **Pattern library** provides reusable solutions
- **Automatic enforcement** via quality gates

### Example
```python
# Bad: Complex function with high cyclomatic complexity
def process_order(order, user, inventory, payment):
    if order.status == "pending":
        if user.is_verified:
            if inventory.has_stock(order.items):
                if payment.is_valid:
                    # ... 10 more nested conditions

# Good: Single responsibility, low complexity
def validate_order(order: Order) -> ValidationResult:
    """Validate order prerequisites."""
    return ValidationResult(
        status_ok=order.status == "pending",
        user_verified=order.user.is_verified,
        stock_available=check_inventory(order.items),
        payment_valid=validate_payment(order.payment)
    )
```

### Metrics
- **Max Complexity:** 10 (enforced)
- **Max Duplication:** 3% (enforced)
- **Max File LOC:** 500 (soft limit)
- **Max Function LOC:** 50 (recommended)

---

## Principle 2: Document Everything

**Philosophy:** Every decision, every architecture choice, every tradeoff must be documented

### What It Means
- Create ADRs (Architecture Decision Records) for all significant decisions
- Document API contracts and interfaces
- Explain the "why" behind non-obvious code
- Maintain up-to-date README and documentation

### How It's Enforced
- **ADR template** provided via `create_adr.py`
- **Initial ADR** created automatically during setup
- **Documentation tracking** in handoffs

### ADR Structure
```markdown
# ADR-XXX: Decision Title

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Rejected | Deprecated
**Principle:** Document Everything (Principle 2)

## Context
What is the situation requiring a decision?

## Decision
What did we decide to do?

## Alternatives Considered
What other options did we evaluate?

## Consequences
What are the positive and negative outcomes?

## Implementation
How will this be implemented?
```

### When to Create an ADR
- Choosing a framework or library
- Changing architecture patterns
- Adding new dependencies
- Modifying build or deployment process
- Making security or performance tradeoffs

---

## Principle 3: Check Before Create

**Philosophy:** Search for existing solutions before writing new code

### What It Means
- **ALWAYS** search the codebase before implementing new functionality
- Check pattern library for established solutions
- Analyze reuse potential (target: ≥70%)
- Document why reuse wasn't possible if creating new code

### How It's Enforced
- **Automatic enforcement** via `enforce_check_before_create.py`
- **30-minute cache** prevents re-implementation without search
- **Multi-language support** (Python, TypeScript, JavaScript, Rust, Go, Java)
- **Search tool** via `search_codebase.py`

### Workflow
```bash
# 1. Search for existing code (REQUIRED)
python .ai-workspace/scripts/search_codebase.py "email validation"

# 2. Analyze reuse potential
python .ai-workspace/scripts/analyze_reuse.py "email validation" new_file.py

# 3. If creating new code, document why reuse failed
python .ai-workspace/scripts/create_adr.py
```

### Reuse Thresholds
- **90%+ match** → Use existing code directly
- **70-89% match** → Create wrapper/adapter
- **50-69% match** → Requires ADR to justify choice
- **<50% match** → Implement new code + document why

---

## Principle 4: Phase Gate Reviews

**Philosophy:** Quality gates must be passed before proceeding to next phase

### What It Means
- Follow structured development phases
- Each phase has entry and exit criteria
- Gates cannot be skipped (sequential enforcement)
- **MUST** test with REAL production data (not mocks)

### Development Phases
```
Discovery → Design → Implementation → Verification → Integration
```

### Phase Gates

#### 1. Discovery Gate
**Exit Criteria:**
- Problem clearly defined
- Requirements documented
- Constraints identified
- Success criteria established

#### 2. Design Gate
**Exit Criteria:**
- Architecture documented (ADR created)
- API contracts defined
- Data models specified
- Reuse analysis completed (≥0% documented)

#### 3. Implementation Gate
**Exit Criteria:**
- Code implemented and reviewed
- Unit tests passing (≥85% coverage)
- Integration tests passing
- Code documented (docstrings/comments)

#### 4. Verification Gate
**Exit Criteria:**
- ✅ **TESTED WITH REAL PRODUCTION DATA** (not mocks!)
- Performance benchmarks met
- Security review completed
- Edge cases validated

#### 5. Integration Gate
**Exit Criteria:**
- Integrated with existing systems
- Deployed to staging/production
- Monitoring and alerts configured
- Rollback plan documented

### How It's Enforced
- **Phase gate validation** via `validate_gates.py`
- **Gate files** track completion in `.agent-workspace/handoffs/`
- **Pre-commit hooks** enforce quality thresholds

### Example Gate File
```json
{
  "phase": "verification",
  "task_name": "user-authentication",
  "timestamp": "2025-10-07T10:30:00",
  "artifacts": {
    "real_data_test": {
      "completed": true,
      "used_real_data": true,
      "data_source": "production-sanitized-dump",
      "sample_size": 10000,
      "path": "tests/integration/test_auth_real_data.py"
    }
  },
  "exit_criteria_met": [true, true, true, true]
}
```

---

## Principle 5: Plan Before Act

**Philosophy:** Think → Research → Plan → Execute

### What It Means
- Create detailed task plan before coding
- Research existing solutions and patterns
- Break down complex tasks into steps
- Document approach and reasoning

### How It's Enforced
- **Planning tool** via `plan_task.py`
- **Task breakdown** required for complex features
- **Research phase** before implementation

### Planning Workflow
```bash
# 1. Create task plan
python .ai-workspace/scripts/plan_task.py "implement-user-auth"

# Creates:
# .agent-workspace/tasks/plan-implement-user-auth.md
```

### Task Plan Structure
```markdown
# Task Plan: Feature Name

## 1. Research Phase
- [ ] Search existing auth implementations
- [ ] Review security best practices
- [ ] Check pattern library

## 2. Design Phase
- [ ] Create ADR for auth strategy
- [ ] Design data models
- [ ] Define API contracts

## 3. Implementation Phase
- [ ] Implement core auth logic
- [ ] Write unit tests (≥85% coverage)
- [ ] Write integration tests

## 4. Verification Phase
- [ ] Test with REAL user data
- [ ] Security audit
- [ ] Performance testing

## 5. Integration Phase
- [ ] Deploy to staging
- [ ] Set up monitoring
- [ ] Document rollback plan
```

---

## How Principles Work Together

### Example: Adding User Authentication

```
┌─────────────────────────────────────────────┐
│ Principle 5: Plan Before Act                │
│ → Create task plan                          │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ Principle 3: Check Before Create            │
│ → Search for existing auth implementations  │
│ → Find 2 similar patterns (75% match)       │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ Principle 2: Document Everything            │
│ → Create ADR-015: Choose OAuth2 + JWT       │
│ → Document why not using found patterns     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ Principle 1: Less is More                   │
│ → Reuse OAuth2 library (not writing own)    │
│ → Keep auth functions simple (<50 LOC)      │
│ → Extract reusable JWT utility              │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ Principle 4: Phase Gate Reviews             │
│ → Discovery: Requirements documented ✅     │
│ → Design: ADR created, API defined ✅       │
│ → Implementation: Code + tests ✅           │
│ → Verification: REAL data tested ✅         │
│ → Integration: Deployed + monitored ✅      │
└─────────────────────────────────────────────┘
```

---

## Enforcement Summary

### Automatic
- **Pre-commit hooks** enforce complexity, duplication, file size
- **Check-Before-Create** blocks code creation without search (30-min cache)
- **Phase gates** prevent skipping development phases

### Manual (Tool-Assisted)
- **ADR creation** via `create_adr.py`
- **Task planning** via `plan_task.py`
- **Reuse analysis** via `search_codebase.py` and `analyze_reuse.py`
- **Gate validation** via `validate_gates.py`

---

## Success Metrics

### Principle 1: Less is More
- **Target:** ≤10 complexity, <3% duplication, ≤500 LOC/file
- **Measured:** Pre-commit hook, radon, jscpd

### Principle 2: Document Everything
- **Target:** 100% decisions have ADRs
- **Measured:** ADR count in `.agent-workspace/decisions/`

### Principle 3: Check Before Create
- **Target:** ≥70% reuse rate
- **Measured:** `analyze_reuse.py` reports

### Principle 4: Phase Gate Reviews
- **Target:** 100% gate compliance, 0 skipped phases
- **Measured:** Gate files in `.agent-workspace/handoffs/`

### Principle 5: Plan Before Act
- **Target:** 100% complex tasks have plans
- **Measured:** Task plans in `.agent-workspace/tasks/`

---

## References

- **AI Workspace README:** `.ai-workspace/README.md`
- **Script Documentation:** `.ai-workspace/scripts/`
- **Pattern Library:** `.ai-workspace/cursor/patterns/`
- **ADR Template:** Via `python .ai-workspace/scripts/create_adr.py`

---

**These principles are the foundation of AI Workspace's approach to sustainable, high-quality software development.**
