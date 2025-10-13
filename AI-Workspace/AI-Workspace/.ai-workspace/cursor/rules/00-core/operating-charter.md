---
description: AI Coding Agent Operating Charter - Universal quality standards and workflow principles
applies_to: all
priority: critical
---

# AI Coding Agent Operating Charter

**Objective:** Minimize code, maximize quality. Every action must align with these principles.

---

## 🎯 Core Principles

### 1. Less is More

**Goal:** Smallest, clearest solution

**Quality Thresholds:**
- Cyclomatic complexity ≤ 10 per function
- Code duplication < 3% in new code
- Prefer deletion/refactoring over new code when possible

**Reuse-First Order:**
1. Standard library
2. Approved dependencies
3. Internal shared code
4. Net-new code (last resort)

**Before Writing Code:**
- Search existing codebase semantically
- Check if wrapper/adapter can reuse ≥70% of existing code
- Document why reuse failed if creating new code

---

### 2. Document Everything

**Every Module/File Must Include:**
- Purpose & usage description
- Sample inputs and outputs
- Links to dependency documentation

**Every Significant Change Requires ADR:**
- Problem statement
- Alternatives considered with trade-offs
- Blast radius analysis
- Rationale for build vs. reuse
- Rollback plan

**ADR Location:** `.agent-workspace/decisions/ADR-XXX-title.md`

---

### 3. Check Before Create

**Mandatory Reuse Workflow:**

```
BEFORE writing any code:
1. Run semantic search of codebase
2. Audit dependencies (license, health, maintenance)
3. Scan for interface compatibility
4. If reuse ≥70% fit → implement wrapper/adapter
5. If new code required → document why reuse failed in ADR
```

**Decision Tree:**
- ≥70% match → Create wrapper/adapter
- <70% match → Implement new + document in ADR
- Always prefer composition over duplication

---

### 4. Phase Gate Reviews

Code cannot progress without passing each gate:

#### **Discovery Gate**
- [ ] Problem clearly stated
- [ ] Constraints documented
- [ ] Success metrics defined
- [ ] Reuse report completed

#### **Design Gate**
- [ ] Minimal API sketch created
- [ ] Complexity budget set (ΔLOC, cyclomatic)
- [ ] Test strategy documented
- [ ] Dependencies identified

#### **Implementation Gate**
- [ ] Working code
- [ ] Type hints complete
- [ ] Files ≤500 LOC
- [ ] Functions ≤50 LOC

#### **Verification Gate**
- [ ] **Real data validation** (never empty mocks)
- [ ] ALL failures tracked (no "tests passed" blanket statements)
- [ ] Results verified BEFORE linting
- [ ] Exit code 1 on any failure, 0 only if ALL pass

#### **Integration Gate**
- [ ] Backward compatibility verified
- [ ] Performance check (≤2% regression allowed)
- [ ] Rollback procedure verified
- [ ] Documentation updated

---

### 5. Plan Before Act

**Before Starting Any Task:**

1. **Think** - Understand the problem fully
2. **Research** - Search existing solutions
3. **Structure** - Create detailed plan
4. **Document** - Save plan to `docs/memory_bank/tasks/`

**Plan Must Include:**
- Problem statement
- Research notes
- Proposed approach
- Verification method
- Phase gates checklist

**No coding begins until plan is documented and reviewed.**

---

## ⚙️ Operating Standards

### Architecture

- **Function-first design** - Classes only for state, validation models, or known patterns
- **No conditional imports** - Dependencies must be explicit in dependency file
- **Type hints required** - All parameters & return values (concrete types > Any)
- **File size limit** - ≤500 LOC per file

### Validation Discipline

- **Every file must include validation block** (adapt to framework):
  - Scripts: `if __name__ == "__main__"` with real data
  - APIs: `/health/validate` endpoint with real data checks
  - Libraries: Example usage in docstring with real data

- **Validation comes BEFORE static analysis**
- **Use real data, not mocks** - Empty test data is forbidden
- **3+ consecutive validation failures** → mandatory external research + documented findings

### Testing Strategy

- **Test Pyramid:**
  - 60% Unit tests (fast, isolated, mock external deps)
  - 30% Integration tests (API level, mock external services only)
  - 10% E2E tests (full stack, real infrastructure)

- **Test Requirements:**
  - Mirror production structure
  - Assertions verify specific expected values
  - MagicMock allowed for external deps, forbidden for core logic
  - Tests are future-proofing (run after validation passes)

### Execution Rules

- **Run scripts:** Use project's package manager (uv, npm, cargo, etc.)
- **Environment variables:** Use proper env var management
- **Max 500 LOC per file** - If exceeded, must justify in PR
- **No single function >50 LOC** - Refactor into smaller functions

### Logging & CLI

- **Logging:** Use structured logging (loguru for Python, winston for JS)
- **CLI Apps:** Use proper CLI framework (typer for Python, commander for JS)
- **Output:** JSON for machine-readable, rich text for human-readable

### Dependency Discipline

**95/5 Rule:** Use 95% of package features, customize only 5%

**Before Adding Dependency:**
- Search for existing solution in current deps
- Check license compatibility
- Verify maintenance status (commits in last 6 months)
- Document decision in ADR
- Add to dependency file with version pinning

---

## ✅ Compliance Checklist

Before marking task complete, confirm:

1. [ ] All files have headers with purpose, docs, inputs/outputs
2. [ ] Validation with real data produces expected results
3. [ ] Type hints used consistently
4. [ ] No `asyncio.run()` inside async functions
5. [ ] Modules < 500 LOC
6. [ ] Functions < 50 LOC
7. [ ] ADR exists for all design/implementation choices
8. [ ] Validation tracks all failures with proper exit codes
9. [ ] No unconditional success messages
10. [ ] Documentation and tests mirror usage examples
11. [ ] If 3+ failures occurred, external research is documented

---

## 📊 Quality KPIs

### Code Quality
- Test coverage ≥ 85% lines, ≥ 70% branches (for changed code)
- Mutation score ≥ 70%
- Cyclomatic complexity ≤ 10 per function
- Code duplication < 3%

### Process Quality
- Refactor rate ≥ 15% of PRs reduce LOC/complexity
- No performance regression > 2% on benchmarks
- Mean cycle time: ≤ 2 days from idea → merged

### Validation Quality
- All tests pass with real data
- Exit code 0 only when 100% pass
- No untested code paths in critical functions

---

## 🔧 Automation & Tools

### Pre-commit Hooks
- Complexity check (radon, lizard)
- Duplication check (jscpd)
- Type check (mypy, tsc)
- Lint check (ruff, eslint)
- Format check (black, prettier)

### CI Pipeline Stages
1. **Plan & ADR Check** - Verify ADR exists for significant changes
2. **Validation** - Run with real data, multiple cases, error paths
3. **Static Analysis** - Type, lint, duplication, complexity scan
4. **Security** - Dependency audit, SAST scan
5. **Performance** - Benchmark against baseline

### Learning Loop
- **3 consecutive validation failures** → External research required
- **Findings logged** in ADR & Task Blueprint
- **Nightly scan** → Suggest top 5 code reduction/refactor opportunities

---

## 🎓 Best Practices

### When Creating New Code

```python
# ❌ BAD: Jump straight to implementation
def process_order(order_id):
    # ... 200 lines of code

# ✅ GOOD: Plan, research, then implement
# 1. Check if similar function exists in codebase
# 2. Create ADR documenting decision
# 3. Write tests first
# 4. Implement in small, focused functions
def process_order(order_id: int) -> OrderResult:
    """Process customer order with validation.

    Args:
        order_id: Unique order identifier

    Returns:
        OrderResult with status and details

    Raises:
        OrderNotFoundError: If order doesn't exist

    Example:
        >>> result = process_order(12345)
        >>> assert result.status == "completed"
    """
    order = _fetch_order(order_id)  # ≤10 LOC
    validated = _validate_order(order)  # ≤10 LOC
    return _execute_order(validated)  # ≤10 LOC
```

### When Refactoring

```python
# ❌ BAD: Big bang refactor
# Rewrite entire module at once

# ✅ GOOD: Incremental refactor
# 1. Add tests for existing behavior
# 2. Extract one function
# 3. Verify tests still pass
# 4. Commit
# 5. Repeat
```

---

## 📝 Templates

### ADR Template
See `.agent-workspace/decisions/ADR-template.md`

### Task Blueprint Template
See `.agent-workspace/tasks/task-template.md`

### Phase Gate Checklist
See `.agent-workspace/phase-gates/template.json`

---

**Remember:** Quality is not negotiable. Every shortcut creates technical debt.
Every line of code is a liability. Write less, test more, refactor constantly.
