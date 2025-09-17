# AI Coding Agent Operating Charter

**Objective:** Minimize code, maximize quality. Every action must align with these principles.  

---

## 🎯 Core Principles (5)

1. **Less is More**
   - Strive for the smallest, clearest solution.
   - Enforce complexity and duplication thresholds:
     - Cyclomatic complexity ≤ 10 per function.
     - < 3% duplication in new code.
   - Prefer deletion/refactoring over new code when possible.
   - Reuse-first order: stdlib → approved dependencies → internal → net-new.

2. **Document Everything**
   - Every module/file includes:
     - Purpose & usage description.
     - Sample inputs and outputs.
     - Links to dependency docs.
   - Every change includes a **Decision Record (ADR)** capturing:
     - Problem, alternatives, trade-offs, blast radius.
     - Rationale for build vs reuse.
     - Rollback plan.

3. **Check Before Create**
   - Always run **reuse workflow** before writing code:
     - Semantic search of codebase.
     - Dependency audit (license, health, maintenance).
     - Interface compatibility scan.
   - If reuse ≥ 70% fit → implement wrapper/adaptor.
   - If new code is required → document why reuse failed.

4. **Phase Gate Reviews**
   Code cannot progress without passing each gate:
   - **Discovery:** Problem, constraints, success metrics, reuse report.
   - **Design:** Minimal API sketch, complexity/LOC budget, test strategy.
   - **Implementation:** Working code, type hints, ≤500 LOC per file.
   - **Verification:**
     - Real data validation (never mocks).
     - Track ALL failures; no “tests passed” blanket statements.
     - Results before linting.
     - Exit code 1 on any failure; 0 only if all pass.
   - **Integration:** Backward compatibility, performance check, rollback verified.

5. **Plan Before Act**
   - The agent must always **think, research, and produce a structured plan** before execution.
   - Plans must include: problem statement, research notes, proposed approach, verification method.
   - Plans are stored under `docs/memory_bank/tasks/`.
   - No coding begins until the plan is documented, reviewed, and phase gates are defined.
   - Every output must trace back to the initial plan, ensuring an organized and intentional result.

---

## ⚙️ Operating Standards

- **Architecture:**
  - Function-first design; classes only for state, validation models, or known patterns.
  - No conditional imports — dependencies must be explicit in `pyproject.toml`.
  - Type hints required for all parameters & return values (concrete > Any).
  - ≤500 LOC per file.

- **Validation Discipline:**
  - Every file must include a `__main__` block that validates functionality with real data.
  - Validation comes **before** static analysis or linting.
  - If 3+ consecutive validation failures occur → mandatory external research + documented findings.

- **Testing:**
  - Tests mirror production structure.
  - Assertions must verify specific expected values.
  - `MagicMock` and mocking of core functions forbidden.
  - Tests are a future-proofing step — validation always comes first.

- **Execution Rules:**
  - Run scripts with `uv run`.
  - Use environment variables with `env VAR_NAME="value"`.
  - Max 500 LOC per file; no single function should exceed maintainability thresholds.

- **Logging & CLI:**
  - Use **loguru** for logging.
  - Use **typer** for CLI apps (`cli.py`).

- **Dependency Discipline:**
  - Apply **95/5 rule**: use 95% package features, customize only 5%.
  - Add dependencies only after research; must be justified in ADR.

---

## ✅ Compliance Checklist

Before task completion, the agent confirms:
1. All files include headers with purpose, docs, inputs/outputs.
2. Validation with real data produces expected results.
3. Type hints used consistently.
4. No `asyncio.run()` inside functions.
5. Module < 500 LOC.
6. ADR exists for all design/implementation choices.
7. Validation tracks all failures and exits correctly.
8. No unconditional success messages.
9. Documentation and tests mirror usage examples.
10. If 3+ failures, external research is documented.

---

## 📊 Quality KPIs

- Test coverage ≥ 85% lines, ≥ 70% branches (for changed code).
- Mutation score ≥ 70%.
- Refactor rate ≥ 15% of PRs reduce LOC/complexity.
- No performance regression > 2% on benchmarks.
- Mean cycle time: ≤ 2 days from idea → merged.

---

## 🔧 Recommended Optimizations for Agent Capability

- **Task Blueprint Template (1 page):**
  - Problem, constraints, success metrics
  - Reuse report summary (top 3 matches + fit %)
  - Minimal API (signatures), data flow sketch, budgets (ΔLOC, complexity), test strategy
  - Risk/rollback & release notes skeleton

- **Reuse-first Automation:**
  - Semantic code search + dep audit before coding.
  - Wrapper/adaptor generation if match ≥70%.

- **Quality Budgets per Change Set:**
  - ΔLOC budget, cyclomatic complexity ≤10, duplication <3%.
  - PR includes before/after table.

- **CI Pipeline Stages:**
  1. Plan & ADR presence check
  2. Validation (real data, multiple cases, error paths)
  3. Static (type, lint) + dup/complexity scan
  4. Security & deps audit

- **Validation Playbook:**
  - Explicit expected vs actual comparisons.
  - Multiple cases: normal, edge, error.
  - Exit non-zero if any test fails; always show summary.

- **Learning Loop:**
  - If 3 consecutive validation failures → external research required, findings logged in ADR & Task Blueprint.
  - Nightly scan: suggest top 5 code reduction/refactor opportunities.

---
