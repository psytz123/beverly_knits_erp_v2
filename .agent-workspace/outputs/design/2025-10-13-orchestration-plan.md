# Multi-Agent Execution Plan – Beverly Knits ERP v2

**Author:** tech-lead-orchestrator  
**Date:** 2025-10-13T08:38:30Z  
**Objective:** Coordinate specialized agents to close critical ERP readiness gaps while maintaining Agent Workspace Protocol v2.

---

## Phase Overview

| Phase | Window | Goals | Primary Deliverables |
|-------|--------|-------|----------------------|
| 0. Mobilize & Quick Wins | Week 0 | Enforce real eFab data path, establish health validation, unblock containers | Updated launch scripts, health-check script, working Docker/Compose |
| 1. Platform Foundations | Weeks 1–2 | Harden delivery pipeline, migrate secrets, prep PostgreSQL rollout | CI/CD pipeline, secrets integration, migration playbook |
| 2. Service Refactor Track | Weeks 3–5 | Begin strangler migration for inventory & production services, add observability | Service boundary design, FastAPI services, tracing/monitoring |
| 3. Data Pipeline Hardening | Weeks 4–6 | Automate SharePoint sync, align analytics with PostgreSQL/Turso | Graph API ingestion job, validation tests |
| 4. Governance & Documentation | Weeks 5–6 | Reconcile docs, establish release governance, publish SLO dashboard | Updated docs, release checklist, KPI dashboards |

---

## Task Breakdown & Agent Assignments

1. **Enforce live eFab proxy & health validation**  
   - Agent: `backend-developer`  
   - Scope: Update `start_dashboard.*` to launch `efab_api_server.py`, add runtime health probe, integrate smoke tests.  
   - Deliverables: Patched scripts, automated health-check script, execution notes.

2. **Repair container toolchain**  
   - Agent: `platform-engineer`  
   - Scope: Generate `requirements.txt`, fix Dockerfile entrypoint, validate `docker compose`.  
   - Deliverables: Updated Docker artifacts, validation report.

3. **Establish CI/CD pipeline**  
   - Agent: `devops-engineer`  
   - Scope: Expand GitHub Actions with lint/test/build/deploy gates, coverage thresholds, artifact storage.  
   - Deliverables: Revised `build.yml`, pipeline runbook.

4. **Design strangler roadmap & extract inventory service**  
   - Agent: `microservices-architect`  
   - Scope: Define service boundaries, contract tests, feature flag rollout plan, inventory service skeleton.  
   - Deliverables: ADRs, service design doc, initial service repo layout.

5. **Database migration & data integrity**  
   - Agent: `database-administrator`  
   - Scope: Author PostgreSQL schema/migration scripts, connection pooling config, rollback plan.  
   - Deliverables: Migration scripts, DBA runbook, validation checklist.

6. **Automate SharePoint data ingestion**  
   - Agent: `data-engineer`  
   - Scope: Replace browser workflow with Graph API job, add retry/backoff, data validation tests.  
   - Deliverables: Ingestion service code, monitoring hooks, test evidence.

7. **Secrets management & security hardening**  
   - Agent: `security-engineer`  
   - Scope: Integrate secrets manager, rotate credentials, audit scripts, implement logging & rate limiting.  
   - Deliverables: Secrets runbook, updated configs, security assessment.

8. **Testing reorganization & coverage uplift**  
   - Agent: `qa-expert`  
   - Scope: Restructure tests, stabilize fixtures, introduce coverage reporting, map to CI pipeline.  
   - Deliverables: Testing strategy doc, coverage report, updated test harness.

9. **Documentation reconciliation & release governance**  
   - Agent: `technical-writer`  
   - Scope: Align status docs with reality, update guides, publish release checklist & KPI definitions.  
   - Deliverables: Revised documentation set, release checklist, communication plan.

---

## Execution Ordering

1. **Phase 0 (Sequential within week)**  
   - Task 1 → Task 2 → Task 3 (handoff dependent)
2. **Phase 1 & 2 (Mixed)**  
   - After Task 3 completes, run Tasks 4 & 5 in parallel.  
   - Task 6 starts once Task 5 delivers migration scaffolding.
3. **Phase 3 & 4 (Parallel, then sequential)**  
   - Tasks 7 & 8 run in parallel post Task 5.  
   - Task 9 finalizes after Tasks 1–8 produce outputs.

---

## Dependencies & Risk Notes

- Task 1 outputs gate accurate testing for Tasks 3 & 8.  
- Task 5 forms prerequisite for Task 6 (data ingestion) and Task 7 (secrets & DB credential rotation).  
- Feature flag strategy from Task 4 must align with CI/CD policies from Task 3 before rollout.  
- Documentation (Task 9) must incorporate security decisions (Task 7) and testing metrics (Task 8).

---

## Monitoring & Reporting

- `agent-health-monitor` to track agent progress.  
- Weekly checkpoint: update `.agent-workspace/context/project-context.md` with status deltas.  
- Trigger `@context-compressor` if context assets exceed 50KB.
