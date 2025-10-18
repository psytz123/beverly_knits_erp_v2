# QA Coverage Uplift Plan

**Agent:** qa-expert  
**Date:** 2025-10-13T10:05:00Z

## Immediate Fixes
- Updated `tests/execute_all_tests.py` to set `PYTHONPATH` to the repository root so pytest discovers application modules consistently on all platforms.
- Added root `.coveragerc` to align coverage collection with the CI gate (targets `src/` and enables branch coverage).

## Test Suite Reorganization Roadmap
1. **Inventory Tests**: Move ad-hoc tests from repository root into `tests/unit/` or `tests/integration/` (tracked in gap register).
2. **Contract Tests**: When microservices land, house parity checks under `tests/contracts/` with explicit markers.
3. **Performance/Load**: Keep `tests/performance/` but gate it behind `pytest -m performance` so CI focuses on unit/integration.

## Coverage Targets
- Maintain ≥80% line coverage in CI (already enforced via workflow).
- Introduce branch coverage tracking (`branch = True`) for critical modules (`src/api`, `src/core`).

## Regression Checklist (to deliver in follow-up sprint)
- Smoke: `/api/health`, `/api/yarn-intelligence` with live data (existing script `system_health_check.py`).
- Integration: orchestrated flows hitting inventory analytics and forecasting endpoints.
- Rate limiting: exercise multiple requests to confirm 429 path and logging.

## Next Actions
- QA to triage scattered tests and create tickets per module.
- DevOps to export coverage artifacts for review on each PR (already configured).
- Product to schedule load test run post-migration.
