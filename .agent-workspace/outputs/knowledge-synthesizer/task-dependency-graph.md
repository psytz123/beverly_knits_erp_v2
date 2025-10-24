# Task Dependency Graph - Beverly Knits ERP v2

**Knowledge Synthesizer - Execution Planning**
**Date:** 2025-10-24

---

## Critical Path Analysis

**Total Duration:** 12 weeks (60 working days)
**Critical Path:** Security → Auth → Testing → Service Layer → Refactoring
**Parallel Work Opportunities:** 8 tasks can run in parallel after Week 2

---

## Phase 1: Security Lockdown (Week 1)

### Critical Path Tasks (Sequential)

```
START
  │
  ├─[1A] Fix Password Hashing (4h) ─────────┐
  │      Priority: P0 - BLOCKING           │
  │      Risk: 10/10                        │
  │      Dependencies: None                 │
  │                                         │
  ├─[1B] Fix SQL Injection (12h) ──────────┤
  │      Priority: P0 - BLOCKING           │
  │      Risk: 9/10                         │
  │      Dependencies: None                 │
  │                                         │
  └─[1C] Complete Secrets Manager (6h) ────┘
         Priority: P0 - BLOCKING
         Risk: 8/10
         Dependencies: None
                    │
                    ▼
         [1D] Implement Basic Auth (16h)
              Priority: P0 - BLOCKING
              Risk: 10/10
              Dependencies: 1A, 1B, 1C
                    │
                    ▼
         WEEK 1 GATE: Security Baseline
              Validation:
              - Passwords use bcrypt ✓
              - No SQL injection ✓
              - Secrets secured ✓
              - Auth enforced ✓
                    │
                    ▼
              PROCEED TO PHASE 2
```

### Parallel Tasks (Can run simultaneously)

```
[1E] Verify CORS Config (3h)        [1F] Verify Rate Limiting (2h)
     Priority: P1                        Priority: P1
     Risk: 7/10                          Risk: 7/10
     Dependencies: None                   Dependencies: None
          │                                    │
          └────────────────┬───────────────────┘
                           │
                           ▼
                    Week 1 Complete
```

**Week 1 Effort:** 43 hours total (5.4 days)
**Team:** 2 developers can complete in 3 days if working in parallel

---

## Phase 2: Testing Infrastructure (Week 2)

### Critical Path

```
Week 1 Gate Passed
        │
        ▼
[2A] Setup pytest Infrastructure (4h)
     Priority: P0 - BLOCKING
     Dependencies: None
        │
        ├──────────────────┬──────────────────┬────────────────┐
        │                  │                  │                │
        ▼                  ▼                  ▼                ▼
[2B] API Tests (24h)  [2C] DB Tests (16h)  [2D] Contract   [2E] CI/CD
     Priority: P0          Priority: P0          Framework       Pipeline (8h)
     Deps: 2A, 1D         Deps: 2A              (12h)           Deps: 2A
        │                  │                  Priority: P0      Priority: P0
        │                  │                  Deps: 2A          │
        │                  │                  │                 │
        └──────────────────┴──────────────────┴─────────────────┘
                           │
                           ▼
                  WEEK 2 GATE: Test Infrastructure
                       Validation:
                       - pytest runs ✓
                       - 40% coverage ✓
                       - CI enforces tests ✓
                           │
                           ▼
                     PROCEED TO PHASE 3
```

**Week 2 Effort:** 64 hours (8 days)
**Team:** 2 developers, 3-4 days with parallel work

---

## Phase 3: Architecture Refactoring (Weeks 3-4)

### Critical Path

```
Week 2 Gate Passed
        │
        ▼
[3A] Design Service Layer Interfaces (12h)
     Priority: P0 - BLOCKS ALL REFACTORING
     Dependencies: 2B, 2C (need tests to validate)
        │
        ├────────────────────────┬───────────────────┐
        │                        │                   │
        ▼                        ▼                   ▼
[3B] Extract Business      [3C] Exception      [3D] Refactor API
     Logic (40h)                Hierarchy (4h)      Layer (16h)
     Priority: P0               Priority: P1        Priority: P0
     Deps: 3A                   Deps: None          Deps: 3A
     LONGEST TASK               │                   │
        │                       │                   │
        │                       ▼                   │
        │              [3E] Replace Exception   │
        │                   Handlers (20h)         │
        │                   Priority: P1           │
        │                   Deps: 3C               │
        │                       │                   │
        ├───────────────────────┴───────────────────┤
        │                                           │
        ▼                                           ▼
[3F] Remove Code Duplication (16h)          Tests Still Pass?
     Priority: P1                                  │
     Deps: 3A, 3B                                 │
        │                                          │
        └──────────────────────────────────────────┘
                           │
                           ▼
                  WEEK 4 GATE: Architecture Refactored
                       Validation:
                       - Service layer exists ✓
                       - API < 1000 LOC ✓
                       - Duplication < 5% ✓
                       - All tests pass ✓
                           │
                           ▼
                     PROCEED TO PHASE 4
```

**Week 3-4 Effort:** 92 hours (11.5 days)
**Critical Path:** 3A → 3B (52h sequential minimum)
**Parallel Opportunities:** 3C, 3D can run parallel to 3B

---

## Phase 4: Quality & Completeness (Weeks 5-8)

### Parallel Work Streams (HIGH PARALLELIZATION)

```
Week 4 Gate Passed
        │
        ├──────────────┬──────────────┬──────────────┬──────────────┐
        │              │              │              │              │
        ▼              ▼              ▼              ▼              ▼
Stream 1:        Stream 2:      Stream 3:      Stream 4:      Stream 5:
Type Safety      Code Style     Testing        Docs           Cleanup
(18h)            (8h)           (64h)          (8h)           (30h)
    │                │              │              │              │
    ▼                ▼              ▼              ▼              ▼
[4A] Add         [4C] Black     [4F] DB        [4I] API       [4K] Complete
Type Hints       Format         Tests          Docs           Stubs
(8h)             (4h)           (16h)          (8h)           (20h)
    │                │              │                             │
    ▼                ▼              ▼                             ▼
[4B] Add         [4D] PEP 8     [4G] Forecast                [4L] Extract
mypy to CI       Fixes          Tests                        Constants
(2h)             (4h)           (32h)                        (4h)
    │                               │                             │
    ▼                               ▼                             ▼
[4E] Standardize                [4H] Integration              [4M] Fix
Docstrings                      Tests                         Logging
(8h)                            (16h)                         (2h)
    │                               │                             │
    └───────────────┬───────────────┴───────────────┬─────────────┘
                    │                               │
                    ▼                               ▼
              WEEK 8 GATE: Production Ready
                   Validation:
                   - 80% coverage ✓
                   - 100% type hints ✓
                   - PEP 8 compliant ✓
                   - Stubs complete ✓
                        │
                        ▼
                  PROCEED TO PHASE 5
```

**Week 5-8 Effort:** 128 hours (16 days)
**Parallelization:** 5 streams can run simultaneously
**Timeline:** 3-4 weeks with 2 developers

---

## Phase 5: Modernization (Months 2-3)

### Sequential with Some Parallelization

```
Week 8 Gate Passed
        │
        ▼
[5A] Consolidate SQLite Databases (24h)
     Priority: P0 - BLOCKS MIGRATION
     Dependencies: None
        │
        ▼
[5B] PostgreSQL Migration (16h)
     Priority: P0 - BLOCKS MICROSERVICES
     Dependencies: 5A
        │
        ├────────────────────────┬─────────────────────┐
        │                        │                     │
        ▼                        ▼                     ▼
[5C] Connection Pool       [5D] Inventory         [5E] Production
     Setup (4h)                 Service (40h)          Service (40h)
     Priority: P1               Priority: P0           Priority: P0
     Deps: 5B                   Deps: 5B               Deps: 5B
        │                        │                     │
        │                        ▼                     │
        │                   [5F] Contract          │
        │                        Tests (12h)          │
        │                        Priority: P0         │
        │                        Deps: 5D, 5E         │
        │                        │                     │
        └────────────────────────┴─────────────────────┘
                                 │
                                 ▼
                    [5G] Observability Stack (20h)
                         Priority: P1
                         Deps: 5D, 5E
                                 │
                                 ▼
                    [5H] Performance Monitoring (12h)
                         Priority: P1
                         Deps: 5G
                                 │
                                 ▼
                    [5I] Load Testing (16h)
                         Priority: P1
                         Deps: 5H
                                 │
                                 ▼
                    MONTH 3 GATE: Modernization Complete
                         Validation:
                         - PostgreSQL live ✓
                         - Microservices deployed ✓
                         - Contract tests pass ✓
                         - p95 latency ≤ 100ms ✓
                                 │
                                 ▼
                          PROJECT COMPLETE
```

**Month 2-3 Effort:** 160 hours (20 days)
**Critical Path:** 5A → 5B → 5D/5E → 5F → 5G → 5H → 5I (148h)
**Parallelization:** 5D and 5E can run in parallel (saves 40h)

---

## Blocking Relationships Summary

### What Blocks What

| Blocker | Blocks | Reason | Hours Blocked |
|---------|--------|--------|---------------|
| **Password Hash Fix** | API Auth | Can't authenticate users safely | 16h |
| **API Auth** | API Tests | Can't test authenticated endpoints | 24h |
| **Test Infrastructure** | All Refactoring | Can't validate changes | 156h |
| **Service Layer** | Deduplication | Need centralized logic first | 16h |
| **Service Layer** | Monolith Refactor | Need target architecture | 40h |
| **DB Tests** | PostgreSQL Migration | Can't validate data integrity | 40h |
| **PostgreSQL** | Microservices | Services need shared DB | 80h |
| **Contract Tests** | Service Deployment | Can't validate service parity | 80h |

**Total Hours Blocked by Test Infrastructure:** 220 hours (79% of total work)

---

## Parallel Execution Opportunities

### Week 1 (Limited Parallelization)

```
Developer A          Developer B
-------------        -------------
Password Hash (4h)   Secrets Manager (6h)
SQL Injection (12h)  CORS Verify (3h)
                     Rate Limit Verify (2h)
                     Basic Auth Part 1 (8h)
-------------        -------------
16h                  19h

Then collaborate on:
Basic Auth Part 2 (8h) - pair programming
```

**Timeline:** 3 days with 2 developers

### Week 2 (High Parallelization)

```
Developer A          Developer B
-------------        -------------
pytest Setup (4h)    Contract Framework (12h)
API Tests (24h)      CI/CD Pipeline (8h)
-------------        -------------
28h                  20h

Run in parallel:
DB Tests (16h) - can be split between both
```

**Timeline:** 4 days with 2 developers

### Weeks 3-4 (Medium Parallelization)

```
Developer A               Developer B
------------------        ------------------
Service Layer Design (6h) Exception Hierarchy (4h)
Business Logic Extract   API Layer Refactor (16h)
(40h - LONG TASK)        Exception Replacement (20h)
                         Code Deduplication (16h)
------------------        ------------------
46h                      56h
```

**Timeline:** 6 days with 2 developers (A does heavy lifting on 3B)

### Weeks 5-8 (Very High Parallelization)

```
Developer A          Developer B
-------------        -------------
Stream 1: Type       Stream 3: Testing
- Type hints (8h)    - DB tests (16h)
- mypy CI (2h)       - Forecast tests (32h)
- Docstrings (8h)    - Integration (16h)
-------------        -------------
Stream 2: Style      Stream 4: Docs + Cleanup
- Black (4h)         - API docs (8h)
- PEP 8 (4h)         - Stubs (20h)
                     - Constants (4h)
                     - Logging (2h)
-------------        -------------
26h                  98h
```

**Timeline:** Developer B is overloaded - needs task redistribution
**Adjusted:** Move some Stream 4 tasks to Developer A
**Result:** 62h each, 8 days duration

### Months 2-3 (Medium Parallelization)

```
Developer A                    Developer B
-------------------------      -------------------------
SQLite Consolidation (24h)
PostgreSQL Migration (16h)
Connection Pool (4h)
                               Inventory Service (40h)
                               Production Service (40h)
Contract Tests (12h)
Observability (20h)
Performance Monitoring (12h)
Load Testing (16h)
-------------------------      -------------------------
104h                          80h
```

**Timeline:** 13 days with 2 developers

---

## Resource Loading Chart

### Developer Hours per Week

| Week | Dev A | Dev B | Total | Max Capacity (80h) | Utilization |
|------|-------|-------|-------|--------------------|-------------|
| 1    | 24h   | 19h   | 43h   | 80h                | 54% |
| 2    | 28h   | 36h   | 64h   | 80h                | 80% |
| 3    | 46h   | 28h   | 74h   | 80h                | 93% |
| 4    | 0h    | 28h   | 28h   | 80h                | 35% (spillover) |
| 5-8  | 62h   | 62h   | 124h  | 320h (4 weeks)     | 39% |
| 9-12 | 104h  | 80h   | 184h  | 320h (4 weeks)     | 58% |

**Observations:**
- Week 3 is at 93% capacity (high risk)
- Week 4 has only 28h (opportunity to pull forward Week 5 work)
- Weeks 5-8 are under-utilized (39%) - can compress to 2 weeks
- Overall project can complete in 10 weeks instead of 12

---

## Risk Mitigation: Buffer Tasks

### If Tasks Exceed Estimates

**Buffer Pool (40 hours reserved):**

1. **If Week 1 slips:** Pull in 1E, 1F from Week 1 to Week 2 (5h freed)
2. **If Week 2 slips:** Reduce API tests from 24h to 16h (test only critical endpoints)
3. **If Week 3 slips:** Defer 3F (deduplication) to Phase 4 (16h freed)
4. **If Week 5-8 slips:** Defer 4E (docstrings) and 4M (logging) to Phase 5 (10h freed)
5. **If Month 2-3 slips:** Defer 5E (Production Service) to future sprint (40h freed)

**Total Buffer:** 87 hours can be deferred without breaking critical path

---

## Recommended Execution Sequence

### Optimal Task Order (Minimizes Idle Time)

**Week 1:**
1. Password Hash → SQL Injection → Basic Auth (Dev A: 32h)
2. Secrets Manager → CORS → Rate Limit (Dev B: 11h)
3. Pair on Basic Auth completion (Both: 8h)

**Week 2:**
1. pytest Setup (Dev A: 4h) → API Tests (Dev A: 24h)
2. Contract Framework (Dev B: 12h) → CI/CD (Dev B: 8h)
3. DB Tests split (Both: 8h each)

**Week 3:**
1. Service Layer Design (Both: 12h pair programming)
2. Business Logic Extract (Dev A: 40h solo - THE BLOCKER)
3. API Refactor + Exceptions (Dev B: 36h parallel work)

**Week 4:**
1. Deduplication (Dev B: 16h) - Dev A done early
2. Exception Replacement (Dev B: 20h)
3. Dev A starts Phase 4 early (Type hints: 8h)

**Weeks 5-8:**
1. Parallel streams run simultaneously
2. Rebalance: Dev A takes Stream 1+2+partial 4, Dev B takes Stream 3+partial 4
3. Both complete ~62h over 4 weeks

**Months 2-3:**
1. Sequential: SQLite → PostgreSQL (Dev A: 40h)
2. Parallel: Inventory Service (Dev B: 40h) during migration
3. Sequential: Contract Tests → Observability → Monitoring (Dev A: 44h)
4. Parallel: Production Service (Dev B: 40h) during observability work

---

## Gates & Validation Checkpoints

### Week 1 Gate (Friday EOD)
**Automated Checks:**
- [ ] `pytest tests/security/test_password_hashing.py` → PASS
- [ ] `sqlmap -u "http://localhost:5000/api/test" --batch` → No vulnerabilities
- [ ] `grep -r "hashlib.sha256" src/` → 0 results (except tests)
- [ ] `curl -X GET http://localhost:5000/api/inventory` → 401 Unauthorized

**Manual Checks:**
- [ ] Code review: Auth implementation follows OAuth2 patterns
- [ ] Penetration test: Try to bypass auth with crafted requests
- [ ] Secrets audit: No hardcoded secrets in `git log`

**Go/No-Go Decision:** If any automated check fails, DO NOT proceed to Week 2

### Week 2 Gate (Friday EOD)
**Automated Checks:**
- [ ] `pytest -v --cov=src --cov-report=term` → Coverage ≥ 40%
- [ ] `pytest tests/` → 0 failures
- [ ] CI pipeline shows green build with test results

**Manual Checks:**
- [ ] Review test fixtures for maintainability
- [ ] Validate contract tests capture current API behavior
- [ ] Check test execution time (should be < 30 seconds)

**Go/No-Go Decision:** If coverage < 40%, spend Weekend on more tests before Week 3

### Week 4 Gate (Friday EOD)
**Automated Checks:**
- [ ] `pylint src/services/` → Score ≥ 8.0
- [ ] `pytest tests/` → All tests still pass
- [ ] `radon cc src/api/ -a` → Average complexity < 10

**Manual Checks:**
- [ ] Code review: Service layer has clear interfaces
- [ ] Architecture review: Monolithic file split successfully
- [ ] Duplication check: `jscpd src/` → < 5% duplication

**Go/No-Go Decision:** If tests fail, roll back refactoring and debug

### Week 8 Gate (Friday EOD)
**Automated Checks:**
- [ ] `pytest --cov=src --cov-report=html` → Coverage ≥ 80%
- [ ] `mypy src/ --strict` → 0 errors
- [ ] `black src/ --check` → 0 files to reformat
- [ ] `bandit -r src/` → 0 high severity issues

**Manual Checks:**
- [ ] All stubs implemented (search for `pass` statements)
- [ ] API documentation published and accessible
- [ ] Integration tests cover end-to-end scenarios

**Go/No-Go Decision:** Must pass all checks before PostgreSQL migration

### Month 3 Gate (Friday EOD)
**Automated Checks:**
- [ ] Contract tests between monolith and microservices → PASS
- [ ] Load test: `locust -f tests/load/test_api.py` → p95 < 100ms
- [ ] Database migration: Row count validation queries → All match

**Manual Checks:**
- [ ] Microservices deployed to staging environment
- [ ] Observability dashboard shows metrics
- [ ] Rollback procedure tested successfully

**Go/No-Go Decision:** Production deployment approved only if all checks pass

---

## Summary: Critical Path

**Longest Sequential Chain:** 164 hours (20.5 days)

```
Password Hash (4h)
    ↓
SQL Injection (12h)
    ↓
Basic Auth (16h)
    ↓
pytest Setup (4h)
    ↓
API Tests (24h)
    ↓
Service Layer Design (12h)
    ↓
Business Logic Extract (40h) ← LONGEST SINGLE TASK
    ↓
SQLite Consolidation (24h)
    ↓
PostgreSQL Migration (16h)
    ↓
Contract Tests (12h)
    ↓
Total: 164h critical path
```

**With 2 Developers:** 164h ÷ 2 = 82 hours (10.25 days minimum)
**With Buffer (25%):** 13 days minimum timeline
**Recommended:** 12 weeks to allow for testing, validation, and unexpected issues

---

## Execution Recommendation

**Start Date:** Next Monday (Week 1)
**End Date:** 12 weeks later (Month 3)
**Team:** 2 developers + 0.5 QA + 0.25 DevOps
**Budget:** $30,000
**Risk Level:** MEDIUM (manageable with gates)

**Next Action:** Print this document, assign Week 1 tasks, schedule gate review.

---

**Document:** Task Dependency Graph
**Generated:** 2025-10-24
**Purpose:** Execution planning for tech lead
