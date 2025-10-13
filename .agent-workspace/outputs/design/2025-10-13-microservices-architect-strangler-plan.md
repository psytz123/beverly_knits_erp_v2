# Strangler Migration Blueprint – Inventory & Production Domains

**Agent:** microservices-architect  
**Date:** 2025-10-13T09:20:00Z

---

## 1. Domain Boundary Definition

| Service | Scope | Primary Data Sources | External Dependencies | Initial Endpoints |
|---------|-------|----------------------|-----------------------|-------------------|
| `inventory-service` | Yarn inventory analytics, planning balance calculations, shortage risk scoring | `inventory_items`, `inventory_transactions`, ML forecasts | Forecast service (read-only), Auth gateway | `GET /inventory/items`, `GET /inventory/items/{id}`, `GET /inventory/summary`, `POST /inventory/recalculate` (async trigger) |
| `production-service` | Work order scheduling, machine capacity planning, production KPIs | `production_orders`, `machine_capacity`, `bom_mappings` | Inventory service (read-only), ERP auth | `GET /production/orders`, `POST /production/schedule`, `GET /production/capacity` |

Key rules:
- Shared read-only data moves via REST/gRPC contracts; no shared database schemas.
- Each service owns its schema; cross-service reads go through HTTP + background cache layer.
- Feature flags (`api_consolidation_enabled`, `inventory_service_enabled`) govern routing through API gateway.

---

## 2. Strangler Phasing

1. **Phase A – Sidecar Services (Weeks 3–4)**
   - Deploy `inventory-service` alongside monolith with read-only endpoints.
   - Mirror traffic via API gateway shadow mode; compare responses with monolith for 2 weeks.
   - Capture contract-test baselines (see section 3).

2. **Phase B – Progressive Cutover (Weeks 5–6)**
   - Flip feature flag to route `GET` requests to microservice; POST/PUT remain monolith.
   - Monitor error budget (≤1% failure) via Prometheus gauge.
   - Begin syncing derived caches into Redis for report latency parity.

3. **Phase C – Production Service Extraction (Week 7+)**
   - Extract production scheduling endpoints using same approach.
   - Replace monolith functions with thin proxies; retire code once coverage passes.

---

## 3. Contract Testing Strategy

- **Golden Record Set:** Snapshot of 25 representative yarn SKUs (critical/high/medium/low risk) exported nightly from staging DB; stored in `tests/contracts/fixtures/inventory_snapshot.json`.
- **Contract Test Flow:**
  1. Call monolith endpoint (`/api/inventory-intelligence-enhanced`).
  2. Call microservice endpoint (`/inventory/items`) using identical query params.
  3. Validate structural parity (fields, data types) and tolerance thresholds:
     - `planning_balance` delta ≤ 0.5%
     - `risk_level` category match strict equality
     - Response time target ≤ 200ms p95
- **Tooling:** Use `pytest` with `schemathesis` for schema validation and `jsondiff` for tolerance-based diff.

---

## 4. Feature Flag Rollout

| Flag | Owner | Default | Description |
|------|-------|---------|-------------|
| `inventory_service_shadow_mode` | Platform team | `true` | Sends mirrored requests to microservice without impacting clients. |
| `inventory_service_enabled` | Product owner | `false` | Routes live traffic to microservice `GET` endpoints once contract tests pass. |
| `inventory_write_enabled` | Product owner | `false` | Transfers write operations after asynchronous job parity achieved. |

Implementation checklist:
- Extend `src/config/feature_flags.py` with new flags, default `false`.
- Update API gateway middleware to consult flags per route.
- Instrument logging to capture shadow discrepancies (disabled by default to avoid noise).

---

## 5. Inventory Service Architecture Skeleton

```
services/
  inventory_service/
    app/
      __init__.py
      main.py          # FastAPI application entrypoint
      api/
        __init__.py
        routers.py     # Route definitions
        dependencies.py
      domain/
        __init__.py
        dto.py         # Data transfer objects
        service.py     # Business orchestration (to be populated)
    config/
      settings.py      # Pydantic-based configuration (env driven)
    tests/
      __init__.py
      contracts/
        test_inventory_contracts.py  # Placeholder with TODO markers
    pyproject.toml      # Enables isolated packaging & tooling
```

Key decisions:
- FastAPI for alignment with lightweight service needs and async support.
- Pydantic models define contract and guard against shape drift.
- `tests/contracts` executes against both monolith and microservice to verify parity using environment variables: `MONOLITH_BASE_URL`, `SERVICE_BASE_URL`.

---

## 6. Deployment Integration

- Docker image inherits from repo root image; `services/inventory_service/Dockerfile` can reuse multi-stage build once service matures.
- Compose overlay `docker-compose.inventory.yaml` (future) will add service container + Postgres DB instance.
- CI pipeline will gain job `inventory-contract-tests` depending on `docker-build` once endpoints implemented.

---

## 7. Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| Shadow traffic reveals discrepancies > tolerance | Hold flag enablement, analyze diff logs, patch service logic before retry. |
| Monolith DB schema changes | Maintain migration pipeline that updates both monolith and service schemas; share Alembic migrations via artifact. |
| Operational overhead | Use Helm umbrella chart to deploy services with shared observability stack; integrate with existing Prometheus exporters. |

---

## 8. Next Actions

1. Database administrator: finalize PostgreSQL schema & migrations compatible with inventory service tables.
2. Data engineer: ensure data pipeline populates new schema in parallel.
3. Backend teams: implement contract tests and gradually port logic from `services.inventory_analyzer_service`.
4. DevOps: extend pipeline with contract-test job once service endpoints live.

---
