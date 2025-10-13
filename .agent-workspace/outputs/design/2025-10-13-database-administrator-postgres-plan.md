# PostgreSQL Migration Plan – Beverly Knits ERP v2

**Author:** database-administrator  
**Date:** 2025-10-13T09:35:00Z

## 1. Target Schema Summary
- Inventory domain: `inventory_items`, `inventory_transactions`, `inventory_snapshots`.
- Production domain: `production_orders`, `machine_capacity`, `production_schedule`.
- Migration bookkeeping: `service_migrations` for idempotent tracking.
- All tables created by `database/migrations/20251013_initial_schema.sql` with matching rollback script.

## 2. Data Migration Strategy
1. Export legacy SQLite tables to CSV (use `sqlite_setup.py` or pandas).
2. Load CSVs into staging tables within PostgreSQL (`COPY staging_*`).
3. Transform into normalized schema using SQL scripts or ETL notebooks, preserving `yarn_id` integrity.
4. Record applied migration in `service_migrations` with file checksum (e.g., `sha256sum database/migrations/20251013_initial_schema.sql`).

## 3. Connection Pooling Configuration
- `src/database/connection_pool.py` now reads host/port/database/user/password plus `DB_MIN_POOL_SIZE`, `DB_POOL_SIZE`, `DB_CONNECT_TIMEOUT`, `DB_CONNECTION_OPTIONS`.
- Default pool: min 2 / max 10 connections, extendable via environment.
- ThreadedConnectionPool is initialized once per process; health logging emits host/db/min/max for observability.

### Recommended Production Settings
```
DB_MIN_POOL_SIZE=5
DB_POOL_SIZE=20
DB_CONNECT_TIMEOUT=15
DB_CONNECTION_OPTIONS='-c statement_timeout=45000'
```

## 4. Verification Queries
```sql
SELECT COUNT(*) FROM inventory_items;
SELECT yarn_id, planning_balance FROM inventory_items WHERE risk_level = 'CRITICAL' LIMIT 10;
SELECT COUNT(*) FROM production_orders WHERE status <> 'completed';
```
Compare counts against pre-migration SQLite exports (±1 row tolerance for in-flight changes).

## 5. Rollback Procedure
1. Disable microservice feature flags (`inventory_service_enabled=false`).
2. Stop dependent services accessing PostgreSQL.
3. Execute `database/migrations/20251013_initial_schema_down.sql`.
4. Restore SQLite-backed operations and document rollback in incident log.

## 6. Next Steps
- Coordinate with data-engineer to automate SharePoint ingestion into new schema.
- Provide connection details to microservices team for integration testing.
- Add scheduled `pg_dump` backup job (daily) once database is live.
