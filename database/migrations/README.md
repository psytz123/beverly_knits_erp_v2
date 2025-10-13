# Database Migration Guide – Beverly Knits ERP v2

## Overview
These migrations establish the canonical PostgreSQL schema that replaces the legacy SQLite layout and prepares the system for microservice extraction.

- `20251013_initial_schema.sql` – Creates inventory, production, and scheduling tables plus migration bookkeeping.
- `20251013_initial_schema_down.sql` – Rolls back the schema to a clean state (use only in non-production environments).

## Prerequisites
1. PostgreSQL 14+ with superuser capable of creating schemas and extensions.
2. Environment variables (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) configured for the target instance.
3. Backup of existing SQLite data for validation and potential rollback.

## Applying Migrations
```bash
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" \
  -f database/migrations/20251013_initial_schema.sql
```

### Data Migration Checklist
1. Export SQLite tables using `.mode csv` / `.output` or Python scripts.
2. Load CSV data into staging tables using `
COPY staging_inventory FROM 'inventory.csv' CSV HEADER;
`.
3. Use stored procedures or Python ETL to transform data into new normalized tables.
4. Record migration execution in `service_migrations` table with checksum of executed file.

## Verification Queries
```sql
-- Validate table presence
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('inventory_items','inventory_transactions','production_orders');

-- Check yarn coverage
SELECT COUNT(DISTINCT yarn_id) FROM inventory_items;

-- Ensure migration recorded
SELECT * FROM service_migrations WHERE migration_name = '20251013_initial_schema.sql';
```

## Rollback Procedure
1. Confirm no dependent services rely on the new schema (microservices disabled).
2. Execute `20251013_initial_schema_down.sql` against the target database.
3. Restore SQLite-backed services if required.
4. Document rollback event in incident log and plan remediation.

## Future Steps
- Introduce migration tooling (Alembic/Flyway) to manage versioning automatically.
- Add data quality checks comparing SQLite exports with PostgreSQL counts/aggregations.
- Schedule routine backups via `pg_dump` stored under secure storage (S3/Turso backup bucket).
