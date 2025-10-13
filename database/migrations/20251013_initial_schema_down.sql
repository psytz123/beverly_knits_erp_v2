BEGIN;

DROP TABLE IF EXISTS production_schedule CASCADE;
DROP TABLE IF EXISTS machine_capacity CASCADE;
DROP TABLE IF EXISTS production_orders CASCADE;
DROP TABLE IF EXISTS inventory_snapshots CASCADE;
DROP TABLE IF EXISTS inventory_transactions CASCADE;
DROP TABLE IF EXISTS inventory_items CASCADE;
DROP TABLE IF EXISTS service_migrations CASCADE;

COMMIT;
