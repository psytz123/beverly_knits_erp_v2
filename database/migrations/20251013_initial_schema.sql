BEGIN;

CREATE TABLE IF NOT EXISTS inventory_items (
    id SERIAL PRIMARY KEY,
    yarn_id VARCHAR(32) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    supplier TEXT,
    color TEXT,
    theoretical_balance NUMERIC(14,2) NOT NULL DEFAULT 0,
    planning_balance NUMERIC(14,2) NOT NULL DEFAULT 0,
    allocated NUMERIC(14,2) NOT NULL DEFAULT 0,
    on_order NUMERIC(14,2) NOT NULL DEFAULT 0,
    cost_per_pound NUMERIC(10,4),
    risk_level VARCHAR(16) NOT NULL DEFAULT 'LOW',
    last_synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS inventory_transactions (
    id BIGSERIAL PRIMARY KEY,
    yarn_id VARCHAR(32) NOT NULL REFERENCES inventory_items(yarn_id) ON DELETE CASCADE,
    transaction_type VARCHAR(32) NOT NULL,
    quantity NUMERIC(14,2) NOT NULL,
    transaction_date TIMESTAMPTZ NOT NULL,
    source_reference VARCHAR(128),
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inventory_transactions_yarn_date
    ON inventory_transactions (yarn_id, transaction_date DESC);

CREATE TABLE IF NOT EXISTS inventory_snapshots (
    id BIGSERIAL PRIMARY KEY,
    yarn_id VARCHAR(32) NOT NULL REFERENCES inventory_items(yarn_id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    theoretical_balance NUMERIC(14,2) NOT NULL,
    planning_balance NUMERIC(14,2) NOT NULL,
    allocated NUMERIC(14,2) NOT NULL,
    on_order NUMERIC(14,2) NOT NULL,
    risk_level VARCHAR(16) NOT NULL,
    UNIQUE (yarn_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS production_orders (
    id BIGSERIAL PRIMARY KEY,
    order_number VARCHAR(50) NOT NULL UNIQUE,
    style_number VARCHAR(50) NOT NULL,
    customer TEXT,
    quantity INTEGER NOT NULL,
    due_date DATE NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'scheduled',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS machine_capacity (
    id BIGSERIAL PRIMARY KEY,
    machine_id VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    shift_hours NUMERIC(6,2) NOT NULL,
    efficiency NUMERIC(5,2) NOT NULL DEFAULT 1.0,
    downtime_hours NUMERIC(6,2) NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS production_schedule (
    id BIGSERIAL PRIMARY KEY,
    production_order_id BIGINT NOT NULL REFERENCES production_orders(id) ON DELETE CASCADE,
    machine_id VARCHAR(50) NOT NULL REFERENCES machine_capacity(machine_id) ON DELETE CASCADE,
    scheduled_start TIMESTAMPTZ NOT NULL,
    scheduled_end TIMESTAMPTZ NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'planned',
    UNIQUE (production_order_id, machine_id, scheduled_start)
);

CREATE TABLE IF NOT EXISTS service_migrations (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(64) NOT NULL,
    migration_name VARCHAR(128) NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum VARCHAR(64) NOT NULL,
    UNIQUE (service_name, migration_name)
);

COMMIT;
