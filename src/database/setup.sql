-- Beverly Knits ERP Database Setup
-- PostgreSQL with TimescaleDB for time-series data
-- Created: 2025-08-29

-- Create database (run as superuser)
-- CREATE DATABASE beverly_knits_erp;
-- \c beverly_knits_erp;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS production;
CREATE SCHEMA IF NOT EXISTS api;

-- =====================================================
-- MASTER DATA TABLES (No daily refresh)
-- =====================================================

-- Suppliers table
CREATE TABLE IF NOT EXISTS production.suppliers (
    supplier_id SERIAL PRIMARY KEY,
    supplier_code VARCHAR(50) UNIQUE NOT NULL,
    supplier_name VARCHAR(255) NOT NULL,
    contact_info TEXT,
    lead_time_days INTEGER DEFAULT 30,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Yarns master table
CREATE TABLE IF NOT EXISTS production.yarns (
    yarn_id SERIAL PRIMARY KEY,
    desc_id INTEGER UNIQUE NOT NULL,
    supplier_id INTEGER REFERENCES production.suppliers(supplier_id),
    yarn_description VARCHAR(500),
    blend VARCHAR(255),
    yarn_type VARCHAR(100),
    color VARCHAR(100),
    cost_per_pound DECIMAL(10,2),
    minimum_order_qty DECIMAL(10,2),
    lead_time_days INTEGER,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Styles master table
CREATE TABLE IF NOT EXISTS production.styles (
    style_id SERIAL PRIMARY KEY,
    style_number VARCHAR(100) UNIQUE NOT NULL,
    fstyle_number VARCHAR(100),
    style_description VARCHAR(500),
    category VARCHAR(100),
    customer_code VARCHAR(50),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bill of Materials table
CREATE TABLE IF NOT EXISTS production.style_bom (
    bom_id SERIAL PRIMARY KEY,
    style_id INTEGER REFERENCES production.styles(style_id),
    yarn_id INTEGER REFERENCES production.yarns(yarn_id),
    bom_percent DECIMAL(5,2) NOT NULL CHECK (bom_percent > 0 AND bom_percent <= 100),
    unit VARCHAR(20) DEFAULT 'lbs',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(style_id, yarn_id)
);

-- Customers table
CREATE TABLE IF NOT EXISTS production.customers (
    customer_id SERIAL PRIMARY KEY,
    customer_code VARCHAR(50) UNIQUE NOT NULL,
    customer_name VARCHAR(255) NOT NULL,
    contact_info TEXT,
    payment_terms VARCHAR(50),
    credit_limit DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fabric specifications table
CREATE TABLE IF NOT EXISTS production.fabric_specs (
    spec_id SERIAL PRIMARY KEY,
    style_id INTEGER REFERENCES production.styles(style_id),
    fabric_id VARCHAR(50),
    finish_code VARCHAR(50),
    overall_width DECIMAL(8,2),
    cuttable_width DECIMAL(8,2),
    oz_per_linear_yd DECIMAL(8,3),
    yards_per_pound DECIMAL(8,3),
    gsm DECIMAL(8,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- TIME-SERIES TABLES (Daily refresh)
-- =====================================================

-- Yarn inventory time-series
CREATE TABLE IF NOT EXISTS production.yarn_inventory_ts (
    inventory_id BIGSERIAL,
    snapshot_date DATE NOT NULL,
    yarn_id INTEGER REFERENCES production.yarns(yarn_id),
    desc_id INTEGER NOT NULL,
    theoretical_balance DECIMAL(12,2),
    allocated DECIMAL(12,2),
    on_order DECIMAL(12,2),
    planning_balance DECIMAL(12,2) GENERATED ALWAYS AS 
        (theoretical_balance - allocated + on_order) STORED,
    consumed_qty DECIMAL(12,2),
    weeks_of_supply DECIMAL(8,2),
    cost_per_pound DECIMAL(10,2),
    data_source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (inventory_id, snapshot_date)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('production.yarn_inventory_ts', 'snapshot_date', 
    if_not_exists => TRUE);

-- Fabric inventory stages time-series
CREATE TABLE IF NOT EXISTS production.fabric_inventory_ts (
    inventory_id BIGSERIAL,
    snapshot_date DATE NOT NULL,
    style_id INTEGER REFERENCES production.styles(style_id),
    fstyle_number VARCHAR(100),
    inventory_stage VARCHAR(10) CHECK (inventory_stage IN ('F01', 'G00', 'G02', 'I01')),
    quantity_lbs DECIMAL(12,2),
    quantity_yds DECIMAL(12,2),
    location VARCHAR(100),
    status VARCHAR(50),
    customer_code VARCHAR(50),
    data_source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (inventory_id, snapshot_date)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('production.fabric_inventory_ts', 'snapshot_date',
    if_not_exists => TRUE);

-- Sales orders time-series
CREATE TABLE IF NOT EXISTS production.sales_orders_ts (
    order_id BIGSERIAL,
    snapshot_date DATE NOT NULL,
    so_number VARCHAR(50) NOT NULL,
    customer_id INTEGER REFERENCES production.customers(customer_id),
    style_id INTEGER REFERENCES production.styles(style_id),
    fstyle_number VARCHAR(100),
    style_number VARCHAR(100),
    order_status VARCHAR(50),
    uom VARCHAR(20),
    quantity_ordered DECIMAL(12,2),
    quantity_picked DECIMAL(12,2),
    quantity_shipped DECIMAL(12,2),
    balance DECIMAL(12,2),
    available_qty DECIMAL(12,2),
    ship_date DATE,
    po_number VARCHAR(100),
    data_source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (order_id, snapshot_date)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('production.sales_orders_ts', 'snapshot_date',
    if_not_exists => TRUE);

-- Knit orders (production) time-series
CREATE TABLE IF NOT EXISTS production.knit_orders_ts (
    knit_order_id BIGSERIAL,
    snapshot_date DATE NOT NULL,
    ko_number VARCHAR(50) NOT NULL,
    style_id INTEGER REFERENCES production.styles(style_id),
    style_number VARCHAR(100),
    start_date DATE,
    quoted_date DATE,
    qty_ordered_lbs DECIMAL(12,2),
    g00_lbs DECIMAL(12,2),
    shipped_lbs DECIMAL(12,2),
    balance_lbs DECIMAL(12,2),
    seconds_lbs DECIMAL(12,2),
    machine VARCHAR(50),
    po_number VARCHAR(100),
    data_source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (knit_order_id, snapshot_date)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('production.knit_orders_ts', 'snapshot_date',
    if_not_exists => TRUE);

-- Yarn demand planning time-series
CREATE TABLE IF NOT EXISTS production.yarn_demand_ts (
    demand_id BIGSERIAL,
    snapshot_date DATE NOT NULL,
    yarn_id INTEGER REFERENCES production.yarns(yarn_id),
    desc_id INTEGER NOT NULL,
    week_number INTEGER,
    week_date DATE,
    demand_qty DECIMAL(12,2),
    forecast_qty DECIMAL(12,2),
    actual_qty DECIMAL(12,2),
    data_source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (demand_id, snapshot_date)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('production.yarn_demand_ts', 'snapshot_date',
    if_not_exists => TRUE);

-- =====================================================
-- DATA QUALITY AND AUDIT TABLES
-- =====================================================

CREATE TABLE IF NOT EXISTS production.data_quality_log (
    log_id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    table_name VARCHAR(100),
    issue_type VARCHAR(100),
    issue_description TEXT,
    affected_records INTEGER,
    severity VARCHAR(20) CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    resolved BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS production.etl_log (
    etl_id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(20) CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED', 'WARNING')),
    records_processed INTEGER,
    records_inserted INTEGER,
    records_updated INTEGER,
    errors_count INTEGER,
    error_details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Yarn indexes
CREATE INDEX idx_yarns_desc_id ON production.yarns(desc_id);
CREATE INDEX idx_yarns_supplier ON production.yarns(supplier_id);

-- Style indexes
CREATE INDEX idx_styles_style_number ON production.styles(style_number);
CREATE INDEX idx_styles_fstyle_number ON production.styles(fstyle_number);

-- BOM indexes
CREATE INDEX idx_bom_style ON production.style_bom(style_id);
CREATE INDEX idx_bom_yarn ON production.style_bom(yarn_id);

-- Time-series indexes
CREATE INDEX idx_yarn_inv_date ON production.yarn_inventory_ts(snapshot_date DESC);
CREATE INDEX idx_yarn_inv_yarn ON production.yarn_inventory_ts(yarn_id);
CREATE INDEX idx_fabric_inv_date ON production.fabric_inventory_ts(snapshot_date DESC);
CREATE INDEX idx_fabric_inv_style ON production.fabric_inventory_ts(style_id);
CREATE INDEX idx_fabric_inv_stage ON production.fabric_inventory_ts(inventory_stage);
CREATE INDEX idx_so_date ON production.sales_orders_ts(snapshot_date DESC);
CREATE INDEX idx_so_number ON production.sales_orders_ts(so_number);
CREATE INDEX idx_ko_date ON production.knit_orders_ts(snapshot_date DESC);
CREATE INDEX idx_ko_number ON production.knit_orders_ts(ko_number);

-- =====================================================
-- MATERIALIZED VIEWS FOR API PERFORMANCE
-- =====================================================

-- Current inventory summary view
CREATE MATERIALIZED VIEW api.current_inventory_summary AS
SELECT 
    y.desc_id,
    y.yarn_description,
    yi.theoretical_balance,
    yi.allocated,
    yi.on_order,
    yi.planning_balance,
    yi.weeks_of_supply,
    CASE 
        WHEN yi.planning_balance < 0 THEN 'CRITICAL'
        WHEN yi.weeks_of_supply < 2 THEN 'WARNING'
        ELSE 'OK'
    END as status
FROM production.yarns y
LEFT JOIN LATERAL (
    SELECT * FROM production.yarn_inventory_ts
    WHERE yarn_id = y.yarn_id
    ORDER BY snapshot_date DESC
    LIMIT 1
) yi ON true
WHERE y.is_active = true;

CREATE INDEX idx_current_inv_status ON api.current_inventory_summary(status);

-- Current fabric pipeline view
CREATE MATERIALIZED VIEW api.fabric_pipeline AS
SELECT 
    s.style_number,
    s.fstyle_number,
    s.style_description,
    SUM(CASE WHEN fi.inventory_stage = 'F01' THEN fi.quantity_lbs ELSE 0 END) as finished_lbs,
    SUM(CASE WHEN fi.inventory_stage = 'G00' THEN fi.quantity_lbs ELSE 0 END) as greige_g00_lbs,
    SUM(CASE WHEN fi.inventory_stage = 'G02' THEN fi.quantity_lbs ELSE 0 END) as greige_g02_lbs,
    SUM(CASE WHEN fi.inventory_stage = 'I01' THEN fi.quantity_lbs ELSE 0 END) as inspection_lbs,
    MAX(fi.snapshot_date) as last_updated
FROM production.styles s
LEFT JOIN LATERAL (
    SELECT * FROM production.fabric_inventory_ts
    WHERE style_id = s.style_id
    AND snapshot_date = (SELECT MAX(snapshot_date) FROM production.fabric_inventory_ts)
) fi ON true
GROUP BY s.style_id, s.style_number, s.fstyle_number, s.style_description;

-- =====================================================
-- FUNCTIONS FOR BUSINESS LOGIC
-- =====================================================

-- Function to calculate yarn requirements for a style
CREATE OR REPLACE FUNCTION production.calculate_yarn_requirements(
    p_style_id INTEGER,
    p_quantity DECIMAL
)
RETURNS TABLE (
    yarn_id INTEGER,
    desc_id INTEGER,
    required_qty DECIMAL,
    bom_percent DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        b.yarn_id,
        y.desc_id,
        (p_quantity * b.bom_percent / 100.0)::DECIMAL as required_qty,
        b.bom_percent
    FROM production.style_bom b
    JOIN production.yarns y ON b.yarn_id = y.yarn_id
    WHERE b.style_id = p_style_id;
END;
$$ LANGUAGE plpgsql;

-- Function to convert pounds to yards
CREATE OR REPLACE FUNCTION production.convert_lbs_to_yards(
    p_style_id INTEGER,
    p_lbs DECIMAL
)
RETURNS DECIMAL AS $$
DECLARE
    v_yards_per_pound DECIMAL;
BEGIN
    SELECT yards_per_pound INTO v_yards_per_pound
    FROM production.fabric_specs
    WHERE style_id = p_style_id
    LIMIT 1;
    
    IF v_yards_per_pound IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN p_lbs * v_yards_per_pound;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- DATA RETENTION POLICY (12 months)
-- =====================================================

-- Create policy to drop old data after 12 months
SELECT add_retention_policy('production.yarn_inventory_ts', INTERVAL '12 months', 
    if_not_exists => TRUE);
SELECT add_retention_policy('production.fabric_inventory_ts', INTERVAL '12 months',
    if_not_exists => TRUE);
SELECT add_retention_policy('production.sales_orders_ts', INTERVAL '12 months',
    if_not_exists => TRUE);
SELECT add_retention_policy('production.knit_orders_ts', INTERVAL '12 months',
    if_not_exists => TRUE);
SELECT add_retention_policy('production.yarn_demand_ts', INTERVAL '12 months',
    if_not_exists => TRUE);

-- =====================================================
-- REFRESH MATERIALIZED VIEWS FUNCTION
-- =====================================================

CREATE OR REPLACE FUNCTION api.refresh_all_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY api.current_inventory_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY api.fabric_pipeline;
END;
$$ LANGUAGE plpgsql;

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA production TO PUBLIC;
GRANT USAGE ON SCHEMA api TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA production TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA api TO PUBLIC;

-- Create comment documentation
COMMENT ON SCHEMA production IS 'Main production data schema for Beverly Knits ERP';
COMMENT ON SCHEMA api IS 'API views and functions for dashboard integration';
COMMENT ON TABLE production.yarns IS 'Master yarn catalog with specifications';
COMMENT ON TABLE production.yarn_inventory_ts IS 'Time-series yarn inventory snapshots';
COMMENT ON TABLE production.fabric_inventory_ts IS 'Time-series fabric inventory across all stages (F01, G00, G02, I01)';