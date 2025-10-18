-- Fabric Inventory Inquiry System - Database Schema
-- Creates tables for tracking fabric inventory across production stages
-- Compatible with Turso/SQLite

BEGIN;

-- Fabric inventory tracking by stage
CREATE TABLE IF NOT EXISTS fabric_inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fabric_id TEXT NOT NULL,              -- F ID or G ID from fabric specs
    fabric_type TEXT NOT NULL CHECK(fabric_type IN ('finished', 'greige')),
    stage TEXT NOT NULL CHECK(stage IN ('G00', 'G02', 'I01', 'F01')),
    quantity_yards REAL DEFAULT 0,
    quantity_lbs REAL DEFAULT 0,
    rolls INTEGER DEFAULT 0,
    location TEXT,                        -- Warehouse location
    lot_number TEXT,                      -- Lot/batch identifier
    grade TEXT,                           -- Quality grade (A, B, C, etc.)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(fabric_id, stage, lot_number)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_fabric_inv_fabric_id
    ON fabric_inventory(fabric_id);

CREATE INDEX IF NOT EXISTS idx_fabric_inv_stage
    ON fabric_inventory(stage);

CREATE INDEX IF NOT EXISTS idx_fabric_inv_type
    ON fabric_inventory(fabric_type);

CREATE INDEX IF NOT EXISTS idx_fabric_inv_location
    ON fabric_inventory(location);

-- Fabric movement history table
CREATE TABLE IF NOT EXISTS fabric_movements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fabric_id TEXT NOT NULL,
    from_stage TEXT,                      -- NULL if receiving into system
    to_stage TEXT NOT NULL,
    quantity_yards REAL NOT NULL,
    quantity_lbs REAL NOT NULL,
    movement_date TEXT DEFAULT CURRENT_TIMESTAMP,
    operator TEXT,                        -- User who performed movement
    reference_doc TEXT,                   -- PO, transfer order, work order, etc.
    notes TEXT,
    CHECK(quantity_yards >= 0),
    CHECK(quantity_lbs >= 0)
);

-- Indexes for movement history
CREATE INDEX IF NOT EXISTS idx_fabric_mov_fabric
    ON fabric_movements(fabric_id);

CREATE INDEX IF NOT EXISTS idx_fabric_mov_date
    ON fabric_movements(movement_date DESC);

CREATE INDEX IF NOT EXISTS idx_fabric_mov_stages
    ON fabric_movements(from_stage, to_stage);

-- Trigger to update updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_fabric_inventory_timestamp
AFTER UPDATE ON fabric_inventory
FOR EACH ROW
BEGIN
    UPDATE fabric_inventory
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

COMMIT;
