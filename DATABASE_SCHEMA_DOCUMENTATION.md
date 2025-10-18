# Beverly Knits ERP - Database Schema Documentation

## Database Overview
- **ORM**: SQLAlchemy 2.0+
- **Primary Database**: PostgreSQL (Production) / SQLite (Development)
- **Schema Version**: 2.0
- **Last Updated**: 2025-01-18

## Entity Relationship Diagram

```
┌──────────────────┐         ┌────────────────────┐
│   CFVersion      │────┬────│  YarnRequirement   │
│  (Core Fabric)   │    │    │    (BOM Items)     │
└──────────────────┘    │    └────────────────────┘
         │              │              │
         │              │              │
         ▼              │              ▼
┌──────────────────┐    │    ┌────────────────────┐
│ ProductionOrder  │    │    │   YarnInventory    │
│                  │    │    │  (Stock Levels)    │
└──────────────────┘    │    └────────────────────┘
         │              │
         │              │
         ▼              │
┌──────────────────┐    │    ┌────────────────────┐
│MachineAssignment │    └────│     APISync        │
│                  │         │  (Sync Tracking)   │
└──────────────────┘         └────────────────────┘
```

## Table Specifications

### 1. cf_versions (Core Fabric/Style Versions)

**Purpose**: Master data for fabric styles and specifications

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY | Auto-increment ID |
| version_id | INTEGER | UNIQUE, INDEX | External version identifier |
| style_number | VARCHAR(50) | INDEX | Style/SKU number |
| description | TEXT | | Product description |
| customer_code | VARCHAR(50) | | Customer identifier |
| fabric_type | VARCHAR(100) | | Type of fabric |
| construction | VARCHAR(50) | | Fabric construction details |
| width | FLOAT | | Fabric width (inches) |
| weight | FLOAT | | Fabric weight (oz/yd²) |
| status | VARCHAR(20) | | Active/Inactive/Discontinued |
| created_at | DATETIME | DEFAULT NOW | Creation timestamp |
| updated_at | DATETIME | ON UPDATE NOW | Last modification |
| api_data | JSON | | Raw API response storage |

**Indexes**:
- `idx_cf_style_customer` on (style_number, customer_code)

**Relationships**:
- Has many `yarn_requirements`
- Has many `production_orders`

---

### 2. yarn_requirements (Bill of Materials)

**Purpose**: Define yarn requirements for each fabric style

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY | Auto-increment ID |
| cf_version_id | INTEGER | FOREIGN KEY | Reference to cf_versions |
| yarn_code | VARCHAR(50) | INDEX | Yarn SKU/code |
| yarn_description | TEXT | | Yarn description |
| supplier | VARCHAR(100) | | Supplier name |
| color | VARCHAR(50) | | Yarn color |
| quantity_required | DECIMAL(15,3) | | Required quantity per unit |
| unit_of_measure | VARCHAR(10) | | UOM (KG, LBS, etc.) |
| cost_per_unit | DECIMAL(10,4) | | Unit cost |
| lead_time_days | INTEGER | | Procurement lead time |
| created_at | DATETIME | DEFAULT NOW | Creation timestamp |
| updated_at | DATETIME | ON UPDATE NOW | Last modification |

**Constraints**:
- UNIQUE constraint on (cf_version_id, yarn_code)

**Indexes**:
- `idx_yarn_supplier` on (yarn_code, supplier)

**Relationships**:
- Belongs to `cf_versions`
- Has many `yarn_inventory` levels

---

### 3. yarn_inventory (Current Stock Levels)

**Purpose**: Track real-time inventory levels by location

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY | Auto-increment ID |
| yarn_requirement_id | INTEGER | FOREIGN KEY | Reference to yarn_requirements |
| yarn_code | VARCHAR(50) | INDEX | Yarn SKU/code |
| location | VARCHAR(50) | | Warehouse/location |
| quantity_on_hand | DECIMAL(15,3) | CHECK >= 0 | Physical inventory |
| quantity_allocated | DECIMAL(15,3) | | Reserved for orders |
| quantity_available | DECIMAL(15,3) | | Available to promise |
| reorder_point | DECIMAL(15,3) | | Min stock level |
| reorder_quantity | DECIMAL(15,3) | | Standard order qty |
| last_received_date | DATE | | Last receipt date |
| last_counted_date | DATE | | Last physical count |
| created_at | DATETIME | DEFAULT NOW | Creation timestamp |
| updated_at | DATETIME | ON UPDATE NOW | Last modification |

**Constraints**:
- UNIQUE constraint on (yarn_code, location)
- CHECK constraint: quantity_on_hand >= 0

**Relationships**:
- Belongs to `yarn_requirements`

---

### 4. production_orders (Manufacturing Orders)

**Purpose**: Track production orders and their status

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY | Auto-increment ID |
| order_number | VARCHAR(50) | UNIQUE, INDEX | Order identifier |
| cf_version_id | INTEGER | FOREIGN KEY | Reference to cf_versions |
| customer_po | VARCHAR(50) | | Customer PO number |
| quantity_ordered | DECIMAL(15,3) | | Order quantity |
| quantity_produced | DECIMAL(15,3) | DEFAULT 0 | Completed quantity |
| unit_of_measure | VARCHAR(10) | | UOM |
| due_date | DATE | | Order due date |
| start_date | DATE | | Production start date |
| status | VARCHAR(20) | | Order status |
| priority | INTEGER | | Priority level (1-10) |
| work_center | VARCHAR(50) | | Production area |
| machine_id | VARCHAR(50) | | Assigned machine |
| created_at | DATETIME | DEFAULT NOW | Creation timestamp |
| updated_at | DATETIME | ON UPDATE NOW | Last modification |

**Indexes**:
- `idx_order_status_due` on (status, due_date)
- `idx_order_machine` on (machine_id, work_center)

**Relationships**:
- Belongs to `cf_versions`
- Has many `machine_assignments`

---

### 5. machine_assignments (Production Scheduling)

**Purpose**: Track machine assignments and scheduling

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY | Auto-increment ID |
| production_order_id | INTEGER | FOREIGN KEY | Reference to production_orders |
| machine_id | VARCHAR(50) | INDEX | Machine identifier |
| work_center | VARCHAR(50) | | Work center location |
| assigned_date | DATETIME | | Assignment timestamp |
| start_time | DATETIME | | Scheduled start |
| end_time | DATETIME | | Scheduled end |
| estimated_hours | FLOAT | | Estimated duration |
| actual_hours | FLOAT | | Actual duration |
| status | VARCHAR(20) | | Assignment status |
| operator_id | VARCHAR(50) | | Operator assignment |
| created_at | DATETIME | DEFAULT NOW | Creation timestamp |
| updated_at | DATETIME | ON UPDATE NOW | Last modification |

**Indexes**:
- `idx_machine_schedule` on (machine_id, start_time, end_time)

**Relationships**:
- Belongs to `production_orders`

---

### 6. api_sync (External API Synchronization)

**Purpose**: Track and audit external API synchronization

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY | Auto-increment ID |
| sync_id | VARCHAR(50) | UNIQUE, DEFAULT UUID | Sync identifier |
| endpoint | VARCHAR(200) | | API endpoint |
| sync_type | VARCHAR(20) | | full/incremental/manual |
| started_at | DATETIME | DEFAULT NOW | Sync start time |
| completed_at | DATETIME | | Sync completion time |
| status | VARCHAR(20) | | running/success/failed |
| records_processed | INTEGER | DEFAULT 0 | Total records processed |
| records_created | INTEGER | DEFAULT 0 | New records created |
| records_updated | INTEGER | DEFAULT 0 | Records updated |
| records_failed | INTEGER | DEFAULT 0 | Failed records |
| error_message | TEXT | | Error details if failed |
| metadata | JSON | | Additional sync metadata |
| created_at | DATETIME | DEFAULT NOW | Creation timestamp |

**Indexes**:
- `idx_sync_status_time` on (status, started_at)

---

### 7. yarn_demand_reports (Demand Analysis)

**Purpose**: Store parsed demand reports and forecasts

| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY | Auto-increment ID |
| report_date | DATE | INDEX | Report generation date |
| yarn_code | VARCHAR(50) | INDEX | Yarn SKU/code |
| yarn_description | TEXT | | Yarn description |
| current_inventory | DECIMAL(15,3) | | Current stock level |
| allocated_quantity | DECIMAL(15,3) | | Reserved quantity |
| on_order_quantity | DECIMAL(15,3) | | Purchase orders in transit |
| planned_usage | DECIMAL(15,3) | | Forecasted usage |
| projected_balance | DECIMAL(15,3) | | Projected inventory |
| reorder_suggestion | DECIMAL(15,3) | | Suggested order quantity |
| lead_time_days | INTEGER | | Lead time for procurement |
| created_at | DATETIME | DEFAULT NOW | Creation timestamp |
| updated_at | DATETIME | ON UPDATE NOW | Last modification |

**Indexes**:
- Index on report_date
- Index on yarn_code

---

## Database Operations

### Common Queries

#### 1. Get Net Requirements
```sql
SELECT
    yr.yarn_code,
    yr.quantity_required * po.quantity_ordered AS gross_requirement,
    COALESCE(yi.quantity_available, 0) AS available_inventory,
    (yr.quantity_required * po.quantity_ordered) - COALESCE(yi.quantity_available, 0) AS net_requirement
FROM production_orders po
JOIN yarn_requirements yr ON yr.cf_version_id = po.cf_version_id
LEFT JOIN yarn_inventory yi ON yi.yarn_code = yr.yarn_code
WHERE po.status = 'ACTIVE';
```

#### 2. Machine Utilization
```sql
SELECT
    machine_id,
    COUNT(*) as total_assignments,
    SUM(actual_hours) as total_hours,
    AVG(actual_hours / estimated_hours) as efficiency
FROM machine_assignments
WHERE start_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY machine_id;
```

#### 3. Inventory Turnover
```sql
SELECT
    yarn_code,
    SUM(quantity_on_hand) as total_inventory,
    AVG(planned_usage) as avg_daily_usage,
    SUM(quantity_on_hand) / NULLIF(AVG(planned_usage), 0) as days_of_inventory
FROM yarn_inventory yi
JOIN yarn_demand_reports ydr USING(yarn_code)
GROUP BY yarn_code;
```

### Database Migrations

Migrations are managed through SQLAlchemy Alembic:
```bash
# Create migration
alembic revision -m "Add new column"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Performance Optimizations

1. **Connection Pooling**: Configured in `database/connection_pool.py`
   - Pool size: 10
   - Max overflow: 20
   - Pool timeout: 30s

2. **Query Optimization**:
   - Use of compound indexes for frequent JOIN operations
   - Materialized views for complex aggregations
   - Query result caching with Redis

3. **Batch Operations**:
   - Bulk inserts for data imports
   - Batch updates for inventory adjustments
   - Transaction batching for consistency

### Data Integrity

1. **Foreign Key Constraints**: Enforced at database level
2. **Check Constraints**: Validate data ranges
3. **Unique Constraints**: Prevent duplicates
4. **Triggers**: Audit trail for critical changes

### Backup Strategy

1. **Daily Backups**: Full database backup at 2 AM
2. **Transaction Logs**: Continuous archival
3. **Retention**: 30 days for daily, 1 year for monthly
4. **Recovery**: Point-in-time recovery capability

---

*Documentation Version: 1.0*
*Last Updated: 2025-01-18*