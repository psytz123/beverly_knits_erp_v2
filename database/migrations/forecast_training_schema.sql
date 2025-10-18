-- Forecast Training Schema Migration
-- Creates tables for tracking ML model training, accuracy, and blend weights
-- Created: 2025-10-18

-- ============================================================================
-- TRAINING HISTORY TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS forecast_training_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    styles_trained INTEGER NOT NULL,
    avg_accuracy REAL,
    training_time_seconds REAL,
    model_weights TEXT, -- JSON string of model weights
    status TEXT CHECK(status IN ('success', 'failed', 'skipped')) DEFAULT 'success',
    error_message TEXT,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_history_date ON forecast_training_history(training_date DESC);

-- ============================================================================
-- FORECAST BLEND WEIGHTS HISTORY TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS forecast_blend_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    effective_date DATE NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('ml_historical', 'sales_team', 'customer_commitment', 'market_intelligence')),
    weight REAL NOT NULL CHECK(weight >= 0 AND weight <= 1),
    reason TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_blend_weights_date ON forecast_blend_weights(effective_date DESC);
CREATE INDEX IF NOT EXISTS idx_blend_weights_source ON forecast_blend_weights(source);

-- ============================================================================
-- FORECAST ACCURACY TRACKING TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS forecast_accuracy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    style TEXT NOT NULL,
    week_number INTEGER NOT NULL CHECK(week_number >= 1 AND week_number <= 53),
    forecast_date DATE NOT NULL,
    forecasted_quantity REAL NOT NULL,
    actual_quantity REAL,
    source TEXT NOT NULL,
    error_pct REAL,
    absolute_error REAL,
    tracked_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_forecast_accuracy_style ON forecast_accuracy(style);
CREATE INDEX IF NOT EXISTS idx_forecast_accuracy_date ON forecast_accuracy(forecast_date DESC);
CREATE INDEX IF NOT EXISTS idx_forecast_accuracy_source ON forecast_accuracy(source);
CREATE INDEX IF NOT EXISTS idx_forecast_accuracy_week ON forecast_accuracy(week_number);

-- ============================================================================
-- EXTERNAL FORECASTS TABLE (for uploaded sales team/customer forecasts)
-- ============================================================================
CREATE TABLE IF NOT EXISTS external_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    style TEXT NOT NULL,
    week_number INTEGER NOT NULL CHECK(week_number >= 1 AND week_number <= 53),
    forecasted_yards REAL NOT NULL CHECK(forecasted_yards >= 0),
    confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
    source_name TEXT NOT NULL,
    notes TEXT,
    uploaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    uploaded_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_external_forecasts_style ON external_forecasts(style);
CREATE INDEX IF NOT EXISTS idx_external_forecasts_week ON external_forecasts(week_number);
CREATE INDEX IF NOT EXISTS idx_external_forecasts_source ON external_forecasts(source_name);
CREATE INDEX IF NOT EXISTS idx_external_forecasts_uploaded ON external_forecasts(uploaded_at DESC);

-- ============================================================================
-- MODEL PERFORMANCE METRICS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL CHECK(model_name IN ('prophet', 'xgboost', 'arima', 'ensemble')),
    metric_date DATE NOT NULL,
    mape REAL, -- Mean Absolute Percentage Error
    rmse REAL, -- Root Mean Squared Error
    mae REAL,  -- Mean Absolute Error
    accuracy REAL CHECK(accuracy >= 0 AND accuracy <= 1),
    sample_size INTEGER,
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_performance_model ON model_performance_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_date ON model_performance_metrics(metric_date DESC);

-- ============================================================================
-- INITIAL DATA - Default Blend Weights
-- ============================================================================
INSERT OR IGNORE INTO forecast_blend_weights (id, effective_date, source, weight, reason)
VALUES
    (1, DATE('now'), 'ml_historical', 0.40, 'Initial default weight'),
    (2, DATE('now'), 'sales_team', 0.35, 'Initial default weight'),
    (3, DATE('now'), 'customer_commitment', 0.20, 'Initial default weight'),
    (4, DATE('now'), 'market_intelligence', 0.05, 'Initial default weight');

-- ============================================================================
-- VIEWS FOR REPORTING
-- ============================================================================

-- Latest blend weights by source
CREATE VIEW IF NOT EXISTS v_latest_blend_weights AS
SELECT
    source,
    weight,
    effective_date,
    reason,
    created_at
FROM forecast_blend_weights
WHERE (source, effective_date) IN (
    SELECT source, MAX(effective_date)
    FROM forecast_blend_weights
    GROUP BY source
)
ORDER BY weight DESC;

-- Recent training history
CREATE VIEW IF NOT EXISTS v_recent_training AS
SELECT
    training_date,
    styles_trained,
    avg_accuracy,
    training_time_seconds,
    status,
    CASE
        WHEN avg_accuracy >= 0.90 THEN 'Excellent'
        WHEN avg_accuracy >= 0.85 THEN 'Good'
        WHEN avg_accuracy >= 0.80 THEN 'Fair'
        ELSE 'Poor'
    END as performance_rating
FROM forecast_training_history
ORDER BY training_date DESC
LIMIT 10;

-- Forecast accuracy summary by source
CREATE VIEW IF NOT EXISTS v_accuracy_by_source AS
SELECT
    source,
    COUNT(*) as forecast_count,
    AVG(ABS(error_pct)) as avg_error_pct,
    AVG(absolute_error) as avg_absolute_error,
    MIN(forecast_date) as first_forecast,
    MAX(forecast_date) as last_forecast
FROM forecast_accuracy
WHERE actual_quantity IS NOT NULL
GROUP BY source
ORDER BY avg_error_pct ASC;

-- Styles with poor forecast accuracy
CREATE VIEW IF NOT EXISTS v_problematic_styles AS
SELECT
    style,
    COUNT(*) as forecast_count,
    AVG(ABS(error_pct)) as avg_error_pct,
    MAX(ABS(error_pct)) as max_error_pct
FROM forecast_accuracy
WHERE actual_quantity IS NOT NULL
GROUP BY style
HAVING avg_error_pct > 20  -- >20% average error
ORDER BY avg_error_pct DESC
LIMIT 50;

-- ============================================================================
-- COMMENTS
-- ============================================================================

-- This schema supports:
-- 1. Tracking ML model training history and performance
-- 2. Managing blend weight adjustments over time
-- 3. Monitoring forecast accuracy by source and style
-- 4. Storing external forecasts from sales team/customers
-- 5. Performance metrics for individual ML models
-- 6. Views for quick reporting and analysis
