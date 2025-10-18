# Forecast Training System Guide

## Overview

The forecast training system enables automated machine learning model training, accuracy tracking, and blend weight optimization for sales forecasting. This guide explains how to use the training features to improve forecast accuracy over time.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           WeeklyForecastGenerator                    │
│  ┌──────────────────────────────────────────────┐  │
│  │ EnhancedForecastingEngine                     │  │
│  │  - Prophet, XGBoost, ARIMA models             │  │
│  │  - Ensemble prediction                         │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │ AutomaticRetrainingSystem                     │  │
│  │  - Loads training data from Turso             │  │
│  │  - Manages retraining schedule                │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │ ForecastBlender                                │  │
│  │  - Combines ML + External forecasts           │  │
│  │  - Auto-tunes blend weights                   │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │ ForecastAccuracyTracker                        │  │
│  │  - Tracks forecast vs actual                  │  │
│  │  - Recommends weight adjustments              │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Initial Training

Train ML models for the first time:

```bash
# Apply database schema (first time only)
python scripts/apply_forecast_training_schema.py

# Trigger initial training via API
curl -X POST http://localhost:5000/api/forecast/train \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

### 2. Check Training Status

```bash
curl http://localhost:5000/api/forecast/training-status
```

Response:
```json
{
  "timestamp": "2025-10-18T14:30:00",
  "needs_training": false,
  "last_training_date": "2025-10-15T02:00:00",
  "days_since_training": 3,
  "ensemble_weights": {
    "prophet": 0.40,
    "xgboost": 0.35,
    "arima": 0.25
  },
  "blend_weights": {
    "ml_historical": 0.40,
    "sales_team": 0.35,
    "customer_commitment": 0.20,
    "market_intelligence": 0.05
  }
}
```

### 3. Auto-Tune Blend Weights

Automatically adjust blend weights based on recent accuracy:

```bash
curl -X POST http://localhost:5000/api/forecast/tune-weights
```

## API Endpoints

### Training Endpoints

#### POST /api/forecast/train
Trigger ML model training

**Request:**
```json
{
  "force": false,  // Force retraining even if models are up to date
  "styles": ["STYLE001", "STYLE002"]  // Optional: specific styles to train
}
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-10-18T...",
  "styles_trained": 45,
  "model_accuracies": {
    "prophet": 0.87,
    "xgboost": 0.89,
    "arima": 0.85
  },
  "weight_adjustments": {
    "status": "adjusted",
    "improvement_pct": 3.2,
    "old_weights": {...},
    "new_weights": {...}
  },
  "average_accuracy": 0.87
}
```

#### GET /api/forecast/training-status
Get current training status and metrics

**Response:**
```json
{
  "timestamp": "2025-10-18T...",
  "needs_training": false,
  "last_training_date": "2025-10-15T...",
  "days_since_training": 3,
  "ensemble_weights": {...},
  "blend_weights": {...},
  "accuracy_report": {...}
}
```

#### POST /api/forecast/tune-weights
Manually trigger blend weight tuning

**Request:**
```json
{
  "lookback_weeks": 13  // Number of weeks to analyze
}
```

**Response:**
```json
{
  "status": "adjusted",
  "improvement_pct": 3.2,
  "old_weights": {
    "ml_historical": 0.40,
    "sales_team": 0.35,
    ...
  },
  "new_weights": {
    "ml_historical": 0.45,
    "sales_team": 0.30,
    ...
  },
  "changes": [
    {
      "source": "ml_historical",
      "old_weight": 0.40,
      "new_weight": 0.45,
      "reason": "ML historical weight increased by 12.5% (MAPE: 12.3%, Hit Rate: 82.1%)",
      "mape": 12.3
    }
  ]
}
```

#### GET /api/forecast/accuracy-history
Get forecast accuracy history

**Query Parameters:**
- `weeks`: Number of weeks to look back (default: 13)
- `source`: Filter by forecast source (optional)

**Response:**
```json
{
  "period": "Last 13 weeks",
  "sources_analyzed": 4,
  "source_performance": {
    "ml_historical": {
      "mape": 12.5,
      "bias": -2.3,
      "hit_rate": 0.78,
      "rmse": 150.2,
      "sample_size": 45
    },
    "sales_team": {...},
    ...
  },
  "best_performer": {
    "source": "customer_commitment",
    "mape": 8.2
  },
  "worst_performer": {
    "source": "market_intelligence",
    "mape": 18.5
  }
}
```

## Training Workflow

### Manual Training

1. **Check if training is needed:**
   ```python
   from src.forecasting.weekly_forecast_generator import WeeklyForecastGenerator

   generator = WeeklyForecastGenerator()
   if generator.needs_training():
       print("Training needed!")
   ```

2. **Trigger training:**
   ```python
   results = generator.train_models(force_retrain=True)
   print(f"Trained {results['styles_trained']} styles")
   print(f"Average accuracy: {results['average_accuracy']:.1%}")
   ```

3. **Check accuracy and tune weights:**
   ```python
   # Get training status
   status = generator.get_training_status()

   # If accuracy is good, weights will auto-tune
   # Weight tuning happens automatically during training
   # But you can also trigger it manually:
   weight_results = generator._tune_blend_weights()
   ```

### Automatic Training

The system automatically checks if training is needed on initialization. Models are considered outdated if:
- Never trained before (last_training_date is None)
- More than 7 days since last training (for weekly schedule)
- More than 1 day since last training (for daily schedule)
- More than 30 days since last training (for monthly schedule)

You'll see a warning on startup:
```
⚠️ ML models need training! Call train_models() or trigger via API endpoint.
```

## Accuracy Tracking

### How It Works

1. **Forecast Generation:** When generating forecasts, each forecast is tracked
2. **Actual Comparison:** When actual sales data comes in, it's compared with forecasts
3. **Error Calculation:** MAPE, bias, hit rate, and RMSE are calculated
4. **Weight Adjustment:** If a source consistently performs better, its weight increases

### Metrics Explained

- **MAPE** (Mean Absolute Percentage Error): Average forecast error as percentage (lower is better)
- **Bias**: Tendency to over-forecast (positive) or under-forecast (negative)
- **Hit Rate**: Percentage of forecasts within ±10% of actual
- **RMSE** (Root Mean Squared Error): Standard deviation of forecast errors

### Viewing Accuracy Reports

```python
report = generator.accuracy_tracker.generate_accuracy_report(lookback_weeks=13)

print(f"Sources analyzed: {report['sources_analyzed']}")
print(f"Best performer: {report['best_performer']['source']} (MAPE: {report['best_performer']['mape']:.1f}%)")

for source, metrics in report['source_performance'].items():
    print(f"{source}:")
    print(f"  MAPE: {metrics['mape']:.1f}%")
    print(f"  Hit Rate: {metrics['hit_rate']*100:.1f}%")
    print(f"  Sample Size: {metrics['sample_size']}")
```

## Blend Weight Tuning

### Default Weights

```python
{
    'ml_historical': 0.40,      # ML from historical sales
    'sales_team': 0.35,          # Sales team forecast
    'customer_commitment': 0.20, # Customer forward orders
    'market_intelligence': 0.05  # Industry data
}
```

### Auto-Tuning Process

1. **Calculate Performance:** Analyze accuracy of each source over lookback period
2. **Compute New Weights:** Weights proportional to inverse MAPE
3. **Apply Smoothing:** 70% new weight + 30% old weight (prevents wild swings)
4. **Check Improvement:** Only apply if improvement > 2%
5. **Store History:** Save weight changes to database

### Manual Weight Adjustment

```python
# Recommend weight adjustments
recommendations = generator.accuracy_tracker.recommend_weight_adjustments(
    current_weights=generator.blender.source_weights,
    lookback_weeks=13
)

if recommendations['overall_improvement'] > 2.0:
    # Apply recommended weights
    generator.blender.source_weights = recommendations['recommended_weights']
    print(f"Weights adjusted: {recommendations['overall_improvement']:.1f}% improvement")
```

## Database Schema

### Tables

1. **forecast_training_history**: Tracks each training session
2. **forecast_blend_weights**: History of weight adjustments
3. **forecast_accuracy**: Forecast vs actual tracking
4. **external_forecasts**: Uploaded external forecasts
5. **model_performance_metrics**: Individual model performance

### Views

- `v_latest_blend_weights`: Current weights by source
- `v_recent_training`: Last 10 training sessions
- `v_accuracy_by_source`: Accuracy summary by source
- `v_problematic_styles`: Styles with >20% average error

## Best Practices

### Training Frequency

- **Weekly**: Good for most scenarios (default)
- **Daily**: For rapidly changing demand patterns
- **Monthly**: For stable, predictable products

### Accuracy Monitoring

- Check accuracy reports weekly
- Investigate styles with >20% MAPE
- Compare forecast sources to identify best performers

### Weight Tuning

- Auto-tune after every training session
- Manual tune if you notice accuracy drops
- Review weight changes in database for audit trail

### Data Quality

- Ensure historical sales data is complete
- Validate external forecasts before upload
- Remove outliers that skew training

## Troubleshooting

### Issue: Models not training

**Symptoms:** `needs_training()` returns True but training fails

**Solutions:**
1. Check if historical_sales table has data
2. Verify at least 10 records per style
3. Check database permissions
4. Look for errors in logs

### Issue: Low accuracy after training

**Symptoms:** Average accuracy < 80%

**Solutions:**
1. Increase lookback period (180→365 days)
2. Check for seasonality in data
3. Verify data quality (no missing weeks)
4. Try different ensemble weights

### Issue: Weights not tuning

**Symptoms:** `_tune_blend_weights()` returns "unchanged"

**Solutions:**
1. Improvement may be < 2% threshold
2. Not enough forecast accuracy data
3. Check `forecast_accuracy` table has records

### Issue: Training takes too long

**Symptoms:** Training > 5 minutes

**Solutions:**
1. Reduce number of styles (use `styles` parameter)
2. Decrease lookback period
3. Check database query performance
4. Consider batch processing

## Advanced Features

### Custom Training Schedule

```python
# Initialize with custom schedule
auto_retrain = AutomaticRetrainingSystem(
    retrain_schedule='daily',
    retrain_day='monday',
    retrain_hour=3
)

# Start automatic scheduler
auto_retrain.start()
```

### Style-Specific Training

```python
# Train only high-volume styles
high_volume_styles = ['STYLE001', 'STYLE002', 'STYLE003']
results = generator.train_models(styles=high_volume_styles)
```

### Export Training History

```python
import json
from pathlib import Path

# Training history is auto-saved to:
history_file = Path('forecast_training_history.json')

with open(history_file, 'r') as f:
    history = json.load(f)

# Analyze trends
for session in history[-5:]:  # Last 5 sessions
    print(f"Date: {session['timestamp']}")
    print(f"Accuracy: {session['avg_accuracy']:.1%}")
    print(f"Styles: {session['styles_trained']}")
```

## Summary

The forecast training system provides:

✅ **Automated Training** - Models retrain on schedule
✅ **Accuracy Tracking** - Monitor forecast vs actual performance
✅ **Auto-Tuning** - Weights adjust based on accuracy
✅ **API Access** - Trigger training and check status via REST API
✅ **Database Persistence** - All training history saved to Turso
✅ **Multi-Source Blending** - Combine ML + external forecasts optimally

For questions or issues, check the logs or consult the development team.
