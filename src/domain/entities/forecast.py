"""Forecast domain entity for demand prediction."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, date
from enum import Enum


class ForecastModel(Enum):
    """ML models available for forecasting."""
    ARIMA = "arima"
    PROPHET = "prophet"
    LSTM = "lstm"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"


class ForecastConfidence(Enum):
    """Confidence levels for forecasts."""
    HIGH = "high"  # > 90% accuracy
    MEDIUM = "medium"  # 70-90% accuracy
    LOW = "low"  # < 70% accuracy


@dataclass
class ForecastPoint:
    """Single point in a forecast timeline."""
    date: date
    predicted_value: float
    lower_bound: float
    upper_bound: float
    confidence_interval: float = 0.95


@dataclass
class DemandForecast:
    """Domain entity representing a demand forecast."""
    
    forecast_id: str
    style_id: str
    model: ForecastModel
    created_at: datetime
    horizon_days: int
    predictions: List[ForecastPoint] = field(default_factory=list)
    confidence: ForecastConfidence = ForecastConfidence.MEDIUM
    accuracy_score: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error
    rmse: float = 0.0  # Root Mean Square Error
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_predicted_demand(self) -> float:
        """Calculate total demand over forecast horizon."""
        return sum(p.predicted_value for p in self.predictions)
    
    @property
    def average_daily_demand(self) -> float:
        """Calculate average daily demand."""
        if not self.predictions:
            return 0.0
        return self.total_predicted_demand / len(self.predictions)
    
    @property
    def peak_demand_date(self) -> Optional[date]:
        """Find date with highest predicted demand."""
        if not self.predictions:
            return None
        peak = max(self.predictions, key=lambda p: p.predicted_value)
        return peak.date
    
    @property
    def peak_demand_value(self) -> float:
        """Get highest predicted demand value."""
        if not self.predictions:
            return 0.0
        return max(p.predicted_value for p in self.predictions)
    
    def get_demand_for_date(self, target_date: date) -> Optional[float]:
        """Get predicted demand for specific date."""
        for point in self.predictions:
            if point.date == target_date:
                return point.predicted_value
        return None
    
    def get_demand_for_period(self, start_date: date, end_date: date) -> float:
        """Get total predicted demand for a period."""
        total = 0.0
        for point in self.predictions:
            if start_date <= point.date <= end_date:
                total += point.predicted_value
        return total
    
    def get_confidence_bounds(self) -> Dict[str, float]:
        """Get confidence bounds for the entire forecast."""
        if not self.predictions:
            return {'lower': 0.0, 'upper': 0.0}
        
        return {
            'lower': sum(p.lower_bound for p in self.predictions),
            'upper': sum(p.upper_bound for p in self.predictions)
        }
    
    def is_high_confidence(self) -> bool:
        """Check if forecast has high confidence."""
        return self.confidence == ForecastConfidence.HIGH or self.accuracy_score > 0.9
    
    def needs_retraining(self, days_old: int = 7, accuracy_threshold: float = 0.7) -> bool:
        """Check if model needs retraining."""
        age_days = (datetime.now() - self.created_at).days
        return age_days > days_old or self.accuracy_score < accuracy_threshold
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'forecast_id': self.forecast_id,
            'style_id': self.style_id,
            'model': self.model.value,
            'created_at': self.created_at.isoformat(),
            'horizon_days': self.horizon_days,
            'predictions': [
                {
                    'date': p.date.isoformat(),
                    'value': p.predicted_value,
                    'lower_bound': p.lower_bound,
                    'upper_bound': p.upper_bound
                }
                for p in self.predictions
            ],
            'total_predicted_demand': self.total_predicted_demand,
            'average_daily_demand': self.average_daily_demand,
            'confidence': self.confidence.value,
            'accuracy_score': self.accuracy_score,
            'mape': self.mape,
            'rmse': self.rmse,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DemandForecast':
        """Create from dictionary."""
        predictions = []
        for pred_data in data.get('predictions', []):
            predictions.append(ForecastPoint(
                date=date.fromisoformat(pred_data['date']),
                predicted_value=pred_data['value'],
                lower_bound=pred_data['lower_bound'],
                upper_bound=pred_data['upper_bound']
            ))
        
        return cls(
            forecast_id=data['forecast_id'],
            style_id=data['style_id'],
            model=ForecastModel(data['model']),
            created_at=datetime.fromisoformat(data['created_at']),
            horizon_days=data['horizon_days'],
            predictions=predictions,
            confidence=ForecastConfidence(data.get('confidence', 'medium')),
            accuracy_score=data.get('accuracy_score', 0.0),
            mape=data.get('mape', 0.0),
            rmse=data.get('rmse', 0.0),
            metadata=data.get('metadata', {})
        )


@dataclass
class ForecastAccuracy:
    """Domain entity for tracking forecast accuracy."""
    
    model: ForecastModel
    style_id: Optional[str]
    period_start: date
    period_end: date
    predicted_values: List[float]
    actual_values: List[float]
    mape: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0  # Mean Absolute Error
    r_squared: float = 0.0
    
    def calculate_metrics(self):
        """Calculate all accuracy metrics."""
        if not self.predicted_values or not self.actual_values:
            return
        
        n = min(len(self.predicted_values), len(self.actual_values))
        
        # Calculate MAPE
        mape_sum = 0
        mae_sum = 0
        rmse_sum = 0
        
        for i in range(n):
            actual = self.actual_values[i]
            predicted = self.predicted_values[i]
            
            if actual != 0:
                mape_sum += abs((actual - predicted) / actual)
            
            mae_sum += abs(actual - predicted)
            rmse_sum += (actual - predicted) ** 2
        
        self.mape = (mape_sum / n) * 100 if n > 0 else 0
        self.mae = mae_sum / n if n > 0 else 0
        self.rmse = (rmse_sum / n) ** 0.5 if n > 0 else 0
        
        # Calculate R-squared
        if n > 0:
            mean_actual = sum(self.actual_values[:n]) / n
            ss_tot = sum((a - mean_actual) ** 2 for a in self.actual_values[:n])
            ss_res = rmse_sum
            
            self.r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def get_confidence_level(self) -> ForecastConfidence:
        """Determine confidence level based on accuracy metrics."""
        if self.mape < 10 and self.r_squared > 0.9:
            return ForecastConfidence.HIGH
        elif self.mape < 30 and self.r_squared > 0.7:
            return ForecastConfidence.MEDIUM
        else:
            return ForecastConfidence.LOW