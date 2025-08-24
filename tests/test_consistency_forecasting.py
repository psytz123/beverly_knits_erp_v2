#!/usr/bin/env python3
"""
Test cases for Consistency-Based Forecasting
Tests the consistency scoring and adaptive forecasting methods.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, patch, MagicMock

from core.beverly_comprehensive_erp import SalesForecastingEngine


class TestConsistencyForecasting(unittest.TestCase):
    """Test cases for consistency-based forecasting functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = SalesForecastingEngine()
        
        # Create sample sales data with different consistency patterns
        base_date = datetime.now() - timedelta(days=180)
        
        # Consistent style - low variation
        self.consistent_data = pd.DataFrame({
            'Style#': ['STYLE001'] * 24,
            'Invoice Date': [base_date + timedelta(days=i*7) for i in range(24)],
            'Yds_ordered': [100 + np.random.normal(0, 5) for _ in range(24)]  # Mean 100, std 5
        })
        
        # Variable style - medium variation
        self.variable_data = pd.DataFrame({
            'Style#': ['STYLE002'] * 24,
            'Invoice Date': [base_date + timedelta(days=i*7) for i in range(24)],
            'Yds_ordered': [100 + np.random.normal(0, 20) for _ in range(24)]  # Mean 100, std 20
        })
        
        # Highly variable style - high variation
        self.highly_variable_data = pd.DataFrame({
            'Style#': ['STYLE003'] * 24,
            'Invoice Date': [base_date + timedelta(days=i*7) for i in range(24)],
            'Yds_ordered': [100 + np.random.normal(0, 50) for _ in range(24)]  # Mean 100, std 50
        })
        
        # Seasonal pattern
        self.seasonal_data = pd.DataFrame({
            'Style#': ['STYLE004'] * 24,
            'Invoice Date': [base_date + timedelta(days=i*7) for i in range(24)],
            'Yds_ordered': [100 + 30 * np.sin(2 * np.pi * i / 12) for i in range(24)]  # Seasonal pattern
        })
        
        # Trending data
        self.trending_data = pd.DataFrame({
            'Style#': ['STYLE005'] * 24,
            'Invoice Date': [base_date + timedelta(days=i*7) for i in range(24)],
            'Yds_ordered': [80 + i * 2 for i in range(24)]  # Linear growth
        })
        
    def test_consistency_score_calculation(self):
        """Test consistency score calculation for different patterns"""
        
        # Test consistent data (CV should be low, score should be high)
        consistent_score = self.engine.calculate_consistency_score(
            self.consistent_data['Yds_ordered'].values
        )
        self.assertGreater(consistent_score, 0.8, "Consistent data should have high consistency score")
        
        # Test variable data (CV should be medium, score should be medium)
        variable_score = self.engine.calculate_consistency_score(
            self.variable_data['Yds_ordered'].values
        )
        self.assertGreater(variable_score, 0.3, "Variable data score should be > 0.3")
        self.assertLess(variable_score, 0.8, "Variable data score should be < 0.8")
        
        # Test highly variable data (CV should be high, score should be low)
        highly_variable_score = self.engine.calculate_consistency_score(
            self.highly_variable_data['Yds_ordered'].values
        )
        self.assertLess(highly_variable_score, 0.5, "Highly variable data should have low consistency score")
        
    def test_consistency_score_edge_cases(self):
        """Test consistency score with edge cases"""
        
        # Test with constant values (perfect consistency)
        constant_data = np.array([100] * 10)
        score = self.engine.calculate_consistency_score(constant_data)
        self.assertEqual(score, 1.0, "Constant data should have perfect consistency score of 1.0")
        
        # Test with single value
        single_value = np.array([100])
        score = self.engine.calculate_consistency_score(single_value)
        self.assertEqual(score, 1.0, "Single value should have consistency score of 1.0")
        
        # Test with empty array
        empty_data = np.array([])
        score = self.engine.calculate_consistency_score(empty_data)
        self.assertEqual(score, 0.0, "Empty data should have consistency score of 0.0")
        
        # Test with very high variation (CV > 1)
        high_variation = np.array([10, 100, 5, 200, 1])
        score = self.engine.calculate_consistency_score(high_variation)
        self.assertEqual(score, 0.0, "Very high variation should have consistency score of 0.0")
        
    @patch('beverly_comprehensive_erp.SalesForecastingEngine.ml_forecast')
    @patch('beverly_comprehensive_erp.SalesForecastingEngine._calculate_weighted_average')
    def test_forecast_with_consistency_high(self, mock_weighted_avg, mock_ml_forecast):
        """Test forecasting with high consistency (uses ML)"""
        
        # Setup mocks
        mock_ml_forecast.return_value = pd.Series([110] * 90, index=pd.date_range(start=datetime.now(), periods=90))
        mock_weighted_avg.return_value = pd.Series([105] * 90)
        
        # Test with consistent data (score > 0.7)
        forecast = self.engine.forecast_with_consistency(self.consistent_data, horizon_days=90)
        
        # Should use ML forecast for high consistency
        mock_ml_forecast.assert_called_once()
        mock_weighted_avg.assert_not_called()
        
        self.assertIsNotNone(forecast)
        self.assertIn('forecast', forecast)
        self.assertIn('consistency_score', forecast)
        self.assertIn('method_used', forecast)
        self.assertEqual(forecast['method_used'], 'ml_forecast')
        self.assertGreater(forecast['consistency_score'], 0.7)
        
    @patch('beverly_comprehensive_erp.SalesForecastingEngine.ml_forecast')
    @patch('beverly_comprehensive_erp.SalesForecastingEngine._calculate_weighted_average')
    def test_forecast_with_consistency_medium(self, mock_weighted_avg, mock_ml_forecast):
        """Test forecasting with medium consistency (uses weighted average)"""
        
        # Setup mocks
        mock_weighted_avg.return_value = pd.Series([105] * 90)
        
        # Create data with medium consistency (0.3 < score < 0.7)
        medium_data = pd.DataFrame({
            'Style#': ['STYLE_MED'] * 10,
            'Invoice Date': pd.date_range(start=datetime.now() - timedelta(days=70), periods=10, freq='W'),
            'Yds_ordered': [100, 90, 110, 85, 115, 95, 105, 92, 108, 88]  # Medium variation
        })
        
        forecast = self.engine.forecast_with_consistency(medium_data, horizon_days=90)
        
        # Should use weighted average for medium consistency
        mock_weighted_avg.assert_called_once()
        
        self.assertIsNotNone(forecast)
        self.assertEqual(forecast['method_used'], 'weighted_average')
        self.assertGreater(forecast['consistency_score'], 0.3)
        self.assertLess(forecast['consistency_score'], 0.7)
        
    def test_forecast_with_consistency_low(self):
        """Test forecasting with low consistency (reacts to orders only)"""
        
        # Create highly variable data (score < 0.3)
        low_consistency_data = pd.DataFrame({
            'Style#': ['STYLE_LOW'] * 10,
            'Invoice Date': pd.date_range(start=datetime.now() - timedelta(days=70), periods=10, freq='W'),
            'Yds_ordered': [10, 200, 5, 150, 20, 180, 8, 160, 12, 190]  # Very high variation
        })
        
        forecast = self.engine.forecast_with_consistency(low_consistency_data, horizon_days=90)
        
        self.assertIsNotNone(forecast)
        self.assertEqual(forecast['method_used'], 'reactive')
        self.assertLess(forecast['consistency_score'], 0.3)
        
        # Reactive method should return very conservative forecast
        if 'forecast' in forecast and len(forecast['forecast']) > 0:
            # Check that forecast values are conservative (using recent average)
            recent_avg = low_consistency_data['Yds_ordered'].tail(4).mean()
            forecast_avg = forecast['forecast'].mean()
            self.assertLessEqual(forecast_avg, recent_avg * 1.5)  # Should not exceed 150% of recent average
            
    def test_weighted_average_calculation(self):
        """Test the weighted average calculation helper"""
        
        # Test with sample data
        forecast = self.engine._calculate_weighted_average(self.consistent_data, horizon_days=30)
        
        self.assertIsNotNone(forecast)
        self.assertEqual(len(forecast), 30)
        
        # Weighted average should be close to recent values
        recent_avg = self.consistent_data['Yds_ordered'].tail(4).mean()
        forecast_avg = forecast.mean()
        self.assertAlmostEqual(forecast_avg, recent_avg, delta=20)
        
    def test_portfolio_consistency_analysis(self):
        """Test portfolio-wide consistency analysis"""
        
        # Combine all test data
        portfolio_data = pd.concat([
            self.consistent_data,
            self.variable_data,
            self.highly_variable_data,
            self.seasonal_data,
            self.trending_data
        ])
        
        # Mock the sales data
        with patch.object(self.engine, 'sales_data', portfolio_data):
            analysis = self.engine.analyze_portfolio_consistency()
            
            self.assertIsNotNone(analysis)
            self.assertIn('style_consistency', analysis)
            self.assertIn('summary', analysis)
            
            # Check that all styles are analyzed
            self.assertEqual(len(analysis['style_consistency']), 5)
            
            # Check summary statistics
            summary = analysis['summary']
            self.assertIn('total_styles', summary)
            self.assertIn('high_consistency_count', summary)
            self.assertIn('medium_consistency_count', summary)
            self.assertIn('low_consistency_count', summary)
            self.assertIn('avg_consistency', summary)
            
            # Verify categorization
            for style, info in analysis['style_consistency'].items():
                self.assertIn('consistency_score', info)
                self.assertIn('category', info)
                self.assertIn('recommended_method', info)
                
                # Check that categorization is correct
                if info['consistency_score'] > 0.7:
                    self.assertEqual(info['category'], 'high')
                    self.assertEqual(info['recommended_method'], 'ml_forecast')
                elif info['consistency_score'] > 0.3:
                    self.assertEqual(info['category'], 'medium')
                    self.assertEqual(info['recommended_method'], 'weighted_average')
                else:
                    self.assertEqual(info['category'], 'low')
                    self.assertEqual(info['recommended_method'], 'reactive')
                    
    def test_seasonal_pattern_detection(self):
        """Test that seasonal patterns affect consistency scoring appropriately"""
        
        # Seasonal data should have medium consistency due to predictable pattern
        score = self.engine.calculate_consistency_score(
            self.seasonal_data['Yds_ordered'].values
        )
        
        # Seasonal patterns should result in medium consistency
        self.assertGreater(score, 0.2, "Seasonal data should not have very low consistency")
        self.assertLess(score, 0.8, "Seasonal data should not have very high consistency")
        
    def test_trending_pattern_detection(self):
        """Test that trending patterns are handled correctly"""
        
        # Trending data should have lower consistency due to changing mean
        score = self.engine.calculate_consistency_score(
            self.trending_data['Yds_ordered'].values
        )
        
        # Trending patterns should result in lower consistency
        self.assertLess(score, 0.7, "Trending data should not have high consistency")
        
    def test_confidence_thresholds(self):
        """Test that confidence thresholds are working correctly"""
        
        # Test boundary conditions around thresholds (0.3 and 0.7)
        
        # Create data with consistency score exactly at 0.7
        boundary_high_data = pd.DataFrame({
            'Style#': ['STYLE_B1'] * 10,
            'Invoice Date': pd.date_range(start=datetime.now() - timedelta(days=70), periods=10, freq='W'),
            'Yds_ordered': [100, 93, 107, 95, 105, 98, 102, 96, 104, 99]  # CV ≈ 0.3, score ≈ 0.7
        })
        
        with patch('beverly_comprehensive_erp.SalesForecastingEngine.ml_forecast') as mock_ml:
            mock_ml.return_value = pd.Series([100] * 90)
            forecast = self.engine.forecast_with_consistency(boundary_high_data, horizon_days=90)
            
            # At exactly 0.7, should still use ML
            if forecast['consistency_score'] >= 0.7:
                self.assertEqual(forecast['method_used'], 'ml_forecast')
            else:
                self.assertIn(forecast['method_used'], ['weighted_average', 'reactive'])
                
    def test_integration_with_main_forecast_method(self):
        """Test integration with the main forecast method"""
        
        # Mock the necessary components
        with patch.object(self.engine, 'sales_data', self.consistent_data):
            with patch.object(self.engine, 'ml_engines', {'sklearn': Mock()}):
                
                # Test that forecast method can use consistency-based approach
                result = self.engine.forecast_with_consistency(
                    self.consistent_data,
                    horizon_days=30
                )
                
                self.assertIsNotNone(result)
                self.assertIn('consistency_score', result)
                self.assertIn('method_used', result)
                self.assertIn('confidence_level', result)
                
    def test_empty_and_invalid_data_handling(self):
        """Test handling of empty and invalid data"""
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame(columns=['Style#', 'Invoice Date', 'Yds_ordered'])
        result = self.engine.forecast_with_consistency(empty_df, horizon_days=30)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['consistency_score'], 0.0)
        self.assertEqual(result['method_used'], 'reactive')
        
        # Test with insufficient data points
        insufficient_data = pd.DataFrame({
            'Style#': ['STYLE_X'],
            'Invoice Date': [datetime.now()],
            'Yds_ordered': [100]
        })
        
        result = self.engine.forecast_with_consistency(insufficient_data, horizon_days=30)
        self.assertIsNotNone(result)
        self.assertIn('method_used', result)


class TestConsistencyIntegration(unittest.TestCase):
    """Integration tests for consistency-based forecasting with the ERP system"""
    
    @patch('beverly_comprehensive_erp.pd.read_csv')
    @patch('beverly_comprehensive_erp.pd.read_excel')
    def test_real_data_consistency_analysis(self, mock_read_excel, mock_read_csv):
        """Test with realistic data patterns"""
        
        # Create realistic sales data
        sales_data = pd.DataFrame({
            'Document': ['SO001', 'SO002', 'SO003'] * 30,
            'Invoice Date': pd.date_range(start='2024-01-01', periods=90, freq='D'),
            'Customer': ['CUST1'] * 90,
            'Style#': ['STYLE001'] * 30 + ['STYLE002'] * 30 + ['STYLE003'] * 30,
            'Yds_ordered': np.concatenate([
                np.random.normal(100, 10, 30),  # Consistent style
                np.random.normal(100, 30, 30),  # Variable style
                np.random.normal(100, 60, 30),  # Highly variable style
            ]),
            'Unit Price': [10.0] * 90
        })
        
        mock_read_csv.return_value = sales_data
        mock_read_excel.return_value = pd.DataFrame()  # Empty for other files
        
        engine = SalesForecastingEngine()
        engine.sales_data = sales_data
        
        # Analyze portfolio
        analysis = engine.analyze_portfolio_consistency()
        
        self.assertIsNotNone(analysis)
        self.assertEqual(len(analysis['style_consistency']), 3)
        
        # Verify that different styles have different consistency scores
        scores = [info['consistency_score'] for info in analysis['style_consistency'].values()]
        self.assertGreater(max(scores) - min(scores), 0.2, "Should have variation in consistency scores")


if __name__ == '__main__':
    unittest.main()