#!/usr/bin/env python3
"""
External Forecast Loader
Loads and validates external sales forecasts from multiple sources:
- Sales team Excel/CSV files
- Customer commitment APIs
- External forecasting systems
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ExternalForecastLoader:
    """
    Load and validate external sales forecasts from multiple sources
    Standardizes all formats to unified structure for blending
    """

    def __init__(self):
        """Initialize external forecast loader"""
        self.standard_columns = ['style', 'week_number', 'forecasted_yards', 'confidence', 'notes', 'source']
        self.validation_results = []

    def load_sales_team_forecast(
        self,
        file_path: str,
        source_name: str = 'sales_team'
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Load Excel/CSV forecast from sales team

        Args:
            file_path: Path to Excel or CSV file
            source_name: Name of this forecast source

        Returns:
            Standardized forecast dict:
            {
                style: {
                    week_num: {
                        'yards': float,
                        'confidence': float,
                        'source': str,
                        'notes': str
                    }
                }
            }

        Expected file format:
            style,week_number,forecasted_yards,confidence,notes
            STYLE001,42,1000,0.85,Customer indicated interest
            STYLE001,43,1050,0.82,Based on Q4 pipeline
        """
        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                logger.error(f"File not found: {file_path}")
                return {}

            # Load file based on extension
            if file_path_obj.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path_obj.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                logger.error(f"Unsupported file format: {file_path_obj.suffix}")
                return {}

            logger.info(f"Loaded external forecast file: {file_path} ({len(df)} rows)")

            # Validate and standardize
            validated_df = self._validate_dataframe(df, source_name)

            if validated_df.empty:
                logger.warning("No valid records found after validation")
                return {}

            # Convert to standard format
            forecast_dict = self._dataframe_to_forecast_dict(validated_df, source_name)

            logger.info(
                f"Loaded {len(forecast_dict)} styles from {source_name}, "
                f"{sum(len(weeks) for weeks in forecast_dict.values())} week forecasts"
            )

            return forecast_dict

        except Exception as e:
            logger.exception(f"Error loading sales team forecast from {file_path}: {e}")
            return {}

    def load_customer_commitments(
        self,
        customer_id: str,
        commitments: List[Dict]
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Load forward commitments from customer (higher confidence)

        Args:
            customer_id: Customer identifier
            commitments: List of commitment dicts with style, week, quantity

        Returns:
            Standardized forecast dict with high confidence (0.90-0.95)

        Example commitments:
        [
            {'style': 'STYLE001', 'week': 42, 'yards': 1000, 'notes': 'Confirmed PO pending'},
            {'style': 'STYLE002', 'week': 43, 'yards': 500, 'notes': 'Verbal commitment'}
        ]
        """
        try:
            forecast_dict: Dict[str, Dict[int, Dict[str, Any]]] = {}

            for commitment in commitments:
                style = commitment.get('style')
                week_num = commitment.get('week')
                yards = commitment.get('yards', commitment.get('quantity', 0))
                notes = commitment.get('notes', f'Customer {customer_id} commitment')

                # Customer commitments have higher confidence
                confidence = commitment.get('confidence', 0.92)

                if not style or not week_num or yards <= 0:
                    logger.warning(f"Invalid commitment: {commitment}")
                    continue

                if style not in forecast_dict:
                    forecast_dict[style] = {}

                forecast_dict[style][week_num] = {
                    'yards': float(yards),
                    'confidence': float(confidence),
                    'source': f'customer_{customer_id}',
                    'notes': notes
                }

            logger.info(
                f"Loaded {len(forecast_dict)} styles from customer {customer_id}, "
                f"{sum(len(weeks) for weeks in forecast_dict.values())} commitments"
            )

            return forecast_dict

        except Exception as e:
            logger.exception(f"Error loading customer commitments: {e}")
            return {}

    def load_api_forecast(
        self,
        api_data: Dict,
        source_name: str = 'external_api'
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Load forecast from external API response

        Args:
            api_data: JSON response from external API
            source_name: Name of the API source

        Returns:
            Standardized forecast dict

        Expected API format:
        {
            "forecast_date": "2024-10-18",
            "forecasts": [
                {
                    "style": "STYLE001",
                    "week_number": 42,
                    "forecasted_yards": 1000,
                    "confidence": 0.85
                }
            ]
        }
        """
        try:
            forecasts_list = api_data.get('forecasts', [])

            if not forecasts_list:
                logger.warning("No forecasts in API response")
                return {}

            # Convert to DataFrame for validation
            df = pd.DataFrame(forecasts_list)

            # Rename columns to standard names if needed
            column_mapping = {
                'quantity': 'forecasted_yards',
                'week': 'week_number'
            }
            df = df.rename(columns=column_mapping)

            # Validate
            validated_df = self._validate_dataframe(df, source_name)

            # Convert to dict
            forecast_dict = self._dataframe_to_forecast_dict(validated_df, source_name)

            logger.info(
                f"Loaded {len(forecast_dict)} styles from {source_name} API, "
                f"{sum(len(weeks) for weeks in forecast_dict.values())} forecasts"
            )

            return forecast_dict

        except Exception as e:
            logger.exception(f"Error loading API forecast: {e}")
            return {}

    def load_from_turso(
        self,
        turso_client,
        source_name: Optional[str] = None
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Load external forecasts from Turso database

        Args:
            turso_client: TursoClient instance
            source_name: Optional filter by source name

        Returns:
            Standardized forecast dict
        """
        try:
            # Query external forecasts from Turso
            if source_name:
                query = """
                    SELECT style, week_number, forecasted_yards, confidence, notes, source_name
                    FROM external_forecasts
                    WHERE source_name = ?
                    ORDER BY style, week_number
                """
                result = turso_client.execute(query, (source_name,))
            else:
                query = """
                    SELECT style, week_number, forecasted_yards, confidence, notes, source_name
                    FROM external_forecasts
                    ORDER BY style, week_number
                """
                result = turso_client.execute(query)

            if not result or 'rows' not in result or not result['rows']:
                logger.info(f"No external forecasts found in Turso" +
                          (f" for source '{source_name}'" if source_name else ""))
                return {}

            # Convert to DataFrame
            df = pd.DataFrame(
                result['rows'],
                columns=[col['name'] for col in result['columns']]
            )

            # Rename source_name to source for consistency
            if 'source_name' in df.columns:
                df = df.rename(columns={'source_name': 'source'})

            # Validate
            validated_df = self._validate_dataframe(df, source_name or 'turso')

            if validated_df.empty:
                return {}

            # Convert to dict
            forecast_dict = self._dataframe_to_forecast_dict(
                validated_df,
                source_name or 'turso'
            )

            logger.info(
                f"Loaded {len(forecast_dict)} styles from Turso, "
                f"{sum(len(weeks) for weeks in forecast_dict.values())} forecasts"
            )

            return forecast_dict

        except Exception as e:
            logger.exception(f"Error loading forecasts from Turso: {e}")
            return {}

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        source_name: str
    ) -> pd.DataFrame:
        """
        Validate and clean forecast dataframe

        Args:
            df: Raw dataframe from external source
            source_name: Source identifier

        Returns:
            Validated and cleaned dataframe
        """
        validation_log = {
            'source': source_name,
            'total_rows': len(df),
            'valid_rows': 0,
            'issues': []
        }

        # Check required columns
        required_cols = ['style', 'week_number', 'forecasted_yards']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.error(f"Available columns: {list(df.columns)}")
            validation_log['issues'].append(f"Missing columns: {missing_cols}")
            self.validation_results.append(validation_log)
            return pd.DataFrame()

        # Create clean dataframe
        clean_df = df.copy()

        # Add default confidence if missing
        if 'confidence' not in clean_df.columns:
            clean_df['confidence'] = 0.75  # Default medium confidence

        # Add default notes if missing
        if 'notes' not in clean_df.columns:
            clean_df['notes'] = ''

        # Remove rows with missing critical data
        initial_count = len(clean_df)
        clean_df = clean_df.dropna(subset=['style', 'week_number', 'forecasted_yards'])
        dropped_nulls = initial_count - len(clean_df)
        if dropped_nulls > 0:
            validation_log['issues'].append(f"Dropped {dropped_nulls} rows with null values")

        # Validate week numbers (1-53)
        invalid_weeks = clean_df[
            (clean_df['week_number'] < 1) | (clean_df['week_number'] > 53)
        ]
        if len(invalid_weeks) > 0:
            logger.warning(f"Found {len(invalid_weeks)} rows with invalid week numbers")
            validation_log['issues'].append(f"Invalid week numbers: {len(invalid_weeks)}")
            clean_df = clean_df[
                (clean_df['week_number'] >= 1) & (clean_df['week_number'] <= 53)
            ]

        # Validate forecasted_yards (must be positive)
        invalid_yards = clean_df[clean_df['forecasted_yards'] <= 0]
        if len(invalid_yards) > 0:
            logger.warning(f"Found {len(invalid_yards)} rows with zero/negative yards")
            validation_log['issues'].append(f"Non-positive yards: {len(invalid_yards)}")
            clean_df = clean_df[clean_df['forecasted_yards'] > 0]

        # Validate confidence (0-1)
        clean_df['confidence'] = clean_df['confidence'].clip(0, 1)

        # Check for unrealistic values (statistical outliers)
        outliers = self._detect_outliers(clean_df)
        if outliers:
            validation_log['issues'].append(f"Outliers detected: {len(outliers)}")
            logger.warning(f"Detected {len(outliers)} potential outliers")

        validation_log['valid_rows'] = len(clean_df)
        self.validation_results.append(validation_log)

        logger.info(
            f"Validation complete: {validation_log['valid_rows']}/{validation_log['total_rows']} "
            f"rows valid, {len(validation_log['issues'])} issues"
        )

        return clean_df

    def _detect_outliers(
        self,
        df: pd.DataFrame,
        iqr_multiplier: float = 3.0
    ) -> List[int]:
        """
        Detect statistical outliers in forecasted_yards

        Args:
            df: Dataframe to check
            iqr_multiplier: IQR multiplier for outlier detection

        Returns:
            List of row indices with outliers
        """
        try:
            if len(df) < 4:
                return []  # Need at least 4 rows for IQR

            yards = df['forecasted_yards']
            Q1 = yards.quantile(0.25)
            Q3 = yards.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - (iqr_multiplier * IQR)
            upper_bound = Q3 + (iqr_multiplier * IQR)

            outliers = df[
                (yards < lower_bound) | (yards > upper_bound)
            ].index.tolist()

            return outliers

        except Exception as e:
            logger.warning(f"Error detecting outliers: {e}")
            return []

    def _dataframe_to_forecast_dict(
        self,
        df: pd.DataFrame,
        source_name: str
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Convert validated dataframe to standard forecast dict format

        Args:
            df: Validated dataframe
            source_name: Source identifier

        Returns:
            Nested dict: {style: {week: {yards, confidence, source, notes}}}
        """
        forecast_dict: Dict[str, Dict[int, Dict[str, Any]]] = {}

        for _, row in df.iterrows():
            style = str(row['style']).strip()
            week_num = int(row['week_number'])
            yards = float(row['forecasted_yards'])
            confidence = float(row.get('confidence', 0.75))
            notes = str(row.get('notes', ''))

            if style not in forecast_dict:
                forecast_dict[style] = {}

            forecast_dict[style][week_num] = {
                'yards': yards,
                'confidence': confidence,
                'source': source_name,
                'notes': notes
            }

        return forecast_dict

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get comprehensive validation report for all loaded forecasts

        Returns:
            Report with validation statistics and issues
        """
        if not self.validation_results:
            return {'message': 'No forecasts loaded yet'}

        total_rows = sum(v['total_rows'] for v in self.validation_results)
        valid_rows = sum(v['valid_rows'] for v in self.validation_results)
        all_issues = []

        for result in self.validation_results:
            for issue in result['issues']:
                all_issues.append(f"{result['source']}: {issue}")

        return {
            'sources_loaded': len(self.validation_results),
            'total_rows': total_rows,
            'valid_rows': valid_rows,
            'validation_rate': f"{(valid_rows/total_rows*100):.1f}%" if total_rows > 0 else "0%",
            'issues': all_issues,
            'details': self.validation_results
        }


def main():
    """Test external forecast loader"""

    loader = ExternalForecastLoader()

    # Test 1: Create sample CSV
    print("=== Test 1: Load Sample CSV ===")
    sample_data = pd.DataFrame([
        {'style': 'STYLE001', 'week_number': 42, 'forecasted_yards': 1000, 'confidence': 0.85, 'notes': 'Q4 pipeline'},
        {'style': 'STYLE001', 'week_number': 43, 'forecasted_yards': 1050, 'confidence': 0.82, 'notes': 'Growth trend'},
        {'style': 'STYLE002', 'week_number': 42, 'forecasted_yards': 500, 'confidence': 0.90, 'notes': 'Customer commitment'},
    ])

    # Save to temp CSV
    temp_file = 'test_forecast.csv'
    sample_data.to_csv(temp_file, index=False)

    forecast = loader.load_sales_team_forecast(temp_file, 'test_sales_team')
    print(f"Loaded forecast for {len(forecast)} styles")
    for style, weeks in forecast.items():
        print(f"\n{style}:")
        for week, data in sorted(weeks.items()):
            print(f"  Week {week}: {data['yards']:.0f} yards (conf: {data['confidence']:.2f})")

    # Test 2: Customer commitments
    print("\n=== Test 2: Customer Commitments ===")
    commitments = [
        {'style': 'STYLE003', 'week': 44, 'yards': 800, 'notes': 'Confirmed PO pending'},
        {'style': 'STYLE003', 'week': 45, 'yards': 850, 'notes': 'Follow-on order'}
    ]

    customer_forecast = loader.load_customer_commitments('CUST001', commitments)
    print(f"Loaded customer commitments for {len(customer_forecast)} styles")

    # Test 3: Validation report
    print("\n=== Test 3: Validation Report ===")
    report = loader.get_validation_report()
    print(f"Sources loaded: {report['sources_loaded']}")
    print(f"Total rows: {report['total_rows']}")
    print(f"Valid rows: {report['valid_rows']} ({report['validation_rate']})")
    if report['issues']:
        print(f"Issues found: {len(report['issues'])}")

    # Cleanup
    Path(temp_file).unlink()


if __name__ == "__main__":
    main()
