#!/usr/bin/env python3
"""
Train Forecast Models Script
Triggers ML model training for sales forecasting
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forecasting.weekly_forecast_generator import WeeklyForecastGenerator
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Train forecast models and display results
    """
    try:
        logger.info("=" * 80)
        logger.info("FORECAST MODEL TRAINING")
        logger.info("=" * 80)

        # Initialize forecast generator
        logger.info("\n[1/4] Initializing Weekly Forecast Generator...")
        generator = WeeklyForecastGenerator(forecast_weeks=13)

        # Check current training status
        logger.info("\n[2/4] Checking current training status...")
        status = generator.get_training_status()

        print("\nCurrent Status:")
        print(f"  Needs Training: {status['needs_training']}")
        print(f"  Last Training: {status.get('last_training_date', 'Never')}")
        print(f"  Days Since: {status.get('days_since_training', 'N/A')}")

        # Trigger training
        logger.info("\n[3/4] Starting model training...")
        logger.info("This may take a few minutes depending on data volume...")

        results = generator.train_models(force_retrain=True)

        # Display results
        logger.info("\n[4/4] Training Complete!")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("TRAINING RESULTS")
        print("=" * 80)

        if results['status'] == 'success':
            print(f"✅ Status: {results['status'].upper()}")
            print(f"\nTraining Metrics:")
            print(f"  Styles Trained: {results['styles_trained']}")
            print(f"  Average Accuracy: {results['average_accuracy']:.1%}")
            print(f"  Training Time: {results['timestamp']}")

            print(f"\nModel Accuracies:")
            for model, accuracy in results.get('model_accuracies', {}).items():
                print(f"  {model}: {accuracy:.1%}")

            print(f"\nWeight Adjustments:")
            weight_adj = results.get('weight_adjustments', {})
            print(f"  Status: {weight_adj.get('status', 'N/A')}")
            if weight_adj.get('status') == 'adjusted':
                print(f"  Improvement: {weight_adj.get('improvement_pct', 0):.1f}%")
                print(f"\n  Old Weights:")
                for source, weight in weight_adj.get('old_weights', {}).items():
                    print(f"    {source}: {weight:.2f}")
                print(f"\n  New Weights:")
                for source, weight in weight_adj.get('new_weights', {}).items():
                    print(f"    {source}: {weight:.2f}")

        elif results['status'] == 'skipped':
            print(f"ℹ️  Status: {results['status'].upper()}")
            print(f"  Reason: {results['reason']}")
            print(f"  Last Training: {results.get('last_training', 'Unknown')}")

        elif results['status'] == 'failed':
            print(f"❌ Status: {results['status'].upper()}")
            print(f"  Reason: {results['reason']}")
            print(f"  Styles Found: {results.get('styles_found', 0)}")

        else:
            print(f"❌ Status: ERROR")
            print(f"  Error: {results.get('error', 'Unknown error')}")

        print("=" * 80)

        # Save detailed results to file
        results_file = Path('training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nDetailed results saved to: {results_file}")

        return 0 if results['status'] == 'success' else 1

    except Exception as e:
        logger.exception(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
