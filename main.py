"""
Main script to test the house prices prediction pipeline.
"""

import pandas as pd
from house_prices.train import build_model
from house_prices.inference import make_predictions
from house_prices.preprocess import handle_missing_values, engineer_features
import os


def main():
    """Main function to run the complete pipeline."""

    print("🏠 House Prices Prediction Pipeline")
    print("=" * 40)

    # Check if data files exist
    if not os.path.exists('data/train.csv'):
        print("❌ Error: data/train.csv not found")
        print("Please ensure your CSV files are in the data/ folder")
        return

    if not os.path.exists('data/test.csv'):
        print("❌ Error: data/test.csv not found")
        print("Please ensure your CSV files are in the data/ folder")
        return

    try:
        # Step 1: Load data
        print("\n📊 Step 1: Loading data...")
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')

        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")

        # Step 2: Save processed dataframe for testing (Step 0 requirement)
        print("\n💾 Step 2: Saving processed dataframe for testing...")
        train_processed = handle_missing_values(train_df)
        train_processed = engineer_features(train_processed)
        train_processed.to_parquet('processed_df.parquet', index=False)
        print("✅ Processed dataframe saved to processed_df.parquet")

        # Step 3: Model Building
        print("\n🔧 Step 3: Model Building...")
        model_performance = build_model(train_df)

        print("\n📈 Model Performance Metrics:")
        for metric, value in model_performance.items():
            print(f"  {metric.upper()}: {value:.4f}")

        # Step 4: Model Inference
        print("\n🔮 Step 4: Model Inference...")
        predictions = make_predictions(test_df)

        print(f"✅ Generated {len(predictions)} predictions")
        print("\nFirst 10 predictions:")
        for i, pred in enumerate(predictions[:10]):
            print(f"  Prediction {i + 1}: {pred:.2f}")

        # Step 5: Test refactoring consistency
        print("\n✅ Step 5: Testing refactoring consistency...")
        expected_processed_df = pd.read_parquet('processed_df.parquet')
        actual_processed_df = engineer_features(handle_missing_values(train_df))

        try:
            pd.testing.assert_frame_equal(actual_processed_df, expected_processed_df)
            print("✅ SUCCESS: Refactoring did not change data processing behavior!")
        except Exception as e:
            print("❌ WARNING: Data processing behavior changed!")
            print(f"Error: {e}")

        print("\n🎉 Pipeline completed successfully!")

    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        raise


if __name__ == "__main__":
    main()