"""
Model inference module for house prices prediction.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Any

from house_prices.preprocess import (
    handle_missing_values, engineer_features, get_feature_types, transform_features
)


def load_artifacts(artifacts_path: str = 'models') -> Tuple[Any, Any, Any]:
    """
    Load model and preprocessors from disk.

    Args:
        artifacts_path (str): Path to load artifacts from

    Returns:
        Tuple: Loaded model, scaler, and encoder
    """
    model = joblib.load(os.path.join(artifacts_path, 'model.joblib'))
    scaler = joblib.load(os.path.join(artifacts_path, 'scaler.joblib'))
    encoder = joblib.load(os.path.join(artifacts_path, 'encoder.joblib'))

    return model, scaler, encoder


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the trained model.

    Args:
        input_data (pd.DataFrame): Input data for prediction

    Returns:
        np.ndarray: Array of predictions
    """
    # Load artifacts
    model, scaler, encoder = load_artifacts()

    # Preprocess data
    data_clean = handle_missing_values(input_data)
    data_engineered = engineer_features(data_clean)

    # Get feature types
    numerical_features, categorical_features = get_feature_types(data_engineered)

    # Transform features
    data_processed = transform_features(
        data_engineered, numerical_features, categorical_features, scaler, encoder
    )

    # Make predictions
    predictions = model.predict(data_processed)

    return predictions