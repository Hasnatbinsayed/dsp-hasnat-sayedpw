"""
Model training module for house prices prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import Dict

from house_prices.preprocess import (
    handle_missing_values, engineer_features, get_feature_types,
    fit_preprocessors, transform_features, split_data
)


def train_model(features: np.ndarray, target: pd.Series) -> RandomForestRegressor:
    """
    Train a Random Forest model.

    Args:
        features (np.ndarray): Training features
        target (pd.Series): Training target

    Returns:
        RandomForestRegressor: Trained model
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(features, target)
    return model


def evaluate_model(
    model: RandomForestRegressor,
    features: np.ndarray,
    target: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        model (RandomForestRegressor): Trained model
        features (np.ndarray): Test features
        target (pd.Series): Test target

    Returns:
        Dict[str, float]: Dictionary with evaluation metrics
    """
    predictions = model.predict(features)

    mse = mean_squared_error(target, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target, predictions)
    r2 = r2_score(target, predictions)

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def save_artifacts(
    model: RandomForestRegressor,
    scaler: StandardScaler,
    encoder: OneHotEncoder,
    artifacts_path: str = 'models'
) -> None:
    """
    Save model and preprocessors to disk.

    Args:
        model (RandomForestRegressor): Trained model
        scaler (StandardScaler): Fitted scaler
        encoder (OneHotEncoder): Fitted encoder
        artifacts_path (str): Path to save artifacts
    """
    os.makedirs(artifacts_path, exist_ok=True)

    joblib.dump(model, os.path.join(artifacts_path, 'model.joblib'))
    joblib.dump(scaler, os.path.join(artifacts_path, 'scaler.joblib'))
    joblib.dump(encoder, os.path.join(artifacts_path, 'encoder.joblib'))


def build_model(data: pd.DataFrame, target_column: str = 'SalePrice') -> Dict[str, float]:
    """
    Build and train the complete model pipeline.

    Args:
        data (pd.DataFrame): Training data
        target_column (str): Name of the target column

    Returns:
        Dict[str, float]: Dictionary with model performance metrics
    """
    # Split data early to avoid data leakage
    features_train, features_test, target_train, target_test = split_data(
        data, target_column
    )

    # Preprocess training data
    features_train_clean = handle_missing_values(features_train)
    features_train_engineered = engineer_features(features_train_clean)

    # Get feature types and fit preprocessors
    numerical_features, categorical_features = get_feature_types(
        features_train_engineered
    )
    scaler, encoder = fit_preprocessors(
        features_train_engineered, numerical_features, categorical_features
    )

    # Transform training data
    features_train_processed = transform_features(
        features_train_engineered, numerical_features,
        categorical_features, scaler, encoder
    )

    # Train model
    model = train_model(features_train_processed, target_train)

    # Preprocess and transform test data
    features_test_clean = handle_missing_values(features_test)
    features_test_engineered = engineer_features(features_test_clean)
    features_test_processed = transform_features(
        features_test_engineered, numerical_features,
        categorical_features, scaler, encoder
    )

    # Evaluate model
    performance = evaluate_model(model, features_test_processed, target_test)

    # Save artifacts
    save_artifacts(model, scaler, encoder)

    return performance