"""
Data preprocessing module for house prices prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    return pd.read_csv(file_path)


def split_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Args:
        data (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    features = data.drop(columns=[target_column])
    target = data[target_column]

    return train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        data (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    data_clean = data.copy()

    numerical_columns = data_clean.select_dtypes(include=[np.number]).columns
    categorical_columns = data_clean.select_dtypes(include=['object']).columns

    for column in numerical_columns:
        if data_clean[column].isna().any():
            data_clean[column].fillna(data_clean[column].median(), inplace=True)

    for column in categorical_columns:
        if data_clean[column].isna().any():
            most_frequent = data_clean[column].mode()
            fill_value = most_frequent[0] if len(most_frequent) > 0 else 'Unknown'
            data_clean[column].fillna(fill_value, inplace=True)

    return data_clean


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.

    Args:
        data (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with new features
    """
    data_engineered = data.copy()

    if 'TotalBsmtSF' in data_engineered.columns:
        data_engineered['TotalArea'] = data_engineered['TotalBsmtSF']
        if '1stFlrSF' in data_engineered.columns:
            data_engineered['TotalArea'] += data_engineered['1stFlrSF']
        if '2ndFlrSF' in data_engineered.columns:
            data_engineered['TotalArea'] += data_engineered['2ndFlrSF']

    if 'YearBuilt' in data_engineered.columns:
        data_engineered['HouseAge'] = 2023 - data_engineered['YearBuilt']

    if 'FullBath' in data_engineered.columns:
        data_engineered['TotalBathrooms'] = data_engineered['FullBath']
        if 'HalfBath' in data_engineered.columns:
            data_engineered['TotalBathrooms'] += 0.5 * data_engineered['HalfBath']

    return data_engineered


def get_feature_types(data: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical features.

    Args:
        data (pd.DataFrame): Input dataframe

    Returns:
        Tuple: Lists of numerical and categorical column names
    """
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    return numerical_features, categorical_features


def fit_preprocessors(
    training_data: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str]
) -> Tuple[StandardScaler, OneHotEncoder]:
    """
    Fit scaler and encoder on training data.

    Args:
        training_data (pd.DataFrame): Training features
        numerical_features (List[str]): Numerical column names
        categorical_features (List[str]): Categorical column names

    Returns:
        Tuple: Fitted scaler and encoder objects
    """
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    if numerical_features:
        scaler.fit(training_data[numerical_features])

    if categorical_features:
        encoder.fit(training_data[categorical_features])

    return scaler, encoder


def transform_features(
    data: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    scaler: StandardScaler,
    encoder: OneHotEncoder
) -> np.ndarray:
    """
    Transform features using fitted preprocessors.

    Args:
        data (pd.DataFrame): Input data
        numerical_features (List[str]): Numerical column names
        categorical_features (List[str]): Categorical column names
        scaler (StandardScaler): Fitted scaler
        encoder (OneHotEncoder): Fitted encoder

    Returns:
        np.ndarray: Transformed features array
    """
    processed_features = []

    if numerical_features:
        numerical_data = scaler.transform(data[numerical_features])
        processed_features.append(numerical_data)

    if categorical_features:
        categorical_data = encoder.transform(data[categorical_features])
        processed_features.append(categorical_data)

    if processed_features:
        return np.hstack(processed_features)

    return np.array([])