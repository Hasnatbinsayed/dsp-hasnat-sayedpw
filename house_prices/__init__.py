"""
House Prices Prediction Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from house_prices.preprocess import load_data, split_data
from house_prices.train import build_model
from house_prices.inference import make_predictions

__all__ = [
    'load_data',
    'split_data', 
    'build_model',
    'make_predictions'
]