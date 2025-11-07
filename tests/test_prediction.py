from src.make_predictions import make_predictions
import os

def test_prediction_output():
    make_predictions("data/test.csv", "data/predictions.csv")
    assert os.path.exists("data/predictions.csv")
