# Model Building Section
import pandas as pd
from house_prices.train import build_model

# Load and train model
training_df = pd.read_csv('../data/train.csv')
model_performance_dict = build_model(training_df)
print("Model Performance:")
print(model_performance_dict)