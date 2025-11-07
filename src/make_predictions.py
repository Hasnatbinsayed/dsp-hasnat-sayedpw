import pandas as pd
import joblib

def make_predictions(input_path, output_path):
    model = joblib.load("models/model.pkl")
    df = pd.read_csv(input_path)
    predictions = model.predict(df)
    pd.DataFrame(predictions, columns=["prediction"]).to_csv(output_path, index=False)
