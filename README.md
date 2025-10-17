# dsp-firstname-sayed
Author: hasnat bin sayed

## Project layout
- `data/` : place your `train.csv` and `test.csv` here (sample files included)
- `house_prices/` : preprocessing, training and inference modules
- `models/` : saved model and preprocessing objects (created after training)
- `main.py` : run this file to train, evaluate and run inference

## Quick start (local / PyCharm)
1. Create a virtualenv and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run:
   ```bash
   python main.py
   ```
3. Output predictions saved to `models/test_predictions.csv`.

## Notes
- Encoder: `OneHotEncoder` persisted in `models/encoder.joblib`
- Scaler: `StandardScaler` persisted in `models/scaler.joblib`
- Model: `Ridge` persisted in `models/model.joblib`
