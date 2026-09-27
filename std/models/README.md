# Models Directory

This directory stores serialised machine learning models, scalers, and pipelines created by `src/train_model.py`.

## Files Saved Here

- `fraud_model.joblib`: Serialized trained machine learning model (e.g. Balanced Random Forest or Cost-Sensitive Logistic Regression / Classifier).
- `scaler.joblib`: Preprocessing scaler (e.g. `RobustScaler` or `StandardScaler`) fitted on training data for scaling `Time` and `Amount`.
- `model_metadata.json`: Model version, training parameters, feature list, and evaluation metrics summary.

## Training

To retrain the model and regenerate these artifacts, run:
```bash
python src/train_model.py
```
