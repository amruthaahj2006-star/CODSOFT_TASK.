"""
predict.py
Inference engine for Credit Card Fraud Detection.
Supports single-transaction and batch CSV predictions with probability scores and risk levels.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.joblib")
DEFAULT_SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.joblib")
DEFAULT_META_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_metadata.json")


def load_model(model_path: str = None, scaler_path: str = None):
    """
    Loads and returns (model, scaler).
    Automatically resolves model path (checks best_model.joblib or fraud_model.joblib).
    """
    import shutil
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    best_p = os.path.join(models_dir, "best_model.joblib")
    fraud_p = os.path.join(models_dir, "fraud_model.joblib")

    # If best_model doesn't exist but fraud_model exists, copy it so both exist
    if not os.path.exists(best_p) and os.path.exists(fraud_p):
        try:
            shutil.copyfile(fraud_p, best_p)
        except Exception:
            pass

    if model_path is None:
        if os.path.exists(best_p):
            model_path = best_p
        elif os.path.exists(fraud_p):
            model_path = fraud_p
        else:
            model_path = best_p

    if scaler_path is None:
        scaler_path = os.path.join(models_dir, "scaler.joblib")

    if not os.path.exists(model_path):
        if os.path.exists(fraud_p):
            model_path = fraud_p
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_transaction(model, scaler, feature_inputs: dict, threshold: float = 0.50):
    """
    Predicts transaction legitimacy.
    Returns (label, proba).
    """
    df = pd.DataFrame([feature_inputs])
    
    # Ensure Time is present with reasonable default
    if "Time" not in df.columns:
        df["Time"] = 0.0

    # Scale columns using exact features the scaler was trained on
    if hasattr(scaler, "feature_names_in_"):
        scaler_cols = list(scaler.feature_names_in_)
        for col in scaler_cols:
            if col not in df.columns:
                df[col] = 0.0
        df[scaler_cols] = scaler.transform(df[scaler_cols])
    else:
        scale_cols = [c for c in ["Amount", "Time"] if c in df.columns]
        if scale_cols:
            df[scale_cols] = scaler.transform(df[scale_cols])

    # Reorder columns to match trained model exact features and order
    if hasattr(model, "feature_names_in_"):
        model_cols = list(model.feature_names_in_)
        for f in model_cols:
            if f not in df.columns:
                df[f] = 0.0
        X = df[model_cols]
    else:
        X = df

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])
    else:
        proba = float(model.predict(X)[0])

    label = "Fraudulent" if proba >= threshold else "Legitimate"
    return label, proba


class FraudPredictor:
    """Production-grade inference class for fraud classification."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
        meta_path: str = DEFAULT_META_PATH,
        threshold: float = 0.50
    ):
        self.model_path = os.path.abspath(model_path)
        self.scaler_path = os.path.abspath(scaler_path)
        self.meta_path = os.path.abspath(meta_path)
        self.threshold = threshold

        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[ERROR] Trained model file not found at: {self.model_path}\n"
                f"Please run 'python src/train_model.py' first to generate model artifacts."
            )
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"[ERROR] Scaler file not found at: {self.scaler_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
                self.feature_names = self.metadata.get("features", [])
        else:
            self.metadata = {}
            # Fallback default feature names
            self.feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_scaled = df.copy()
        cols_to_scale = [c for c in ["Amount", "Time"] if c in df_scaled.columns]
        if cols_to_scale:
            df_scaled[cols_to_scale] = self.scaler.transform(df_scaled[cols_to_scale])
        return df_scaled

    @staticmethod
    def classify_risk(probability: float) -> str:
        if probability < 0.20:
            return "LOW"
        elif probability < 0.50:
            return "MODERATE"
        elif probability < 0.80:
            return "HIGH"
        else:
            return "CRITICAL"

    def predict_single(self, transaction_dict: dict) -> dict:
        """
        Predicts fraud for a single transaction.

        Args:
            transaction_dict: Dictionary containing feature key-values.
        Returns:
            Dictionary with prediction (0 or 1), label, fraud probability, and risk level.
        """
        # Ensure all required features are present
        missing = [f for f in self.feature_names if f not in transaction_dict]
        if missing:
            raise ValueError(f"Transaction data missing required features: {missing}")

        df_input = pd.DataFrame([transaction_dict])[self.feature_names]
        df_scaled = self._scale_features(df_input)

        if hasattr(self.model, "predict_proba"):
            fraud_prob = float(self.model.predict_proba(df_scaled)[0, 1])
        else:
            fraud_prob = float(self.model.predict(df_scaled)[0])

        is_fraud = int(fraud_prob >= self.threshold)
        label = "Fraudulent" if is_fraud == 1 else "Legitimate"
        risk_level = self.classify_risk(fraud_prob)

        return {
            "prediction": is_fraud,
            "label": label,
            "fraud_probability": round(fraud_prob, 4),
            "fraud_probability_pct": round(fraud_prob * 100, 2),
            "risk_level": risk_level,
            "threshold_used": self.threshold,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts fraud for a batch DataFrame of transactions.

        Args:
            df: DataFrame containing the transaction features.
        Returns:
            Original DataFrame augmented with 'Fraud_Probability', 'Prediction', 'Status', and 'Risk_Level'.
        """
        # Drop Class column if it exists in input
        X = df.copy()
        if "Class" in X.columns:
            X = X.drop(columns=["Class"])

        # Check features
        for f in self.feature_names:
            if f not in X.columns:
                raise ValueError(f"Input batch is missing required feature '{f}'")

        X_ordered = X[self.feature_names]
        X_scaled = self._scale_features(X_ordered)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[:, 1]
        else:
            probs = self.model.predict(X_scaled).astype(float)

        preds = (probs >= self.threshold).astype(int)

        result_df = df.copy()
        result_df["Fraud_Probability"] = np.round(probs, 4)
        result_df["Fraud_Probability_Pct"] = np.round(probs * 100, 2)
        result_df["Prediction"] = preds
        result_df["Status"] = np.where(preds == 1, "Fraudulent", "Legitimate")
        result_df["Risk_Level"] = [self.classify_risk(p) for p in probs]

        return result_df


def main():
    print("=" * 60)
    print("      CREDIT CARD FRAUD INFERENCE TESTER")
    print("=" * 60)

    try:
        predictor = FraudPredictor()
        print(f"[+] Loaded Model Type: {predictor.metadata.get('model_type', 'ML Model')}")
        print(f"[+] Features expected: {len(predictor.feature_names)}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize predictor: {e}")
        return

    # 1. Test Single Legitimate Sample
    legit_sample = {f"V{i}": 0.05 for i in range(1, 29)}
    legit_sample["Time"] = 45000.0
    legit_sample["Amount"] = 49.99

    res_legit = predictor.predict_single(legit_sample)
    print("\n--- Test Sample 1: Normal Everyday Transaction ---")
    print(f"Amount       : ${legit_sample['Amount']}")
    print(f"Prediction   : {res_legit['prediction']} ({res_legit['label']})")
    print(f"Probability  : {res_legit['fraud_probability_pct']}%")
    print(f"Risk Level   : {res_legit['risk_level']}")

    # 2. Test Single Suspicious Sample (Known PCA shift patterns)
    fraud_sample = {f"V{i}": 0.0 for i in range(1, 29)}
    fraud_sample["Time"] = 90000.0
    fraud_sample["Amount"] = 999.00
    fraud_sample["V4"] = 3.8
    fraud_sample["V10"] = -4.5
    fraud_sample["V11"] = 3.2
    fraud_sample["V12"] = -5.0
    fraud_sample["V14"] = -6.8
    fraud_sample["V17"] = -4.2

    res_fraud = predictor.predict_single(fraud_sample)
    print("\n--- Test Sample 2: Suspicious Anomalous Transaction ---")
    print(f"Amount       : ${fraud_sample['Amount']}")
    print(f"Prediction   : {res_fraud['prediction']} ({res_fraud['label']})")
    print(f"Probability  : {res_fraud['fraud_probability_pct']}%")
    print(f"Risk Level   : {res_fraud['risk_level']}")

    # 3. Test Batch File if sample exists
    sample_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_transactions.csv")
    if os.path.exists(sample_file):
        print(f"\n--- Testing Batch File: {sample_file} ---")
        df_sample = pd.read_csv(sample_file)
        results = predictor.predict_batch(df_sample)
        flagged = (results["Prediction"] == 1).sum()
        print(f"[+] Total Transactions Tested : {len(results)}")
        print(f"[+] Flagged as Fraudulent     : {flagged}")
        print(f"[+] Cleared as Legitimate     : {len(results) - flagged}")

    print("\n[+] Inference test finished successfully!")


if __name__ == "__main__":
    main()
