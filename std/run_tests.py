"""
run_tests.py
End-to-end smoke tests for Credit Card Fraud Detection pipeline and inference engine.
"""

import os
import sys
import unittest
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import inspect_dataset, load_dataset
from src.predict import FraudPredictor


class TestCreditCardFraudPipeline(unittest.TestCase):

    def test_01_dataset_loading_and_inspection(self):
        """Test dataset loading and dynamic schema inspection."""
        data_path = os.path.join("data", "creditcard.csv")
        self.assertTrue(os.path.exists(data_path), "Dataset file creditcard.csv should exist.")
        
        df = load_dataset(data_path, auto_create_sample_if_missing=False)
        report = inspect_dataset(df)
        
        self.assertGreater(report["num_rows"], 0)
        self.assertEqual(report["target_column"], "Class")
        self.assertIn("Amount", report["columns"])
        self.assertIn("Time", report["columns"])
        self.assertIn("V1", report["columns"])
        self.assertIn("V28", report["columns"])

    def test_02_model_artifacts_exist(self):
        """Test that saved model, scaler, and metadata files exist."""
        self.assertTrue(os.path.exists(os.path.join("models", "fraud_model.joblib")))
        self.assertTrue(os.path.exists(os.path.join("models", "scaler.joblib")))
        self.assertTrue(os.path.exists(os.path.join("models", "model_metadata.json")))

    def test_03_inference_single_prediction(self):
        """Test single transaction prediction output format and risk scoring."""
        predictor = FraudPredictor()
        
        sample = {f"V{i}": 0.0 for i in range(1, 29)}
        sample["Time"] = 3600.0
        sample["Amount"] = 25.50
        
        res = predictor.predict_single(sample)
        self.assertIn("prediction", res)
        self.assertIn(res["prediction"], [0, 1])
        self.assertIn("fraud_probability", res)
        self.assertTrue(0.0 <= res["fraud_probability"] <= 1.0)
        self.assertIn(res["risk_level"], ["LOW", "MODERATE", "HIGH", "CRITICAL"])

    def test_04_inference_batch_prediction(self):
        """Test batch transaction prediction on sample CSV."""
        predictor = FraudPredictor()
        batch_path = os.path.join("data", "sample_transactions.csv")
        self.assertTrue(os.path.exists(batch_path), "sample_transactions.csv should exist.")
        
        df_batch = pd.read_csv(batch_path)
        res_df = predictor.predict_batch(df_batch)
        
        self.assertEqual(len(res_df), len(df_batch))
        self.assertIn("Fraud_Probability", res_df.columns)
        self.assertIn("Prediction", res_df.columns)
        self.assertIn("Risk_Level", res_df.columns)
        self.assertIn("Status", res_df.columns)


if __name__ == "__main__":
    unittest.main()
