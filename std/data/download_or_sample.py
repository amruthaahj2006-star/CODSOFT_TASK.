"""
download_or_sample.py
Utility script to verify dataset presence or generate realistic benchmark data.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import inspect_dataset, print_inspection_report, generate_realistic_synthetic_dataset
import pandas as pd

DATASET_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")
SAMPLE_TEST_PATH = os.path.join(os.path.dirname(__file__), "sample_transactions.csv")


def main():
    print("=" * 60)
    print("      DATASET SETUP & VERIFICATION UTILITY")
    print("=" * 60)

    if os.path.exists(DATASET_PATH):
        print(f"[+] Found dataset at: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        report = inspect_dataset(df)
        print_inspection_report(report)
    else:
        print(f"[!] Dataset not found at: {DATASET_PATH}")
        print("[*] Generating realistic benchmark dataset (20,000 transactions, ~60 frauds)...")
        df = generate_realistic_synthetic_dataset(DATASET_PATH, n_samples=20000, fraud_rate=0.003)
        report = inspect_dataset(df)
        print_inspection_report(report)

    # Also generate sample_transactions.csv for testing Streamlit batch upload
    print(f"[*] Creating sample test transactions at: {SAMPLE_TEST_PATH}")
    sample_df = df.sample(n=min(50, len(df)), random_state=42).copy()
    sample_df.to_csv(SAMPLE_TEST_PATH, index=False)
    print(f"[+] Sample test transactions generated ({len(sample_df)} rows).")


if __name__ == "__main__":
    main()
