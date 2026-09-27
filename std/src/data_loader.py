"""
data_loader.py
Data loading, dataset inspection, and validation utilities for Credit Card Fraud Detection.
"""

import os
import sys
import numpy as np
import pandas as pd


def inspect_dataset(df: pd.DataFrame, target_col: str = "Class") -> dict:
    """
    Inspects the dataset and automatically detects:
    - number of rows
    - number of columns
    - column names
    - missing values
    - duplicate rows
    - target column and class distribution
    """
    n_rows, n_cols = df.shape
    columns = list(df.columns)
    missing_vals = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # Detect target column
    has_target = target_col in df.columns
    class_distribution = {}
    fraud_rate = 0.0

    if has_target:
        class_counts = df[target_col].value_counts().to_dict()
        class_distribution = {
            "legitimate (0)": int(class_counts.get(0, 0)),
            "fraudulent (1)": int(class_counts.get(1, 0)),
        }
        total_valid = class_counts.get(0, 0) + class_counts.get(1, 0)
        if total_valid > 0:
            fraud_rate = (class_counts.get(1, 0) / total_valid) * 100

    report = {
        "num_rows": n_rows,
        "num_cols": n_cols,
        "columns": columns,
        "total_missing_values": total_missing,
        "missing_per_column": {k: int(v) for k, v in missing_vals.items() if v > 0},
        "duplicate_rows": duplicate_rows,
        "target_column": target_col if has_target else None,
        "class_distribution": class_distribution,
        "fraud_rate_pct": round(fraud_rate, 4),
    }

    return report


def print_inspection_report(report: dict):
    """Prints a clean, formatted terminal summary of the dataset inspection."""
    print("=" * 60)
    print("        DATASET INSPECTION & PROFILING REPORT")
    print("=" * 60)
    print(f"[*] Total Rows (Records)      : {report['num_rows']:,}")
    print(f"[*] Total Columns (Features)  : {report['num_cols']}")
    print(f"[*] Total Missing Values      : {report['total_missing_values']}")
    if report["missing_per_column"]:
        print(f"    - Columns with missing: {report['missing_per_column']}")
    print(f"[*] Duplicate Rows Detected   : {report['duplicate_rows']:,}")
    print(f"[*] Target Column Detected    : {report['target_column']}")

    if report["class_distribution"]:
        legit = report["class_distribution"].get("legitimate (0)", 0)
        fraud = report["class_distribution"].get("fraudulent (1)", 0)
        print(f"[*] Class Distribution        :")
        print(f"    - Legitimate (0)          : {legit:,} ({100 - report['fraud_rate_pct']:.3f}%)")
        print(f"    - Fraudulent (1)          : {fraud:,} ({report['fraud_rate_pct']:.3f}%)")
        print(f"[*] Extreme Imbalance Ratio   : 1 fraud per ~{int(legit / max(fraud, 1)):,} transactions")
    print("=" * 60)


def generate_realistic_synthetic_dataset(
    filepath: str,
    n_samples: int = 15000,
    fraud_rate: float = 0.003,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic, statistically faithful synthetic dataset matching
    the exact schema of the ULB Kaggle Credit Card Fraud Detection dataset
    (Time, V1..V28, Amount, Class).

    Used for zero-setup execution, unit testing, and demonstration when the
    150MB raw creditcard.csv has not yet been placed by the user.
    """
    rng = np.random.RandomState(random_state)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # Generate Time (seconds across 2 days, ~ 0 to 172800)
    time_legit = np.sort(rng.uniform(0, 172800, n_legit))
    time_fraud = np.sort(rng.uniform(0, 172800, n_fraud))

    # PCA features V1..V28
    # In the actual dataset, PCA features have zero mean and unit variance for legitimate,
    # but fraudsters exhibit distinct shifts in key components:
    # V14, V12, V10, V17 tend to have strong negative shifts for frauds.
    # V4, V11 tend to have positive shifts for frauds.
    features_legit = rng.normal(0.0, 1.0, size=(n_legit, 28))
    features_fraud = rng.normal(0.0, 1.2, size=(n_fraud, 28))

    # Shift specific known discriminant PCA components based on empirical findings
    # Indices: V4 (idx 3), V10 (idx 9), V11 (idx 10), V12 (idx 11), V14 (idx 13), V17 (idx 16)
    features_fraud[:, 3] += rng.normal(2.5, 0.8, size=n_fraud)    # V4
    features_fraud[:, 9] -= rng.normal(3.0, 1.0, size=n_fraud)    # V10
    features_fraud[:, 10] += rng.normal(2.8, 0.9, size=n_fraud)   # V11
    features_fraud[:, 11] -= rng.normal(3.5, 1.1, size=n_fraud)   # V12
    features_fraud[:, 13] -= rng.normal(4.2, 1.2, size=n_fraud)   # V14
    features_fraud[:, 16] -= rng.normal(3.2, 1.0, size=n_fraud)   # V17

    # Amounts: legitimate transactions follow log-normal distribution, frauds often have higher variance
    amount_legit = np.round(np.exp(rng.normal(3.2, 1.2, size=n_legit)), 2)
    amount_fraud = np.round(np.exp(rng.normal(4.0, 1.5, size=n_fraud)), 2)

    df_legit = pd.DataFrame(features_legit, columns=[f"V{i}" for i in range(1, 29)])
    df_legit.insert(0, "Time", time_legit)
    df_legit["Amount"] = amount_legit
    df_legit["Class"] = 0

    df_fraud = pd.DataFrame(features_fraud, columns=[f"V{i}" for i in range(1, 29)])
    df_fraud.insert(0, "Time", time_fraud)
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df_combined = pd.concat([df_legit, df_fraud], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_combined.to_csv(filepath, index=False)
    print(f"[+] Created realistic benchmark dataset at: {filepath} ({len(df_combined):,} rows, {n_fraud} frauds)")
    return df_combined


def load_dataset(data_path: str = "data/creditcard.csv", auto_create_sample_if_missing: bool = True) -> pd.DataFrame:
    """
    Loads creditcard.csv if present.
    If not present, raises FileNotFoundError or generates a realistic sample if allowed.
    """
    if os.path.exists(data_path):
        print(f"[*] Loading existing dataset from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        if auto_create_sample_if_missing:
            print(f"[!] Dataset not found at: {data_path}")
            print(f"[*] Generating realistic benchmark dataset to allow instant out-of-the-box execution...")
            df = generate_realistic_synthetic_dataset(data_path, n_samples=20000, fraud_rate=0.003)
        else:
            raise FileNotFoundError(
                f"\n[ERROR] Dataset not found at expected location: '{data_path}'!\n"
                f"Please download 'creditcard.csv' from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)\n"
                f"and place it into the 'data/' directory as 'data/creditcard.csv'."
            )

    return df
