"""
train_model.py
Machine Learning Model Training Pipeline for Credit Card Fraud Detection.
Handles extreme class imbalance, feature scaling, model evaluation, and artifact serialization.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
import joblib

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_dataset, inspect_dataset, print_inspection_report

# Configure matplotlib for headless environment
plt.switch_backend("Agg")
sns.set_theme(style="whitegrid", palette="muted")


def preprocess_data(df: pd.DataFrame, target_col: str = "Class"):
    """
    Cleans data, validates required features, separates X and y, and fits RobustScaler on Time and Amount.
    """
    # 1. Clean duplicates if present
    initial_count = len(df)
    df_clean = df.drop_duplicates().copy()
    num_dups = initial_count - len(df_clean)
    if num_dups > 0:
        print(f"[*] Removed {num_dups} duplicate transactions.")

    if target_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df_clean.columns)}")

    # 2. Extract X and y
    y = df_clean[target_col].astype(int)
    X = df_clean.drop(columns=[target_col])

    feature_cols = list(X.columns)

    # 3. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[*] Train set size: {X_train.shape[0]:,} transactions ({sum(y_train)} frauds)")
    print(f"[*] Test set size : {X_test.shape[0]:,} transactions ({sum(y_test)} frauds)")

    # 4. Scale 'Amount' and 'Time' using RobustScaler (resilient to severe outliers)
    scaler = RobustScaler()
    cols_to_scale = [col for col in ["Amount", "Time"] if col in X_train.columns]

    if cols_to_scale:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    else:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains models with cost-sensitive class balancing to handle extreme fraud imbalance.
    Returns dictionary of trained models.
    """
    print("[*] Training Machine Learning Models...")

    # 1. Balanced Logistic Regression (Linear baseline with class weights)
    print("    - Training Balanced Logistic Regression...")
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    lr_model.fit(X_train, y_train)

    # 2. Balanced Random Forest (Nonlinear ensemble with tree-level subsample balancing)
    print("    - Training Balanced Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    return {"Logistic Regression": lr_model, "Random Forest": rf_model}


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """
    Evaluates model performance with imbalanced classification metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model_name": model_name,
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1_score": float(round(f1, 4)),
        "roc_auc": float(round(roc_auc, 4)),
        "pr_auc": float(round(pr_auc, 4)),
        "confusion_matrix": cm,
    }

    print(f"\n[*] --- Evaluation: {model_name} ---")
    print(f"    - Precision : {prec:.4f}")
    print(f"    - Recall    : {rec:.4f}")
    print(f"    - F1-Score  : {f1:.4f}")
    print(f"    - ROC-AUC   : {roc_auc:.4f}")
    print(f"    - PR-AUC    : {pr_auc:.4f}")

    return metrics, y_proba


def generate_and_save_visualizations(
    results_dir: str,
    y_test: pd.Series,
    rf_proba: np.ndarray,
    rf_pred: np.ndarray,
    rf_model,
    feature_cols: list,
    df_raw: pd.DataFrame
):
    """
    Generates and saves professional evaluation figures into the results directory.
    """
    os.makedirs(results_dir, exist_ok=True)

    # 1. Class Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    class_counts = df_raw["Class"].value_counts()
    colors = ["#2b5c8f", "#d9534f"]
    bars = ax.bar(["Legitimate (0)", "Fraudulent (1)"], class_counts.values, color=colors, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + (max(class_counts.values) * 0.01),
                f"{int(h):,} ({h/len(df_raw)*100:.2f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Credit Card Transactions Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Transactions", fontsize=11)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "class_distribution.png"))
    plt.close(fig)

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, rf_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Legit (0)", "Pred Fraud (1)"],
        yticklabels=["Actual Legit (0)", "Actual Fraud (1)"],
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Confusion Matrix (Random Forest)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close(fig)

    # 3. ROC and Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, rf_proba)
    roc_val = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {roc_val:.3f})")
    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, rf_proba)
    pr_auc_val = average_precision_score(y_test, rf_proba)
    ax2.plot(recall_vals, precision_vals, color="#d9534f", lw=2, label=f"PR Curve (AUC = {pr_auc_val:.3f})")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "pr_roc_curve.png"))
    plt.close(fig)

    # 4. Feature Importance Plot
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        top_features = [feature_cols[i] for i in indices]
        top_values = importances[indices]

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        sns.barplot(x=top_values, y=top_features, palette="viridis", ax=ax)
        ax.set_title("Top 15 Most Discriminant Features for Fraud Detection", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Random Forest Feature Importance Score", fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, "feature_importance.png"))
        plt.close(fig)

    print(f"[+] Successfully saved visualization plots to: {results_dir}")


def main():
    print("=" * 60)
    print("   CREDIT CARD FRAUD DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "creditcard.csv")
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load and Inspect Dataset
    df = load_dataset(data_path, auto_create_sample_if_missing=True)
    report = inspect_dataset(df)
    print_inspection_report(report)

    # 2. Preprocessing & Train-Test Split
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess_data(df, target_col="Class")

    # 3. Train Models
    models = train_models(X_train, y_train)

    # 4. Evaluate Models
    eval_results = {}
    best_model_name = "Random Forest"
    best_model = models[best_model_name]

    for name, model in models.items():
        metrics, proba = evaluate_model(model, X_test, y_test, name)
        eval_results[name] = metrics
        if name == best_model_name:
            rf_proba = proba
            rf_pred = model.predict(X_test)

    # 5. Save Artifacts
    model_path = os.path.join(models_dir, "fraud_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")
    metrics_path = os.path.join(results_dir, "evaluation_metrics.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_type": type(best_model).__name__,
        "features": feature_cols,
        "scaled_columns": ["Amount", "Time"],
        "scaler_type": type(scaler).__name__,
        "dataset_rows": report["num_rows"],
        "fraud_rate_pct": report["fraud_rate_pct"],
        "metrics": eval_results[best_model_name],
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    with open(metrics_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"[+] Model saved to: {model_path}")
    print(f"[+] Scaler saved to: {scaler_path}")
    print(f"[+] Metadata saved to: {meta_path}")

    # 6. Generate and Save Visualizations
    generate_and_save_visualizations(
        results_dir=results_dir,
        y_test=y_test,
        rf_proba=rf_proba,
        rf_pred=rf_pred,
        rf_model=best_model,
        feature_cols=feature_cols,
        df_raw=df
    )

    print("\n[+] Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
