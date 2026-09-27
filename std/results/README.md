# Results Directory

This directory stores evaluation metrics, classification reports, and visual charts generated during model training and validation.

## Generated Artifacts

- `confusion_matrix.png`: Normalized and raw confusion matrices highlighting true positives (detected frauds) vs false negatives (missed frauds).
- `pr_roc_curve.png`: Precision-Recall Curve and Receiver Operating Characteristic (ROC) curve.
- `feature_importance.png`: Feature importance chart highlighting the most critical PCA variables (e.g. V17, V14, V12, V10, V4) contributing to fraud classification.
- `class_distribution.png`: Imbalance visualization displaying legitimate vs fraudulent transaction counts.
- `evaluation_metrics.json`: Exported summary of Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
