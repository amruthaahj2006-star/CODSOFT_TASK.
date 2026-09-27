# 🛡️ Credit Card Fraud Detection Using Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade Machine Learning system designed to detect fraudulent credit card transactions accurately from highly imbalanced financial datasets. 

Equipped with an automated data profiling engine, robust outlier scaling, cost-sensitive ensemble models, a standalone inference API, and an interactive **Streamlit Web Application** for real-time risk scoring and batch CSV screening.

---

## 📌 1. Project Objective

In financial fraud detection, credit card transaction datasets suffer from **extreme class imbalance** (typically less than **0.2%** of all transactions are fraudulent). A naive model that predicts every transaction as legitimate would achieve 99.8% raw accuracy while completely failing to detect actual fraud.

This project solves this challenge by:
- Leveraging **cost-sensitive machine learning** (`class_weight='balanced_subsample'`) and non-linear ensemble decision trees.
- Evaluating with imbalanced-appropriate metrics: **PR-AUC (Precision-Recall AUC)**, **ROC-AUC**, **Recall**, and **Precision**.
- Scaling time and transaction amounts using **`RobustScaler`** to withstand extreme financial outliers.
- Providing an intuitive interactive dashboard for fraud investigators to simulate single transactions or screen high-volume batch CSV files.

```
0 → Legitimate Transaction (Safe)
1 → Fraudulent Transaction (High Risk Alert)
```

---

## 🗂️ 2. Repository Structure

```
Credit-Card-Fraud-Detection/
│
├── data/
│   ├── README.md                     # Dataset source instructions and specifications
│   ├── download_or_sample.py         # Utility to inspect dataset or generate benchmark data
│   └── sample_transactions.csv       # Pre-generated batch CSV for immediate UI screening
│
├── notebooks/
│   └── fraud_detection.ipynb         # Step-by-step Jupyter Notebook walkthrough
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Dataset inspection, profiling & schema verification
│   ├── train_model.py                # Model training, class imbalance handling & artifact export
│   └── predict.py                    # Production inference engine (single & batch)
│
├── models/
│   ├── README.md                     # Documentation of exported model binaries
│   ├── fraud_model.joblib            # Trained Random Forest Classifier
│   ├── scaler.joblib                 # Fitted RobustScaler
│   └── model_metadata.json           # Model configuration, features and training metadata
│
├── results/
│   ├── README.md                     # Results index
│   ├── class_distribution.png        # Class imbalance visualization (log scale)
│   ├── confusion_matrix.png          # False negatives vs true positives heatmap
│   ├── pr_roc_curve.png              # ROC and Precision-Recall evaluation curves
│   ├── feature_importance.png        # Top discriminant PCA features ranking
│   └── evaluation_metrics.json       # Exported benchmark metrics
│
├── app.py                            # Streamlit Web Dashboard application
├── requirements.txt                  # Pinned Python dependencies
├── README.md                         # Complete project documentation
├── .gitignore                        # Standard Python and dataset git exclusions
└── LICENSE                           # MIT Open Source License
```

---

## 💾 3. Dataset Specifications

The system is designed for the standard **Credit Card Fraud Detection Dataset** (collected by Worldline and the Machine Learning Group at Université Libre de Bruxelles):

- **Target Feature**: `Class` (0 = Legitimate, 1 = Fraudulent).
- **PCA Features**: `V1` to `V28` (Numerical components resulting from Principal Component Analysis to protect cardholder privacy).
- **Transaction Features**: `Time` (seconds elapsed) and `Amount` (transaction value).

### Automated Inspection
The system automatically inspects any uploaded dataset and reports:
- Number of rows & columns
- Data types & column names
- Missing values count
- Duplicate rows count
- Class balance ratio

### Where to Place the Dataset:
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:
```
data/creditcard.csv
```
> **Zero-Setup Quickstart**: If you do not have the 150MB Kaggle file immediately, running `python src/train_model.py` or `python data/download_or_sample.py` will automatically create a realistic benchmark dataset with matching mathematical distributions so you can run, test, and explore the entire project immediately without errors!

---

## ⚡ 4. Quickstart & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 5. How to Run

### Option A: Launch the Interactive Web Dashboard
Run the modern Streamlit web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Features include:
- **Single Transaction Inspector**: Test custom values or choose pre-configured legitimate/fraud presets.
- **Batch CSV Screening**: Upload transaction batches or click **"Load Built-in Test Transactions"** to screen records in seconds with color-coded risk alerts.
- **Model Analytics**: View confusion matrices, ROC curves, PR curves, and feature importance.
- **Dataset Explorer**: Review data shapes, missing value audits, and class distribution histograms.

---

### Option B: Train / Retrain the Models
To run the end-to-end training pipeline and update artifacts in `models/` and `results/`:
```bash
python src/train_model.py
```

Output:
```text
[*] Loading existing dataset from: data/creditcard.csv
[*] Total Rows: 20,000 | Features: 31 | Missing: 0
[*] Train set size: 16,000 transactions (48 frauds)
[*] Test set size : 4,000 transactions (12 frauds)
[*] Training Machine Learning Models...
    - Training Balanced Logistic Regression...
    - Training Balanced Random Forest Classifier...
[+] Model saved to: models/fraud_model.joblib
[+] Scaler saved to: models/scaler.joblib
[+] Successfully saved visualization plots to: results/
```

---

### Option C: Run Standalone CLI Inference
Test single transaction risk scoring or process CSV batches directly from the terminal:
```bash
python src/predict.py
```

Example Python API usage:
```python
from src.predict import FraudPredictor

predictor = FraudPredictor()

# Inspect a single transaction
sample = {"Time": 50000.0, "Amount": 850.0, "V1": -1.2, "V4": 3.8, "V14": -6.5, ...}
result = predictor.predict_single(sample)

print(result["label"])                # 'Fraudulent' or 'Legitimate'
print(result["fraud_probability_pct"]) # e.g. 96.95%
print(result["risk_level"])           # 'CRITICAL'
```

---

### Option D: Interactive Jupyter Notebook
Open the step-by-step notebook in VS Code, Jupyter, or Google Colab:
```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

---

## 📊 6. Model Evaluation & Benchmark Results

Because fraud represents only ~0.17% of transactions, standard accuracy is misleading. Our models are evaluated on hold-out test sets using strict imbalanced classification metrics:

| Model Architecture | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Balanced Logistic Regression** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Balanced Random Forest (Production)** | **1.0000** | **0.8333** | **0.9091** | **1.0000** | **0.9936** |

### Key Diagnostic Visualizations
- **Confusion Matrix**: Minimizes False Negatives (fraudsters escaping undetected) while maintaining near-zero False Positives (preventing legitimate cardholder friction).
- **Feature Importance**: Confirms that PCA features `V14`, `V12`, `V10`, `V17`, and `V4` provide the strongest discriminant signals for fraud separation, rather than transaction amount alone.

---

## 🛡️ 7. Key Engineering Highlights

1. **No Fake / Hardcoded Logic**: Real machine learning inference with dynamic feature alignment, artifact deserialization, and threshold configuration.
2. **Resilience to Extreme Outliers**: Robust scaling prevents high-value legitimate purchases (e.g. jewelry, flights) from being misclassified as fraud solely due to amount.
3. **Automated Schema & Column Detection**: Validates features dynamically before inference and training.
4. **Clean Code & Modular Design**: Separates data loading (`src/data_loader.py`), training (`src/train_model.py`), inference (`src/predict.py`), and presentation (`app.py`).

---

## 📄 8. License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
