import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="Advanced Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
}

.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.markdown(
    '<div class="title">💳 Advanced Credit Card Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Transaction Risk Analysis</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# LOAD MODEL & SCALER
# ---------------------------------------------------
MODEL_PATHS = [
    "models/fraud_model.joblib",
    "models/best_model.joblib",
    "models/fraud_model.pkl",
    "models/credit_card_model.pkl",
    "models/model.pkl",
    "fraud_model.joblib",
    "best_model.joblib",
    "fraud_model.pkl",
    "credit_card_model.pkl",
    "model.pkl"
]

SCALER_PATHS = [
    "models/scaler.joblib",
    "models/scaler.pkl",
    "scaler.joblib",
    "scaler.pkl"
]

model = None
model_path = None
scaler = None
metadata = {}

for path in MODEL_PATHS:
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            model_path = path
            break
        except Exception:
            continue

for path in SCALER_PATHS:
    if os.path.exists(path):
        try:
            scaler = joblib.load(path)
            break
        except Exception:
            continue

meta_path = "models/model_metadata.json"
if os.path.exists(meta_path):
    try:
        import json
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    except Exception:
        pass

if model is None:
    st.error("❌ Model file not found.")
    st.info(
        "Please run `python src/train_model.py` or place your trained `.joblib` / `.pkl` model inside the `models` folder."
    )
    st.stop()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("⚙️ Project Information")

st.sidebar.success("✅ Model Loaded Successfully")

st.sidebar.write("### Model")
st.sidebar.write(type(model).__name__)

if scaler is not None:
    st.sidebar.write("### Scaler")
    st.sidebar.write(type(scaler).__name__)

st.sidebar.write("### Features")
st.sidebar.write("Time + V1–V28 + Amount (30 features)")

st.sidebar.write("### Detection")
st.sidebar.write("Fraud / Genuine")

# Threshold slider
threshold = st.sidebar.slider(
    "🚨 Fraud Decision Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Transactions with probability >= threshold are classified as Fraud."
)

# ---------------------------------------------------
# SESSION STATE & PRESETS
# ---------------------------------------------------
if "preset_applied" not in st.session_state:
    st.session_state["preset_applied"] = None
if "time_val" not in st.session_state:
    st.session_state["time_val"] = 0.0
if "amount_val" not in st.session_state:
    st.session_state["amount_val"] = 0.0
for i in range(1, 29):
    if f"v_input_{i}" not in st.session_state:
        st.session_state[f"v_input_{i}"] = 0.0

def load_preset(preset_type: str):
    if preset_type == "legit":
        st.session_state["time_val"] = 45000.0
        st.session_state["amount_val"] = 49.99
        for i in range(1, 29):
            st.session_state[f"v_input_{i}"] = 0.05
    elif preset_type == "fraud":
        st.session_state["time_val"] = 90000.0
        st.session_state["amount_val"] = 999.00
        fraud_vals = {4: 3.8, 10: -4.5, 11: 3.2, 12: -5.0, 14: -6.8, 17: -4.2}
        for i in range(1, 29):
            st.session_state[f"v_input_{i}"] = fraud_vals.get(i, 0.0)
    elif preset_type == "reset":
        st.session_state["time_val"] = 0.0
        st.session_state["amount_val"] = 0.0
        for i in range(1, 29):
            st.session_state[f"v_input_{i}"] = 0.0

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------
st.header("🔍 Transaction Feature Inputs")
st.write(
    "Enter transaction details or click a preset below to populate test values."
)

# Preset buttons
col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
with col_p1:
    if st.button("🟢 Load Normal Sample", use_container_width=True):
        load_preset("legit")
        st.rerun()
with col_p2:
    if st.button("🔴 Load Fraud Sample", use_container_width=True):
        load_preset("fraud")
        st.rerun()
with col_p3:
    if st.button("🔄 Reset Inputs to Zero", use_container_width=True):
        load_preset("reset")
        st.rerun()

st.markdown("---")

# Transaction Amount and Time
col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    st.subheader("💰 Transaction Amount")
    amount = st.number_input(
        "Amount ($)",
        min_value=0.0,
        value=float(st.session_state["amount_val"]),
        format="%.2f",
        key="amount_widget"
    )
    st.session_state["amount_val"] = amount

with col_meta2:
    st.subheader("⏱️ Transaction Time")
    time_val = st.number_input(
        "Time (Seconds elapsed)",
        min_value=0.0,
        value=float(st.session_state["time_val"]),
        format="%.2f",
        key="time_widget"
    )
    st.session_state["time_val"] = time_val

# V1 - V28 Inputs
st.subheader("🧬 PCA Components (V1 – V28)")
features_v = []
columns = st.columns(4)

for i in range(1, 29):
    with columns[(i - 1) % 4]:
        val = st.number_input(
            f"V{i}",
            value=float(st.session_state[f"v_input_{i}"]),
            format="%.6f",
            key=f"v_widget_{i}"
        )
        st.session_state[f"v_input_{i}"] = val
        features_v.append(val)

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------
st.divider()

predict_button = st.button(
    "🚨 Detect Fraud",
    type="primary",
    use_container_width=True
)

# ---------------------------------------------------
# PREDICTION & EVALUATION
# ---------------------------------------------------
if predict_button:

    try:
        # Build feature dictionary
        feature_dict = {f"V{i}": features_v[i - 1] for i in range(1, 29)}
        feature_dict["Time"] = time_val
        feature_dict["Amount"] = amount

        # Align with model's expected features
        expected_cols = metadata.get(
            "features",
            ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        )
        df_input = pd.DataFrame([feature_dict])

        for col in expected_cols:
            if col not in df_input.columns:
                df_input[col] = 0.0
        df_input = df_input[expected_cols]

        # Apply RobustScaler to Amount and Time
        if scaler is not None:
            scaled_cols = metadata.get("scaled_columns", ["Amount", "Time"])
            cols_to_scale = [c for c in scaled_cols if c in df_input.columns]
            if cols_to_scale:
                df_input[cols_to_scale] = scaler.transform(df_input[cols_to_scale])

        # Model Inference
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df_input)[0]
            if len(probabilities) == 2:
                prob_val = float(probabilities[1])
            else:
                prob_val = float(max(probabilities))
            probability = prob_val * 100
            prediction = 1 if prob_val >= threshold else 0
        else:
            prediction = int(model.predict(df_input)[0])

        # Risk Classification
        if probability is not None:
            if probability >= 80.0:
                risk = "CRITICAL RISK"
            elif probability >= 50.0:
                risk = "HIGH RISK"
            elif probability >= 20.0:
                risk = "MODERATE RISK"
            else:
                risk = "LOW RISK"
        else:
            risk = "HIGH RISK" if prediction == 1 else "LOW RISK"

        # ------------------------------------------------
        # RESULT DISPLAY
        # ------------------------------------------------
        st.divider()
        st.header("📊 Fraud Detection Result")

        if prediction == 1:
            st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
        else:
            st.success("✅ GENUINE TRANSACTION")

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Prediction",
                "FRAUD" if prediction == 1 else "GENUINE"
            )
        with col2:
            if probability is not None:
                st.metric(
                    "Fraud Probability",
                    f"{probability:.2f}%"
                )
            else:
                st.metric("Fraud Probability", "N/A")
        with col3:
            st.metric("Risk Level", risk)

        # Progress bar
        if probability is not None:
            st.subheader("📈 Fraud Probability Gauge")
            st.progress(min(int(probability), 100))

            if probability >= 70:
                st.warning(
                    "⚠️ High probability of fraud. Immediate card block or verification required."
                )
            elif probability >= 40:
                st.info(
                    "ℹ️ Medium probability of fraud. Multi-factor authentication recommended."
                )
            else:
                st.success("✅ Low probability of fraud. Safe to process.")

        # Transaction Summary
        st.subheader("📋 Evaluated Transaction Summary")
        summary_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        transaction_data = pd.DataFrame(
            [[feature_dict.get(c, 0.0) for c in summary_cols]],
            columns=summary_cols
        )
        st.dataframe(transaction_data, use_container_width=True)

        # ------------------------------------------------
        # SAVE TO HISTORY LOG
        # ------------------------------------------------
        history_file = "transaction_history.csv"
        result_data = transaction_data.copy()
        result_data["Prediction"] = "FRAUD" if prediction == 1 else "GENUINE"
        if probability is not None:
            result_data["Fraud Probability"] = f"{probability:.2f}%"
        result_data["Risk Level"] = risk

        if os.path.exists(history_file):
            try:
                old_data = pd.read_csv(history_file)
                updated_data = pd.concat([old_data, result_data], ignore_index=True)
            except Exception:
                updated_data = result_data
        else:
            updated_data = result_data

        updated_data.to_csv(history_file, index=False)
        st.success("💾 Transaction successfully logged to history!")

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# ---------------------------------------------------
# VIEW TRANSACTION HISTORY
# ---------------------------------------------------
history_file = "transaction_history.csv"
if os.path.exists(history_file):
    with st.expander("📜 View Audit Log & Transaction History"):
        try:
            hist_df = pd.read_csv(history_file)
            st.dataframe(hist_df, use_container_width=True)
            st.download_button(
                "📥 Download History Log (CSV)",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="transaction_history.csv",
                mime="text/csv"
            )
        except Exception:
            st.caption("Unable to load history file.")

# ---------------------------------------------------
# BATCH CSV SCREENING
# ---------------------------------------------------
with st.expander("📁 Batch Transaction Screening (Upload CSV or Test Sample)"):
    st.write("Screen multiple transactions simultaneously using the trained model.")
    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        load_sample = st.button("⚡ Load Built-in Test Sample (50 Transactions)", key="batch_sample_btn")
    with b_col2:
        uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"], key="batch_upload")

    batch_df = None
    sample_path = os.path.join("data", "sample_transactions.csv")

    if load_sample and os.path.exists(sample_path):
        batch_df = pd.read_csv(sample_path)
    elif uploaded_csv is not None:
        try:
            batch_df = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if batch_df is not None:
        st.write(f"**Loaded Records:** `{len(batch_df)}` rows | **Columns:** `{len(batch_df.columns)}`")
        if st.button("🚀 Run Batch Prediction", type="primary", key="run_batch_btn"):
            with st.spinner("Processing batch transactions..."):
                try:
                    X_batch = batch_df.copy()
                    if "Class" in X_batch.columns:
                        X_batch = X_batch.drop(columns=["Class"])

                    expected_cols = metadata.get("features", ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
                    for col in expected_cols:
                        if col not in X_batch.columns:
                            X_batch[col] = 0.0
                    X_in = X_batch[expected_cols].copy()

                    if scaler is not None:
                        scaled_cols = metadata.get("scaled_columns", ["Amount", "Time"])
                        cols_to_scale = [c for c in scaled_cols if c in X_in.columns]
                        if cols_to_scale:
                            X_in[cols_to_scale] = scaler.transform(X_in[cols_to_scale])

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_in)[:, 1]
                    else:
                        probs = model.predict(X_in).astype(float)

                    preds = (probs >= threshold).astype(int)
                    screened_df = batch_df.copy()
                    screened_df["Fraud_Probability_%"] = np.round(probs * 100, 2)
                    screened_df["Prediction"] = np.where(preds == 1, "FRAUD", "GENUINE")
                    screened_df["Risk_Level"] = [
                        "CRITICAL" if p >= 0.8 else ("HIGH" if p >= 0.5 else ("MODERATE" if p >= 0.2 else "LOW"))
                        for p in probs
                    ]

                    total_tx = len(screened_df)
                    fraud_tx = int((preds == 1).sum())
                    legit_tx = total_tx - fraud_tx

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Transactions", total_tx)
                    m2.metric("Flagged Fraud", fraud_tx, delta=f"{(fraud_tx/total_tx)*100:.1f}%", delta_color="inverse")
                    m3.metric("Cleared Legitimate", legit_tx)

                    # Sorted display
                    sorted_res = screened_df.sort_values(by="Fraud_Probability_%", ascending=False)
                    st.dataframe(sorted_res, use_container_width=True)

                    st.download_button(
                        "📥 Download Screened Results (CSV)",
                        data=sorted_res.to_csv(index=False).encode("utf-8"),
                        file_name="screened_transactions.csv",
                        mime="text/csv"
                    )
                except Exception as ex:
                    st.error(f"Error processing batch: {ex}")

# ---------------------------------------------------
# MODEL ANALYTICS & VISUALIZATIONS
# ---------------------------------------------------
with st.expander("📊 Model Analytics & Evaluation Charts"):
    st.write("Hold-out test benchmark metrics and visualization charts.")

    if "metrics" in metadata:
        m = metadata["metrics"]
        met1, met2, met3, met4, met5 = st.columns(5)
        met1.metric("Model", m.get("model_name", "Random Forest"))
        met2.metric("Precision", f"{m.get('precision', 1.0):.2%}")
        met3.metric("Recall", f"{m.get('recall', 0.833):.2%}")
        met4.metric("PR-AUC", f"{m.get('pr_auc', 0.9936):.4f}")
        met5.metric("ROC-AUC", f"{m.get('roc_auc', 1.0):.4f}")

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        if os.path.exists("results/confusion_matrix.png"):
            st.markdown("##### 🎯 Confusion Matrix")
            st.image("results/confusion_matrix.png", use_container_width=True)
        if os.path.exists("results/feature_importance.png"):
            st.markdown("##### 🔍 Feature Importance Ranking")
            st.image("results/feature_importance.png", use_container_width=True)

    with v_col2:
        if os.path.exists("results/pr_roc_curve.png"):
            st.markdown("##### 📈 ROC & PR Curves")
            st.image("results/pr_roc_curve.png", use_container_width=True)
        if os.path.exists("results/class_distribution.png"):
            st.markdown("##### ⚖️ Imbalance Distribution")
            st.image("results/class_distribution.png", use_container_width=True)