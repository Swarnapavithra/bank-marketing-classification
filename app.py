import streamlit as st
import pandas as pd
import sys
import importlib.util

st.write("Python version:", sys.version)
st.write("Joblib module found:", importlib.util.find_spec("joblib"))

import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Classification")
st.write("Compare five machine learning models for predicting term deposit subscriptions.")

# Load saved preprocessing artifact
preprocessor = joblib.load("preprocessor.pkl")

model_names = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl"
}

selected_model = st.selectbox(
    "Select a classification model",
    list(model_names.keys())
)

uploaded_file = st.file_uploader(
    "Upload test CSV (must include target column 'y')",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if "y" not in data.columns:
            st.error("The uploaded CSV must contain the target column 'y'.")
        else:
            X = data.drop(columns=["y", "duration"], errors="ignore")
            y_true = data["y"]

            if set(y_true.dropna().unique()) - {"yes", "no"}:
                st.error("Target column 'y' must contain only 'yes' and 'no'.")
            elif y_true.isna().any():
                st.error("Target column 'y' contains missing values.")
            else:
                X_processed = preprocessor.transform(X)

                model_path = os.path.join(
                    "models", model_names[selected_model]
                )
                model = joblib.load(model_path)

                y_pred = model.predict(X_processed)
                y_prob = model.predict_proba(X_processed)[:, 1]

                metrics = {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "AUC": roc_auc_score(
                        (y_true == "yes").astype(int), y_prob
                    ),
                    "Precision": precision_score(
                        y_true, y_pred, pos_label="yes", zero_division=0
                    ),
                    "Recall": recall_score(
                        y_true, y_pred, pos_label="yes", zero_division=0
                    ),
                    "F1 Score": f1_score(
                        y_true, y_pred, pos_label="yes", zero_division=0
                    ),
                    "MCC": matthews_corrcoef(y_true, y_pred)
                }

                st.subheader(f"Evaluation: {selected_model}")

                cols = st.columns(3)
                for i, (metric, value) in enumerate(metrics.items()):
                    cols[i % 3].metric(metric, f"{value:.4f}")

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(
                    y_true, y_pred, labels=["no", "yes"]
                )
                st.dataframe(
                    pd.DataFrame(
                        cm,
                        index=["Actual no", "Actual yes"],
                        columns=["Predicted no", "Predicted yes"]
                    )
                )

                st.subheader("Classification Report")
                report = classification_report(
                    y_true,
                    y_pred,
                    labels=["no", "yes"],
                    zero_division=0,
                    output_dict=True
                )
                st.dataframe(pd.DataFrame(report).transpose())

    except Exception as e:
        st.error(f"Unable to evaluate the uploaded file: {e}")

st.caption("Models are loaded from saved artifacts. Uploaded data is evaluated without retraining.")