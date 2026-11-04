import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Load the preprocessing script to recreate the test dataset
exec(open("preprocess.py").read())

# Store evaluation results
results = []

model_names = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "svm"
]

for name in model_names:

    print(f"\nEvaluating {name}...")

    # Load saved model
    model = joblib.load(f"models/{name}.pkl")

    # Generate predictions
    y_pred = model.predict(X_test_processed)

    # Generate prediction probabilities for AUC
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    # Calculate metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, pos_label="yes"),
        "Recall": recall_score(y_test, y_pred, pos_label="yes"),
        "F1 Score": f1_score(y_test, y_pred, pos_label="yes"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

# Display results
results_df = pd.DataFrame(results)

print("\nMODEL EVALUATION RESULTS")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("model_evaluation_results.csv", index=False)

print("\nResults saved to model_evaluation_results.csv")