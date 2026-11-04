import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the fitted preprocessing artifacts and split data
exec(open("preprocess.py").read())

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=42)
}

os.makedirs("models", exist_ok=True)

# Train and save models
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train_processed, y_train)

    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"models/{filename}")

    print(f"{name} trained and saved.")

print("\nAll five models trained and saved successfully!")