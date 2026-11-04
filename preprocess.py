import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

# Separate features and target
X = df.drop(columns=["y", "duration"])
y = df["y"]

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create preprocessing pipelines
numerical_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Fit on training data and transform both datasets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

print("\nTraining data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)

feature_names = preprocessor.get_feature_names_out()

print("\nTransformed feature names:")
print(feature_names)

print("\nTotal transformed features:", len(feature_names))
print("\nPreprocessing completed successfully!")

joblib.dump(preprocessor, "preprocessor.pkl")
print("\nPreprocessor saved successfully!")

# Save raw test data for Streamlit application
test_data = X_test.copy()
test_data["y"] = y_test

test_data.to_csv("test_data.csv", index=False)

print("\nTest data saved successfully!")