# bank-marketing-classification
Machine learning classification of bank marketing data using Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and SVM. Includes model evaluation using Accuracy, AUC, Precision, Recall, F1-score, and MCC, with an interactive Streamlit application for testing saved models.

## Project Overview
This project uses machine learning classification algorithms to predict whether a customer will subscribe to a term deposit based on bank marketing campaign data.

The project implements five classification models and evaluates their performance using multiple classification metrics. An interactive Streamlit application allows users to upload test data, select a model, and view its evaluation results.

## Dataset
- **Source:** UCI Bank Marketing Dataset
- **File:** `bank-full.csv`
- **Records:** 45,211
- **Original features:** 16 input features and 1 target variable
- **Target variable:** `y` (yes/no)

The `duration` feature was excluded because it is only known after the marketing call and would not be available when predicting before contact.

## Data Preprocessing
- Train-test split: 80:20
- Stratified sampling to preserve target distribution
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- ColumnTransformer for combined preprocessing
- Preprocessing fitted only on training data to prevent data leakage

## Machine Learning Models
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Model Evaluation

The models were evaluated using Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8933 | 0.7717 | 0.6632 | 0.1786 | 0.2815 | 0.3065 |
| Decision Tree | 0.8308 | 0.6094 | 0.2948 | 0.3204 | 0.3071 | 0.2111 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Gaussian Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| SVM | 0.8933 | 0.7183 | 0.6555 | 0.1853 | 0.2889 | 0.3098 |

## Streamlit Application
The application supports:
- Uploading a test CSV file containing the target column `y`
- Selecting one of the five trained models
- Evaluating uploaded data using saved models
- Displaying Accuracy, AUC, Precision, Recall, F1-score, and MCC
- Displaying a confusion matrix and classification report

The application loads previously trained models and preprocessing artifacts. It does not retrain models using uploaded test data.

## Project Structure
```text
Bank Marketing/
├── app.py
├── preprocess.py
├── train_models.py
├── evaluate_models.py
├── requirements.txt
├── preprocessor.pkl
├── test_data.csv
├── model_evaluation_results.csv
├── data/
│   └── bank-full.csv
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── svm.pkl
```

## Run Locally

1. Clone the repository:
```bash
git clone https://github.com/Swarnapavithra/bank-marketing-classification.git
cd bank-marketing-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the application:
```bash
streamlit run app.py
```

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn
- Joblib
