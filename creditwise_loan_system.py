# CreditWise Loan System
# Loan Approval Prediction using Machine Learning

import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# 1. LOAD DATA
df = pd.read_csv("loan_approval_data.csv")


# 2. DATA PREPROCESSING

# Separate column types
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['number']).columns

# Handle missing values
num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

# Drop unnecessary column
df = df.drop("Applicant_ID", axis=1)

# Label Encoding
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

# One-Hot Encoding
cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Gender", "Employer_Category"]

ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(cols),
    index=df.index
)

df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)


# 3. FEATURE ENGINEERING
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2


# 4. TRAIN-TEST SPLIT
X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio", "Applicant_Income"])
y = df["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. MODEL TRAINING FUNCTION
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 6. MODEL TRAINING

# Logistic Regression
evaluate_model(LogisticRegression(), X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression")

# KNN
evaluate_model(KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled, y_train, y_test, "KNN")

# Naive Bayes
evaluate_model(GaussianNB(), X_train_scaled, X_test_scaled, y_train, y_test, "Naive Bayes")


# FINAL RESULT
print("\nBest Model: Naive Bayes (based on Precision Score)")