# Model training and evaluation
from src.data_processing import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=2000, random_state=42)
    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"Logistic Regression Cross-Validation Accuracy: {scores.mean():.4f}")
    # Train final model on full training data
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


# if __name__ == "__main__":
#     train_df = load_data("../data/train.csv")
#     train_df = fill_missing(train_df)
#     print("\nAfter Filling Missing Values:")
#     missing(train_df)
#     train_df["TotalIncome"] = train_df["ApplicantIncome"] + train_df["CoapplicantIncome"]
#     log_transform(train_df, ["ApplicantIncome", "TotalIncome", "LoanAmount"])
#
#     train_df, label_encoders, scaler = preprocess_data(train_df)
#     # Save the processed dataset for future use
#     train_df.to_csv("../data/train_preprocessed.csv", index=False)
#     X_train, X_val, y_train, y_val = split_data(train_df)
#     model = logistic_regression(X_train, y_train)
#     evaluate_model(model, X_val, y_val)
