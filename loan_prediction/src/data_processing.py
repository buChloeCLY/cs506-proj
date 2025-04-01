# Handles data loading and preprocessing
import pandas as pd
import numpy as np
from src.data_analysis import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def fill_missing(df):
    df = df.copy()
    mode_values = df[['Credit_History', 'Self_Employed', 'Dependents', 'Loan_Amount_Term', 'Gender', 'Married']].mode().iloc[0]
    df.fillna(mode_values, inplace=True)
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

    return df

def log_transform(df, columns):
    transformed_df = df.copy()
    for column in columns:
        transformed_df[column] = np.log1p(transformed_df[column])  # log1p(x)=log(1+x) to handle zero values
        plt.figure(figsize=(6, 4))
        sns.histplot(transformed_df[column], bins=20, kde=True, color="#3182bd")
        plt.title(f"Histogram of Log Transformed {column}")
        plt.xlabel(f"Log {column}")
        plt.ylabel("Frequency")
        plt.show()

    return transformed_df

def preprocess_data(df):
    df = df.copy()
    # Remove Loan_ID
    df = df.drop(columns=['Loan_ID'], errors='ignore')
    # Encode categorical variables
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    label_encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Store encoder for future use
    # Scale numerical features
    numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'TotalIncome']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df, label_encoders, scaler  # Return scaler for later use

def split_data(df):
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    train_df = load_data("../data/train.csv")
    test_df = load_data("../data/test.csv")
    # Display the first 5 rows
    # print("Train Data Head:\n", train_df.head())
    # print("Train Data Columns:\n", train_df.columns)
    # print("Test Data Head:\n", test_df.head())
    # print("Test Data Columns:\n", test_df.columns)
    # handel missing values
    # missing(train_df)
    # missing(test_df)
    # train_df = fill_missing(train_df)
    # test_df = fill_missing(test_df)
    # print("\nAfter Filling Missing Values:")
    # missing(train_df)
    # missing(test_df)
    # train_df["TotalIncome"] = train_df["ApplicantIncome"] + train_df["CoapplicantIncome"]
    # log_transform(train_df, ["ApplicantIncome", "TotalIncome", "LoanAmount"])
    # test_df["TotalIncome"] = test_df["ApplicantIncome"] + test_df["CoapplicantIncome"]
    # log_transform(test_df, ["ApplicantIncome", "TotalIncome", "LoanAmount"])





