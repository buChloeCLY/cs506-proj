# Main script to execute the project
from src.data_processing import *
from src.data_analysis import *
from src.model_training import *

def main():
    train_path = '../loan_prediction/data/train.csv'
    test_path = '../loan_prediction/data/test.csv'
    train_df = load_data(train_path)
    test_df = load_data(test_path)

    # Print basic info
    print("Train Data Head:\n", train_df.head())
    print("\nTrain Data Columns:\n", train_df.columns)
    print("\nTest Data Head:\n", test_df.head())
    print("\nTest Data Columns:\n", test_df.columns)

    # Perform EDA on training set
    print("\nAnalyze the training set:")
    analyze_data(train_df)
    analyze_target_variable(train_df)
    categorical_features = ['Gender', 'Married', 'Self_Employed', 'Credit_History']
    analyze_categorical_features(train_df, categorical_features)
    ordinal_features = ['Dependents', 'Education', 'Property_Area']
    analyze_ordinal_features(train_df, ordinal_features)
    applicantIncome_by_education(train_df)
    bivariate_stacked_bar(train_df, "Gender", "Gender vs Loan Approval")
    bivariate_stacked_bar(train_df, "Education", "Education vs Approval")
    bivariate_stacked_bar(train_df, "Self_Employed", "Self_Employed vs Approval")
    bivariate_stacked_bar(train_df, "Married", "Marital StatusLoan vs Approval")
    bivariate_stacked_bar(train_df, "Credit_History", "Credit History vs Loan Approval")
    correlation_heatmap(train_df.copy())
    income_vs_loan_status(train_df)
    scatter_correlation(train_df, "ApplicantIncome", "LoanAmount", "Applicant Income vs Loan Amount")

    # Handel missing values
    print("\nBefore Filling Missing Values:")
    missing(train_df)
    missing(test_df)
    train_df=fill_missing(train_df)
    test_df=fill_missing(test_df)
    print("\nAfter Filling Missing Values:")
    missing(train_df)
    missing(test_df)

    # Handel outliers
    outlier_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    detect_outliers(train_df, outlier_columns)

    # Creat new feature
    train_df["TotalIncome"] = train_df["ApplicantIncome"] + train_df["CoapplicantIncome"]
    log_transform(train_df, ["ApplicantIncome", "TotalIncome", "LoanAmount"])
    test_df["TotalIncome"] = test_df["ApplicantIncome"] + test_df["CoapplicantIncome"]
    log_transform(test_df, ["ApplicantIncome", "TotalIncome", "LoanAmount"])

    train_df, label_encoders, scaler = preprocess_data(train_df)
    #print(train_df.dtypes)
    # Save the processed dataset for future use
    train_df.to_csv("../loan_prediction/data/train_preprocessed.csv", index=False)
    X_train, X_val, y_train, y_val = split_data(train_df)
    #print(X_train.dtypes)
    # Convert Binned Categories to Numeric Labels
    X_train["IncomeBin"] = X_train["IncomeBin"].cat.codes
    X_train["TotalIncomeBin"] = X_train["TotalIncomeBin"].cat.codes
    X_val["IncomeBin"] = X_val["IncomeBin"].cat.codes
    X_val["TotalIncomeBin"] = X_val["TotalIncomeBin"].cat.codes

    # logistic_regression
    model = logistic_regression(X_train, y_train)
    evaluate_model(model, X_val, y_val)


if __name__ == "__main__":
    main()
