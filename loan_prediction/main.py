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
    bivariate_stacked_bar(train_df, "Married", "Marital Status vs Loan Approval")
    bivariate_stacked_bar(train_df, "Credit_History", "Credit History vs Loan Approval")
    correlation_heatmap(train_df.copy())
    interactive_correlation_heatmap(train_df.copy())
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
    # Save the processed dataset for future use
    train_df.to_csv("../loan_prediction/data/train_preprocessed.csv", index=False)
    X_train, X_val, y_train, y_val = split_data(train_df)
    # # Convert Binned Categories to Numeric Labels
    X_train["IncomeBin"] = X_train["IncomeBin"].cat.codes
    X_train["TotalIncomeBin"] = X_train["TotalIncomeBin"].cat.codes
    X_val["IncomeBin"] = X_val["IncomeBin"].cat.codes
    X_val["TotalIncomeBin"] = X_val["TotalIncomeBin"].cat.codes

    # logistic_regression
    print("\nTraining Logistic Regression")
    basic_model = logistic_regression(X_train, y_train)
    basic_cv = cross_validate_model(basic_model, X_train, y_train)
    basic_model.fit(X_train, y_train)
    lr_feature_coefficient(basic_model, X_train.columns)
    tuned_model, best_cv = tune_logistic_regression(X_train, y_train)
    tuned_model.fit(X_train, y_train)
    val_acc_tuned = evaluate_model(tuned_model, X_val, y_val, "Logistic Regression")
    record_accuracy("Logistic Regression", basic_cv, best_cv, val_acc_tuned)

    # Decision Tree
    print("\nTraining Decision Tree")
    basic_model = decision_tree(X_train, y_train)
    basic_cv = cross_validate_model(basic_model, X_train, y_train)
    basic_model.fit(X_train, y_train)
    tuned_model, best_cv = tune_decision_tree(X_train, y_train)
    tuned_model.fit(X_train, y_train)
    val_acc_tuned = evaluate_model(tuned_model, X_val, y_val, "Decision Tree")
    record_accuracy("Decision Tree", basic_cv, best_cv, val_acc_tuned)
    visualize_tree(tuned_model,feature_names=X_train.columns)

    # Random Forest
    print("\nTraining Random Forest")
    basic_model = random_forest(X_train, y_train)
    basic_cv = cross_validate_model(basic_model, X_train, y_train)
    basic_model.fit(X_train, y_train)
    tuned_model, best_cv = tune_random_forest(X_train, y_train)
    tuned_model.fit(X_train, y_train)
    val_acc_tuned = evaluate_model(tuned_model, X_val, y_val, "Random Forest")
    record_accuracy("Random Forest", basic_cv, best_cv, val_acc_tuned)
    plot_feature_importance(tuned_model, X_train.columns,"Tuned Random Forest")
    interactive_feature_importance(tuned_model, X_train.columns, title="Random Forest Feature Importance")

    # XGBoost
    print("\nTraining XGBoost")
    basic_model = xgboost(X_train, y_train)
    basic_cv = cross_validate_model(basic_model, X_train, y_train)
    basic_model.fit(X_train, y_train)
    tuned_model, best_cv = tune_xgboost(X_train, y_train)
    tuned_model.fit(X_train, y_train)
    val_acc_tuned = evaluate_model(tuned_model, X_val, y_val, "XGBoost")
    record_accuracy("XGBoost", basic_cv, best_cv, val_acc_tuned)
    plot_feature_importance(tuned_model, X_train.columns, "Tuned XGBoost")
    interactive_feature_importance(tuned_model, X_train.columns, title="XGBoost Feature Importance")

    # SVM
    print("\nTraining SVM")
    basic_model = svm(X_train, y_train)
    basic_cv = cross_validate_model(basic_model, X_train, y_train)
    basic_model.fit(X_train, y_train)
    tuned_model, best_cv = tune_svm(X_train, y_train)
    tuned_model.fit(X_train, y_train)
    val_acc_tuned = evaluate_model(tuned_model, X_val, y_val, "SVM")
    record_accuracy("SVM", basic_cv, best_cv, val_acc_tuned)
    svm_vis = SVC(kernel="linear", C=1.0, random_state=42)
    svm_vis.fit(X_val[["Credit_History", "TotalIncome"]], y_val)
    svm_decision_boundary(svm_vis, X_val, y_val, ["Credit_History", "TotalIncome"])

    # Final Results
    accuracy_df = pd.DataFrame(accuracy_table)
    print("\nModel Accuracy Comparison:")
    print(accuracy_df.sort_values("Validation Accuracy", ascending=False).to_string(index=False))
    plot_model_accuracy(accuracy_df)
    evaluate_df = pd.DataFrame(evaluate_table)
    print("\nModel Performance Comparison:")
    print(evaluate_df.sort_values("F1 Score", ascending=False).to_string(index=False))
    interactive_model_comparison(accuracy_df)

    # Predict
    test_df_original = test_df.copy()
    test_df, _, _ = preprocess_data(test_df)
    # Create IncomeBin and TotalIncomeBin for test data
    test_df['IncomeBin'] = pd.qcut(test_df['ApplicantIncome'], q=4, precision=0)
    test_df["TotalIncomeBin"] = pd.qcut(test_df["TotalIncome"], q=4, precision=0)
    # Encode the bins
    test_df["IncomeBin"] = test_df["IncomeBin"].cat.codes
    test_df["TotalIncomeBin"] = test_df["TotalIncomeBin"].cat.codes
    X_test = test_df.drop(columns=['Loan_ID'], errors='ignore')
    # Align test columns with train columns
    X_test = X_test[X_train.columns]
    tuned_model, best_cv = tune_logistic_regression(X_train, y_train)
    tuned_model.fit(X_train, y_train)
    test_predictions = tuned_model.predict(X_test)
    test_predictions = ['Y' if pred == 1 else 'N' for pred in test_predictions]
    submission_df = pd.DataFrame({
        'Loan_ID': test_df_original['Loan_ID'],
        'Loan_Status': test_predictions
    })
    submission_df.to_csv("../loan_prediction/data/test_predictions.csv", index=False)
    print("Test predictions saved.")

if __name__ == "__main__":
    main()
