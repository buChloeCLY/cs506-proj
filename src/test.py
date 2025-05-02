# Model training and evaluation
from src.data_processing import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_table = [] #collect accuracy
evaluate_table = [] #collect results

def logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=2000, random_state=42)
    return model

def tune_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'max_iter': [1000]
    }
    model = LogisticRegression(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_, grid.best_score_

def lr_feature_coefficient(model, feature_names):
    coef = model.coef_[0]
    coef_df = pd.DataFrame({'Feature': feature_names,'Coefficient': coef}).sort_values(by='Coefficient', ascending=False)
    plt.figure(figsize=(10, 6))
    colors = coef_df['Coefficient'].apply(lambda x: 'green' if x > 0 else 'red')
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.title('Logistic Regression Feature Coefficient')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    return model

def tune_decision_tree(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 4, 6],
        'criterion': ['gini', 'entropy']
    }
    model = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_, grid.best_score_

def visualize_tree(model, feature_names):
    plt.figure(figsize=(20, 12))
    plot_tree(model,feature_names=feature_names,class_names=['Not Approved', 'Approved'],filled=True,proportion=True,fontsize=10)
    plt.title("Decision Tree Structure",fontsize=16)
    plt.tight_layout()
    plt.show()

def random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 8],
        'min_samples_split': [2, 5, 8],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_, grid.best_score_

def plot_feature_importance(model, feature_names, model_name="Model"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(importances)), importances[indices], align='center', color="#396EB0")
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance for {model_name}')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()


def xgboost(X_train, y_train):
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    return model

def tune_xgboost(X_train, y_train):
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200]
    }
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_, grid.best_score_

def svm(X_train, y_train):
    model = SVC(kernel='rbf', probability=True, random_state=42)
    return model

def tune_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    model = SVC(probability=True, random_state=42)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_, grid.best_score_

def svm_decision_boundary(model_class, X, y, feature_names):
    # Only select the two features
    X_plot = X[feature_names].copy()
    y_plot = y.copy()
    # Scale features
    scaler = StandardScaler()
    X_plot_scaled = scaler.fit_transform(X_plot)
    # Retrain a new SVM model on the selected features
    model = model_class
    model.fit(X_plot_scaled, y_plot)
    # Create a meshgrid
    x_min, x_max = X_plot_scaled[:, 0].min() - 1, X_plot_scaled[:, 0].max() + 1
    y_min, y_max = X_plot_scaled[:, 1].min() - 1, X_plot_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
    # Predict over the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z,alpha=0.3)
    plt.scatter(X_plot_scaled[:, 0], X_plot_scaled[:, 1], c=y_plot, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("SVM Decision Boundary (2 features)")
    plt.show()

def cross_validate_model(model, X_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    return scores.mean()

def evaluate_model(model, X_val, y_val, model_name=None):
    y_pred = model.predict(X_val)
    metrics = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1": f1_score(y_val, y_pred)
    }
    if model_name:
        evaluate_table.append({"Model": model_name, **metrics})
    return metrics["Accuracy"]


def record_accuracy(model_name, cv_score, best_cv_score, val_accuracy):
    accuracy_table.append({
        "Model": model_name,
        "CV Accuracy": round(cv_score, 6),  # Prevent floating point weirdness
        "Best CV Accuracy": round(best_cv_score, 6),
        "Validation Accuracy": round(val_accuracy, 6)
    })


def train_and_evaluate_model(model_func, tune_func, X_train, y_train, X_val, y_val, model_name):
    print(f"\nTraining {model_name}")
    # Basic model
    basic_model = model_func(X_train, y_train)
    basic_cv = cross_validate_model(basic_model, X_train, y_train)
    basic_model.fit(X_train, y_train)
    basic_val_acc = evaluate_model(basic_model, X_val, y_val)
    # Tuned model
    tuned_model, best_cv = tune_func(X_train, y_train)
    tuned_model.fit(X_train, y_train)
    tuned_val_acc = evaluate_model(tuned_model, X_val, y_val, model_name)
    # Record both basic and tuned results
    record_accuracy(f"{model_name} (Basic)", basic_cv, basic_cv, basic_val_acc)
    record_accuracy(model_name, basic_cv, best_cv, tuned_val_acc)
    return tuned_model

def plot_model_accuracy(accuracy_df):
    ax = accuracy_df.plot(x='Model', y=['CV Accuracy', 'Best CV Accuracy', 'Validation Accuracy'],kind='bar', figsize=(14, 8),color=['#8BB8E8', '#2774AE', '#005587'])
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_df = load_data("../data/train.csv")
    train_df = fill_missing(train_df)
    #print("\nAfter Filling Missing Values:")
    missing(train_df)
    train_df["TotalIncome"] = train_df["ApplicantIncome"] + train_df["CoapplicantIncome"]
    #log_transform(train_df, ["ApplicantIncome", "TotalIncome", "LoanAmount"])

    train_df, label_encoders, scaler = preprocess_data(train_df)
    # Save the processed dataset for future use
    train_df.to_csv("../data/train_preprocessed.csv", index=False)
    X_train, X_val, y_train, y_val = split_data(train_df)

    # Model training pipeline
    models = [
        (logistic_regression, tune_logistic_regression, "Logistic Regression"),
        (decision_tree, tune_decision_tree, "Decision Tree"),
        (random_forest, tune_random_forest, "Random Forest"),
        (xgboost, tune_xgboost, "XGBoost"),
        (svm, tune_svm, "SVM")
    ]

    for model_func, tune_func, name in models:
        model = train_and_evaluate_model(
            model_func, tune_func,
            X_train, y_train, X_val, y_val, name
        )

        # Model-specific visualizations
        if name == "Logistic Regression":
            lr_feature_coefficient(model, X_train.columns)
        elif name == "Decision Tree":
            visualize_tree(model, X_train.columns)
        elif name in ["Random Forest", "XGBoost"]:
            plot_feature_importance(model, X_train.columns, name)
        elif name == "SVM":
            svm_vis = SVC(kernel="linear", C=1.0, random_state=42)
            svm_decision_boundary(svm_vis, X_val, y_val, ["Credit_History", "TotalIncome"])

    # Results analysis
    accuracy_df = pd.DataFrame(accuracy_table)
    print("\nModel Accuracy Comparison:")
    print(accuracy_df.sort_values("Validation Accuracy", ascending=False).to_string(index=False))
    plot_model_accuracy(accuracy_df)

    evaluate_df = pd.DataFrame(evaluate_table)
    print("\nModel Performance Comparison:")
    print(evaluate_df.sort_values("F1 Score", ascending=False).to_string(index=False))





