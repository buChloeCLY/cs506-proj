# Performs exploratory data analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import *
from sklearn.preprocessing import LabelEncoder

def analyze_data(df):
    print("Dataset Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

def analyze_target_variable(df):
    # Target Variable: Loan_Status
    plt.figure(figsize=(7, 6))
    palette = {"Y": "#396EB0", "N": "#FFAB5B"}
    sns.countplot(x="Loan_Status", hue="Loan_Status", data=df, palette=palette, legend=False)
    # Calculate count and proportion
    total = len(df)
    counts = df['Loan_Status'].value_counts()
    proportions = (counts / total * 100).round(1)
    # Create legend labels
    labels = [f"{k}: {v} ({proportions[k]}%)" for k, v in counts.items()]
    # Add legend with counts and proportions
    plt.legend(labels=labels, title="Loan Approved", loc='upper right')
    plt.title('Loan Status Distribution')
    plt.xlabel("Loan Approved (Y/N)")
    plt.ylabel("Count")
    plt.show()

def analyze_categorical_features(df, categorical_features):
    palette = ["#396EB0", "#FFAB5B"]
    for feature in categorical_features:
        plt.figure(figsize=(7, 6))
        # Count the occurrences and calculate proportions
        counts = df[feature].value_counts().sort_index()  # Ensure correct order
        proportions = counts / counts.sum() * 100
        # Create a bar plot
        ax = sns.barplot(x=counts.index.astype(str), hue=counts.index.astype(str), y=counts.values, palette=palette, legend=False)
        # Add labels
        for i, (count, prop) in enumerate(zip(counts.values, proportions)):
            ax.text(i, count + 2, f"{count} ({prop:.1f}%)", ha='center', fontsize=12)
        plt.title(f"{feature} Distribution", fontsize=14)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.show()

def analyze_ordinal_features(df, ordinal_features):
    palette = ["#2171b5", "#6baed6", "#bdd7e7", "#eff3ff"]
    for feature in ordinal_features:
        plt.figure(figsize=(7, 6))
        # Count occurrences and calculate proportions
        counts = df[feature].value_counts().sort_index()
        proportions = counts / counts.sum() * 100
        # Select colors dynamically based on the number of unique categories
        num_categories = len(counts)
        colors = palette[:num_categories]  # Pick colors accordingly
        # Create a bar plot with explicit category order
        ax = sns.barplot(x=counts.index.astype(str), hue=counts.index.astype(str), y=counts.values, palette=colors, legend=False)
        # Add labels
        for i, (count, prop) in enumerate(zip(counts.values, proportions)):
            ax.text(i, count + 2, f"{count} ({prop:.1f}%)", ha='center', fontsize=12)
        plt.title(f"{feature} Distribution", fontsize=14)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.show()

def analyze_numerical_features(df, numerical_features):
    hist_color = "#3182bd"
    box_color = "#9ecae1"
    for feature in numerical_features:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Histogram
        sns.histplot(df[feature].dropna(), bins=30, kde=True, color=hist_color, ax=axes[0])
        axes[0].set_title(f"Histogram of {feature}", fontsize=14)
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel("Frequency")
        # Boxplot
        sns.boxplot(x=df[feature], color=box_color, ax=axes[1])
        axes[1].set_title(f"Box Plot of {feature}", fontsize=14)
        plt.tight_layout()
        plt.show()

def applicantIncome_by_education(df):
        # Box plot of ApplicantIncome by Education
        plt.figure(figsize=(7, 6))
        sns.boxplot(x="Education", hue="Education", y="ApplicantIncome", data=df, palette=["#3182bd", "#9ecae1"])
        plt.title("Box Plot of Applicant Income by Education", fontsize=14)
        plt.xlabel("Education")
        plt.ylabel("Applicant Income")
        plt.show()

def bivariate_stacked_bar(df, feature, title):
    colors = ["#396EB0", "#FFAB5B"]
    # Stacked bar plot of Loan_Status grouped by feature
    crosstab = pd.crosstab(df[feature], df["Loan_Status"], normalize="index") * 100
    crosstab.plot(kind="bar", stacked=True, color=colors, figsize=(7, 6))
    plt.title(title, fontsize=14)
    plt.ylabel("Percentage")
    plt.xlabel(feature)
    plt.legend(["Not Approved", "Approved"], title="Loan Status")
    plt.xticks(rotation=0)
    plt.show()
    # Perform Chi-Square Test
    chi2, p, _, _ = chi2_contingency(pd.crosstab(df[feature], df["Loan_Status"]))
    print(f"Chi-Square Test for {feature}: p-value = {p:.4f}")
    if p < 0.05:
        print("Significant relationship\n")
    else:
        print("No significant relationship\n")

def income_vs_loan_status(df):
    # Mean Applicant Income vs Loan Status
    colors = ["#396EB0", "#FFAB5B"]
    mean_income = df.groupby("Loan_Status")["ApplicantIncome"].mean()
    plt.figure(figsize=(7, 6))
    ax = sns.barplot(x=mean_income.index, hue=mean_income.index, y=mean_income.values, palette=colors,legend=False)
    # Add text annotations
    for i, v in enumerate(mean_income):
        ax.text(i, v + 200, f"{v:.0f}", ha="center", fontsize=12)
    plt.xlabel("Loan Status")
    plt.ylabel("Mean Applicant Income")
    plt.title("Mean Applicant Income vs Loan Status")
    # Adjust y-limit for better visibility
    plt.ylim(0, max(mean_income) + 1000)

    # Applicant Income Bin vs Loan Status
    df['IncomeBin'] = pd.qcut(df['ApplicantIncome'], q=4, precision=0) # Create bins based on income percentiles
    # Calculate approval rate per bin
    bin_analysis = df.groupby('IncomeBin', observed=False)['Loan_Status'].value_counts(normalize=True).unstack()
    bin_analysis.columns = ['Not Approved', 'Approved']
    bin_analysis.plot(kind='bar', stacked=True, color=colors)
    plt.xlabel('Applicant Income Bins')
    plt.ylabel('Approval Rate')
    plt.title('Applicant Income Bin vs Loan Status')
    plt.xticks(rotation=45)
    plt.legend(title='Loan Status', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Total Income vs Loan Status
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["TotalIncomeBin"] = pd.qcut(df["TotalIncome"], q=4, precision=0)
    bin_analysis_total = df.groupby("TotalIncomeBin", observed=False)["Loan_Status"].value_counts(normalize=True).unstack()
    bin_analysis_total.columns = ["Not Approved", "Approved"]
    bin_analysis_total.plot(kind="bar", stacked=True, color=colors)
    plt.xlabel("Total Income Bins")
    plt.ylabel("Approval Rate")
    plt.title("Total Income Bin vs Loan Status")
    plt.xticks(rotation=45)
    plt.legend(title="Loan Status", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

def scatter_correlation(df, feature_x, feature_y, title):
    # Drop NaNs in both columns, keeping lengths equal
    df_clean = df.dropna(subset=[feature_x, feature_y])
    # Plot scatter plot with regression line
    plt.figure(figsize=(7, 6))
    sns.regplot(x=df_clean[feature_x], y=df_clean[feature_y], line_kws={"color": "#FFAB5B"})
    plt.title(title)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()
    # Compute Pearson Correlation
    corr, p = pearsonr(df_clean[feature_x].dropna(), df_clean[feature_y].dropna())
    print(f"Pearson Correlation ({feature_x} vs {feature_y}): {corr:.4f}, p-value = {p:.4f}")
    if p < 0.05:
        print(f"Significant {'positive' if corr > 0 else 'negative'} correlation\n")
    else:
        print("No significant correlation\n")

def correlation_heatmap(df):
    df_encoded = df.copy()  # Work on a copy to prevent modifying the original
    df_encoded["Loan_Status"] = df_encoded["Loan_Status"].map({"N": 0, "Y": 1})
    df_encoded["Dependents"] = df_encoded["Dependents"].replace({"3+": 3}).astype(float)

    categorical_columns = df_encoded.select_dtypes(include=["object"]).columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

    plt.figure(figsize=(14, 15))
    sns.heatmap(df_encoded.corr(), annot=True, cmap=sns.cubehelix_palette(rot=0.5), fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Features", fontsize=16)
    plt.show()


def missing(df):
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    missing_rate = (df.isnull().sum() / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Rate (%)': missing_rate
    })
    # Sort the DataFrame by missing rate in descending order
    missing_info = missing_info.sort_values(by='Missing Rate (%)', ascending=False)
    print(missing_info)

def detect_outliers(df, columns):
    for column in columns:
        # Calculate outliers using IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        print(f"\nOutliers in {column}:")
        print(f"Number of outliers: {len(outliers)}")
        print(f"Outlier values:\n{outliers[column].values}")

# if __name__ == "__main__":
#     train_df = pd.read_csv("../data/train.csv")
#     test_df = pd.read_csv("../data/test.csv")
#     analyze_data(train_df)
#     analyze_target_variable(train_df)
#     categorical_features = ['Gender', 'Married', 'Self_Employed', 'Credit_History']
#     analyze_categorical_features(train_df, categorical_features)
#     ordinal_features = ['Dependents', 'Education', 'Property_Area']
#     analyze_ordinal_features(train_df, ordinal_features)
#     numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
#     analyze_numerical_features(train_df, numerical_features)
#     applicantIncome_by_education(train_df)
#     bivariate_stacked_bar(train_df, "Gender", "Gender vs Loan Approval")
#     bivariate_stacked_bar(train_df, "Education", "Education vs Loan Approval")
#     bivariate_stacked_bar(train_df, "Self_Employed", "Self_Employed vs Loan Approval")
#     bivariate_stacked_bar(train_df, "Married", "Marital StatusLoan vs Loan Approval")
#     bivariate_stacked_bar(train_df, "Credit_History", "Credit History vs Loan Approval")
#     correlation_heatmap(train_df.copy())
#     income_vs_loan_status(train_df)
#     scatter_correlation(train_df, "ApplicantIncome", "LoanAmount", "Applicant Income vs Loan Amount")
#     missing(train_df)
#     missing(test_df)
#     outlier_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
#     detect_outliers(train_df, outlier_columns)





