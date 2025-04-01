**Loan Eligibility and Loan Amount Prediction** 

**Link for presentation:** https://youtu.be/HDyQraQMVGg

**1. Project Overview**

This project focuses on predicting loan eligibility based on customer financial records and demographic details. The main objectives are:

- **Loan Eligibility Prediction (Classification)**: Determine whether a loan should be approved (Loan_Status).
- **Loan Amount Estimation (Regression - Future Work)**: Predict the loan amount that can be safely approved.

The dataset includes 13 features describing 614 loan applicants: Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status. So far, we have completed data analysis, hypothesis validation, data processing, feature engineering, and training and evaluation of logistic regression models.

 

**2. Data Analysis**

**2.1 Dataset Characteristics**

- **Initial Exploration:**

- - Dataset shape: (614, 13)
  - Variables: 12 features + 1 target (Loan_Status)
  - Categorical features: Gender, Married, Self_Employed, Credit_History, Loan_Status.
  - Ordinal features: Dependents, Education, Property_Area.
  - Numerical features: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term.

**2.2 Preliminary Visualizations**

**Univariate Analysis:**

- **Target Variable (Loan_Status) Distribution**: A bar plot to observe the ratio of approved vs. rejected     loans. The loan of 422 (68.7%) people out of 614 was approved.

- **Categorical Feature Distributions**:     Bar plots with counts and proportions for: Gender, Married, Self_Employed, Credit_History, Dependents, Education, Property_Area

- **Numerical Feature Distributions**:

- - Histograms and Box plots for ApplicantIncome, CoapplicantIncome, LoanAmount and TotalIncome.
  - Logarithmic transformation applied to skewed features.

**Bivariate Analysis:**

- **Categorical / Numerical Features vs. Loan Status**: Stacked percentage bar chart of approval status distributed by different feature categories.
- **Scatterplot with regression line**: Applicant's income positively correlates with approved loan amount.

**Multivariate Analysis:**

- **Correlation Heatmap**: Identified relationships between features. The most relevant variables are     (Applicantlcome-LoanAmount) and (Credit_History-LoanStatus).

**2.3 Hypothesis Testing**

Through data analysis, we validated the previously proposed hypothesis and found that:

- Applicants with a credit history are more likely to be approved for loans.
- Married applicants are more likely to be approved for loans  than unmarried applicants. 
- The loan amount approved is positively correlated with the applicant’s income. 
- The applicant's income does not affect loan approval, which is contrary to hypothesis.

 

**3. Data Processing**

To ensure high-quality input data for modeling, we applied the following preprocessing steps:

**3.1** **Handling Missing Values**

- Missingness report: Highest in Credit_History (about 8%); Other features <5% missing.
- Used **mode imputation** for categorical features: Credit_History  with mode, Missing values for Self_Employed, Dependents, Loan_Amount_Term, Gender, Married.
- Used **median imputation** for numerical feature Loan     Amount. (The data distribution of LoanAmount indicates the presence of outliers in the loan amount, so mean imputation is not used because the mean is greatly affected by outliers)

**3.2 Feature Engineering**

- **Created TotalIncome** = ApplicantIncome + CoapplicantIncome.
- **Categorized Income Levels**: Binned income into four equal-width bins: low, average, high, and very high.
- **Encoded Categorical Variables** using LabelEncoder.

- - e.g., Gender ["Male", "Female"] into numerical values ([0, 1]).
  - Special handling: '3+' → 3 in Dependents.

**3.3 Handling Outliers**

- **Log Transformation** applied to ApplicantIncome, CoapplicantIncome, LoanAmount and TotalIncome to reduce skewness.

- - Before/after plots showing normalized distributions.

**3.4 Data Normalization**

- **Standardized numerical features** (ApplicantIncome, CoapplicantIncome, LoanAmount, TotalIncome, Loan_Amount_Term) using StandardScaler.

- 

**4. Data Modeling**

**4.1 Model Selection**

We implemented a **logistic regression model** as the baseline based on its interpretability and fast training. Future improvements will include Random Forest, Decision Tree, and SVM.

**4.2 Model Training**

- **80-20 Train-Test Split**: The dataset was split into training (80%) and validation (20%) sets.
- **Stratified K-Fold Cross-Validation (5 folds)**: Ensures that each fold maintains the same proportion of loan approvals and denials as the full dataset. 

- **Logistic Regression Hyperparameters**: max_iter=2000, random_state=42
- **Feature Scaling Applied** before training.

**4.3 Model Evaluation** 

- **Cross-Validation Accuracy (Logistic Regression)**: Achieved 79.83%.

- **Model Evaluation on Validation Set**:

- - **Accuracy**: 86.18%
  - **Precision**: 84%
  - **Recall**: 98.82%
  - **F1-Score**: 90.81%

 

**5. Preliminary Results**

**5.1 Strong Predictors:**

- Credit_History (p<0.001)
- ApplicantIncome vs LoanAmount: positive correlation (Pearson Correlation=0.5709)

**5.2 Surprising Insights:**

- Applicant income does not affect the chances of loan approval.
- Self-Employed status not significant (p=1.00)

**5.3 Limitations:**

- Complexity and accuracy of the model needs improvement.
- Propose new features that may affect the target variable and explore them.
- Perform interactive visualization.
