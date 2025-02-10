# **Project Proposal: Loan Eligibility and Loan Amount Prediction**

**Project Description:**  
Predicting credit risk is one of the important applications of data science in the financial industry. Lending institutions face the challenge of balancing risk and profitability: approving loans for creditworthy customers while minimizing defaults. This project aims to build a predictive system for loan eligibility and loan amount estimation. The project is intended to assist lending institutions in making data-driven decisions by predicting two key outcomes:
1. **Loan Eligibility:** Determine whether a customer should be approved for a loan based on their financial and demographic profile.  
2. **Loan Amount:** Estimate the loan amount that can be safely approved for eligible customers.

The project will involve analyzing a dataset from Kaggle containing detailed information about loan applicants, including their income, credit history, loan amount, and other relevant features. The project will involve data collection, data cleaning, feature extraction, model training, and data visualization to gain insights into credit risk factors.

**Project Goals:**  
By analyzing financial and demographic factors, building and comparing multiple machine learning models, we aim to identify the best-performing model for each task and provide actionable insights to lending institutions. The primary objectives of this project are:
1. **Loan Eligibility Prediction**: Build a classification model to predict whether a customer is eligible for a loan.  
2. **Loan Amount Estimation**: Build a regression model to estimate the loan amount that can be safely approved for eligible customers.  
3. **Data Visualization & Insights**: Generate meaningful visualizations to explore relationships between features and their impact on loan eligibility and loan amount.  
4. **Hypothesis Testing**: Propose and validate hypotheses related to loan approvals, such as the impact of credit history, income levels, and employment status on approval.

**Data Collection:**  
The dataset used in this project is the Loan Prediction Dataset from Kaggle ([https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?resource=download\&select=train\_u6lujuX\_CVtuZ9i.csv](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?resource=download&select=train_u6lujuX_CVtuZ9i.csv)). This dataset includes 13 features describing 641 loan applicants:
* **Demographic Information**: Gender, Marital Status, Number of Dependents, Education, Employment Status.  
* **Financial Information**: Applicant Income, Co-applicant Income, Credit History.  
* **Loan Details**: Loan Amount, Loan Term, Property Area, Loan Status (Approved/Not Approved).
The dataset is already pre-collected, so no external data collection is required. However, data cleaning and preprocessing will be performed to handle missing values, outliers, and categorical encoding.

**Modeling Approach:**  
We will build and compare multiple models for classification and regression tasks to identify the best-performing model.
1. **Loan Eligibility Prediction (Classification):**  
* Algorithms to Compare:  
  * Logistic Regression  
  * Random Forest   
  * K-Nearest Neighbors  
  * Decision Tree   
  * Support Vector Machines  
* Evaluation Metrics: Accuracy, Precision, Recall, F1-Score.  

2. **Loan Amount Prediction (Regression):**  
* Algorithms to Compare:  
  * Linear Regression  
  * Random Forest  
  * K-Nearest Neighbors  
  * Decision Trees  
  * Support Vector Machines  
* Evaluation Metrics: Mean Absolute Error, Mean Squared Error, R-squared.

**Feature Engineering:**
* Create new features like Total Income (Applicant Income \+ Co-applicant Income) and Loan-to-Income Ratio (Loan Amount / Total Income).  
* Encode categorical variables (e.g., Gender, Married, Education) using label encoding.  
* Normalize numerical features (e.g., Applicant Income, Loan Amount).

**Data Visualization:**  
We plan to include at least 5 key visualizations to explore the dataset and model results, such as:
* Univariate Analysis: Histogram of Applicant Income to understand income distribution.  
* Bivariate Analysis: Scatter plot of Applicant Income vs. Loan Amount to explore their relationship.  
* Multivariate Analysis: Heatmap of correlations between numerical features (e.g., Applicant Income, Loan Amount, Credit History).  
* Impact of Features on Loan Eligibility: Bar plot showing the approval rates for different credit history groups.  
* Hypothesis Testing: Test hypotheses like “Applicants with higher incomes are more likely to be approved for loans.”

**Test Plan:**
1. **Train-Test Split:** We will split the dataset into training (80%) and testing (20%) sets to evaluate model performance.  
2. **Cross-Validation:** Use k-fold cross-validation (e.g., k=5) to ensure the models generalize well to unseen data.  
3. **Evaluation Metrics:** Compare models based on the metrics mentioned above for both classification and regression tasks.

**Hypotheses to Test:**
* Hypothesis 1: Applicants with higher incomes are more likely to be approved for loans.

  Test: Compare approval rates across income brackets.
* Hypothesis 2: Applicants with a credit history are more likely to be approved for loans.

  Test: Compare approval rates for applicants with and without a credit history.
* Hypothesis 3: Married applicants are more likely to be approved for loans than unmarried applicants.

  Test: Compare approval rates by marital status.
* Hypothesis 4: The loan amount approved is positively correlated with the applicant’s income.

  Test: Use a scatter plot and correlation analysis to explore the relationship.
