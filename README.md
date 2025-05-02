**Loan Eligibility Prediction System Report**

 

**Link for presentation:** https://youtu.be/kPAYT03o1PU 

 

**How to Build and Run the Code**

- **Clone the repository:**

git clone https://github.com/buChloeCLY/cs506-proj.git

cd loan-prediction

- **Install dependencies:**

make install

- **Run the tests:**

make test

- **Run the test**

make run

The output will display the analysis, model performance comparison, and will generate prediction file: Preprocessed data (data/train_preprocessed.csv) and Test predictions (data/test_predictions.csv).

 

**1. Project Introduction**

This project aims to build a predictive system for loan eligibility using financial and demographic data. By analyzing key features such as credit history, income, and employment status, we train and compare multiple machine learning models to determine the best-performing classifier. The system provides actionable insights to lending institutions and automates loan approval decisions. The main objectives are:

- **Loan Eligibility Prediction:** Binary classification (approved/rejected) based on applicant profiles.
- **Data Analysis & Visualization:** Explore relationships between features and loan approval.
- **Hypothesis Testing:** Propose and validate assumptions about factors influencing approvals.

The dataset includes 13 features describing 614 loan applicants: Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status.

 

**2. Project Logic**

- **Data Loading:** Load train and test datasets.
- **EDA and Visualization:** Perform exploratory data analysis to understand feature distributions and relationships. Then verify the proposed hypothesis.
- **Data Preprocessing:** Handle missing values, outliers, and feature engineering.
- **Feature Engineering:** Create new features. Encode categorical variables using label encoding. Normalize numerical features.
- **Modeling:** Train, Cross-Validation, tune and evaluate multiple machine learning models.
- **Prediction:** Use the best model to predict test set loan approvals.

 

**3. Data Analysis**

**3.1 Dataset Characteristics**

- **Initial Exploration:**

- - Dataset shape: (614, 13)
  - Variables: 12 features + 1 target (Loan_Status)
  - Categorical features: Gender, Married, Self_Employed, Credit_History, Loan_Status.
  - Ordinal features: Dependents, Education, Property_Area.
  - Numerical features: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term.

**3.2 Visualizations**

**Univariate Analysis:**

- **Target Variable (Loan_Status) Distribution**: A bar plot to observe the ratio of approved vs. rejected loans. The loan of 422 (68.7%) people out of 614 was approved.

- **Categorical Feature Distributions**: Bar plots with counts and proportions for: Gender, Married, Self_Employed, Credit_History, Dependents, Education, Property_Area

- **Numerical Feature Distributions**:

- - Histograms and Box plots for ApplicantIncome, CoapplicantIncome, LoanAmount and TotalIncome.
  - Logarithmic transformation applied to skewed features.

**Bivariate Analysis:**

- **Categorical / Numerical Features vs. Loan Status**: Stacked percentage bar chart of approval status distributed by different feature categories.
- **Scatterplot with regression line**: Applicant income positively correlates with approved loan amount.

**Multivariate Analysis:**

- **Correlation Heatmap**: Identified     relationships between features. The most relevant variables are (Applicantlcome LoanAmount) and (Credit_Cistory-LoanStatus).

**Model Insights** (Result analysis in Data Modeling)**:**

- Feature Coefficient for Logistic Regression.
- Feature Importance for Random Forest and XGBoost.
- Tree Structure for Decision Tree.
- Decision Boundary for SVM: 2D visualization for Credit_History and TotalIncome.
- Model Accuracy Comparison: Accuracy, Precision, Recall, F1 Score.

Interactive and dynamic visualization allows clicking, zooming in and out of plots and hovering to display accurate values.

**3.3 Hypothesis Testing**

Through data analysis, we validated the previously proposed hypothesis and found that:

- **The applicant's income does not affect loan approval**: The average income of those who have received loan approval     (5384) is not significantly different from that of those who have not received loan approval (5486). Considering the existence of individual outlier data, we divided the applicant's income into four equally wide groups. The loan approval rate for each group is approximately 60%, with no significant difference. It can be inferred that the applicant's income does not affect the opportunity for loan approval, which contradicts the hypothesis.
- **Applicants with a credit history are more likely to be approved for loans**: We analyzed the credit history and loan status, and the bar chart showed that out of 422 people who received loan approval, 378 had a credit history, accounting for 89.57%. And the p-value obtained from the credit history Chi-Square test is 0, confirming a significant relationship between credit history and loan status.
- **Married applicants are more likely to be approved for loans than unmarried applicants**: Analyzing marital and loan status, it was found that married applicants have a significantly higher approval rate (76.01%) compared to unmarried applicants (62.91%). And the p-value obtained from the marital status Chi-Square test is 0.0344, indicating the hypothesis is statistically validated.
- **The loan amount approved is positively correlated with the applicant’s income**: We calculated that the Pearson correlation between the applicant's income and loan amount was 0.5709, with a p-value of 0. The data shows a statistically significant, moderately strong positive correlation between the two features, which confirms the hypothesis.

 

**4. Data Processing & Feature Engineering**

To ensure high-quality input data for modeling, we applied the following preprocessing steps:

**4.1** **Handling Missing Values**

- Missingness report: Highest in Credit_History (about 8%); Other features <5% missing.
- Used **mode imputation** for categorical features: Credit_Sistory with mode, Missing values for Self_Employed, Dependents, Loan_Amount_Term, Gender, Married.
- Used **median imputation** for numerical feature Loan Amount. (The data distribution of LoanAmount indicates the presence of outliers in the loan amount, so mean imputation is not used because the mean is greatly affected by outliers)



**4.2 Feature Engineering**

- **Created TotalIncome** = ApplicantIncome + CoapplicantIncome. 
- **Categorized Income Levels**: Binned income into four equal-width bins: low, average, high, and very high.
- **Encoded Categorical Variables** using LabelEncoder.

- - e.g., Gender ["Male", "Female"] into numerical values ([0, 1]).
  - Special handling: '3+' → 3 in Dependents.

**4.3 Handling Outliers**

- **Log Transformation** applied to ApplicantIncome, CoapplicantIncome, LoanAmount and TotalIncome to reduce skewness.

- - Before/after plots showing normalized distributions.

**4.4 Data Normalization**

- **Standardized numerical features** (ApplicantIncome, CoapplicantIncome, LoanAmount, TotalIncome, Loan_Amount_Term) using StandardScaler. 

 

**5. Data Modeling**

**5.1 Model Selection**

- **Logistic Regression**: It is a simple, interpretable baseline model for binary classification problems. It assumes a linear relationship between the independent variables and the log-odds of the dependent variable, which makes it a good starting point for initial model comparisons.
- **Decision Tree**: It can capture non-linear relationships in the data without requiring much preprocessing. It is also highly interpretable, allowing clear visualization of decision-making paths, which is particularly useful for understanding feature importance.
- **Random Forest**: It builds upon Decision Trees by constructing an ensemble of trees, improving generalization through averaging. Random Forests reduce the risk of overfitting and can handle a large number of input variables efficiently, making them a powerful model for structured data.
- **XGBoost**: It is a gradient-boosted decision tree model. It has high performance and the ability to handle imbalanced datasets in structured data tasks, as well as built-in regularization to prevent overfitting. It is particularly good at handling class imbalance and complex interactions between variables.
- **Support Vector Machine**: It works well with clear margin separation in high-dimensional spaces, making it suitable for both linear and non-linear classification problems and can be very effective when the number of features is large relative to the number of samples.

**5.2 Model Training & Tuning**

- **80-20 Train-Test Split**: The dataset was split into training (80%) and validation (20%) sets.
- **Stratified K-Fold Cross-Validation (5 folds)**: Ensures that each fold maintains the same proportion of loan approvals and denials as the full dataset. 

- **Hyperparameters Tuning**:

- - Logistic Regression: C, penalty, solver, max_iter.
  - Decision Tree: max_depth, min_samples_split, min_samples_leaf, criterion.
  - Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf.
  - XGBoost: max_depth, learning_rate, n_estimators.
  - SVM: C, gamma, kernel.

**5.3 Model Evaluation** 

- **Accuracy**: measures the proportion of correct predictions (both approved and rejected loans).
- **Precision:** measures the proportion of correctly predicted approvals among all predicted approvals. High precision reduces the risk of approving bad loans, which is crucial for minimizing financial loss. 
- **Recall:** measures the proportion of actual approved loans that the model correctly identifies. High recall ensures fewer qualified applicants are wrongly rejected, improving customer satisfaction.
- **F1 Score:** balances the trade-off between precision and recall, and is especially useful when the dataset may be slightly imbalanced (i.e., the number of approvals and denials are not exactly 50/50). Here, the F1 Score serves as a good overall metric because it reflects both the cost of false positives (bad approvals) and false negatives (missed good applicants).

 

**6. Results &** **Analysis**

**6.1 Model Visualization & Insights**

- **Logistic Regression Coefficient Analysis:** It assigns coefficients to features that directly reflect the direction and relative strength of their influence on the loan approval outcome. In the results, Credit_History has the highest positive coefficient, confirming that applicants with a good credit history significantly increase the odds of loan approval. Married and TotalIncomeBin also show strong positive effects, suggesting lenders favor stable marital status and higher income brackets. On the negative side, Loan_Amount_Term has the largest negative coefficient, indicating that longer loan terms reduce approval likelihood, perhaps due to greater uncertainty in long-term repayments. Self_Employed and CoapplicantIncome have negative weights, indicating self-employed applicants or those relying on coapplicants might face stricter scrutiny. Surprisingly, Education negatively impacts approval, this could reflect dataset bias or confounding variables for example, graduates might have higher debt.
- **Decision Tree Structure Analysis:** It explains how decisions are made based on key financial attributes. The root node splits the dataset based on credit_history ≤ 0.5, reflecting its pivotal role in loan approval decisions. Entropy of 0.897 suggesting a relatively high level of uncertainty. The value distribution (13.4% rejections vs. 68.6% approvals) shows that among applicants at this node, a majority are approved. The second layer introduces TotalIncome as a critical feature. The left subtree has low entropy and a stark value imbalance, shows that nearly all applicants in this segment are denied loans. This reflects that applicants with both poor credit and very low income are deemed high-risk. At the third level, the tree introduces ApplicantIncome as the next criterion. This additional check refines predictions by separating applicants with similar total income but differing personal incomes, suggesting that individual earning capacity still contributes meaningfully beyond combined household income. The nodes of the last layer represent the final decision points of the model. The low entropy at most leaves reflects confidence in classification, and the values provide interpretable evidence supporting the model's decisions.
- **Random Forest Feature Importance Analysis:** It measures feature importance by how much each feature reduces impurity across all trees. Credit_History ranks highest again, showing its consistent predictive power across models. However, its ranking diverges afterward, emphasizing TotalIncome (unbinned) over binned income, suggesting raw income is more informative for tree-based splits. ApplicantIncome and LoanAmount rank highly, reinforcing that individual earnings and loan size directly affect decisions. Notably, Married and Education have low importance here, contrasting with logistic regression. This discrepancy arises because Random Forests capture non-linear interactions, for example, income and marriage combined might matter more than marriage alone, while logistic regression isolates linear effects.
- **XGBoost Feature Importance Analysis:** It uses gradient boosting to build a sequence of optimized trees, presents a more refined view of feature contribution. While Credit_History remains the top predictor consistent across all models, but XGBoost gives unexpected weight to Self_Employed and Education, ranking them second and third. This could indicate that self-employment status and education interact strongly with other features, for example, self-employed applicants might need higher incomes to qualify. TotalIncome remains important, though less dominant than in Random Forest, while Married and CoapplicantIncome gain prominence, hinting at nuanced relationships, for example, married applicants with coapplicants might be safer bets.
- **SVM Decision Boundary:** Two important features (Credit_Sistory and TotalIncome) were selected, and the decision boundary plot shows how SVM separates these two classes. The left area represents applicants without credit history, most of whom have not been approved. The right area shows that most applicants with credit history have been approved. And because Credit_Sistory only has two unique values, the points are stacked vertically. The SVM decision boundary is mainly divided between Credit_Sistory 0 and 1, and is less concerned with income.
- **Model Accuracy Comparison**: The bar chart shows the comparison between CV Accuracy (average accuracy during cross-validation), Best CV Accuracy (best cross-validation score achieved during hyperparameter tuning), and Validation Accuracy (accuracy on the held-out validation set). These metrics offer insight into the models’ generalization capabilities and robustness. The specific analysis is provided below.

**6.2 Model Performance**

| Model  Accuracy Comparison | CV Accuracy | Best CV Accuracy | Accuracy on Validation Set |
| -------------------------- | ----------- | ---------------- | -------------------------- |
| Logistic Regression        | 0.798289    | 0.798371         | 0.853659                   |
| SVM                        | 0.792208    | 0.798371         | 0.853659                   |
| Decision Tree              | 0.763678    | 0.786127         | 0.845528                   |
| Random Forest              | 0.777922    | 0.798392         | 0.845528                   |
| XGBoost                    | 0.743373    | 0.798371         | 0.845528                   |

- Logistic Regression and SVM emerged as the top performers, both achieving identical validation accuracy scores of 85.37%. This suggests that for this particular dataset, simpler linear models may generalize better to unseen data compared to more complex tree-based models. It also indicates that the feature engineering and data preprocessing steps effectively shaped the data for this model.

- The tree-based models Decision Tree, Random Forest, and XGBoost showed slightly lower performance, with validation accuracies clustering around 84.55%. Interestingly, Random Forest achieved a higher Best CV Accuracy (79.83%) compared to Decision Tree (78.61%) and XGBoost (79.84%), highlighting the strength of ensemble methods in reducing overfitting and variance. Despite this, these models did not translate their strong CV scores into superior validation performance, suggesting possible slight overfitting or sensitivity to data splits.

- The cross-validation (CV) accuracy scores, which measure average performance across different data splits, were generally lower than the validation accuracy, indicating that the models may benefit from further regularization or feature refinement to improve generalization. For instance, Logistic Regression's CV accuracy (79.83%) was about 5.5% lower than its validation accuracy, suggesting some degree of overfitting. However, the fact that all models achieved validation accuracies above 84% confirms that the project successfully met its goal of building a reliable loan eligibility predictor.

  

| Model  Performance Comparison | Accuracy  | Precision | Recall   | F1 Score |
| ----------------------------- | --------- | --------- | -------- | -------- |
| Logistic Regression           | 0.853659  | 0.831683  | 0.988235 | 0.903226 |
| SVM                           | 0. 853659 | 0.831683  | 0.988235 | 0.903226 |
| Random Forest                 | 0.845528  | 0.830000  | 0.976471 | 0.897297 |
| XGBoost                       | 0.845528  | 0.830000  | 0.976471 | 0.897297 |
| Decision Tree                 | 0.845528  | 0.836735  | 0.964706 | 0.896175 |

- Both Logistic Regression and SVM stand out with the highest Accuracy (85.37%), and also lead in Recall (98.82%), indicating that they are excellent at correctly identifying the positive class—i.e., applicants who should receive a loan. Their F1 Score (90.32%) is also the highest among all models, showing a strong balance between Precision (83.17%) and Recall. This balance is critical in financial prediction tasks, as it reflects both the ability to detect true positives and to limit false positives, which could lead to lending risk.
- Random Forest and XGBoost follow closely with Accuracy at 84.55% and F1 Scores of 89.73%, showing only marginally weaker performance than Logistic Regression and SVM. While they have slightly lower Recall (97.65%) and Precision (83.00%), these values still indicate strong performance and robust generalization. The Decision Tree, although yielding similar Accuracy (84.55%), shows a bit of imbalance with a slightly lower Recall (96.47%) and marginally higher Precision (83.67%), resulting in the lowest F1 Score among the group at 89.61%.

The reason why different models show similar performance could be:

- **Dominant Predictive Features:** The dataset likely contains the extremely strong predictive features (like Credit_History) that overshadow other variables.  When most models effectively leverage these dominant features, they converge to similar performance levels. This explains why even simple models like logistic regression achieve comparable results to more complex ones.
- **Data Preprocessing Effects:** Common preprocessing steps (like filling missing values in Credit_History) might have homogenized the input space. The feature engineering (like creating TotalIncome) may have provided sufficient signal that all models exploit effectively.
- **Evaluation Set Limitations:** The validation set might be too small or non-representative, causing all models to appear similarly capable. With only 20% of 614 samples (~123 cases), a few identical predictions can make models seem more alike than they truly are. The test set might reveal more differentiation in performance.

**6.3 Final Prediction**

**Logistic Regression** is the preferred final model for test set prediction due to: 

- **High and stable performance**: It performs consistently well across all metrics—Accuracy, Precision, Recall and F1 Score.
- **Simplicity and interpretability**: Coefficients in Logistic Regression directly show feature importance. It is easier to explain to stakeholders, regulators, or decision-makers, which is valuable in finance where model transparency is often require.
- **Computational efficiency**: It is faster and lighter in computational cost compared to SVM, XGBoost, or Random Forest, making it more scalable for future deployment.

**Final Prediction File:** test_predictions.csv generated with final predictions, which includes two columns: Loan_ID from the test dataset and the predictions Loan_Status we made.

 

**7. Conclusion**

This project successfully developed a Loan Eligibility Prediction System capable of accurately classifying applicants based on financial and demographic data. Through comprehensive data preprocessing, including missing value imputation, log transformations, and feature engineering, the dataset was optimized for machine learning. Exploratory Data Analysis revealed key insights, such as the strong influence of Credit_History and Income on loan approvals, which were further validated through statistical testing.

Five machine learning models (Logistic Regression, SVM, Decision Tree, Random Forest, and XGBoost) were trained and evaluated. After rigorous hyperparameter tuning and cross-validation, Logistic Regression emerged as the best model, achieving 85.37% accuracy and a 90.32% F1-score, while maintaining high interpretability and computational efficiency. The final model was deployed to generate predictions (test_predictions.csv), and the demonstration system is ready for practical use.

