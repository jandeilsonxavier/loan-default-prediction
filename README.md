# 📄 Project Documentation

**Project Title:**
Loan Approval Prediction Using Machine Learning Techniques

---

## 1. Introduction

Access to credit is an essential factor for economic growth, enabling individuals and companies to invest, purchase goods, and foster development. However, granting loans involves significant risks for financial institutions, especially regarding defaults.

This project aims to develop Machine Learning models to predict whether a customer is eligible for a loan, using historical customer data and their financial behavior.

The proposed solution can be applied by banks, fintechs, and credit unions to support more assertive decision-making, reducing risks and increasing efficiency in the approval process.

---

## 2. Objectives

**General Objective:**
Develop a high-performance predictive model to identify customers with a higher probability of default.

**Specific Objectives:**

* Perform Exploratory Data Analysis (EDA) to identify patterns and relationships.
* Handle missing and inconsistent data.
* Select and test different classification algorithms.
* Tune hyperparameters to optimize results.
* Evaluate models using appropriate metrics.

---

## 3. Tools Used

**Language:** Python  

**Environment:** Jupyter Notebook

**Libraries:**
* **Data Manipulation:** Pandas, Numpy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Preprocessing:** StandardScaler, LabelEncoder
* **Models:** LogisticRegression, RandomForestClassifier, SGDClassifier, ExtraTreesClassifier
* **Hyperparameter Optimization:** RandomizedSearchCV, GridSearchCV

---

## 4. Dataset

**Source:** Kaggle — Loan Default Prediction Dataset  
**Size:** (Insert number of rows and columns)  
**Target Variable:** loan_status (1 = eligible for loan, 0 = not eligible)  

**Features:**

| Column_name     | Column_type | Data_type | Description |
|-----------------|-------------|-----------|-------------|
| LoanID          | Identifier  | string    | A unique identifier for each loan. |
| Age             | Feature     | integer   | The age of the borrower. |
| Income          | Feature     | integer   | The annual income of the borrower. |
| LoanAmount      | Feature     | integer   | The amount of money being borrowed. |
| CreditScore     | Feature     | integer   | The credit score of the borrower, indicating their creditworthiness. |
| MonthsEmployed  | Feature     | integer   | The number of months the borrower has been employed. |
| NumCreditLines  | Feature     | integer   | The number of credit lines the borrower has open. |
| InterestRate    | Feature     | float     | The interest rate for the loan. |
| Loan Term       | Feature     | integer   | The term length of the loan in months. |
| DTIRatio        | Feature     | float     | The Debt-to-Income ratio, indicating the borrower's debt compared to their income. |
| Education       | Feature     | string    | The highest level of education attained by the borrower (PhD, Master's, Bachelor's, High School). |
| EmploymentType  | Feature     | string    | The type of employment status of the borrower (Full-time, Part-time, Self-employed, Unemployed). |
| MaritalStatus   | Feature     | string    | The marital status of the borrower (Single, Married, Divorced). |
| HasMortgage     | Feature     | string    | Whether the borrower has a mortgage (Yes or No). |
| HasDependents   | Feature     | string    | Whether the borrower has dependents (Yes or No). |
| LoanPurpose     | Feature     | string    | The purpose of the loan (Home, Auto, Education, Business, Other). |
| HasCoSigner     | Feature     | string    | Whether the loan has a co-signer (Yes or No). |
| Default         | Target      | integer   | The binary target variable indicating whether the loan defaulted (1) or not (0). |

**Notes:**

* The dataset shows imbalanced classes, with more customers not eligible for loans.
* Missing values were found in some variables.

---

## 5. Data Preprocessing

Steps performed:

1. **Missing value handling:** Replaced with mean, median, or removed depending on the column.
2. **Categorical encoding:** Applied Label Encoding.
3. **Normalization:** Applied StandardScaler to numerical variables.
4. **Train-test split:** 70% training / 30% testing using `train_test_split` with a fixed `random_state` for reproducibility.

---

## 6. Models Tested

The following algorithms were tested:

* Logistic Regression
* Stochastic Gradient Descent (SGD) with log and hinge loss
* Random Forest Classifier
* Extra Trees Classifier

---

## 7. Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC
* Precision-Recall Curve for threshold analysis

---

## 8. Results

**Example – Logistic Regression (Colab):**

* Accuracy: 0.8031
* Precision: 0.2467
* Recall: 0.3383
* F1-Score: 0.2853
* AUC-ROC: 0.6770

**Example – Logistic Regression (Local Jupyter):**

* Accuracy: 0.7311
* Precision: 0.2255
* Recall: 0.5405
* F1-Score: 0.3183
* AUC-ROC: 0.7123

**Note:** Differences between local and Colab results can be explained by variations in library versions, `random_state` configurations, or preprocessing differences.

---

## 9. Hyperparameter Tuning

RandomizedSearchCV was used to optimize hyperparameters for Random Forest and Extra Trees models, aiming to reduce search time and find promising parameter combinations.

---

## 10. Visualizations

Generated visual outputs:

* Confusion Matrix for each model.
* ROC Curves.
* Precision-Recall Curves highlighting the optimal threshold based on F1-score.

---

## 11. Conclusion

The project demonstrated that performance varies depending on the algorithm and parameter tuning.
Although Random Forest achieved a better balance between Recall and AUC-ROC, simpler models such as Logistic Regression still delivered competitive performance with higher interpretability.

---

## 12. Next Steps

* Explore feature engineering techniques.
* Test boosting-based models (XGBoost, LightGBM, CatBoost).
* Implement stratified cross-validation.
* Deploy the final model via API or Streamlit.

---
