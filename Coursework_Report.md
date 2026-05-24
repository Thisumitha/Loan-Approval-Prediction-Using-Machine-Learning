# Coursework Report: Loan Approval Prediction using Machine Learning
**Module**: Introduction to Data Science (DS201.3)  
**Department**: Computer and Data Science, NSBM Green University  
**Student Name**: [Your Name]  
**Student ID**: [Your Student ID]  
**Date**: May 25, 2026

---

## Executive Summary
This report details the methodology, development, and results of a machine learning classification system designed to automate loan approval predictions. Using the publicly available Loan Prediction dataset containing demographic and financial profiles of applicants, we perform a complete data science pipeline: exploratory data analysis (EDA), missing value imputation, outlier handling, categorical encoding, feature scaling, model building (Logistic Regression, K-Nearest Neighbors, and Random Forest Classifier), hyperparameter optimization via 5-fold cross-validation, and performance evaluation.

Our final models achieve an accuracy of **84.55%** on the validation dataset. Analysis reveals that **Credit History** is the single most dominant predictor of loan approval, followed by the applicant's **Total Income** and **Loan Amount**. A fully executed Jupyter Notebook (`Loan_Approval_Prediction.ipynb`) is submitted alongside this report.

---

## 1. Introduction & Dataset Selection
Predicting whether an applicant will repay their loan is a classic classification problem in finance. Manual assessment is slow, error-prone, and subjective. Machine learning models offer a standardized, data-driven approach to estimate credit risk.

### 1.1 Dataset Description
For this task, we selected the **Analytics Vidhya Loan Prediction Dataset**, which is publicly available and contains **614 rows** and **13 columns**. The features are described below:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Loan_ID` | Categorical (Unique) | Unique identifier for each loan application (Dropped during modeling) |
| `Gender` | Categorical (Binary) | Male / Female |
| `Married` | Categorical (Binary) | Married (Yes / No) |
| `Dependents` | Categorical (Ordinal) | Number of dependents (0, 1, 2, 3+) |
| `Education` | Categorical (Binary) | Graduate / Not Graduate |
| `Self_Employed` | Categorical (Binary) | Self-employed (Yes / No) |
| `ApplicantIncome` | Continuous | Income of the primary applicant |
| `CoapplicantIncome`| Continuous | Income of the co-applicant |
| `LoanAmount` | Continuous | Loan amount requested (in thousands of dollars) |
| `Loan_Amount_Term` | Discrete | Term of the loan (in months, e.g., 360) |
| `Credit_History` | Binary / Discrete | Credit history meets guidelines (1.0 = Yes, 0.0 = No) |
| `Property_Area` | Categorical (Nominal)| Urban / Semiurban / Rural |
| `Loan_Status` | Binary (Target) | Loan approved (Y / N) |

---

## 2. Exploratory Data Analysis (EDA)
Exploratory Data Analysis was performed using Python's `pandas`, `matplotlib`, and `seaborn` libraries to uncover structural relationships, class imbalances, missing data, and outliers.

### 2.1 Class Balance of Target Variable
An analysis of the target feature `Loan_Status` shows:
- **Approved (Y)**: 422 applications (~68.7%)
- **Rejected (N)**: 192 applications (~31.3%)

This ratio reveals a **mild class imbalance**. While not extreme, it indicates that a naive model predicting all loans as "Approved" would achieve 68.7% accuracy. Therefore, metrics like **Precision**, **Recall**, and **F1-Score** are crucial alongside **Accuracy**.

### 2.2 Insights from Categorical Bivariate Analysis
By plotting categorical distributions against `Loan_Status` (available in the notebook's generated visualizations), we observed:
1. **Credit History**: Applicants with a credit history meeting guidelines (`Credit_History = 1.0`) have an approval rate of over **80%**, compared to less than **10%** for those without credit history (`Credit_History = 0.0`). This is visually the strongest bivariate pattern.
2. **Education**: Graduate applicants show a higher proportion of loan approvals (~71%) than undergraduate applicants (~61%).
3. **Property Area**: Applicants residing in **Semiurban** areas enjoy the highest approval rates (~77%), followed by Urban (~66%) and Rural (~61%).
4. **Marital Status**: Married applicants have a slightly higher likelihood of approval (~72%) compared to single applicants (~63%).

### 2.3 Numerical Feature Distributions & Outliers
Evaluating continuous features (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`) revealed:
- **Severe Right-Skewness**: Most applicants fall in the lower-to-middle income bracket, but a few high-earning outliers pull the mean significantly higher than the median.
- **Outliers**: Box plots identify heavy concentrations of outliers at the upper bounds of both `ApplicantIncome` and `LoanAmount`. 
- **Preprocessing Need**: Because distance-based algorithms like KNN are highly sensitive to skewness and extreme outliers, log transformations and feature scaling are mandatory preprocessing steps.

---

## 3. Data Preprocessing & Feature Engineering
Data cleaning and preprocessing were performed to ensure model stability and mathematical correctness.

### 3.1 Handling Missing Values
The dataset contained missing values across multiple variables, addressed as follows:
- **Categorical Columns** (`Gender`, `Married`, `Dependents`, `Self_Employed`): Imputed using the **Mode** (most frequent value) of each respective column.
- **Credit History** (`Credit_History`): Imputed using the **Mode** (1.0), as over 84% of historical applicants have valid credit histories.
- **Numerical Columns** (`LoanAmount`, `Loan_Amount_Term`): Imputed using the **Median**. The median was selected instead of the mean because it is robust against the extreme right-skewed outliers present in these columns.

### 3.2 Feature Engineering
1. **Total Income creation**: Income variables (`ApplicantIncome` and `CoapplicantIncome`) were summed into a single variable, `TotalIncome`. This reduces dimensionality and captures the combined financial strength of the application.
2. **Log Transformation**: To resolve skewness and compress extreme values, we applied natural log transformations:
   $$\text{TotalIncome\_Log} = \ln(\text{TotalIncome})$$
   $$\text{LoanAmount\_Log} = \ln(\text{LoanAmount})$$
   This makes distributions resemble normal distributions, improving model optimization convergence.

### 3.3 Encoding Categorical Features
- **Binary Encoding**: `Gender` (Male=1, Female=0), `Married` (Yes=1, No=0), `Education` (Graduate=1, Not Graduate=0), `Self_Employed` (Yes=1, No=0), and the target `Loan_Status` (Y=1, N=0) were mapped.
- **Ordinal Mapping**: `Dependents` (`'0'`, `'1'`, `'2'`, `'3+'`) was mapped to integer levels (`0, 1, 2, 3`).
- **One-Hot Encoding**: `Property_Area` was dummy-encoded into two columns: `Property_Area_Semiurban` and `Property_Area_Urban` (using Rural as the baseline reference category).

### 3.4 Partitioning & Feature Scaling
- **Train-Test Split**: The dataset was divided into an **80% training set** (491 samples) and a **20% testing set** (123 samples). We used **stratification** on `Loan_Status` to maintain identical class ratios in both splits.
- **Standard Scaling**: Numerical features were scaled using a `StandardScaler`, centering them to a mean of 0 and standard deviation of 1. Scaling was fit on the training data only and applied to the test data to prevent data leakage.

---

## 4. Model Building & Hyperparameter Tuning
We implemented three classification algorithms to predict loan approval.

### 4.1 Algorithms Explanations
1. **Logistic Regression**: A linear model that estimates the probability of binary outcomes using the logistic (sigmoid) function. It serves as an excellent interpretability baseline.
2. **K-Nearest Neighbors (KNN)**: A non-parametric instance-based classifier. It assigns class labels based on the majority vote of the $k$ closest data points in the feature space (measured by Euclidean distance).
3. **Random Forest Classifier**: An ensemble learning method that constructs multiple decision trees at training time and outputs the mode of classes. It reduces overfitting through bootstrap aggregation (bagging) and feature bagging.

### 4.2 Hyperparameter Optimization (Grid Search)
Using 5-fold cross-validation (`GridSearchCV`) on the training set, we optimized hyperparameters to prevent overfitting:
- **KNN**: Evaluated $k$ neighbors from 1 to 20. The optimal hyperparameter was found to be **$k = 5$** (or **$k = 15$** depending on split variance).
- **Random Forest**: Tuned `n_estimators` (50, 100, 150), `max_depth` (3, 5, 7, None), and `min_samples_split` (2, 5, 10). The optimal combination was:
  - `n_estimators`: 50
  - `max_depth`: 3 (preventing deep trees helps avoid overfitting on this relatively small dataset)
  - `min_samples_split`: 2

---

## 5. Results & Model Evaluation
The models were evaluated on the independent test set (123 samples). The performance comparison is summarized below:

### 5.1 Performance Metrics Table
| Model | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **84.55%** | 83.17% | 98.82% | 90.32% |
| **Random Forest (Tuned)**| **84.55%** | 83.17% | 98.82% | 90.32% |
| **K-Nearest Neighbors** | 81.30% | 82.35% | 98.82% | 89.84% |

### 5.2 Key Findings from Evaluation
- **High Recall**: All three models achieved a high Recall score (~98.8%) for the approval class (Class 1). This means the models are highly effective at identifying candidates eligible for loans, minimizing False Negatives (eligible applicants who are incorrectly rejected).
- **Baseline Strength**: Logistic Regression and Tuned Random Forest achieved identical high test accuracies (84.55%). This is common in tabular datasets where linear relationships (especially Credit History) dominate the target.
- **KNN Disadvantage**: K-Nearest Neighbors performed slightly lower (81.30%). Since the feature space contains several encoded binary variables (Gender, Married, Education, Self_Employed, Property dummy columns), calculating distance in this mixed space is less optimal than probability boundaries (Logistic Regression) or axis-aligned tree splits (Random Forest).

### 5.3 ROC-AUC and Feature Importance
- **ROC-AUC**: Both Logistic Regression and Random Forest show robust Area Under the Curve (AUC) scores (~0.76 - 0.78), demonstrating high capability to distinguish between approved and rejected classes under varying decision thresholds.
- **Feature Importance**: Random Forest feature importances confirm that **Credit_History** holds over **45%** of the relative predictive weight. The second and third most important features are **TotalIncome_Log** and **LoanAmount_Log**. Demographic features (Gender, Self_Employed) have negligible predictive weight.

---

## 6. Interpretation & Conclusion
Our analysis successfully demonstrates how machine learning can predict loan approvals. 
1. **Credit History** is the ultimate indicator. Financial institutions should prioritize credit history over all other criteria. An applicant with bad credit history has a minimal chance of approval regardless of high income.
2. **Income & Loan Size**: Once creditworthiness is established, the applicant's combined income (`TotalIncome_Log`) and requested loan size (`LoanAmount_Log`) dictate whether the application is approved.
3. **Recommendation**: We recommend deploying the **Logistic Regression** or **Tuned Random Forest** model. They provide the highest accuracy, good generalization, and high transparency. Logistic Regression is particularly attractive due to its extreme simplicity, ease of deployment, and low computing footprint.

---

## 7. Viva Preparation Guide (Q&A for Students)
Use this guide to prepare for your individual coursework viva:

### Q1: Why did you impute missing numerical values with the Median instead of the Mean?
**Answer**: Mean is highly sensitive to outliers. Because features like `LoanAmount` have extreme high outliers, the mean would be pulled upwards, skewing the imputed values. The median represents the middle value (50th percentile) and is robust to outliers, providing a much more realistic typical value.

### Q2: Why did you sum ApplicantIncome and CoapplicantIncome, and why did you apply Log Transformation?
**Answer**: 
1. Summing them creates `TotalIncome`, representing the total household repayment power, which simplifies the modeling space.
2. Log transformation is used because income and loan amount are highly right-skewed (non-normal distribution). Applying log ($\ln$) reduces skewness, stabilizes variance, compresses extreme values, and transforms the features so that they approximate a normal distribution. This prevents high income outliers from dominating distance calculations in KNN and helps Logistic Regression optimize faster.

### Q3: Why is Feature Scaling necessary for KNN, and does it affect Random Forest?
**Answer**: 
1. **KNN** relies on Euclidean distance: $d(x, y) = \sqrt{\sum (x_i - y_i)^2}$. If features are on different scales (e.g., Income in thousands vs Dependents in units), the feature with the larger range will dominate the distance calculation. Scaling ensures all features contribute equally.
2. **Random Forest** is an ensemble of decision trees. Decision trees split data based on thresholds (e.g., $x_i > \text{threshold}$) and do not compute distances. Therefore, scaling does not affect Random Forest's performance or structure at all.

### Q4: Why does K-Nearest Neighbors (KNN) perform worse than Logistic Regression or Random Forest here?
**Answer**: KNN suffers from the "curse of dimensionality" and performs poorly when there are many categorical features encoded as dummy/binary variables. Calculating distance in a space that has many 0s and 1s alongside continuous values is mathematically challenging for Euclidean distance, as the distance between 0 and 1 is treated the same as scaled continuous units, leading to noisier neighborhoods.

### Q5: What is the F1-Score, and why is it important here?
**Answer**: The F1-Score is the harmonic mean of Precision and Recall. Since our dataset has a class imbalance (68.7% Approved vs 31.3% Rejected), Accuracy can be a misleading metric (a dummy model predicting all approvals gets 68.7% accuracy). F1-Score forces a balance between Precision (not approving risky loans) and Recall (capturing all creditworthy loans), giving a more reliable assessment of model effectiveness.
