# Fraud Detection for E-commerce and Bank Transactions

## Project Overview

This project develops fraud detection solutions for:

1. E-commerce transactions (`Fraud_Data.csv`)
2. Bank credit card transactions (`creditcard.csv`)

The objective is to identify fraudulent transactions while minimizing false positives and false negatives through data preprocessing, feature engineering, class imbalance handling, and fraud pattern analysis.

---

## Business Problem

Fraudulent transactions result in significant financial losses and reduced customer trust.

Both datasets contain highly imbalanced classes, where fraudulent transactions represent only a small fraction of all observations. As a result, traditional accuracy metrics can be misleading. This project focuses on preparing high-quality datasets for machine learning-based fraud detection.

---

## Project Structure

```text
fraud-detection/

├── .github/
│   └── workflows/
│       └── unittests.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   └── modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   └── eda.py
│
├── scripts/
│   └── run_preprocessing.py
│
├── tests/
│
├── models/
│
├── requirements.txt
│
└── README.md
```

---

## Task 1: Data Analysis and Preprocessing

### Data Cleaning

Performed the following preprocessing steps:

* Removed duplicate records
* Handled missing values
* Converted datetime fields to proper datetime format
* Validated data types

---

### Exploratory Data Analysis (EDA)

Conducted exploratory analysis on both datasets:

#### Fraud_Data.csv

* Class distribution analysis
* Purchase value distribution
* Age distribution
* Fraud rate by hour of day
* Fraud rate by day of week
* Fraud rate by country
* Correlation analysis
* Feature-target relationship analysis

#### creditcard.csv

* Class imbalance analysis
* Transaction amount distribution
* Correlation heatmap
* Fraud vs non-fraud comparison

---

### Geolocation Integration

Merged transaction data with country information by:

* Converting IP addresses to integer format
* Performing range-based IP-to-country mapping
* Enriching transactions with geographic information

This enabled country-level fraud analysis.

---

### Feature Engineering

The following fraud detection features were created:

| Feature                   | Description                              |
| ------------------------- | ---------------------------------------- |
| time_since_signup         | Time elapsed between signup and purchase |
| hour_of_day               | Hour when transaction occurred           |
| day_of_week               | Day of transaction                       |
| transaction_count         | Number of transactions by user           |
| device_transaction_count  | Number of transactions per device        |
| time_between_transactions | Time elapsed since previous transaction  |

---

### Data Transformation

Applied:

* One-hot encoding for categorical variables
* Standard scaling for numerical features

---

### Class Imbalance Handling

The fraud dataset exhibited significant class imbalance.

To address this:

* Train-test split was performed before resampling
* SMOTE (Synthetic Minority Oversampling Technique) was applied only to the training set
* Class distributions were documented before and after resampling
* Information leakage was prevented by leaving the test set untouched

---

## Processed Datasets

Generated datasets include:

```text
data/processed/

├── fraud_merged.csv
├── fraud_processed.csv
├── fraud_train_smote.csv
├── fraud_test.csv
└── creditcard_processed.csv
```
## Task 2: Model Building and Training

### Objective

Build, train, evaluate, and compare machine learning models for fraud detection on highly imbalanced datasets.

---

### Train-Test Strategy

A stratified train-test split was used to preserve the original fraud class distribution.

For the Fraud_Data dataset:

* Training set was balanced using SMOTE
* Test set remained untouched to avoid information leakage

---

### Models Evaluated

#### Logistic Regression

Used as the baseline model because:

* Highly interpretable
* Fast training time
* Strong baseline for binary classification

#### Random Forest

Used as the ensemble model because:

* Handles nonlinear relationships
* Robust to noisy features
* Effective for fraud detection problems

Hyperparameter tuning was performed using GridSearchCV.

Parameters explored included:

* n_estimators
* max_depth
* min_samples_split

---

### Evaluation Metrics

Because fraud detection is a highly imbalanced classification problem, accuracy was not used as the primary metric.

Models were evaluated using:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Area Under Precision-Recall Curve (AUC-PR)
* Confusion Matrix

AUC-PR was treated as the primary evaluation metric because it better reflects performance on rare fraud cases.

---

### Cross Validation

Stratified 5-Fold Cross Validation was used to obtain reliable performance estimates.

Reported statistics include:

* Mean F1 Score
* Standard Deviation of F1 Score
* Mean AUC-PR
* Standard Deviation of AUC-PR

---

### Model Comparison

The following models were compared:

| Model               | Precision                 | Recall                    | F1                        | ROC-AUC                   | AUC-PR                    |
| ------------------- | ------------------------- | ------------------------- | ------------------------- | ------------------------- | ------------------------- |
| Logistic Regression | Generated during training | Generated during training | Generated during training | Generated during training | Generated during training |
| Random Forest       | Generated during training | Generated during training | Generated during training | Generated during training | Generated during training |

Complete results are stored in:

```text
models/model_comparison.csv
```

---

### Model Selection

Logistic Regression served as the interpretable baseline model.

Random Forest achieved stronger fraud detection performance across evaluation metrics and cross-validation folds.

Since fraud detection prioritizes detection of minority fraud cases, AUC-PR was used as the primary model selection criterion.

The best-performing model was saved as:

```text
models/best_model.pkl
```

---

## Generated Model Artifacts

```text
models/

├── logistic_regression.pkl
├── random_forest.pkl
├── best_model.pkl
└── model_comparison.csv
```

---

## Running Model Training

To retrain models from scratch:

```bash
python scripts/train_models.py
```

Alternatively, open:

```text
notebooks/modeling.ipynb
```

and execute all cells.
## Task 3: Model Explainability and Business Insights

### Objective

The objective of Task 3 was to interpret the selected fraud detection model using SHAP (SHapley Additive exPlanations) and translate model behavior into actionable business recommendations.

Explainability is critical in fraud detection because financial institutions must understand why transactions are flagged as suspicious. Transparent models help fraud analysts investigate cases more effectively and support trust in automated decision-making systems.

---

### Explainability Approach

The best-performing model from Task 2, Random Forest, was selected for explainability analysis.

The following techniques were used:

#### Built-in Feature Importance

Random Forest feature importance scores were extracted to identify the most influential predictors.

A visualization of the top 10 features was generated.

Output:

```text
models/top10_feature_importance.png
```

---

#### SHAP Analysis

SHAP was used to understand both global model behavior and individual predictions.

Generated visualizations:

```text
models/shap_summary.png
models/shap_bar.png
models/force_tp.html
models/force_fp.html
models/force_fn.html
```

These plots provide insight into:

* Global feature importance
* Direction of feature impact
* Individual transaction-level explanations

---

### Top Fraud Drivers

The Random Forest feature importance analysis identified the following top fraud indicators:

| Feature                  | Importance |
| ------------------------ | ---------- |
| device_transaction_count | 0.405      |
| time_since_signup        | 0.205      |
| ip_address               | 0.042      |
| age                      | 0.041      |
| purchase_value           | 0.038      |

Key findings:

* Device behavior was the strongest fraud signal.
* Fraudulent transactions often occurred shortly after account creation.
* Transaction characteristics such as purchase amount and IP-related features contributed significantly to model decisions.

---

### SHAP Interpretation

#### Global Feature Importance

The SHAP summary plot confirmed that the model primarily relies on:

* Device transaction frequency
* Time since signup
* Purchase value
* IP-related features
* Transaction timing information

The SHAP rankings closely aligned with Random Forest feature importance, increasing confidence in the model's decision-making process.

---

#### True Positive Analysis

A correctly identified fraud case was analyzed using a SHAP force plot.

Key observations:

* High device transaction activity strongly pushed the prediction toward fraud.
* Short account age increased fraud probability.
* Multiple fraud indicators contributed simultaneously to the final prediction.

---

#### False Positive Analysis

A legitimate transaction incorrectly flagged as fraud was examined.

Key observations:

* The transaction displayed several fraud-like characteristics.
* Device and timing features contributed heavily to the fraud prediction.
* This suggests opportunities to reduce customer friction through threshold optimization.

---

#### False Negative Analysis

A missed fraud case was analyzed.

Key observations:

* The transaction resembled legitimate customer behavior.
* Fraud indicators were weaker than in correctly detected fraud cases.
* Additional behavioral features may improve recall.

---

### Comparison of Feature Importance Methods

Both Random Forest feature importance and SHAP identified similar high-impact predictors.

Advantages of SHAP:

* Explains individual predictions.
* Indicates whether a feature increases or decreases fraud risk.
* Provides greater transparency than traditional feature importance alone.

This makes SHAP particularly valuable for fraud investigation and regulatory reporting.

---

### Business Recommendations

Based on feature importance and SHAP analysis, the following recommendations are proposed.

#### 1. Additional Verification for New Accounts

Evidence:

* time_since_signup was the second most influential feature.

Recommendation:

Require additional authentication for purchases occurring shortly after account creation.

Expected Impact:

Reduce account creation fraud and synthetic identity attacks.

---

#### 2. Device-Based Fraud Monitoring

Evidence:

* device_transaction_count was the most important fraud indicator.

Recommendation:

Flag devices associated with unusually high transaction volumes or multiple accounts.

Expected Impact:

Improve detection of coordinated fraud activity.

---

#### 3. Risk-Based Review for High-Value Transactions

Evidence:

* purchase_value consistently contributed to fraud predictions.

Recommendation:

Apply additional fraud screening for large purchases.

Expected Impact:

Reduce financial losses from high-risk transactions.

---

#### 4. Time-Based Monitoring Rules

Evidence:

* hour_of_day and day_of_week influenced model decisions.

Recommendation:

Implement enhanced monitoring during high-risk transaction periods.

Expected Impact:

Improve fraud detection efficiency and analyst resource allocation.

---

### Explainability Artifacts

Generated explainability outputs:

```text
models/

├── top10_feature_importance.png
├── shap_summary.png
├── shap_bar.png
├── force_tp.html
├── force_fp.html
└── force_fn.html
```

---

### Conclusion

Random Forest was selected as the final fraud detection model because it significantly outperformed Logistic Regression across all evaluation metrics.

SHAP analysis demonstrated that fraud predictions are primarily driven by:

* Device transaction activity
* Account age
* Transaction value
* IP-related signals

The explainability analysis provided actionable business insights that can be used to strengthen fraud prevention strategies while minimizing unnecessary customer disruption.
