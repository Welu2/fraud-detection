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
