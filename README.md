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

### Dataset Descriptions

| File                     | Description                             |
| ------------------------ | --------------------------------------- |
| fraud_merged.csv         | Feature-engineered dataset used for EDA |
| fraud_processed.csv      | Encoded and scaled fraud dataset        |
| fraud_train_smote.csv    | Training dataset after SMOTE            |
| fraud_test.csv           | Holdout test dataset                    |
| creditcard_processed.csv | Cleaned and scaled credit card dataset  |

---

## How to Run

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Preprocessing Pipeline

```bash
python3 -m scripts.run_preprocessing
```

### Launch Notebooks

```bash
jupyter notebook
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Jupyter Notebook

---

## Status

✅ Task 1 Completed

Completed deliverables:

* Data cleaning
* Exploratory data analysis
* Geolocation integration
* Feature engineering
* Data transformation
* Class imbalance handling with SMOTE
* Processed datasets ready for machine learning modeling
