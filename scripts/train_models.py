import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# ==================================================
# CONFIG
# ==================================================

DATA_PATH = "../data/processed/creditcard_processed.csv"
# DATA_PATH = "../data/processed/fraud_processed.csv"

RANDOM_STATE = 42

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

TARGET = "Class" if "Class" in df.columns else "class"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("=" * 60)
print("Dataset Shape:", df.shape)
print("Target:", TARGET)
print("Fraud Rate:", round(y.mean(), 5))
print("=" * 60)

# ==================================================
# STRATIFIED TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTrain Class Distribution")
print(y_train.value_counts(normalize=True))

print("\nTest Class Distribution")
print(y_test.value_counts(normalize=True))

# ==================================================
# LOGISTIC REGRESSION BASELINE
# ==================================================

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )
])

logistic_model.fit(X_train, y_train)

log_pred = logistic_model.predict(X_test)
log_prob = logistic_model.predict_proba(X_test)[:, 1]

log_results = {
    "Model": "Logistic Regression",
    "Precision": precision_score(y_test, log_pred),
    "Recall": recall_score(y_test, log_pred),
    "F1": f1_score(y_test, log_pred),
    "ROC_AUC": roc_auc_score(y_test, log_prob),
    "AUC_PR": average_precision_score(y_test, log_prob)
}

print("\nLOGISTIC REGRESSION")
print(classification_report(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))

# ==================================================
# RANDOM FOREST + TUNING
# ==================================================

rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="average_precision",
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("\nBest RF Parameters")
print(grid_search.best_params_)

rf_pred = best_rf.predict(X_test)
rf_prob = best_rf.predict_proba(X_test)[:, 1]

rf_results = {
    "Model": "Random Forest",
    "Precision": precision_score(y_test, rf_pred),
    "Recall": recall_score(y_test, rf_pred),
    "F1": f1_score(y_test, rf_pred),
    "ROC_AUC": roc_auc_score(y_test, rf_prob),
    "AUC_PR": average_precision_score(y_test, rf_prob)
}

print("\nRANDOM FOREST")
print(classification_report(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))

# ==================================================
# STRATIFIED K-FOLD CROSS VALIDATION
# ==================================================

print("\nRunning 5-Fold Stratified Cross Validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "f1": "f1",
    "aucpr": "average_precision"
}

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": best_rf
}

cv_results = []

for model_name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_results.append({
        "Model": model_name,
        "CV_F1_Mean": scores["test_f1"].mean(),
        "CV_F1_Std": scores["test_f1"].std(),
        "CV_AUC_PR_Mean": scores["test_aucpr"].mean(),
        "CV_AUC_PR_Std": scores["test_aucpr"].std()
    })

cv_df = pd.DataFrame(cv_results)

# ==================================================
# MODEL COMPARISON
# ==================================================

metrics_df = pd.DataFrame([
    log_results,
    rf_results
])

comparison = metrics_df.merge(
    cv_df,
    on="Model"
)

print("\nMODEL COMPARISON")
print(comparison)

# ==================================================
# MODEL SELECTION
# ==================================================

best_model_name = comparison.sort_values(
    "AUC_PR",
    ascending=False
).iloc[0]["Model"]

if best_model_name == "Random Forest":
    best_model = best_rf
else:
    best_model = logistic_model

print("\nSELECTED MODEL:", best_model_name)

# ==================================================
# SAVE ARTIFACTS
# ==================================================

joblib.dump(
    logistic_model,
    "../models/logistic_regression.pkl"
)

joblib.dump(
    best_rf,
    "../models/random_forest.pkl"
)

joblib.dump(
    best_model,
    "../models/best_model.pkl"
)

comparison.to_csv(
    "../models/model_comparison.csv",
    index=False
)

print("\nArtifacts saved in models/")
print("Task 2 completed successfully.")