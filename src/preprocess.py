import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


class FraudPreprocessor:

    def __init__(
        self,
        fraud_path,
        ip_path,
        credit_path
    ):

        self.fraud_path = fraud_path
        self.ip_path = ip_path
        self.credit_path = credit_path

    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------

    def load_data(self):

        fraud_df = pd.read_csv(self.fraud_path)
        ip_df = pd.read_csv(self.ip_path)
        credit_df = pd.read_csv(self.credit_path)

        return fraud_df, ip_df, credit_df

    # --------------------------------------------------
    # Fraud Cleaning
    # --------------------------------------------------

    def clean_fraud_data(self, fraud_df):

        fraud_df = fraud_df.copy()

        fraud_df.drop_duplicates(inplace=True)

        fraud_df["signup_time"] = pd.to_datetime(
            fraud_df["signup_time"],
            errors="coerce"
        )

        fraud_df["purchase_time"] = pd.to_datetime(
            fraud_df["purchase_time"],
            errors="coerce"
        )

        fraud_df = fraud_df.dropna(
            subset=[
                "signup_time",
                "purchase_time"
            ]
        )

        return fraud_df

    # --------------------------------------------------
    # Credit Cleaning
    # --------------------------------------------------

    def clean_credit_data(self, credit_df):

        credit_df = credit_df.copy()

        credit_df.drop_duplicates(inplace=True)

        credit_df.fillna(
            credit_df.median(numeric_only=True),
            inplace=True
        )

        return credit_df

    # --------------------------------------------------
    # Merge Country Data
    # --------------------------------------------------

    def merge_country(
        self,
        fraud_df,
        ip_df
    ):

        fraud_df = fraud_df.copy()
        ip_df = ip_df.copy()

        fraud_df["ip_address"] = (
            fraud_df["ip_address"]
            .astype(float)
            .astype(np.int64)
        )

        ip_df["lower_bound_ip_address"] = (
            ip_df["lower_bound_ip_address"]
            .astype(np.int64)
        )

        ip_df["upper_bound_ip_address"] = (
            ip_df["upper_bound_ip_address"]
            .astype(np.int64)
        )

        fraud_df = fraud_df.sort_values(
            "ip_address"
        )

        ip_df = ip_df.sort_values(
            "lower_bound_ip_address"
        )

        merged = pd.merge_asof(
            fraud_df,
            ip_df,
            left_on="ip_address",
            right_on="lower_bound_ip_address",
            direction="backward"
        )

        merged = merged[
            merged["ip_address"]
            <= merged["upper_bound_ip_address"]
        ]

        return merged

    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------

    def engineer_features(self, df):

        df = df.copy()

        df["signup_time"] = pd.to_datetime(
            df["signup_time"],
            errors="coerce"
        )

        df["purchase_time"] = pd.to_datetime(
            df["purchase_time"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "signup_time",
                "purchase_time"
            ]
        )

        # Time since signup

        df["time_since_signup"] = (
            df["purchase_time"]
            - df["signup_time"]
        ).dt.total_seconds()

        # Hour

        df["hour_of_day"] = (
            df["purchase_time"]
            .dt.hour
        )

        # Day of week

        df["day_of_week"] = (
            df["purchase_time"]
            .dt.dayofweek
        )

        # User transaction count

        df["transaction_count"] = (
            df.groupby("user_id")
            ["user_id"]
            .transform("count")
        )

        # Device transaction count

        df["device_transaction_count"] = (
            df.groupby("device_id")
            ["device_id"]
            .transform("count")
        )

        # Transaction velocity

        df = df.sort_values(
            ["user_id", "purchase_time"]
        )

        df["prev_purchase"] = (
            df.groupby("user_id")
            ["purchase_time"]
            .shift(1)
        )

        df["time_between_transactions"] = (
            df["purchase_time"]
            - df["prev_purchase"]
        ).dt.total_seconds()

        df["time_between_transactions"] = (
            df["time_between_transactions"]
            .fillna(0)
        )

        return df

    # --------------------------------------------------
    # Encoding
    # --------------------------------------------------

    def encode_features(self, df):

        categorical_cols = [
            "source",
            "browser",
            "sex",
            "country"
        ]

        df = pd.get_dummies(
            df,
            columns=categorical_cols,
            drop_first=True
        )

        return df

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------

    def scale_features(
        self,
        fraud_df,
        credit_df
    ):

        scaler = StandardScaler()

        fraud_cols = [
            "purchase_value",
            "age",
            "time_since_signup",
            "transaction_count",
            "device_transaction_count",
            "time_between_transactions"
        ]

        fraud_df[fraud_cols] = scaler.fit_transform(
            fraud_df[fraud_cols]
        )

        credit_df[["Amount"]] = scaler.fit_transform(
            credit_df[["Amount"]]
        )

        return fraud_df, credit_df

    # --------------------------------------------------
    # Save Outputs
    # --------------------------------------------------

    def save_processed(
        self,
        fraud_df,
        credit_df
    ):

        fraud_df.to_csv(
            "data/processed/fraud_processed.csv",
            index=False
        )

        credit_df.to_csv(
            "data/processed/creditcard_processed.csv",
            index=False
        )

    # --------------------------------------------------
    # Main Pipeline
    # --------------------------------------------------

    def run(self):

        fraud_df, ip_df, credit_df = (
            self.load_data()
        )

        # Cleaning

        fraud_df = self.clean_fraud_data(
            fraud_df
        )

        credit_df = self.clean_credit_data(
            credit_df
        )

        # Country merge

        fraud_df = self.merge_country(
            fraud_df,
            ip_df
        )

        # Feature engineering

        fraud_df = self.engineer_features(
            fraud_df
        )

        # Save EDA version

        fraud_df.to_csv(
            "data/processed/fraud_merged.csv",
            index=False
        )

        # Encoding

        fraud_df = self.encode_features(
            fraud_df
        )

        # Remove datetime columns before modeling

        fraud_df.drop(
            columns=[
                "signup_time",
                "purchase_time",
                "prev_purchase"
            ],
            errors="ignore",
            inplace=True
        )

        # Scaling

        fraud_df, credit_df = (
            self.scale_features(
                fraud_df,
                credit_df
            )
        )

        # Save

        self.save_processed(
            fraud_df,
            credit_df
        )

        print("Preprocessing Complete")