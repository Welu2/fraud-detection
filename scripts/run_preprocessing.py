from src.preprocess import FraudPreprocessor

processor = FraudPreprocessor(
    fraud_path="data/raw/Fraud_Data.csv",
    ip_path="data/raw/IpAddress_to_Country.csv",
    credit_path="data/raw/creditcard.csv"
)

processor.run()