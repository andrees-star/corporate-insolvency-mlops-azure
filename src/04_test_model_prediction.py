import sys
import joblib
import pandas as pd
from pathlib import Path

sys.path.append("src")

from custom_transformers import PercentileWinsorizer


MODEL_PATH = Path("models/logistic_regression_riesgo24.pkl")
DATA_PATH = Path("data/processed/df_processed.csv")


def main():
    print("Loading model artifact...")

    artifact = joblib.load(MODEL_PATH)

    pipeline = artifact["pipeline"]
    features = artifact["features"]
    threshold = artifact["threshold"]
    target = artifact["target"]

    print(f"Model target: {target}")
    print(f"Threshold: {threshold}")
    print(f"Number of features: {len(features)}")

    print("\nLoading processed data...")
    df = pd.read_csv(DATA_PATH)

    X_sample = df[features].head(1)

    print("\nSample input:")
    print(X_sample)

    probability = pipeline.predict_proba(X_sample)[:, 1][0]
    prediction = int(probability >= threshold)

    print("\nPrediction result:")
    print(f"Risk probability: {probability:.4f}")
    print(f"Risk class: {prediction}")

    if prediction == 1:
        print("Interpretation: The company is classified as risk_24 = 1.")
    else:
        print("Interpretation: The company is classified as risk_24 = 0.")


if __name__ == "__main__":
    main()