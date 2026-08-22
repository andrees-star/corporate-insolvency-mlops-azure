import sys
import joblib
import yaml
import numpy as np
import pandas as pd

from pathlib import Path

# Allow imports from src/
sys.path.append(str(Path(__file__).resolve().parent))

from custom_transformers import PercentileWinsorizer


CONFIG_PATH = Path("config/config.yml")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    print("Starting batch scoring...")

    # Load config
    config = load_config(CONFIG_PATH)

    model_path = Path(config["model"]["output_path"])

    # For now, we score the processed dataset.
    # Later we can change this to data/scoring/new_companies.xlsx
    input_path = Path(config["data"]["processed_path"])

    output_path = Path(
        config["outputs"].get("predictions_output", "outputs/predictions.csv")
    )

    print(f"Model path: {model_path}")
    print(f"Input data path: {input_path}")
    print(f"Output predictions path: {output_path}")

    # Validate files
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input data file not found: {input_path}")

    # Load model artifact
    artifact = joblib.load(model_path)

    pipeline = artifact["pipeline"]
    features = artifact["features"]
    threshold = artifact["threshold"]
    target = artifact["target"]

    print(f"Model target: {target}")
    print(f"Threshold: {threshold}")
    print(f"Number of features expected: {len(features)}")

    # Load data
    df = pd.read_csv(input_path)

    print(f"Input shape: {df.shape}")

    # Validate required features
    missing_features = [col for col in features if col not in df.columns]

    if len(missing_features) > 0:
        raise ValueError(f"Missing required features in input data: {missing_features}")

    # Prepare scoring data
    scoring_df = df.copy()

    X = (
        scoring_df[features]
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )

    # Identify rows with complete features
    valid_mask = X.notna().all(axis=1)

    rows_before = X.shape[0]
    rows_valid = valid_mask.sum()
    rows_dropped = rows_before - rows_valid

    print(f"Rows before scoring: {rows_before}")
    print(f"Rows valid for scoring: {rows_valid}")
    print(f"Rows dropped due to missing values: {rows_dropped}")

    X_valid = X.loc[valid_mask].copy()
    scored_df = scoring_df.loc[valid_mask].copy()

    # Predict probabilities and classes
    probabilities = pipeline.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    scored_df["risk_probability"] = probabilities
    scored_df["risk_class"] = predictions
    scored_df["threshold_used"] = threshold
    scored_df["model_target"] = target

    # Create output folder
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save predictions
    scored_df.to_csv(output_path, index=False)

    print("\nBatch scoring finished successfully.")
    print(f"Predictions saved in: {output_path}")

    print("\nPrediction distribution:")
    print(scored_df["risk_class"].value_counts())

    print("\nTop 10 predictions:")
    print(
        scored_df[
            ["risk_probability", "risk_class", "threshold_used", "model_target"]
        ].head(10)
    )


if __name__ == "__main__":
    main()