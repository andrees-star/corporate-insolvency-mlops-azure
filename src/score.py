import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

from pathlib import Path


# Permite importar custom_transformers.py desde src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from custom_transformers import PercentileWinsorizer


LOCAL_MODEL_PATH = Path("models/logistic_regression_riesgo24.pkl")


def find_model_path():
    """
    Busca el modelo en Azure ML cuando está desplegado.
    Si se ejecuta localmente, usa la ruta local del proyecto.
    """
    azure_model_dir = os.getenv("AZUREML_MODEL_DIR")

    if azure_model_dir:
        model_files = list(
            Path(azure_model_dir).rglob(
                "logistic_regression_riesgo24.pkl"
            )
        )

        if not model_files:
            raise FileNotFoundError(
                "The model file was not found inside "
                f"AZUREML_MODEL_DIR: {azure_model_dir}"
            )

        return model_files[0]

    return LOCAL_MODEL_PATH


def init():
    """
    Azure ejecuta esta función una vez cuando inicia el contenedor.
    """
    global model_artifact
    global pipeline
    global features
    global threshold
    global target

    print("Initializing model...")

    model_path = find_model_path()

    print(f"Loading model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model_artifact = joblib.load(model_path)

    pipeline = model_artifact["pipeline"]
    features = model_artifact["features"]
    threshold = model_artifact["threshold"]
    target = model_artifact["target"]

    print("Model loaded successfully.")
    print(f"Target: {target}")
    print(f"Threshold: {threshold}")
    print(f"Number of features: {len(features)}")


def run(raw_data):
    """
    Recibe empresas en formato JSON y devuelve sus predicciones.
    """
    try:
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        if "inputs" not in data:
            return {
                "error": "Input JSON must contain an 'inputs' key."
            }

        df = pd.DataFrame(data["inputs"])

        missing_features = [
            column
            for column in features
            if column not in df.columns
        ]

        if missing_features:
            return {
                "error": "Missing required features.",
                "missing_features": missing_features
            }

        X = (
            df[features]
            .replace([np.inf, -np.inf], np.nan)
            .copy()
        )

        if X.isna().any().any():
            return {
                "error": (
                    "Input contains missing or invalid values "
                    "in the required features."
                )
            }

        probabilities = pipeline.predict_proba(X)[:, 1]
        predictions = (
            probabilities >= threshold
        ).astype(int)

        results = []

        for probability, prediction in zip(
            probabilities,
            predictions
        ):
            results.append({
                "risk_probability": float(probability),
                "risk_class": int(prediction),
                "threshold_used": float(threshold),
                "model_target": target
            })

        return {
            "predictions": results
        }

    except Exception as error:
        return {
            "error": str(error)
        }