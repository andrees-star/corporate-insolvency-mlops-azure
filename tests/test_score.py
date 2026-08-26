import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

import score
from custom_transformers import PercentileWinsorizer


FEATURES = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "margen_operacional",
    "ractiv",
    "rpatri",
    "activos_pasivos",
    "niven",
    "apalc",
    "apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso",
]


def configure_test_model():
    random_generator = np.random.default_rng(42)

    X = pd.DataFrame(
        random_generator.normal(
            loc=1.0,
            scale=0.5,
            size=(100, len(FEATURES)),
        ),
        columns=FEATURES,
    )

    y = np.array([0] * 80 + [1] * 20)

    test_pipeline = Pipeline(
        steps=[
            (
                "winsor",
                PercentileWinsorizer(
                    feature_names=FEATURES,
                    default_limits=(0.01, 0.99),
                ),
            ),
            (
                "yeojohnson",
                PowerTransformer(
                    method="yeo-johnson",
                    standardize=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    test_pipeline.fit(X, y)

    score.pipeline = test_pipeline
    score.features = FEATURES
    score.threshold = 0.48
    score.target = "riesgo_24"

    return X


def test_score_returns_valid_prediction():
    X = configure_test_model()

    request_data = {
        "inputs": [
            X.iloc[0].to_dict()
        ]
    }

    response = score.run(request_data)

    assert "predictions" in response
    assert len(response["predictions"]) == 1

    prediction = response["predictions"][0]

    assert 0 <= prediction["risk_probability"] <= 1
    assert prediction["risk_class"] in [0, 1]
    assert prediction["threshold_used"] == 0.48
    assert prediction["model_target"] == "riesgo_24"


def test_score_reports_missing_features():
    configure_test_model()

    incomplete_request = {
        "inputs": [
            {
                "raz": 2.0,
                "teso": 0.5,
            }
        ]
    }

    response = score.run(incomplete_request)

    assert "error" in response
    assert response["error"] == "Missing required features."
    assert "missing_features" in response
    assert "rota" in response["missing_features"]
    assert "margen_operacional" in response["missing_features"]