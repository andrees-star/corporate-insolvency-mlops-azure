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

from custom_transformers import PercentileWinsorizer


def test_complete_model_pipeline_generates_valid_probabilities():
    feature_names = [
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

    random_generator = np.random.default_rng(42)

    X = pd.DataFrame(
        random_generator.normal(
            loc=1.0,
            scale=0.5,
            size=(100, len(feature_names)),
        ),
        columns=feature_names,
    )

    # Agregar algunos valores extremos para probar la winsorización
    X.loc[0, "raz"] = 1000.0
    X.loc[1, "margen"] = -500.0

    # Target sintético con dos clases
    y = np.array([0] * 80 + [1] * 20)

    pipeline = Pipeline(
        steps=[
            (
                "winsor",
                PercentileWinsorizer(
                    feature_names=feature_names,
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

    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = pipeline.predict(X)

    assert len(probabilities) == len(X)
    assert len(predictions) == len(X)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert pipeline.named_steps["yeojohnson"].lambdas_.shape[0] == len(
        feature_names
    )