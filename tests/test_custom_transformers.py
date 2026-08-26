import sys
from pathlib import Path

import pandas as pd


# Permite importar módulos desde la carpeta src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from custom_transformers import PercentileWinsorizer


def test_percentile_winsorizer_preserves_shape_and_limits_outliers():
    feature_names = ["raz", "margen"]

    data = pd.DataFrame(
        {
            "raz": [1.0, 2.0, 3.0, 4.0, 1000.0],
            "margen": [-500.0, 0.10, 0.20, 0.30, 0.40],
        }
    )

    transformer = PercentileWinsorizer(
        feature_names=feature_names,
        default_limits=(0.20, 0.80),
    )

    transformed_data = transformer.fit_transform(data)

    assert transformed_data.shape == data.shape
    assert transformed_data.columns.tolist() == feature_names
    assert transformed_data["raz"].max() < data["raz"].max()
    assert transformed_data["margen"].min() > data["margen"].min()
    assert transformed_data.isna().sum().sum() == 0