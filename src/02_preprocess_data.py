import pandas as pd
import numpy as np
import yaml
from pathlib import Path


CONFIG_PATH = Path("config/config.yml")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    print("Starting preprocessing...")

    config = load_config(CONFIG_PATH)

    raw_path = Path(config["data"]["raw_path"])
    processed_path = Path(config["data"]["processed_path"])
    target_name = config["target"]["name"]

    # Features desde YAML
    vars_modelo = config["features"]["vars_modelo"]

    print(f"Raw data path: {raw_path}")
    print(f"Processed data path: {processed_path}")
    print(f"Target: {target_name}")
    print(f"Number of configured features: {len(vars_modelo)}")

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df = pd.read_excel(raw_path)

    print(f"Original shape: {df.shape}")

    columnas_necesarias = vars_modelo + [target_name]
    faltantes = [col for col in columnas_necesarias if col not in df.columns]

    if len(faltantes) > 0:
        raise ValueError(f"Missing columns in raw data: {faltantes}")

    # Base mínima para entrenamiento
    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    base[target_name] = base[target_name].astype(int)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(processed_path, index=False)

    print("\nProcessed dataset created successfully.")
    print(f"Processed shape: {base.shape}")
    print(f"Processed data saved in: {processed_path}")

    print("\nTarget distribution:")
    print(base[target_name].value_counts())

    print("\nPreprocessing finished successfully.")


if __name__ == "__main__":
    main()