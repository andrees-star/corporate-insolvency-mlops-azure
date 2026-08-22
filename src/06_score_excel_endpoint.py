import json
import subprocess
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/scoring/new_companies.xlsx")
REQUEST_PATH = Path("outputs/new_companies_request.json")
OUTPUT_PATH = Path("outputs/new_companies_predictions.xlsx")

ENDPOINT_NAME = "insolvency-risk24-endpoint"

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


def parse_azure_response(raw_response):
    """
    Convierte la respuesta de Azure CLI en un diccionario de Python.

    En algunos casos, Azure CLI devuelve un JSON codificado
    dentro de una cadena JSON. Por eso intentamos decodificarlo
    una segunda vez cuando sea necesario.
    """
    parsed_response = json.loads(raw_response)

    if isinstance(parsed_response, str):
        parsed_response = json.loads(parsed_response)

    return parsed_response


def main():
    print("Starting Excel endpoint scoring...")

    # Validar que exista el Excel
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input Excel file not found: {INPUT_PATH}"
        )

    # Leer empresas nuevas
    df = pd.read_excel(INPUT_PATH)

    print(f"Companies loaded: {len(df)}")
    print(f"Columns received: {len(df.columns)}")

    # Validar las 14 variables requeridas
    missing_features = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required model features: {missing_features}"
        )

    # Validar nulos en las variables del modelo
    null_counts = df[FEATURES].isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]

    if not columns_with_nulls.empty:
        raise ValueError(
            "Missing values found in the required features:\n"
            f"{columns_with_nulls}"
        )

    # Construir solicitud JSON usando solo las 14 variables
    request_data = {
        "inputs": df[FEATURES].to_dict(orient="records")
    }

    REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REQUEST_PATH, "w", encoding="utf-8") as request_file:
        json.dump(
            request_data,
            request_file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Request JSON saved in: {REQUEST_PATH}")
    print(f"Rows sent to endpoint: {len(request_data['inputs'])}")

    # Invocar endpoint de Azure ML
    command = [
        "az",
        "ml",
        "online-endpoint",
        "invoke",
        "--name",
        ENDPOINT_NAME,
        "--request-file",
        str(REQUEST_PATH),
    ]

    print("Sending companies to Azure ML endpoint...")

    completed_process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    response = parse_azure_response(
        completed_process.stdout.strip()
    )

    if "error" in response:
        raise RuntimeError(
            f"Endpoint returned an error: {response['error']}"
        )

    predictions = response.get("predictions", [])

    if len(predictions) != len(df):
        raise ValueError(
            "The number of predictions does not match "
            "the number of companies sent. "
            f"Companies: {len(df)}, predictions: {len(predictions)}"
        )

    predictions_df = pd.DataFrame(predictions)

    # Unir datos originales con resultados
    result_df = pd.concat(
        [
            df.reset_index(drop=True),
            predictions_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Guardar Excel final
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_excel(
        OUTPUT_PATH,
        index=False,
        sheet_name="predictions",
    )

    print("\nScoring finished successfully.")
    print(f"Results saved in: {OUTPUT_PATH}")

    print("\nPrediction distribution:")
    print(result_df["risk_class"].value_counts())

    print("\nScoring summary:")
    summary_columns = []

    if "NIT" in result_df.columns:
        summary_columns.append("NIT")

    if "Razón social de la sociedad_balance" in result_df.columns:
        summary_columns.append(
            "Razón social de la sociedad_balance"
        )

    summary_columns.extend([
        "risk_probability",
        "risk_class",
        "threshold_used",
        "model_target",
    ])

    print(result_df[summary_columns])


if __name__ == "__main__":
    main()
