import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Coordinate preprocessing and training in Azure ML."
    )

    parser.add_argument(
        "--input-data",
        required=True,
        help="Path where Azure ML mounts or downloads the input Data Asset.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where Azure ML stores model and metric artifacts.",
    )

    return parser.parse_args()


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_input_file(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        return input_path

    if input_path.is_dir():
        excel_files = list(input_path.rglob("*.xlsx"))

        if len(excel_files) == 1:
            return excel_files[0]

        if not excel_files:
            raise FileNotFoundError(
                f"No Excel file was found inside: {input_path}"
            )

        raise ValueError(
            "More than one Excel file was found in the input Data Asset: "
            f"{excel_files}"
        )

    raise FileNotFoundError(
        f"Azure ML input path does not exist: {input_path}"
    )


def run_script(script_name):
    script_path = PROJECT_ROOT / "src" / script_name

    print(f"\nRunning: {script_path}")

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def copy_if_exists(source_path, destination_directory):
    source_path = Path(source_path)

    if source_path.exists():
        destination_path = destination_directory / source_path.name
        shutil.copy2(source_path, destination_path)

        print(
            f"Artifact copied: {source_path} -> {destination_path}"
        )
    else:
        print(f"Artifact not found and was not copied: {source_path}")


def main():
    args = parse_arguments()
    config = load_config()

    print("Starting Azure ML training coordinator...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Azure input: {args.input_data}")
    print(f"Azure output: {args.output_dir}")

    # Encontrar el Excel recibido desde el Data Asset
    input_file = resolve_input_file(args.input_data)

    print(f"Training Excel found: {input_file}")

    # Copiar el Excel a la ruta que espera 02_preprocess_data.py
    configured_raw_path = Path(config["data"]["raw_path"])

    if not configured_raw_path.is_absolute():
        configured_raw_path = PROJECT_ROOT / configured_raw_path

    configured_raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        input_file,
        configured_raw_path,
    )

    print(
        "Training Excel copied to configured raw path: "
        f"{configured_raw_path}"
    )

    # Ejecutar los scripts existentes en el orden correcto
    run_script("02_preprocess_data.py")
    run_script("03_train_model.py")

    # Crear la carpeta de salida administrada por Azure ML
    output_directory = Path(args.output_dir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Rutas generadas por los scripts actuales
    model_path = Path(config["model"]["output_path"])
    metrics_path = Path(config["outputs"]["metrics_report"])
    tuning_path = Path(
        config["outputs"].get(
            "tuning_results",
            "outputs/tuning_results.csv",
        )
    )

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not metrics_path.is_absolute():
        metrics_path = PROJECT_ROOT / metrics_path

    if not tuning_path.is_absolute():
        tuning_path = PROJECT_ROOT / tuning_path

    statsmodels_path = (
        PROJECT_ROOT
        / "outputs"
        / "statsmodels_logit_summary.txt"
    )

    # Copiar artefactos para que Azure los conserve al terminar el job
    print("\nCopying training artifacts...")

    copy_if_exists(
        model_path,
        output_directory,
    )

    copy_if_exists(
        metrics_path,
        output_directory,
    )

    copy_if_exists(
        tuning_path,
        output_directory,
    )

    copy_if_exists(
        statsmodels_path,
        output_directory,
    )

    # Copiar la clase custom necesaria para cargar el modelo
    copy_if_exists(
        PROJECT_ROOT / "src" / "custom_transformers.py",
        output_directory,
    )

    print("\nAzure ML training completed successfully.")
    print(f"Artifacts saved in: {output_directory}")


if __name__ == "__main__":
    main()