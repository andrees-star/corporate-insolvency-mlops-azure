import pandas as pd
import yaml
from pathlib import Path


CONFIG_PATH = Path("config/config.yml")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    print("Starting data validation...")

    # Load YAML config
    config = load_config(CONFIG_PATH)

    # Get paths and target from YAML
    data_path = Path(config["data"]["raw_path"])
    output_path = Path(config["outputs"]["validation_report"])
    target_name = config["target"]["name"]

    # Validate data file exists
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    print(f"File found: {data_path}")
    print(f"Target selected from YAML: {target_name}")

    # Read Excel
    df = pd.read_excel(data_path)

    # Create output folder if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Basic information
    rows = df.shape[0]
    columns = df.shape[1]
    first_10_columns = df.columns[:10].tolist()
    data_types = df.dtypes
    missing_values = df.isnull().sum().sort_values(ascending=False).head(20)

    print("\nDataset shape:")
    print(f"Rows: {rows}")
    print(f"Columns: {columns}")

    print("\nFirst 10 columns:")
    print(first_10_columns)

    print("\nMissing values by column:")
    print(missing_values)

    print("\nTarget variable check:")
    if target_name in df.columns:
        print(f"{target_name}: FOUND")
        target_counts = df[target_name].value_counts(dropna=False)
        print(target_counts)
    else:
        print(f"{target_name}: NOT FOUND")
        target_counts = None

    # Save report
    with open(output_path, "w", encoding="utf-8") as report:
        report.write("DATA VALIDATION REPORT\n")
        report.write("======================\n\n")

        report.write(f"Project: {config['project']['name']}\n")
        report.write(f"Version: {config['project']['version']}\n\n")

        report.write(f"Data file: {data_path}\n")
        report.write(f"Target: {target_name}\n")
        report.write(f"Rows: {rows}\n")
        report.write(f"Columns: {columns}\n\n")

        report.write("First 10 columns:\n")
        report.write(str(first_10_columns))
        report.write("\n\n")

        report.write("Data types:\n")
        report.write(str(data_types))
        report.write("\n\n")

        report.write("Top 20 columns with missing values:\n")
        report.write(str(missing_values))
        report.write("\n\n")

        report.write("Target variable check:\n")
        if target_counts is not None:
            report.write(f"{target_name}: FOUND\n")
            report.write(str(target_counts))
            report.write("\n")
        else:
            report.write(f"{target_name}: NOT FOUND\n")

        report.write("\nData validation finished successfully.\n")

    print(f"\nReport saved in: {output_path}")
    print("Data validation finished successfully.")


if __name__ == "__main__":
    main()
