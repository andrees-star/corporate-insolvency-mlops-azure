import argparse
import json
import sys
from pathlib import Path


QUALITY_THRESHOLDS = {
    "roc_auc_validation": 0.85,
    "recall": 0.65,
    "max_error_tipo_I": 0.10,
}

EXPECTED_TARGET = "riesgo_24"
EXPECTED_MODEL_TYPE = "logistic_regression"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate whether a trained model passes quality gates."
    )

    parser.add_argument(
        "--metrics-path",
        required=True,
        help="Path to the model_metrics.json file.",
    )

    return parser.parse_args()


def load_metrics(metrics_path):
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}"
        )

    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def evaluate_quality_gates(metrics):
    failures = []

    required_fields = [
        "target",
        "model_type",
        "roc_auc_validation",
        "recall",
        "error_tipo_I",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in metrics
    ]

    if missing_fields:
        failures.append(
            f"Missing required metric fields: {missing_fields}"
        )

        return failures

    threshold = metrics.get(
        "threshold_final",
        metrics.get("threshold"),
    )

    if threshold is None:
        failures.append(
            "Missing threshold_final or threshold."
        )

    elif not 0 <= float(threshold) <= 1:
        failures.append(
            f"Threshold must be between 0 and 1. Received: {threshold}"
        )

    if metrics["target"] != EXPECTED_TARGET:
        failures.append(
            f"Unexpected target. Expected {EXPECTED_TARGET}, "
            f"received {metrics['target']}."
        )

    if metrics["model_type"] != EXPECTED_MODEL_TYPE:
        failures.append(
            f"Unexpected model type. Expected {EXPECTED_MODEL_TYPE}, "
            f"received {metrics['model_type']}."
        )

    if (
        float(metrics["roc_auc_validation"])
        < QUALITY_THRESHOLDS["roc_auc_validation"]
    ):
        failures.append(
            "ROC-AUC validation is below the approved minimum: "
            f"{metrics['roc_auc_validation']} < "
            f"{QUALITY_THRESHOLDS['roc_auc_validation']}"
        )

    if float(metrics["recall"]) < QUALITY_THRESHOLDS["recall"]:
        failures.append(
            "Recall is below the approved minimum: "
            f"{metrics['recall']} < "
            f"{QUALITY_THRESHOLDS['recall']}"
        )

    if (
        float(metrics["error_tipo_I"])
        > QUALITY_THRESHOLDS["max_error_tipo_I"]
    ):
        failures.append(
            "Type I error exceeds the approved maximum: "
            f"{metrics['error_tipo_I']} > "
            f"{QUALITY_THRESHOLDS['max_error_tipo_I']}"
        )

    return failures


def main():
    args = parse_arguments()
    metrics = load_metrics(args.metrics_path)

    print("\n===== MODEL QUALITY GATES =====")
    print(f"Target: {metrics.get('target')}")
    print(f"Model type: {metrics.get('model_type')}")
    print(
        "ROC-AUC validation: "
        f"{metrics.get('roc_auc_validation')}"
    )
    print(f"Recall: {metrics.get('recall')}")
    print(f"Type I error: {metrics.get('error_tipo_I')}")
    print(
        "Threshold: "
        f"{metrics.get('threshold_final', metrics.get('threshold'))}"
    )

    failures = evaluate_quality_gates(metrics)

    if failures:
        print("\nQUALITY GATE RESULT: FAILED")

        for failure in failures:
            print(f"- {failure}")

        sys.exit(1)

    print("\nQUALITY GATE RESULT: PASSED")
    print(
        "The candidate model meets all approved quality thresholds."
    )


if __name__ == "__main__":
    main()