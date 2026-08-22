import json
import sys
import joblib
import yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

# Allow imports from src/
sys.path.append(str(Path(__file__).resolve().parent))

from custom_transformers import PercentileWinsorizer


CONFIG_PATH = Path("config/config.yml")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calcular_metricas(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "error_tipo_I": float(error_tipo_I),
        "error_tipo_II": float(error_tipo_II),
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn)
    }


def main():
    print("Starting model training...")

    # =====================================================
    # 1. Load config
    # =====================================================
    config = load_config(CONFIG_PATH)

    processed_path = Path(config["data"]["processed_path"])
    model_output_path = Path(config["model"]["output_path"])
    metrics_report_path = Path(config["outputs"]["metrics_report"])
    tuning_results_path = Path(
        config["outputs"].get("tuning_results", "outputs/tuning_results.csv")
    )

    target_name = config["target"]["name"]
    vars_modelo = config["features"]["vars_modelo"]

    random_state = config["training"]["random_state"]
    test_size = config["training"]["test_size"]
    cv_folds = config["training"].get("cv_folds", 5)

    k_values = config.get("manual_weight_tuning", {}).get(
        "k_values",
        [20, 25, 30, 35]
    )

    thresholds = config.get("threshold_tuning", {}).get(
        "thresholds",
        [0.40, 0.42, 0.45, 0.48, 0.50]
    )

    recall_min_primary = config.get("threshold_tuning", {}).get(
        "recall_min_primary",
        0.50
    )

    error_tipo_I_max_primary = config.get("threshold_tuning", {}).get(
        "error_tipo_I_max_primary",
        0.10
    )

    recall_min_fallback = config.get("threshold_tuning", {}).get(
        "recall_min_fallback",
        0.45
    )

    error_tipo_I_max_fallback = config.get("threshold_tuning", {}).get(
        "error_tipo_I_max_fallback",
        0.08
    )

    default_limits_cfg = config.get("winsorization", {}).get("default_limits", {})

    default_limits = (
        default_limits_cfg.get("lower", 0.01),
        default_limits_cfg.get("upper", 0.99)
    )

    cuts_cfg = config.get("winsorization", {}).get("cuts", {})

    cuts_winsor = {
        col: tuple(vals)
        for col, vals in cuts_cfg.items()
    }

    solver = config["hyperparameters"]["logistic_regression"].get(
        "solver",
        "lbfgs"
    )

    max_iter = config["hyperparameters"]["logistic_regression"].get(
        "max_iter",
        5000
    )

    print(f"Processed data path: {processed_path}")
    print(f"Target: {target_name}")
    print(f"Model output path: {model_output_path}")
    print(f"Number of features: {len(vars_modelo)}")
    print(f"Features: {vars_modelo}")

    # =====================================================
    # 2. Load processed data
    # =====================================================
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {processed_path}")

    df = pd.read_csv(processed_path)

    columnas_necesarias = vars_modelo + [target_name]

    faltantes = [
        col for col in columnas_necesarias
        if col not in df.columns
    ]

    if len(faltantes) > 0:
        raise ValueError(f"Missing columns in processed data: {faltantes}")

    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target_name].astype(int).copy()

    print("\n===== FINAL DATASET =====")
    print("Shape:", base.shape)
    print("Positives:", int(y.sum()))
    print("Negatives:", int((y == 0).sum()))
    print("Event rate:", round(y.mean(), 4))

    # =====================================================
    # 3. Train / validation split
    # =====================================================
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TRAIN / VALIDATION SPLIT =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positives train:", int(y_train.sum()))
    print("Positives validation:", int(y_val.sum()))

    # =====================================================
    # 4. Cross-validation tuning
    # =====================================================
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )

    resultados_tuning = []

    print("\n===== CV TUNING: MANUAL WEIGHTS + THRESHOLDS =====")

    for k in k_values:
        pipe = Pipeline(steps=[
            (
                "winsor",
                PercentileWinsorizer(
                    feature_names=vars_modelo,
                    cuts_by_variable=cuts_winsor,
                    default_limits=default_limits
                )
            ),
            (
                "yeojohnson",
                PowerTransformer(
                    method="yeo-johnson",
                    standardize=True
                )
            ),
            (
                "model",
                LogisticRegression(
                    class_weight={0: 1, 1: k},
                    max_iter=max_iter,
                    solver=solver,
                    random_state=random_state
                )
            )
        ])

        y_prob_cv = cross_val_predict(
            pipe,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        roc_auc_cv = roc_auc_score(y_train, y_prob_cv)
        pr_auc_cv = average_precision_score(y_train, y_prob_cv)

        for threshold in thresholds:
            met = calcular_metricas(y_train, y_prob_cv, threshold)
            met["k"] = int(k)
            met["roc_auc_cv"] = float(roc_auc_cv)
            met["pr_auc_cv"] = float(pr_auc_cv)
            resultados_tuning.append(met)

    resultados_df = pd.DataFrame(resultados_tuning)

    print("\n===== TUNING RESULTS =====")
    print(
        resultados_df[
            [
                "k",
                "threshold",
                "precision",
                "recall",
                "f1_score",
                "mcc",
                "error_tipo_I",
                "error_tipo_II",
                "roc_auc_cv",
                "pr_auc_cv",
                "TP",
                "FN",
                "FP",
                "TN"
            ]
        ].round(4)
    )

    # =====================================================
    # 5. Select best configuration
    # =====================================================
    candidatos = resultados_df[
        (resultados_df["recall"] >= recall_min_primary) &
        (resultados_df["error_tipo_I"] <= error_tipo_I_max_primary)
    ].copy()

    if candidatos.empty:
        candidatos = resultados_df[
            (resultados_df["recall"] >= recall_min_fallback) &
            (resultados_df["error_tipo_I"] <= error_tipo_I_max_fallback)
        ].copy()

    if candidatos.empty:
        candidatos = resultados_df.copy()

    mejor = candidatos.sort_values(
        by=["error_tipo_II", "mcc", "f1_score", "pr_auc_cv"],
        ascending=[True, False, False, False]
    ).iloc[0]

    k_final = int(mejor["k"])
    threshold_final = float(mejor["threshold"])

    print("\n===== BEST CONFIGURATION =====")
    print("k_final:", k_final)
    print("threshold_final:", threshold_final)
    print(
        mejor[
            [
                "precision",
                "recall",
                "f1_score",
                "mcc",
                "error_tipo_I",
                "error_tipo_II",
                "roc_auc_cv",
                "pr_auc_cv",
                "TP",
                "FN",
                "FP",
                "TN"
            ]
        ].round(4)
    )

    # =====================================================
    # 6. Fit final pipeline
    # =====================================================
    pipe_final = Pipeline(steps=[
        (
            "winsor",
            PercentileWinsorizer(
                feature_names=vars_modelo,
                cuts_by_variable=cuts_winsor,
                default_limits=default_limits
            )
        ),
        (
            "yeojohnson",
            PowerTransformer(
                method="yeo-johnson",
                standardize=True
            )
        ),
        (
            "model",
            LogisticRegression(
                class_weight={0: 1, 1: k_final},
                max_iter=max_iter,
                solver=solver,
                random_state=random_state
            )
        )
    ])

    pipe_final.fit(X_train, y_train)

    # =====================================================
    # 7. Validation metrics
    # =====================================================
    y_prob_val = pipe_final.predict_proba(X_val)[:, 1]

    met_val = calcular_metricas(
        y_true=y_val,
        y_prob=y_prob_val,
        threshold=threshold_final
    )

    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)

    final_metrics = {
        "target": target_name,
        "model_type": "logistic_regression",
        "k_final": k_final,
        "threshold_final": threshold_final,
        "roc_auc_validation": float(roc_auc_val),
        "pr_auc_validation": float(pr_auc_val),
        **met_val
    }

    print("\n===== VALIDATION METRICS =====")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")

    cm_df = pd.DataFrame(
        [
            [met_val["TP"], met_val["FN"]],
            [met_val["FP"], met_val["TN"]]
        ],
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print(f"\n===== CONFUSION MATRIX VALIDATION threshold={threshold_final} =====")
    print(cm_df)

    # =====================================================
    # 8. Statsmodels explanatory model
    # =====================================================
    try:
        X_train_win = pipe_final.named_steps["winsor"].transform(X_train)
        X_train_t = pipe_final.named_steps["yeojohnson"].transform(X_train_win)

        X_train_t_sm = pd.DataFrame(
            X_train_t,
            columns=vars_modelo,
            index=X_train.index
        )

        X_train_t_sm_const = sm.add_constant(X_train_t_sm)
        pesos_train = np.where(y_train == 1, k_final, 1.0)

        modelo_sm = sm.GLM(
            y_train,
            X_train_t_sm_const,
            family=sm.families.Binomial(),
            freq_weights=pesos_train
        ).fit()

        statsmodels_summary_path = Path("outputs/statsmodels_logit_summary.txt")
        statsmodels_summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(statsmodels_summary_path, "w", encoding="utf-8") as file:
            file.write(str(modelo_sm.summary()))

        print(f"\nStatsmodels summary saved in: {statsmodels_summary_path}")

    except Exception as e:
        print("\nStatsmodels explanatory model failed, but sklearn model was trained.")
        print(f"Error: {e}")

    # =====================================================
    # 9. Save artifacts
    # =====================================================
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_report_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_results_path.parent.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "pipeline": pipe_final,
        "features": vars_modelo,
        "target": target_name,
        "threshold": threshold_final,
        "k_final": k_final,
        "metrics": final_metrics
    }

    joblib.dump(model_artifact, model_output_path)

    with open(metrics_report_path, "w", encoding="utf-8") as file:
        json.dump(final_metrics, file, indent=4)

    resultados_df.to_csv(tuning_results_path, index=False)

    print("\n===== ARTIFACTS SAVED =====")
    print(f"Model saved in: {model_output_path}")
    print(f"Metrics saved in: {metrics_report_path}")
    print(f"Tuning results saved in: {tuning_results_path}")

    print("\nTraining finished successfully.")


if __name__ == "__main__":
    main()