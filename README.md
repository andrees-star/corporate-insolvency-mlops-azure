## MLOps Architecture

The solution implements the following end-to-end workflow:

```text
Historical financial data
        |
        v
Data validation
        |
        v
Feature selection and preprocessing
        |
        v
Leakage-safe training pipeline
  - Winsorization
  - Yeo-Johnson transformation
  - Logistic Regression
        |
        v
Cross-validation and threshold tuning
        |
        v
Serialized model artifact
        |
        v
Azure ML Model Registry
        |
        v
Managed Online Endpoint
        |
        v
Excel file with new companies
        |
        v
Validation of required financial indicators
        |
        v
Conversion of Excel rows to JSON
        |
        v
Risk probabilities and classifications
        |
        v
Excel file with prediction results
```

The architecture covers the complete lifecycle of the corporate insolvency model. Historical financial data is validated and transformed into the 14 financial indicators required by the model. A leakage-safe pipeline applies winsorization and Yeo-Johnson transformation before training the Logistic Regression model.

Cross-validation and threshold tuning are used to evaluate the model and determine the appropriate risk-classification threshold. The resulting model artifact is registered in Azure Machine Learning and can be deployed through a Managed Online Endpoint.

For business use, new company records can be submitted through an Excel file. The required financial indicators are validated and converted into JSON requests. The model then generates an insolvency risk probability and classification, which are returned in a new Excel file.

# Corporate Insolvency Prediction with MLOps on Azure

End-to-end Machine Learning and MLOps project designed to predict corporate insolvency risk in Colombia using financial indicators and Azure Machine Learning.

The project implements a Logistic Regression model for the target `riesgo_24`, where:

- `1` indicates that a company entered a reorganization or liquidation process during 2024.
- `0` indicates that the company did not enter an insolvency process during 2024.

The solution includes data validation, preprocessing, model training, threshold optimization, model registration, real-time deployment, batch scoring, and Excel-based inference.

---

## Business Objective

The objective is to develop an early-warning model that estimates the probability that a company will enter an insolvency process.

The model is designed to support:

- Early-risk identification
- Financial monitoring
- Company prioritization
- Credit-risk analysis
- Portfolio review
- Business decision support

The model should be interpreted as an early-warning tool and not as an automatic insolvency decision mechanism.

---

Dataset

The original dataset was prepared to support two corporate insolvency prediction scenarios. However, the current Machine Learning and MLOps implementation focuses exclusively on the 2024 scenario.

The target variable used in this project is `riesgo_24`, where:

- `1` indicates that a company entered a reorganization or liquidation process during 2024.
- `0` indicates that the company did not enter an insolvency process during 2024.

Companies that had already entered an insolvency process before the end of 2023 were excluded. This ensures that the model predicts new insolvency events during 2024 rather than identifying companies that were already insolvent.

The dataset used for the 2024 modeling exercise contains:

- **31,276 company records**
- **57 original variables**
- Financial statements and financial ratios
- Binary target: `riesgo_24`
- Positive cases: **339**
- Negative cases: **30,937**
- Event rate: approximately **1.08%**

The dataset presents a severe class imbalance because insolvency cases represent only a small percentage of the total observations.

Although the source information was prepared for two prediction scenarios, the results, metrics, model artifacts, Azure Machine Learning workflows, and scoring processes documented in this project correspond only to the 2024 scenario.

## Model and Financial Features

The current model is a:

**Logistic Regression**

The model uses the following 14 financial indicators:

- `raz`
- `teso`
- `rota`
- `margenb`
- `margen`
- `margen_operacional`
- `ractiv`
- `rpatri`
- `activos_pasivos`
- `niven`
- `apalc`
- `apaltot`
- `pasivo_corto_pasivo_total`
- `ctno_ventas_preciso`

The training process includes:

1. Stratified train and validation split
2. Percentile-based winsorization
3. Yeo-Johnson transformation and standardization
4. Logistic Regression training
5. Manual positive-class weight tuning
6. Five-fold stratified cross-validation
7. Decision-threshold optimization
8. Final evaluation on the validation sample

Winsorization and Yeo-Johnson transformations are fitted only on the training data inside the scikit-learn Pipeline. This design reduces the risk of data leakage from the validation sample.

---

## Model Performance

The current model achieved the following results on the validation sample:

| Metric                           | Result |
| -------------------------------- | -----: |
| ROC-AUC                          | 0.9073 |
| PR-AUC                           | 0.1079 |
| Recall                           | 0.7000 |
| Precision                        | 0.0737 |
| F1-score                         | 0.1333 |
| Matthews Correlation Coefficient | 0.2058 |
| Type I Error                     | 0.0955 |
| Type II Error                    | 0.3000 |
| Selected Threshold               |   0.48 |
| Positive-Class Weight            |     35 |

### Validation Confusion Matrix

| Actual class   | Predicted Risk | Predicted No Risk |
| -------------- | -------------: | ----------------: |
| Actual Risk    |             42 |                18 |
| Actual No Risk |            528 |             4,998 |

The model detected **70% of the actual insolvency cases** in the validation sample.

The relatively low precision reflects the severe class imbalance. Therefore, the model is intended to generate an early-warning shortlist for additional financial analysis rather than make fully automated business decisions.

---

## MLOps Architecture

The solution implements the following end-to-end workflow:

The architecture covers the complete lifecycle of the corporate insolvency model. Historical financial data is validated and transformed into the 14 financial indicators required by the model. A leakage-safe pipeline applies winsorization and Yeo-Johnson transformation before training the Logistic Regression model.

Cross-validation and threshold tuning are used to evaluate the model and determine the appropriate risk-classification threshold. The resulting model artifact is registered in Azure Machine Learning and can be deployed through a Managed Online Endpoint.

For business use, new company records can be submitted through an Excel file. The required financial indicators are validated and converted into JSON requests. The model then generates an insolvency risk probability and classification, which are returned in a new Excel file.

The input Excel file can contain additional accounting and descriptive columns. The scoring script selects only the 14 financial indicators required by the model.

The output preserves identifying columns for internal use and adds:

- `risk_probability`
- `risk_class`
- `threshold_used`
- `model_target`

A real scoring test was completed with five companies. The endpoint received the financial indicators, generated the predictions, and returned the results to a new Excel file.

Company identifiers, business names, input files, and individual prediction results are excluded from the public repository to protect confidential information.

---

## Project Structure

```text


```text
MLOPS-INSOLVENCY-PROJECT/
├── .github/
│   └── workflows/
│       ├── azure-oidc-test.yml
│       ├── ci.yml
│       ├── register-environment.yml
│       └── train-model.yml
├── azureml/
│   ├── deployment.yml
│   ├── endpoint.yml
│   └── train-job.yml
├── config/
│   └── config.yml
├── environment/
│   └── conda.yml
├── src/
│   ├── 01_data_check.py
│   ├── 02_preprocess_data.py
│   ├── 03_train_model.py
│   ├── 04_test_model_prediction.py
│   ├── 05_score_new_data.py
│   ├── 06_score_excel_endpoint.py
│   ├── 07_azure_train_job.py
│   ├── 08_quality_gate.py
│   ├── custom_transformers.py
│   └── score.py
├── tests/
│   ├── test_custom_transformers.py
│   ├── test_model_pipeline.py
│   ├── test_score.py
│   └── test_yaml_files.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── scoring/
├── models/
├── model_package/
├── notebooks/
├── outputs/
├── .amlignore
├── .gitignore
├── DECISIONS.md
├── PROGRESS.md
├── README.md
└── requirements-dev.txt
```

### Main Folders

### Main Folders

- `.github/workflows/`: GitHub Actions workflows for continuous integration, Azure authentication, environment registration, automated training, quality gates, and approved-model registration.
- `azureml/`: Azure Machine Learning configurations for training jobs, endpoints, and deployments.
- `config/`: Centralized project, model, preprocessing, and training configuration.
- `environment/`: Conda environment and package dependencies used by Azure Machine Learning.
- `src/`: Data validation, preprocessing, model training, quality-gate, scoring, and inference scripts.
- `tests/`: Automated tests for model components, scoring, pipelines, and YAML configuration files.
- `data/raw/`: Original source datasets used by the project.
- `data/processed/`: Modeling datasets generated by the preprocessing workflow.
- `data/scoring/`: Excel files containing new companies submitted for prediction.
- `models/`: Locally serialized model artifacts.
- `model_package/`: Model deployment package containing the trained model, metrics, and custom transformer.
- `notebooks/`: Exploratory analysis and model experimentation.
- `outputs/`: Validation reports, model metrics, tuning results, and prediction outputs.

Local environments, credentials, temporary files, private artifacts, and datasets not selected for publication are excluded from GitHub through `.gitignore`.

Files that are intentionally published for the project portfolio, such as the public Excel dataset and the article PDF, are managed separately in the portfolio repository.

---

## Technologies

The project uses:

- Python
- Pandas
- NumPy
- scikit-learn
- Statsmodels
- Joblib
- PyYAML
- OpenPyXL
- Azure Machine Learning
- Azure CLI v2
- Azure ML Model Registry
- Azure Managed Online Endpoints
- Git and GitHub

---

## Completed Milestones

- [X] Azure Machine Learning workspace configuration
- [X] Cost budget and resource management
- [X] Data validation
- [X] Reproducible preprocessing
- [X] Leakage-safe winsorization
- [X] Yeo-Johnson transformation
- [X] Logistic Regression training
- [X] Stratified cross-validation
- [X] Class-weight tuning
- [X] Decision-threshold optimization
- [X] Model serialization
- [X] Batch scoring
- [X] Azure ML Model Registry
- [X] Azure ML environment registration
- [X] Managed Online Endpoint deployment
- [X] Real-time JSON inference
- [X] Excel-to-endpoint scoring
- [X] Cost-safe endpoint and deployment deletion

---

## Project Roadmap

The next phases of the project include:

- [ ] Publish the project repository on GitHub
- [ ] Implement automated code and configuration tests
- [ ] Configure GitHub Actions with OpenID Connect
- [ ] Automate Azure ML training jobs
- [ ] Add continuous integration
- [ ] Add controlled model deployment
- [ ] Implement production inference-data collection
- [ ] Monitor data quality and feature drift
- [ ] Configure Azure Monitor alerts
- [ ] Automate scheduled retraining
- [ ] Implement champion and challenger model versions
- [ ] Package the solution with Docker

The final automated workflow will follow this structure:

```text
## Target MLOps Workflow

The target MLOps workflow will follow this structure:

```text
GitHub repository
        |
        v
GitHub Actions
        |
        v
Automated validation and testing
        |
        v
Azure ML training job
        |
        v
Model evaluation and quality gates
        |
        v
Azure ML Model Registry
        |
        v
FastAPI prediction service
        |
        v
Docker container
        |
        v
Controlled model deployment
        |
        v
JSON or Excel-based scoring
        |
        v
Risk probabilities and classifications
        |
        v
Production monitoring
        |
        v
Drift detection or new labeled data
        |
        v
Controlled model retraining
```
