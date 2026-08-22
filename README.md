


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

## Dataset

The original dataset contains:

- **31,276 company records**
- **57 variables**
- Financial statements and financial ratios
- Binary target: `riesgo_24`
- Positive cases: **339**
- Negative cases: **30,937**
- Event rate: approximately **1.08%**

The dataset presents a severe class imbalance because insolvency cases represent only a small percentage of the total observations.

Due to confidentiality and privacy considerations, the original dataset, company identifiers, business names, scoring files, and individual prediction results are not included in this public repository.




---

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
JSON or Excel-based scoring
        |
        v
Risk probabilities and classifications

---

## Excel-Based Scoring

The project includes a practical business workflow for scoring new companies from an Excel file:

```text
Excel file with new companies
        |
        v
Validation of required financial indicators
        |
        v
Conversion of Excel rows to JSON
        |
        v
Azure ML Managed Online Endpoint
        |
        v
Probability and risk classification
        |
        v
Excel file with prediction results
```

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
mlops-insolvency-project/
├── azureml/
│   ├── endpoint.yml
│   └── deployment.yml
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
│   ├── custom_transformers.py
│   └── score.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── scoring/
├── models/
├── model_package/
├── notebooks/
├── outputs/
├── .gitignore
└── README.md
```

### Main Folders

- `azureml/`: Azure ML endpoint and deployment configurations.
- `config/`: centralized project and model configuration.
- `environment/`: Python and package dependencies.
- `src/`: validation, preprocessing, training, scoring, and inference scripts.
- `data/raw/`: original private datasets.
- `data/processed/`: modeling datasets generated by the preprocessing pipeline.
- `data/scoring/`: new companies submitted for prediction.
- `models/`: locally serialized model artifacts.
- `model_package/`: deployment package containing the model and custom transformer.
- `notebooks/`: exploratory analysis and experimentation.
- `outputs/`: validation reports, metrics, tuning results, and predictions.

Private data, trained model binaries, prediction results, backups, and credentials are excluded from GitHub through `.gitignore`.


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

- [x] Azure Machine Learning workspace configuration
- [x] Cost budget and resource management
- [x] Data validation
- [x] Reproducible preprocessing
- [x] Leakage-safe winsorization
- [x] Yeo-Johnson transformation
- [x] Logistic Regression training
- [x] Stratified cross-validation
- [x] Class-weight tuning
- [x] Decision-threshold optimization
- [x] Model serialization
- [x] Batch scoring
- [x] Azure ML Model Registry
- [x] Azure ML environment registration
- [x] Managed Online Endpoint deployment
- [x] Real-time JSON inference
- [x] Excel-to-endpoint scoring
- [x] Cost-safe endpoint and deployment deletion



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
Approved model deployment
        |
        v
Production monitoring
        |
        v
Drift detection or new labeled data
        |
        v
Controlled model retraining
``




```
