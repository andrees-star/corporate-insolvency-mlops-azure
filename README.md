
# Corporate Insolvency Prediction with MLOps on Azure

End-to-end Machine Learning and MLOps project designed to predict corporate insolvency risk in Colombia using financial indicators, Logistic Regression, GitHub Actions, and Azure Machine Learning.

The project estimates the probability that a company will enter a reorganization or liquidation process during 2024.

The target variable is `riesgo_24`, where:

- `1` indicates that a company entered a reorganization or liquidation process during 2024.
- `0` indicates that the company did not enter an insolvency process during 2024.

Companies that had already entered an insolvency process before the end of 2023 were excluded. This ensures that the model predicts new insolvency events rather than identifying companies that were already insolvent.

---

## Business Objective

The objective is to develop an early-warning model that estimates the probability that a company will enter a corporate insolvency process.

The model is designed to support:

- Early-risk identification
- Financial monitoring
- Company prioritization
- Credit-risk analysis
- Portfolio review
- Business decision support

The model must be interpreted as an early-warning and prioritization tool. It is not intended to make fully automated financial, credit, or insolvency decisions.

---

## Dataset

The original financial information was prepared to support two corporate insolvency prediction scenarios. However, the current Machine Learning and MLOps implementation focuses exclusively on the 2024 scenario.

The target variable used in this project is `riesgo_24`.

The dataset used for the 2024 modeling exercise contains:

- **31,276 company records**
- **57 original variables**
- Financial statements and financial ratios
- Binary target: `riesgo_24`
- Positive cases: **339**
- Negative cases: **30,937**
- Event rate: approximately **1.08%**

The dataset presents a severe class imbalance because insolvency cases represent only a small percentage of the total observations.

Although the source information was prepared for two prediction scenarios, the model, metrics, artifacts, Azure Machine Learning workflows, and scoring processes documented in this repository correspond exclusively to the 2024 scenario.

A public Excel file associated with the project will be available through the project portfolio.

---

## Model and Financial Features

The current model is:

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

1. Stratified training and validation split
2. Percentile-based winsorization
3. Yeo-Johnson transformation
4. Standardization
5. Logistic Regression training
6. Manual positive-class weight tuning
7. Five-fold stratified cross-validation
8. Decision-threshold optimization
9. Final evaluation on the validation sample
10. Model serialization

Winsorization, Yeo-Johnson transformation, and standardization are fitted only on the training data inside the scikit-learn Pipeline.

This design reduces the risk of data leakage from the validation sample.

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

Accuracy is not used as the primary evaluation metric because the negative class represents approximately 98.92% of the dataset. In this context, accuracy could present an overly optimistic and misleading view of model performance.

---

## Current MLOps Architecture

The current architecture covers model training, registration, deployment, and Excel-based scoring.

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
  - Percentile-based winsorization
  - Yeo-Johnson transformation
  - Standardization
  - Logistic Regression
        |
        v
Cross-validation and threshold tuning
        |
        v
Model evaluation
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

Historical financial data is validated and transformed into the 14 financial indicators required by the model. A leakage-safe pipeline applies winsorization, Yeo-Johnson transformation, and standardization before training the Logistic Regression model.

Cross-validation and threshold tuning are used to evaluate the model and determine the appropriate risk-classification threshold. The resulting model artifact can be registered in Azure Machine Learning and deployed through a Managed Online Endpoint.

For business use, new company records can be submitted through an Excel file. The required financial indicators are validated and converted into JSON requests. The model generates an insolvency risk probability and classification, which are returned in a new Excel file.

---

## Excel-Based Scoring

The project includes a practical business workflow for scoring new companies from an Excel file.

```text
Excel file with new companies
        |
        v
Validation of required financial indicators
        |
        v
Selection of the 14 model features
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

The input Excel file can contain additional accounting and descriptive columns. The scoring script validates the required information and selects only the 14 financial indicators used by the Logistic Regression model.

The output preserves the available identifying and descriptive columns and adds:

- `risk_probability`
- `risk_class`
- `threshold_used`
- `model_target`

A real scoring test was successfully completed using five company records.

The Azure Machine Learning endpoint received the financial indicators, generated the predictions, and returned the results in a new Excel file.

The endpoint used for the validation test was deleted after the test to control Azure consumption costs.

The public Excel file associated with the project will be available for download through the project portfolio.

---

## Azure Machine Learning

The project uses Azure Machine Learning to manage training, model artifacts, model registration, and deployment.

The current Azure Machine Learning configuration includes:

- Azure Machine Learning workspace
- Registered training Data Asset
- Registered execution environment
- `cpu-cluster` training compute
- YAML training-job definition
- Training artifacts stored as an Azure ML output
- Azure ML Model Registry
- Managed Online Endpoint configuration
- Deployment configuration
- Real-time JSON inference
- Excel-based scoring

### Current Azure Resources

- Data Asset: `insolvency-financial-data:1`
- Environment: `mlops-insolvency-env:2`
- Compute: `cpu-cluster`
- Training job definition: `azureml/train-job.yml`
- Model: Logistic Regression
- Target: `riesgo_24`

The training job uses a registered Azure ML Data Asset instead of uploading the local training file as part of the source-code context.

---

## GitHub Actions and Continuous Integration

The repository includes four GitHub Actions workflows:

- `azure-oidc-test.yml`
- `ci.yml`
- `register-environment.yml`
- `train-model.yml`

These workflows support:

- OpenID Connect authentication between GitHub and Azure
- Continuous integration
- Automated Python testing
- Azure ML environment registration
- Automated Azure ML training
- Training-log streaming
- Training-artifact download
- Model quality gates
- Automatic registration of approved models
- Verification of registered model versions
- Verification of the Azure ML training-job status

The training workflow is designed to follow this sequence:

```text
Repository checkout
        |
        v
Python configuration
        |
        v
Dependency installation
        |
        v
Automated tests
        |
        v
Azure authentication with OIDC
        |
        v
Azure ML training job
        |
        v
Training artifact download
        |
        v
Model quality gates
        |
        v
Approved model registration
        |
        v
Registered model verification
```

---

## Automated Tests

The project currently contains **11 automated tests**.

The tests validate:

- Percentile-based winsorization
- Preservation of input shape
- Outlier limits
- Complete model pipeline behavior
- Valid prediction probabilities
- Scoring responses
- Missing-feature validation
- Required YAML files
- YAML syntax
- Target variable configuration
- Logistic Regression configuration
- The 14 required model features
- Valid winsorization limits

The latest local test execution produced:

```text
11 passed in 2.20s
```

The continuous-integration workflow was also executed successfully in GitHub Actions.

---

## Model Quality Gates

The automated training workflow includes model quality gates.

The quality-gate process evaluates the training metrics before allowing the model to continue to automatic registration.

The purpose of this control is to prevent a newly trained model from being registered only because the training code executed successfully.

A model must satisfy the established quality requirements before it can be approved for automatic registration.

The quality-gate implementation is located in:

- `src/08_quality_gate.py`
- `.github/workflows/train-model.yml`

Any future change to the approval thresholds must include:

- Technical justification
- Test evidence
- Updated documentation
- A recorded decision in `DECISIONS.md`

---

## Project Structure

- `.github/workflows/`: GitHub Actions workflows for continuous integration, Azure authentication, environment registration, training, quality gates, and approved-model registration.
- `azureml/`: Azure Machine Learning training-job, endpoint, and deployment configurations.
- `config/`: Centralized project, model, preprocessing, and training configuration.
- `environment/`: Conda environment and package dependencies used by Azure Machine Learning.
- `src/`: Data validation, preprocessing, training, quality-gate, scoring, and inference scripts.
- `tests/`: Automated tests for model components, pipelines, scoring, and YAML configuration.
- `data/raw/`: Original source datasets used by the project.
- `data/processed/`: Modeling datasets generated by the preprocessing workflow.
- `data/scoring/`: Excel files containing new companies submitted for prediction.
- `models/`: Locally serialized model artifacts.
- `model_package/`: Model deployment package containing model-related artifacts, metrics, and the custom transformer.
- `notebooks/`: Exploratory analysis and model experimentation.
- `outputs/`: Validation reports, model metrics, tuning results, and prediction outputs.
- `.amlignore`: Files and folders excluded from the Azure ML code context.
- `.gitignore`: Local files, datasets, artifacts, credentials, and temporary files excluded from Git.
- `README.md`: Main technical and business documentation.
- `PROGRESS.md`: Detailed checklist of completed and pending project tasks.
- `DECISIONS.md`: Technical and methodological decision log.
- `requirements-dev.txt`: Dependencies used for local testing and continuous integration.

Local environments, credentials, temporary files, private artifacts, and datasets not selected for publication are excluded from GitHub.

Files intentionally published for the project portfolio, including the public Excel file and the article PDF, are managed separately in the portfolio repository.

---

## Technologies

The project currently uses:

- Python
- Pandas
- NumPy
- scikit-learn
- Statsmodels
- Joblib
- PyYAML
- OpenPyXL
- Pytest
- Azure Machine Learning
- Azure CLI
- Azure ML Model Registry
- Azure Managed Online Endpoints
- Git
- GitHub
- GitHub Actions
- OpenID Connect
- YAML

The current Machine Learning model is Logistic Regression.

XGBoost is not part of the current implementation.

FastAPI and Docker are planned components and are not yet implemented.

---

## Current Implementation Status

### Implemented and Validated

- [X] Logistic Regression model
- [X] Data validation
- [X] Leakage-safe preprocessing
- [X] Percentile-based winsorization
- [X] Yeo-Johnson transformation
- [X] Standardization
- [X] Five-fold stratified cross-validation
- [X] Positive-class weight tuning
- [X] Decision-threshold optimization
- [X] Validation-sample evaluation
- [X] Local model serialization
- [X] Azure Machine Learning workspace configuration
- [X] Azure ML Data Asset registration
- [X] Azure ML environment registration
- [X] Managed Online Endpoint deployment test
- [X] Real-time JSON inference test
- [X] Excel-based scoring test with five company records
- [X] Eleven automated tests
- [X] GitHub Actions continuous integration
- [X] OpenID Connect authentication
- [X] Cost-controlled endpoint and deployment deletion

### Implemented and Pending End-to-End Validation

- [ ] Execute the complete automated Azure ML training workflow
- [ ] Verify successful Azure ML training
- [ ] Verify the generated training artifacts
- [ ] Verify the model quality gates
- [ ] Verify automatic approved-model registration
- [ ] Verify the newly registered model version
- [ ] Confirm that `cpu-cluster` returns to 0 nodes

### Planned Development

- [ ] FastAPI prediction service
- [ ] API request and response validation
- [ ] API health-check endpoint
- [ ] API prediction endpoint
- [ ] Automated API tests
- [ ] Docker containerization
- [ ] Docker image validation
- [ ] Controlled deployment workflow
- [ ] Automated endpoint smoke tests
- [ ] Deployment rollback strategy
- [ ] Production inference-data collection
- [ ] Data-quality monitoring
- [ ] Feature-drift monitoring
- [ ] Population Stability Index monitoring
- [ ] Azure Monitor alerts
- [ ] Scheduled model retraining
- [ ] Champion-challenger model management
- [ ] Stable Git release tags

---

## Target MLOps Workflow

The target MLOps workflow represents the complete architecture planned for the project.

It combines components already implemented with additional capabilities that will be developed in future phases.

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
Production data collection
        |
        v
Data quality and feature drift monitoring
        |
        v
New labeled insolvency data
        |
        v
Model performance evaluation
        |
        v
Controlled model retraining
```

The process begins in the GitHub repository, where the source code, automated tests, Azure Machine Learning configurations, and documentation are stored.

GitHub Actions runs automated validation and coordinates the Azure Machine Learning training workflow.

The training metrics are evaluated through model quality gates. Only models that satisfy the established quality requirements can be registered as approved model versions.

FastAPI will provide validated endpoints for corporate insolvency predictions. Docker will package the API, model-loading logic, and required dependencies into a reproducible container.

After deployment, the solution will collect production inference data and monitor data quality and feature drift.

When new labeled insolvency data becomes available, model performance can be evaluated and a controlled retraining process can produce a new candidate model.

Every new candidate must pass the automated tests and quality gates before it can replace the currently approved model.

---

## Project Documentation

The project uses three main documentation files:

### `README.md`

Presents the business objective, dataset, model, metrics, architecture, technologies, implementation status, and target workflow.

### `PROGRESS.md`

Tracks completed tasks, pending tasks, the current phase, and the next technical step.

This is the first file to review when resuming project development.

### `DECISIONS.md`

Records technical and methodological decisions.

Each entry explains:

- What the decision is
- Why it was incorporated
- Which component uses it
- How another person can validate it
- What risks must be considered
- Whether it is implemented, validated, published, or operational

---

## Portfolio Resources

The project will also be presented through the public portfolio.

The portfolio page will provide access to:

- Project details
- GitHub repository
- Public Excel file
- Article PDF
- Model results
- MLOps architecture
- Current implementation status
- Planned FastAPI and Docker development

Portfolio:

`https://andrees-star.github.io/`

Repository:

`https://github.com/andrees-star/corporate-insolvency-mlops-azure`

---

## Current Project Status

Last updated: 2026-09-18

Current phase: Automated Azure Machine Learning training-workflow validation.

Last completed documentation milestones:

- `.amlignore` validated and published
- `PROGRESS.md` created and published
- `DECISIONS.md` created and published
- Eleven automated tests validated locally
- Main README reorganized and updated

Next technical step:

1. Open GitHub Actions.
2. Select `Train Model in Azure ML`.
3. Run the workflow from the `main` branch.
4. Verify the Azure ML training job.
5. Verify the generated artifacts.
6. Verify the model quality gates.
7. Verify automatic model registration.
8. Verify the new registered model version.
9. Confirm that `cpu-cluster` returns to 0 nodes.

Confirmed blockers: None.
