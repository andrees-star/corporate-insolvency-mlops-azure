# Model Card: Corporate Insolvency Risk Prediction

## Model Overview

- **Model name:** Corporate Insolvency Risk Prediction Model
- **Model type:** Logistic Regression
- **Azure ML registered version:** Version 2
- **Target variable:** `riesgo_24`
- **Prediction horizon:** Corporate insolvency events during 2024
- **Project status:** Approved by automated quality gates and registered in Azure Machine Learning
- **Estimated project completion:** 85%

---

## Model Purpose

The model estimates the probability that a Colombian company will enter a reorganization or liquidation process during 2024.

The model is designed as an early-warning and prioritization tool. It supports the identification of companies that may require additional financial analysis.

The model does not determine whether a company is legally insolvent and must not be used as the sole basis for automated credit, financing, investment, or insolvency decisions.

---

## Intended Users

The intended users include:

- Financial-risk analysts
- Credit-risk teams
- Portfolio-monitoring teams
- Business Intelligence teams
- Financial analysts
- Data Science and Machine Learning teams
- Model-risk and validation teams

Users must combine the model output with financial analysis, business context, data-quality review, and human judgment.

---

## Intended Uses

The model can support:

- Early identification of potential corporate insolvency risk
- Prioritization of companies for financial review
- Credit-risk monitoring
- Portfolio segmentation
- Generation of risk-alert lists
- Financial decision support
- Scenario analysis
- Research on corporate insolvency prediction

---

## Out-of-Scope Uses

The model is not intended for:

- Fully automated credit approval or rejection
- Automatic denial of financial products
- Legal determination of corporate insolvency
- Decisions without human review
- Predictions for horizons other than the documented 2024 target
- Use with data that does not contain the required financial indicators
- Use in populations that differ materially from the training population without additional validation
- Replacement of professional financial, accounting, legal, or risk analysis

---

## Target Definition

The binary target is `riesgo_24`:

- `1`: the company entered a reorganization or liquidation process during 2024.
- `0`: the company did not enter an insolvency process during 2024.

Companies that had already entered an insolvency process before the end of 2023 were excluded.

This exclusion ensures that the model predicts new insolvency events during 2024 rather than identifying companies that were already insolvent.

---

## Dataset

### Original Dataset

The original dataset contained:

- **31,276 company records**
- **57 original variables**
- Financial statements
- Financial ratios
- Corporate insolvency target information

### Processed Modeling Dataset

After selecting the required variables, replacing infinite values, and removing incomplete observations, the processed dataset contained:

- **27,926 observations**
- **14 financial indicators**
- **1 binary target**
- **300 positive cases**
- **27,626 negative cases**
- Event rate of approximately **1.07%**

### Training and Validation Split

The processed dataset was divided using a stratified 80/20 split:

- **Training observations:** 22,340
- **Validation observations:** 5,586
- **Positive cases in training:** 240
- **Positive cases in validation:** 60

The validation sample was not used to fit the preprocessing transformations or train the model.

---

## Financial Features

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

---

## Preprocessing and Training Pipeline

The model uses a leakage-safe scikit-learn pipeline containing:

1. Percentile-based winsorization
2. Yeo-Johnson transformation
3. Standardization
4. Logistic Regression

The preprocessing components are fitted only on the training data.

The training process includes:

- Stratified training and validation split
- Five-fold stratified cross-validation
- Manual positive-class weight tuning
- Decision-threshold optimization
- Validation-sample evaluation
- Model serialization

---

## Selected Configuration

The selected model configuration is:

- **Positive-class weight:** 35
- **Decision threshold:** 0.48
- **Cross-validation folds:** 5
- **Random state:** 42
- **Logistic Regression solver:** `lbfgs`
- **Maximum iterations:** 5,000

The threshold and positive-class weight were selected to balance insolvency detection with the number of false-positive alerts.

---

## Validation Performance

The model achieved the following results on the validation sample:

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

| Actual Class   | Predicted Risk | Predicted No Risk |
| -------------- | -------------: | ----------------: |
| Actual Risk    |             42 |                18 |
| Actual No Risk |            528 |             4,998 |

The model identified **70% of the actual insolvency cases** in the validation sample.

The relatively low precision reflects the severe class imbalance and the early-warning purpose of the model. Model alerts require additional financial review.

Accuracy is not used as the primary metric because the negative class represents approximately 98.93% of the processed dataset. Accuracy could therefore provide a misleading view of model performance.

---

## Automated Testing

The project includes 11 automated tests that validate:

- Percentile-based winsorization
- Preservation of input shape
- Limitation of extreme values
- Complete model-pipeline behavior
- Valid probability predictions
- Scoring responses
- Missing-feature detection
- Required YAML files
- YAML syntax
- Target-variable configuration
- Logistic Regression configuration
- The 14 required model features
- Valid winsorization limits

The most recent verified local execution produced:

```text
11 passed
```

The tests are executed in GitHub Actions before the workflow submits a training job to Azure Machine Learning.

If a CI test fails, the workflow stops and the Azure ML training job is not submitted.

---

## Quality Gates

The automated workflow evaluates the trained model before automatic registration.

The quality gates prevent a model from being registered only because the training code executed successfully.

The quality-gate process uses model metrics and predefined approval requirements.

If the model fails the quality gates:

- The workflow stops.
- The model is not approved.
- The model is not registered automatically.

If the model passes:

- The model is approved as a candidate.
- The model artifacts are registered in Azure Machine Learning.
- The registered version is verified by the workflow.

Passing the quality gates does not automatically prove that a new model is superior to the current champion. Formal champion-challenger comparison requires evaluation under the same data, metrics, business criteria, and human-review process.

---

## Azure Machine Learning Registration

The model was trained through an automated Azure Machine Learning job.

The workflow completed the following stages:

```text
GitHub Actions
        |
        v
11 automated tests
        |
        v
OIDC authentication
        |
        v
Azure ML training job
        |
        v
Training artifact generation
        |
        v
Model quality gates
        |
        v
Automatic model registration
        |
        v
Registered-model verification
```

The approved model was registered automatically in Azure Machine Learning as:

```text
Model version 2
```

The `cpu-cluster` returned to 0 nodes after the training process completed.

---

## Generated Artifacts

The Azure Machine Learning training job generated:

- `logistic_regression_riesgo24.pkl`
- `model_metrics.json`
- `tuning_results.csv`
- `statsmodels_logit_summary.txt`
- `custom_transformers.py`

These artifacts support model loading, metric validation, reproducibility, statistical review, and deployment.

---

## Deployment and Scoring

A previous Managed Online Endpoint deployment was validated successfully.

The deployment test included:

- Real-time JSON inference
- Excel-based scoring
- Validation using five company records
- Generation of risk probabilities and classifications
- Creation of a new Excel file with prediction results

The endpoint used for the test was deleted after validation to control Azure consumption costs.

Controlled deployment, rollback, FastAPI, and Docker containerization are planned components of the next project phases.

---

## Limitations

The model has the following limitations:

- Severe class imbalance
- Low precision despite acceptable recall
- A single documented prediction horizon
- Validation based on a single train/validation split
- No independent out-of-time validation
- No external validation dataset
- Potential sensitivity to changes in economic conditions
- Dependence on the quality and completeness of reported financial information
- Potential variation in performance across company sizes, sectors, regions, and reporting practices
- No confirmed production monitoring history
- No formal champion-challenger comparison has been completed
- The registered version 2 is approved, but it has not yet been formally declared the champion model

---

## Ethical and Business Considerations

The model can generate false positives and false negatives.

A false positive may cause an otherwise stable company to be selected for additional review.

A false negative may fail to identify a company that later enters an insolvency process.

Model results must therefore be interpreted with appropriate financial and business context.

Before operational use, performance differences should be evaluated across relevant groups such as:

- Company size
- Economic sector
- Geographic region
- Reporting quality
- Availability of financial information

The model must not be used as the sole basis for adverse financial decisions.

Human review is required before any operational action.

---

## Monitoring

Model monitoring is organized into three layers.

### Model Performance

The primary model metrics are:

- PR-AUC
- ROC-AUC
- Recall
- Precision
- F1-score
- MCC
- Type I Error
- Type II Error
- Confusion matrix

When new labeled insolvency outcomes become available, the model must be evaluated against the established validation metrics.

### Data and Score Stability

Population Stability Index will be calculated for:

- The 14 financial indicators
- The predicted risk score

The initial PSI interpretation criteria are:

- PSI below 0.10: stable distribution
- PSI between 0.10 and 0.25: moderate change requiring review
- PSI above 0.25: significant change requiring investigation

These thresholds are initial operational guidelines and must be reviewed against the project's sample sizes, binning strategy, and observed behavior.

### Operational Monitoring

Future endpoint monitoring will include:

- Prediction latency
- Service availability
- Failed requests
- Compute usage
- Endpoint cost
- Input-validation failures

---

## Maintenance

A PSI value above 0.25 triggers an investigation, not automatic retraining.

PSI identifies distribution changes but does not, by itself, prove that model performance has deteriorated.

Any retraining decision must include:

1. Data-quality validation
2. Investigation of the affected variables
3. Review of economic and business context
4. Availability of new labeled outcomes
5. Model-performance evaluation
6. Training of a challenger model
7. Automated CI tests
8. Quality-gate evaluation
9. Champion-challenger comparison
10. Human review and approval
11. Controlled deployment
12. Rollback capability

A challenger can replace the champion only after demonstrating acceptable performance under the same evaluation conditions and receiving approval.

---

## Human Oversight

Human review is required for:

- Interpretation of risk predictions
- Decisions based on model alerts
- Investigation of data drift
- Approval of model retraining
- Champion-challenger selection
- Model promotion
- Production deployment
- Rollback decisions

The model supports professional judgment but does not replace it.

---

## Versioning

The current Azure Machine Learning registered model is:

```text
Version 2
```

Azure ML model versions and Git repository releases are separate concepts:

- Azure ML model versions identify registered model artifacts.
- Git tags identify stable versions of the complete project code and configuration.

The formal Git version-tagging convention remains pending.

---

## Current Status

- **Model development:** Completed
- **Local validation:** Completed
- **Automated testing:** Completed
- **Azure ML training:** Completed
- **Artifact generation:** Completed
- **Quality-gate validation:** Completed
- **Automatic registration:** Completed
- **Registered-model verification:** Completed
- **Azure ML model version:** 2
- **Complete CI workflow:** Operational
- **Controlled deployment:** Pending
- **Rollback:** Pending
- **FastAPI:** Pending
- **Docker:** Pending
- **PSI monitoring:** Pending
- **Estimated project progress:** 85%

---

## Next Steps

1. Validate and publish this Model Card.
2. Define the Git version-tagging convention.
3. Create the first stable Git tag.
4. Implement controlled deployment of the approved model.
5. Implement rollback.
6. Develop the FastAPI prediction service.
7. Package the prediction service with Docker.
8. Implement PSI monitoring for the 14 variables and risk score.
9. Implement operational endpoint monitoring.
10. Define the champion-challenger governance process.
