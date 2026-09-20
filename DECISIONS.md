# Project Decision Log

This file records the technical and methodological decisions made during the project. Each entry explains what was decided, why it was incorporated, which component uses it, how another person can validate it, and its current implementation status.

## Status Definitions

- **Implemented:** The code, configuration, or document exists in the repository.
- **Validated:** Evidence confirms that the implementation works as expected.
- **Published:** The implementation has been committed and pushed to GitHub.
- **Operational:** The implementation has completed an end-to-end execution in Azure Machine Learning.

---

## 2026-09-10: Create a Centralized Project Progress File

### What it is

`PROGRESS.md` is the centralized checklist for tracking the phases, completed tasks, pending tasks, and current status of the MLOps project.

### Purpose

It allows developers, thesis reviewers, model owners, and future maintainers to understand the project status without depending on chat history.

### Why it was incorporated

The existing `README.md` was not updated after the initial Azure MLOps implementation. The repository did not have a dedicated progress-tracking document.

### Components and people that use it

- Project developers
- Model maintainers
- Thesis reviewers
- MLOps reviewers
- Future contributors

### How to validate it

Compare every completed checklist item against:

- Git commit history
- GitHub Actions workflows
- Automated test results
- Azure Machine Learning jobs
- Registered models and model versions
- Repository files and configurations

### Risks and maintenance

An incorrectly marked task can create a false impression that a component was validated. A task must not be marked as completed based only on intended behavior.

### Status

Implemented locally and prepared in Git. Pending commit and publication to GitHub.

---

## 2026-09-10: Use Logistic Regression as the Initial Production Candidate

### What it is

A Logistic Regression model that estimates the probability associated with the target variable `riesgo_24`.

### Purpose

Provide an interpretable initial model for corporate insolvency early-warning analysis.

### Why it was incorporated

Logistic Regression provides probability estimates, supports interpretation, and establishes a reproducible baseline for the first MLOps implementation.

### Component that uses it

- `src/03_train_model.py`
- `src/07_azure_train_job.py`
- `config/config.yml`
- Azure Machine Learning training workflow

### How to validate it

- Confirm `model_type: logistic_regression` in `config/config.yml`.
- Run the automated tests.
- Execute the Azure ML training job.
- Review the generated metrics and serialized model artifact.

### Status

Implemented and previously evaluated locally. Automated Azure ML workflow validation is pending.

---

## 2026-09-10: Register Only Models Approved by Quality Gates

### What it is

A control in `train-model.yml` that evaluates model metrics before registering a newly trained model.

### Purpose

Prevent a model that does not meet the predefined quality requirements from being registered automatically.

### Why it was incorporated

Successful code execution does not guarantee acceptable model quality. Model registration must depend on measurable approval criteria.

### Component that uses it

- `.github/workflows/train-model.yml`
- `src/08_quality_gate.py`
- Azure ML training artifacts
- Azure ML Model Registry

### How to validate it

1. Run the `Train Model in Azure ML` workflow.
2. Confirm that all 11 automated tests pass.
3. Confirm that the Azure ML training job finishes successfully.
4. Confirm that the quality-gate step evaluates the downloaded metrics.
5. Confirm that registration occurs only when the quality gate passes.
6. Verify the new model version in Azure Machine Learning.

### Risks and maintenance

Changing metric thresholds can alter the approval criteria. Any threshold change must include justification, test evidence, and a documented decision.

### Status

Implemented and published in GitHub. End-to-end execution and automatic registration validation remain pending.

---

## 2026-09-10: Exclude Local and Private Files from the Azure ML Code Context

### What it is

The `.amlignore` file defines which local files and folders are excluded when the project root is uploaded as an Azure ML training-code context.

### Purpose

Reduce unnecessary uploads and prevent private data, local environments, caches, and outdated artifacts from being sent to Azure ML.

### Why it was incorporated

The Azure ML job definition uses `code: ..`, which sends the project root as its code context. Exclusion rules are therefore necessary.

### Component that uses it

- Azure Machine Learning CLI
- `azureml/train-job.yml`
- GitHub Actions training workflow

### How to validate it

- Review `.amlignore`.
- Confirm that the registered Data Asset provides the training data.
- Confirm that the training script generates new model and metric artifacts.
- Confirm that required files under `src/`, `config/`, and `azureml/` remain available.
- Execute the training workflow and verify artifact creation.

### Risks and maintenance

An incorrect exclusion rule could prevent Azure ML from receiving a required file. Any new training dependency must be checked against `.amlignore`.

### Status

Validated, committed as `f46066a`, and published to GitHub. Final confirmation will occur during the complete Azure ML workflow execution.



---

## 2026-09-19: Complete Azure ML Workflow Validation

### Decision

Replace `az ml job stream` with periodic Azure ML job-status checks using `az ml job show`.

### Reason

The first GitHub Actions execution failed with `Incorrect padding` while streaming logs, although the Azure ML training job completed successfully.

The log-streaming failure prevented the workflow from continuing to the quality gates and automatic model registration.

### Implementation

The workflow now checks the Azure ML job status every 30 seconds:

- `Completed`: continues the workflow.
- `Failed` or `Canceled`: stops the workflow.
- Active status: waits and checks again.
- Timeout: stops after approximately 60 minutes.

### Validation Evidence

- 11 automated tests passed locally.
- Azure ML training completed successfully.
- Training artifacts were generated successfully.
- Model quality gates passed.
- The approved model was registered automatically as version `2`.
- The registered model version was verified.
- The complete GitHub Actions workflow finished successfully.
- `cpu-cluster` returned to `0` nodes.

### Result

The complete workflow from GitHub Actions to Azure Machine Learning was validated successfully.

### Status

Implemented, tested, published, and operational.



### Estimated Project Progress

**85% completed**

This estimate reflects the successful end-to-end validation of:

- GitHub Actions automation
- Azure Machine Learning training
- Training artifact generation
- Model quality gates
- Automatic model registration
- Registered model verification
- Compute scale-down to 0 nodes

### Next Steps

1. Update `PROGRESS.md` with the validated workflow results.
2. Define the Git version-tagging convention.
3. Create the first stable Git tag.
4. Implement controlled deployment of approved models.
5. Develop the FastAPI prediction service.
6. Add Docker containerization.
7. Implement production data collection.
8. Implement data-quality and feature-drift monitoring.
9. Configure alerts and controlled model retraining.



---
