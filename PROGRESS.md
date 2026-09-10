
# Project Progress

Last updated: 2026-09-10

## Phase 1: Data and Model Development

- [X] Validate the source dataset
- [X] Create the modeling dataset
- [X] Define target variable `riesgo_24`
- [X] Select 14 financial indicators
- [X] Implement leakage-safe preprocessing
- [X] Implement percentile-based winsorization
- [X] Implement Yeo-Johnson transformation
- [X] Train Logistic Regression model
- [X] Perform five-fold stratified cross-validation
- [X] Tune positive-class weights
- [X] Optimize the decision threshold
- [X] Evaluate the validation sample
- [X] Serialize the trained model

## Phase 2: Azure Machine Learning

- [X] Configure Azure Machine Learning workspace
- [X] Register training Data Asset
- [X] Register Azure ML environment
- [X] Configure `cpu-cluster`
- [X] Create Azure ML training job definition
- [X] Complete previous Managed Online Endpoint test
- [X] Complete previous real-time JSON inference test
- [X] Complete previous Excel-to-endpoint scoring test
- [X] Delete previous endpoint and deployment to control costs
- [ ] Execute the new automated training workflow
- [ ] Verify successful Azure ML training job
- [ ] Verify training artifacts
- [ ] Verify automatic model registration
- [ ] Verify the new registered model version
- [ ] Confirm that `cpu-cluster` returns to 0 nodes

## Phase 3: Continuous Integration and Automation

- [X] Publish repository on GitHub
- [X] Configure GitHub Actions
- [X] Configure authentication with OpenID Connect
- [X] Implement Python continuous integration
- [X] Implement 11 automated tests
- [X] Validate 11 tests locally
- [X] Implement automated Azure ML training workflow
- [X] Add automated model quality gates
- [X] Integrate quality gates into the training workflow
- [X] Add approved-model registration to `train-model.yml`
- [X] Add registered-model verification to `train-model.yml`
- [ ] Validate the complete workflow from GitHub Actions to Azure ML

## Phase 4: Documentation and Version Control

- [X] Configure `.gitignore`
- [X] Configure and validate `.amlignore`
- [X] Publish `.amlignore` in commit `f46066a`
- [ ] Update `README.md`
- [X] Create `PROGRESS.md`
- [ ] Create `DECISIONS.md`
- [ ] Define the version-tagging convention
- [ ] Create the first stable Git tag

## Phase 5: Deployment and Monitoring

- [ ] Implement controlled deployment of approved models
- [ ] Implement production inference-data collection
- [ ] Implement data-quality monitoring
- [ ] Implement feature-drift monitoring
- [ ] Configure Azure Monitor alerts
- [ ] Implement scheduled retraining
- [ ] Implement champion-challenger model management

## Current Project Status

Current phase: Documentation before automated workflow validation
Last completed step: `.amlignore` validated, committed, and published
Local test result: 11 tests passed in 2.20 seconds
Next step: Complete project documentation
Next MLOps execution step: Run `Train Model in Azure ML` from GitHub Actions
Blocked by: No confirmed blocker
