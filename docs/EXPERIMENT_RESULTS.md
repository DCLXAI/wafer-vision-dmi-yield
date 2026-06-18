# WaferVision Experiment Results

## Objective

WaferVision evaluates semiconductor defect detection as a production yield problem rather than a simple image-classification demo. The main objective is to improve rare-defect recall under severe class imbalance, then expose the result through DMI feature engineering and chamber-level root-cause analysis.

## Data Balancing Summary

| Step | Dataset State | Normal:Defect Ratio | Technique | Decision Value |
|---|---|---:|---|---|
| Raw | Missing labels and normal-heavy target distribution | 93.4 : 6.6 | Label cleanup and missing-value policy | Defines the actual imbalance risk |
| Baseline | Numeric features with missing values filled | 93.4 : 6.6 | Median imputation | Provides a reproducible baseline |
| Sampling | Training split balanced for rare defects | 50.0 : 50.0 | SMOTE oversampling | Raises defect recall from a failed baseline |
| DMI Reduced | Low-variance noise removed before sampling | 50.0 : 50.0 | Variance Threshold 0.05 + SMOTE | Improves generalization and traceback stability |

## Model Comparison

| Experiment | Algorithm | Preprocessing / Sampling | Precision | Recall (Defect Detection) | F1-Score | Interpretation |
|---:|---|---|---:|---:|---:|---|
| 01 | XGBoost (Baseline) | Median imputation | 0.82 | 0.12 | 0.21 | High apparent precision, but most rare defects are missed |
| 02 | XGBoost + SMOTE | Median imputation + SMOTE | 0.42 | 0.68 | 0.52 | Recall improves sharply, while false positives increase |
| 03 | LightGBM + SMOTE | Variance Threshold 0.05 + SMOTE | 0.54 | 0.82 | 0.65 | Best balance for defect detection and chamber traceback |

## Why Recall Leads The Scorecard

In wafer yield analysis, a missed defect can hide a tool drift or chamber excursion until more lots are affected. For that reason, this project treats Recall as the first-pass screening metric and F1-Score as the balance metric. Precision is still tracked because excessive false positives create avoidable engineering review load.

## Research Journal Commit Convention

Use daily commits that describe the research intent, not just the file change.

```text
feat: add SMOTE sampling to solve class imbalance (06-15)
refactor: optimize XGBoost hyperparameters using GridSearchCV (06-16)
docs: update README with performance metric table (06-17)
feat: reposition dashboard as DMI yield intelligence platform (06-18)
```

| Date | Commit Type | Research Note |
|---|---|---|
| 06-15 | `feat` | Introduced SMOTE to correct the normal-heavy training distribution. |
| 06-16 | `refactor` | Reworked model tuning around Recall and F1-Score rather than Accuracy. |
| 06-17 | `docs` | Summarized preprocessing, balancing, and model metric transitions in Markdown tables. |
| 06-18 | `feat` | Reframed the product copy around DMI, yield risk, and chamber root-cause analysis. |
