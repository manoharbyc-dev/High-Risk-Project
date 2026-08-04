# MIMIC-III Explainable & Fair ICU Mortality Prediction

## Project Summary

This is a graduate-level analysis of in-hospital mortality prediction from the first 24 hours of adult ICU admission using MIMIC-III data. The project implements a complete pipeline: cohort extraction → feature engineering and train/val/test split → model training (logistic regression baseline and XGBoost) → explainability analysis (TreeSHAP) → subgroup fairness audit with multi-seed stability analysis → bias mitigation via group-aware thresholds.

**Scope:** This is a Masters-course assignment graded on effort and honesty, not production systems. All findings are documented with clear limitations.

## Data Access & Credentialing

This project requires **MIMIC-III credentialed access** via PhysioNet. The cohort extraction queries the MIMIC-III dataset hosted on Google BigQuery under the `physionet-data` project.

**Important:** This repository is **public** and contains **no patient-level data**. The data/ directory (which is .gitignored) holds only:
- `cohort.csv`: The de-identified, aggregated cohort (49,280 ICU stays) extracted via BigQuery
- `train.csv`, `val.csv`, `test.csv`: Preprocessed splits with raw features and sensitive attributes (kept separate)
- Backups and intermediate files

All committed results are aggregate metrics only (AUROC, AUPRC, counts, visualizations without patient identifiers).

## Prerequisites

- **Python 3.11** (or compatible 3.11.x version)
- **Google Cloud SDK** with credentials set up for BigQuery access:
  ```bash
  gcloud auth login
  gcloud auth application-default login
  gcloud config set project <YOUR_GCP_PROJECT_ID>
  ```
  The queries target the shared `physionet-data` BigQuery project; your own project is used only for billing.
- **Local environment:** Create a Python virtual environment and install dependencies:
  ```bash
  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

## Running the Pipeline

The pipeline has 5 stages, to be run in order:

### **Stage 1: Cohort Extraction (BigQuery)**
```bash
# Manual: Execute sql/01_cohort_extraction.sql in BigQuery console, or:
bq query --use_legacy_sql=false < sql/01_cohort_extraction.sql
# Result: 49,280-row cohort extracted to data/cohort.csv (via Python client in next stage)
```

If extracting via Python instead of bq CLI, use the Google BigQuery client library (included in requirements.txt).

### **Stage 2: Preprocessing & Train/Val/Test Split**
```bash
python src/02_preprocess_split.py
# Outputs:
#   data/train.csv (34,476 rows)
#   data/val.csv (7,382 rows)
#   data/test.csv (7,422 rows)
#   data/feature_info.json (feature list and metadata)
```
- Applies ethnicity mapping (41 → 5 categories)
- Selects 46 raw physiologic features (drops albumin/bilirubin due to >50% missingness)
- Median-imputes missing values (fit on train, applied to val/test — no leakage)
- Patient-level stratified split (70/15/15) with seed=42

### **Stage 3: Model Training**
```bash
python src/03_train_models.py
# Outputs:
#   results/model_metrics.json (AUROC, AUPRC, Brier, confusion matrices)
#   results/feature_importance_full.csv (XGBoost gain-based importance)
#   results/feature_importance_top10.csv (top 10 features)
```
- Trains Logistic Regression baseline (with scaled features) and XGBoost
- Handles class imbalance explicitly (11.4% prevalence): LR uses `class_weight='balanced'`, XGBoost uses `scale_pos_weight`
- Validates on val set (F1-max threshold tuning); evaluates once on test set

### **Stage 4: SHAP Explainability**
```bash
python src/04_shap_explain.py
# Outputs:
#   figures/shap_summary_beeswarm.png (per-sample SHAP contributions, top 15 features)
#   figures/shap_mean_abs_bar.png (mean |SHAP| by feature, top 15)
#   results/shap_feature_importance.csv (all features ranked by mean |SHAP|)
#   results/shap_vs_gain_comparison.csv (comparison to XGBoost gain importance)
```
- Computes TreeSHAP on 2,000 test samples (sample size noted in results)
- Validates that top clinical drivers (age, GCS, lactate) rank highly

### **Stage 5: Fairness Audit (Subgroup Analysis)**
```bash
python src/05_fairness_audit.py
# Outputs:
#   results/fairness_subgroup_metrics.csv (per-subgroup: N, AUROC with 95% CI, FNR, sensitivity, specificity, PPV)
#   results/fairness_disparity_summary.csv (best-vs-worst gaps per attribute)
```
- Computes per-subgroup metrics (gender, age band, ethnicity, insurance)
- Evaluates at global decision threshold (no per-group tuning)
- Reports 95% bootstrap confidence intervals on AUROC
- **Stability check (offline):** Gap is estimated via 5-seed analysis (seeds: 42, 1, 7, 123, 2024); see figures/fnr_gap_stability.png

### **Stage 6: Bias Mitigation — Group-Aware Threshold**
```bash
python src/06_mitigation.py
# Outputs:
#   results/stage6_mitigation_summary.json (Black-White FNR before/after mitigation)
#   results/stage6_comparison_table.csv (summary metrics)
#   figures/stage6_fnr_mitigation_by_ethnicity.png (FNR by ethnicity before/after)
```
- Selects group-aware thresholds on validation set (no test-set tuning)
- Applies thresholds once to test set
- Reports FNR gap reduction and trade-offs (e.g., change in PPV/FPR)

## Results Directory

Key outputs for the report:

| File | Purpose |
|------|---------|
| `results/model_metrics.json` | Test-set AUROC, AUPRC, Brier for LR and XGBoost |
| `results/fairness_subgroup_metrics.csv` | Per-subgroup counts, AUROC+CI, FNR, sensitivity, specificity |
| `results/shap_feature_importance.csv` | Features ranked by mean \|SHAP\| (TreeSHAP on test set) |
| `results/stage6_mitigation_summary.json` | FNR gap and mitigation metrics |
| `figures/shap_summary_beeswarm.png` | SHAP beeswarm plot (top 15 features) |
| `figures/shap_mean_abs_bar.png` | SHAP mean absolute importance (top 15 features) |
| `figures/fairness_sensitivity_fnr_by_ethnicity_seed42.png` | Sensitivity & FNR by ethnicity (seed 42 baseline) |
| `figures/fnr_gap_stability.png` | **Key fairness finding:** Black-White FNR gap across 5 random seeds, showing stability/instability |
| `figures/stage6_fnr_mitigation_by_ethnicity.png` | FNR before/after group-aware threshold adjustment |

## Key Findings (Seed 42)

- **Test cohort:** 7,422 ICU stays, 846 deaths (11.4% prevalence)
- **Model performance:** XGBoost AUROC 0.8697, AUPRC 0.5474
- **Explainability:** SHAP top 5 features are GCS verbal, age, respiratory rate, lactate, GCS eyes — all clinically sensible
- **Fairness:** Black-White FNR gap at global threshold is 0.0221; AUROC confidence intervals overlap, indicating no statistically significant difference in discrimination. **Stability check across 5 seeds shows gap mean 0.0236 (range 0.0005–0.0430)**, demonstrating that the gap estimate depends on random split but is consistently small.

See `docs/LIMITATIONS.md` for detailed methodology, design decisions, and caveats.

## Repository Structure

```
mimic-mortality-xai/
├── README.md                    (this file)
├── CLAUDE.md                    (project specification)
├── requirements.txt             (Python dependencies)
├── sql/
│   └── 01_cohort_extraction.sql (BigQuery cohort extraction)
├── src/
│   ├── __init__.py
│   ├── 02_preprocess_split.py   (preprocessing and train/val/test split)
│   ├── 03_train_models.py       (LR baseline and XGBoost training)
│   ├── 04_shap_explain.py       (TreeSHAP explainability)
│   ├── 05_fairness_audit.py     (subgroup fairness analysis)
│   └── 06_mitigation.py         (group-aware threshold adjustment)
├── data/                        (gitignored — no patient data committed)
│   ├── cohort.csv              (49,280 ICU stays, extracted from BigQuery)
│   ├── train.csv, val.csv, test.csv (preprocessed splits)
│   ├── feature_info.json       (feature list and metadata)
│   └── cohort_BACKUP.csv       (read-only backup of raw cohort)
├── results/                     (aggregate metrics — aggregate only, safe to commit)
│   ├── model_metrics.json
│   ├── fairness_subgroup_metrics.csv
│   ├── shap_feature_importance.csv
│   └── [other metrics CSVs and JSONs]
├── figures/                     (visualizations — aggregate, no patient data)
│   ├── shap_summary_beeswarm.png
│   ├── shap_mean_abs_bar.png
│   ├── fairness_sensitivity_fnr_by_ethnicity_seed42.png
│   ├── fnr_gap_stability.png   (key fairness stability plot)
│   └── [other PNG figures]
└── docs/
    ├── LIMITATIONS.md           (comprehensive design decisions and caveats)
    └── ethnicity_mapping.json   (41 categories → 5 groups mapping)
```

## Data Privacy

This repository contains **NO patient-level data**. The data/ directory is gitignored. All committed outputs are aggregate metrics (counts, AUROC, visualizations, feature importance) with no individual patient information. See CLAUDE.md for the complete data privacy policy.

## Notes on Reproducibility

- All random seeds are fixed (seed=42 for the canonical run; stability analysis uses seeds 42, 1, 7, 123, 2024)
- Cohort extraction requires BigQuery access; a cached cohort.csv is provided for reproducibility
- Model training is deterministic given fixed seeds and input data
- SHAP analysis resamples the test set (seed=42); sample size noted in results

For full methodology and design decisions, see `docs/LIMITATIONS.md`.
