# CLAUDE.md — MIMIC-III Explainable & Fair ICU Mortality Prediction

## Project
Solo graduate project for a Masters course "AI in Healthcare". Goal: predict
in-hospital mortality from the first 24 hours of an adult ICU stay using
MIMIC-III, then analyze the model with explainable AI (SHAP) and a
fairness/subgroup disparity analysis, including ONE honest bias-mitigation attempt.

This is a "high-risk" assignment graded on EFFORT and HONESTY, not on achieving
state-of-the-art numbers. A believable graduate-level result with clearly
documented limitations scores higher than an over-polished or over-claimed one.

## Golden rules
1. Masters-student level, not a production system. Prefer clear, standard,
   well-understood methods over clever or exotic ones.
2. Be honest. Never fabricate, round up, or oversell results. If something fails
   or underperforms, say so plainly and record it.
3. Keep a running limitations log at docs/LIMITATIONS.md. Append to it whenever
   we hit a caveat, shortcut, cleaning decision, or failure. This is graded
   material, not a confession.
4. Never invent numbers. Only report metrics actually produced by code that ran.
   If it has not been run, say "not yet run."
5. Plan before large changes. For any multi-file or multi-step task, outline the
   plan and WAIT for my confirmation before writing lots of code.

## DATA PRIVACY — non-negotiable
- MIMIC-III is credentialed data under a PhysioNet Data Use Agreement. This git
  repo is PUBLIC.
- NEVER commit any patient-level data: no MIMIC rows, no cohort CSVs, no
  extracts, no parquet, no SQL result dumps, no notebooks with rendered data
  cells, no figures showing raw patient records.
- All data files live under data/ which is gitignored. Only aggregate,
  de-identified summaries (counts, mean age, AUROC, etc.) may be committed.
- Never commit credentials: no service-account JSON, no API keys, no tokens.
  Use gcloud application-default credentials, never key files in the repo.
- If unsure whether something is safe to commit, ASK me first.

## Tech stack
- Python 3.11 in a local venv at .venv
- Data: MIMIC-III via Google BigQuery, queried with the google-cloud-bigquery
  client and/or the bq CLI
  - Raw tables:      physionet-data.mimiciii_clinical.*
  - Derived/concepts: physionet-data.mimiciii_derived.*
- ML: pandas, numpy, scikit-learn, xgboost; optional small PyTorch MLP as stretch
- Explainability: shap (TreeSHAP for XGBoost); optional captum IG/DeepLIFT for MLP
- Fairness: manual subgroup metrics; optionally fairlearn
- Plotting: matplotlib

## Repo layout (inside mimic-mortality-xai/)
- data/       gitignored — all raw and processed data
- sql/        cohort extraction queries
- src/        python modules: data, features, models, explain, fairness
- notebooks/  optional exploration; strip outputs before commit
- figures/    generated figures for the report — aggregate only
- results/    metrics tables as CSV/JSON — aggregate only
- docs/       LIMITATIONS.md and notes
- README.md
- requirements.txt

## Workflow conventions
- Fixed random seed (42) everywhere for reproducibility.
- Every script writes what it produced to results/ or figures/ and prints a
  short summary.
- Reusable logic goes in src/ as functions; keep scripts thin.
- Patient-level train/validation/test split, no leakage across splits. Report
  test metrics only once, at the very end.
- Clarity over cleverness in code and comments.

## Cohort definition (Session 1 target) — MIMIC-III specifics
- Population: adult (admission_age >= 18) ICU stays in MIMIC-III.
- If a patient has multiple ICU stays, use their FIRST ICU stay only. The derived
  table icustay_detail exposes first_icu_stay / first_hosp_stay flags — use them.
- STRONGLY PREFER the pre-built mimiciii_derived concept tables over raw tables:
  - icustay_detail  -> subject_id, hadm_id, icustay_id, gender, admission_age,
                       ethnicity, first_icu_stay, los, etc. (admission_age here is
                       already cleaned for the >89 date-shift quirk)
  - first-day vitals / labs concept tables (discover exact names via
    INFORMATION_SCHEMA first — likely names include vitals_first_day,
    labs_first_day, gcs_first_day, blood_gas_first_day, or pivoted_* variants).
  If the expected concept tables are missing or empty on BigQuery, fall back to
  computing first-24h aggregates from raw chartevents/labevents, and LOG that we
  had to do so in docs/LIMITATIONS.md.
- AGE QUIRK: MIMIC-III obscures ages > 89 by shifting DOB, making naive age ~300.
  Do NOT compute age naively from DOB. Use icustay_detail.admission_age if present.
  Cap/flag ages >= 90 into a single "90+" band. Log this decision.
- Sensitive attributes to keep as SEPARATE columns (not model features unless we
  deliberately test that): gender, age band, insurance (from ADMISSIONS),
  ethnicity (from ADMISSIONS / icustay_detail).
- Target: in-hospital mortality = ADMISSIONS.hospital_expire_flag for the
  associated hadm_id.
- Session 1 deliverable: one tidy analytic CSV in data/ (one row per included ICU
  stay) + an EDA summary saved to results/eda_summary.txt + row counts logged.

## What NOT to do
- No web frameworks, dashboards, Docker, CI, or deployment. Out of scope.
- Don't chase SOTA. A tuned XGBoost is the ceiling of ambition.
- Don't silently drop rows or impute in surprising ways — log every cleaning
  decision to docs/LIMITATIONS.md.
- Don't commit anything under data
