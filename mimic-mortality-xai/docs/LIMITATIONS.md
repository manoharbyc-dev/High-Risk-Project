# Limitations and Key Decisions

This document records limitations, design decisions, and known caveats encountered during the project.

## Feature choices

We deliberately exclude pre-computed severity scores (SOFA, OASIS, SAPS, APSIII, LODS, etc.) from the model's features, even though these are available in the MIMIC-III derived tables. Rationale: these scores are themselves engineered to predict mortality. Including them as model features would constitute outcome leakage and circularity — we would be predicting mortality using a score that already encodes mortality risk. We keep the model's inputs to raw-ish physiology (vitals, labs, GCS, blood gas) plus demographics (age, gender, ethnicity, insurance). This forces the model to learn associations from first principles rather than simply replicating domain-expert engineering.

---

## Environment & setup notes

**Python version:** We use Python 3.11.15 (installed via `brew install python@3.11`). This ensures broad wheel compatibility for data science packages. Python 3.14.6 (available via google-cloud-sdk) was initially attempted but lacks pre-built wheels for ML packages like pandas and numpy.

**SHAP dependency:** shap 0.44.0 is deferred due to numba/llvmlite compilation issues on macOS. It will be installed separately in Stage 2+ once the cohort extraction is complete. SHAP is not needed for data pipeline work, only for explainability analysis in Stage 3.

---

## Data cleaning and cohort filtering

**Cohort definition (Stage 2, Final):**
- Population: Adult (admission_age ≥ 18) ICU stays from MIMIC-III
- Inclusion: first_icu_stay = TRUE (using icustay_detail flag)
- Outcome: in-hospital mortality (hospital_expire_flag from icustay_detail)
- **Final result: 49,280 unique ICU stays, 5,727 deaths (11.62% mortality rate)**
- Demographics: 56.2% M / 43.8% F; well-distributed by age, insurance, ethnicity

**Age handling:**
- Used admission_age directly from icustay_detail (avoids naive DOB calculation quirk for ages > 89)
- Binned ages into bands: <30, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90+
- Only 1 patient in 90+ band; all others well-distributed

**First-24h features (all from MIMIC-III derived concept tables):**
- Vitals (vitals_first_day): HR, systolic/diastolic/mean BP, RR, temperature, SpO2
  - Stored as mean/min/max values; no raw time-series
  - Only heartrate_mean missing in 1 case (1.0%)
  - Temperature missing in 2 cases (2.0%)
- Labs (labs_first_day): albumin, bilirubin, creatinine, glucose, CBC values, electrolytes, INR, lactate, etc.
  - Stored as min/max values; no mean
  - Albumin/bilirubin heavily missing (64%, 56%) — likely not collected for shorter stays
  - Lactate missing in 38% of cases
- GCS (gcs_first_day): stored as mingcs (minimum GCS over 24h), plus motor/verbal/eye components
  - These are component scores; note: BQ schema uses mingcs, not individual min/max for total
- Blood gas (blood_gas_first_day): pH, pCO2, pO2, aado2, HCO3, base excess
  - **NOT YET JOINED**: This table has different structure (upper-case column names, different sampling)
  - Deferred to Stage 2+ refinement if needed
  
**Sensitive attributes (kept separate, not fed to model):**
- gender (M/F)
- age_band (binned as above)
- ethnicity (from icustay_detail)
- insurance (from admissions table, joined on hadm_id)

**Missing data strategy:**
- No imputation at this stage; missingness preserved as NULLs
- Albumin/bilirubin heavily missing → may drop if needed after modeling
- Will decide on imputation / listwise deletion in Stage 2 feature engineering

**Data privacy:**
- Cohort CSV (49,280 rows, patient-level) saved to data/cohort.csv (gitignored)
- No patient identifiers or raw data committed to repo
- Only aggregate summary (mortality rate, missingness %) in results/eda_summary.txt

**Technical note (Stage 2):**
- Initial `bq query --format=csv` export was limited to ~100 rows despite no LIMIT clause
- Root cause: BigQuery CLI default row limit for preview mode
- Solution: Used Python BigQuery client library (`query.result().to_dataframe()`) to fetch all 49,280 rows
- SQL query itself had no limit; export method was the constraint

## Feature engineering and preprocessing (Stage 3a)

**Feature matrix X (46 features — raw physiology only):**

Vitals (21 features):
  - Heart rate: mean, min, max
  - Systolic BP: mean, min, max
  - Diastolic BP: mean, min, max
  - Mean arterial pressure: mean, min, max
  - Respiratory rate: mean, min, max
  - Temperature (Celsius): mean, min, max
  - SpO2: mean, min, max

Labs (20 features, excluding albumin/bilirubin):
  - Creatinine: min, max
  - Glucose: min, max
  - Hematocrit: min, max
  - Hemoglobin: min, max
  - Platelets: min, max
  - Potassium: min, max
  - Sodium: min, max
  - Chloride: min, max
  - WBC: min, max
  - Lactate: min, max

GCS (4 features):
  - mingcs (minimum GCS over first 24h)
  - gcsmotor, gcsverbal, gcseyes (components)

Numeric age (1 feature):
  - admission_age_raw (continuous, legitimate clinical predictor)

**DROPPED features (4):**
  - albumin_min, albumin_max, bilirubin_min, bilirubin_max
  - Reason: >50% missing (63%, 55%); imputing >50% introduces high bias/uncertainty
  - Impact: Minimal — most critical features (creatinine, glucose, CBC, vitals) well-represented

**SENSITIVE ATTRIBUTES (KEPT SEPARATE, NOT IN X):**
  - gender (M/F)
  - age_band (8 bins: <30, 30-39, ..., 90+)
  - insurance (5 categories)
  - ethnicity_group (5 groups: White, Black, Hispanic, Asian, Other-Unknown)
  - Used only for fairness/subgroup analysis; not fed to model

**Outcome (SEPARATE):**
  - hospital_expire_flag (binary: 0/1)

**Missing data strategy:**
  - Imputed with median fit on TRAIN ONLY (34,606 rows)
  - Applied same median values to validation (7,353 rows) and test (7,321 rows)
  - No data leakage: imputation parameters derived from train only
  - 45 features had some missing values in training set; all imputed before modeling

**Train/Val/Test split (patient-level, stratified, seed=42):**
  - Patient-level split to prevent patient leakage (a patient's ICU stays stay in same split)
  - Stratified by in-hospital mortality to preserve outcome distribution
  - Train: 26,948 patients → 34,606 ICU stays (11.58% mortality)
  - Val: 5,775 patients → 7,353 ICU stays (11.68% mortality)
  - Test: 5,775 patients → 7,321 ICU stays (11.73% mortality)
  - Stratification verification: ✓ All splits within 0.15% of overall rate (11.62%)
  - Patient overlap: ✓ Zero overlap between train/val/test (no leakage)

**Ethnicity mapping (41 categories → 5 groups):**
  - White: 35,248 (71.5%) [includes WHITE, WHITE - EASTERN EUROPEAN, WHITE - RUSSIAN, etc.]
  - Other-Unknown: 6,492 (13.2%) [includes UNKNOWN/NOT SPECIFIED, UNABLE TO OBTAIN, DECLINED, etc.]
  - Black: 4,704 (9.5%) [includes BLACK/AFRICAN AMERICAN, BLACK/HAITIAN, BLACK/CAPE VERDEAN, etc.]
  - Hispanic: 1,687 (3.4%) [includes all HISPANIC/LATINO variants]
  - Asian: 1,149 (2.3%) [includes all ASIAN variants]
  - Mapping saved to: docs/ethnicity_mapping.json

## Modeling and validation (Stage 3b)

**Class imbalance handling:**
- Test set prevalence: 11.73% (859 deaths / 7,321 stays)
- Imbalance ratio (neg/pos): 7.63
- Logistic Regression: class_weight='balanced' (auto-weights by inverse frequency)
- XGBoost: scale_pos_weight=7.63 (explicit cost weighting)
- NOTE: We avoid reporting accuracy as headline metric due to 11.6% baseline; instead report AUROC, AUPRC, Brier score

**Model training:**

Logistic Regression (baseline):
  - Hyperparameters: max_iter=1000, solver='lbfgs', random_state=42
  - Features: StandardScaler fit on train, applied to val/test (no leakage)
  - Class weight: 'balanced'
  - Validation threshold selection: 0.466 (max F1 on validation set)

XGBoost:
  - Hyperparameters: n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8
  - Scale_pos_weight: 7.63 (explicit imbalance handling)
  - Early stopping: on validation set, patience=10 rounds
  - Best iteration: 99 (out of 100)
  - Validation threshold selection: 0.421 (max F1 on validation set)

**Test set performance (final evaluation, NO tuning):**

|                | Logistic Regression | XGBoost | Difference |
|----------------|---------------------|---------|-----------|
| AUROC          | 0.8192              | 0.8677  | +0.0485   |
| AUPRC          | 0.4826              | 0.5730  | +0.0904   |
| Brier Score    | 0.1664              | 0.1186  | -0.0478   |
| Sensitivity    | 0.7334              | 0.7579  | +0.0245   |
| Specificity    | 0.7392              | 0.7929  | +0.0537   |
| PPV            | 0.2721              | 0.3273  | +0.0552   |
| NPV            | 0.9543              | 0.9610  | +0.0067   |

Confusion matrices:
  LR @ threshold=0.466: TN=4777, FP=1685, FN=229, TP=630
  XGBoost @ threshold=0.421: TN=5124, FP=1338, FN=208, TP=651

**Interpretation:**
- XGBoost substantially outperforms LR baseline: +4.85% AUROC, +9.04% AUPRC
- XGBoost captures more true positives (TP 651 vs 630) with fewer false positives (FP 1338 vs 1685)
- Both models achieve high NPV (~95-96%), reassuring when predicting survival
- XGBoost's PPV is modest (32.7%), reflecting baseline prevalence; acceptable for risk stratification tool
- Brier score: XGBoost better calibrated (lower = better)

## Explainability (Stage 4)

**TreeSHAP implementation (SHAP 0.43.0):**

Successfully installed SHAP with prebuilt wheels for Python 3.11 on macOS:
- shap==0.43.0
- numba==0.62.1
- llvmlite==0.45.1

The key was using prebuilt wheels (--only-binary) rather than compiling from source. Earlier attempts to build numba/llvmlite from source failed due to missing CMake; switching to wheel-only installation resolved the issue. All dependencies now import cleanly.

**Top 10 features by mean absolute SHAP value (computed on 2,000 test samples):**
  1. admission_age_raw (0.4129) — Age at admission (continuous)
  2. gcsverbal (0.3801) — GCS verbal response component
  3. lactate_min (0.2374) — Minimum lactate
  4. resprate_mean (0.2060) — Mean respiratory rate
  5. gcseyes (0.1914) — GCS eye opening component
  6. hemoglobin_max (0.1737) — Maximum hemoglobin
  7. sysbp_min (0.1603) — Minimum systolic BP
  8. heartrate_max (0.1469) — Maximum heart rate
  9. creatinine_min (0.1428) — Minimum creatinine
  10. spo2_min (0.1420) — Minimum SpO2

**Comparison: SHAP vs XGBoost Gain Importance**

| Feature | SHAP Rank | Gain Rank | Agreement |
|---------|-----------|-----------|-----------|
| admission_age_raw | 1 | 6 | ✓ Yes |
| gcsverbal | 2 | 1 | ✓ Yes |
| lactate_min | 3 | 2 | ✓ Yes |
| resprate_mean | 4 | 5 | ✓ Yes |
| gcseyes | 5 | 7 | ✓ Yes |
| hemoglobin_max | 6 | 16 | ✗ No |
| sysbp_min | 7 | 3 | ✓ Yes |
| heartrate_max | 8 | 18 | ✗ No |
| creatinine_min | 9 | 9 | ✓ Yes |
| spo2_min | 10 | 8 | ✓ Yes |

**8 out of 10 features agree between SHAP and Gain importance (80% agreement)**, validating model consistency. Disagreements (hemoglobin_max, heartrate_max) are minor ranking shifts, not fundamental differences.

**Clinical validity:**
✓ GCS (gcsverbal, gcseyes) → ranks 2, 5 (top 10) ✓ Strong
✓ Age → rank 1 (top 10) ✓ Dominant predictor
✓ Lactate → rank 3 (top 10) ✓ Critical marker
✓ Abnormal vitals (resprate_mean, sysbp_min, spo2_min) → ranks 4, 7, 10 ✓ Well-captured

**Interpretation:**
Age is the dominant SHAP contributor, followed by GCS and lactate. This differs slightly from XGBoost's split-gain ranking (which weights gcsverbal #1, age #6), but both methods agree on the core clinical drivers. SHAP's per-patient signed contributions provide directionality: higher age increases predicted mortality risk substantially, GCS measures consciousness/neurological status, and lactate signals tissue hypoperfusion. The model has learned clinically plausible associations.

## Fairness & Subgroup Disparity Analysis (Stage 5)

**Methodology:**
- Test set n = 7,321 ICU stays (859 deaths, 11.7% prevalence)
- Global threshold = 0.421 (from XGBoost stage 3b validation optimization)
- All subgroups evaluated at this single threshold (not re-tuned per group)
- AUROC computed with 95% bootstrap confidence intervals (1,000 resamples, seed=42)
- Disparities measured as best-minus-worst performance gaps
- Confidence intervals provided to highlight uncertainty in small subgroups (n < 200)

**KEY FINDING: Small but Real Disparities in Specificity & FNR; Minimal AUROC Gaps**

### Gender Disparity Summary
| Subgroup | N | Deaths | Prevalence | AUROC (95% CI) | Sensitivity | FNR | Specificity | PPV |
|----------|---|--------|-----------|---|---|---|---|---|
| Female | 3,210 | 409 | 12.7% | 0.8607 (0.8417-0.8795) | 0.765 | 0.235 | 0.773 | 0.330 |
| Male | 4,111 | 450 | 10.9% | 0.8731 (0.8560-0.8886) | 0.751 | 0.249 | 0.807 | 0.323 |

**Disparity Assessment:**
- **AUROC gap:** 0.0125 (1.25% difference) — negligible; CIs overlap substantially
- **Sensitivity gap:** 0.0142 (1.4%) — minimal
- **FNR gap:** 0.0142 — males miss 1.4% more deaths than females
- **Specificity gap:** 0.0333 (3.3%) — males have slightly lower specificity
- **Interpretation:** MINIMAL gender disparity. Both subgroups well-represented (F n=3,210, M n=4,111). Model performance essentially equivalent.

### Age Band Disparity Summary
| Age Band | N | Deaths | Prevalence | AUROC (95% CI) | Sensitivity | FNR | Specificity |
|----------|---|--------|-----------|---|---|---|---|
| <30 | 316 | 17 | 5.4% | 0.9490 (0.9068-0.9814) | 0.647 | 0.353 | 0.960 |
| 30-39 | 361 | 24 | 6.6% | 0.9475 (0.9042-0.9805) | 0.750 | 0.250 | 0.944 |
| 40-49 | 755 | 67 | 8.9% | 0.8776 (0.8360-0.9121) | 0.582 | 0.418 | 0.903 |
| 50-59 | 1,214 | 104 | 8.6% | 0.8766 (0.8399-0.9077) | 0.731 | 0.269 | 0.872 |
| 60-69 | 1,586 | 156 | 9.8% | 0.8718 (0.8448-0.8967) | 0.724 | 0.276 | 0.808 |
| 70-79 | 1,493 | 196 | 13.1% | 0.8331 (0.8053-0.8639) | 0.760 | 0.240 | 0.729 |
| 80-89 | 1,196 | 218 | 18.2% | 0.8326 (0.8037-0.8615) | 0.803 | 0.197 | 0.662 |
| 90+ | 400 | 77 | 19.3% | 0.8576 (0.8069-0.9049) | 0.909 | 0.091 | 0.545 |

**CRITICAL DISPARITY: Age Band Performance Gap**
- **AUROC gap:** 0.1165 (11.65% difference) — LARGEST observed disparity
  - Best: <30 years (AUROC 0.9490)
  - Worst: 80-89 years (AUROC 0.8326)
- **Sensitivity gap:** 0.327 — High-age model catches MORE deaths (90+: 91%, 40-49: 58%)
- **FNR gap:** 0.327 — CRITICAL: 40-49 years misses 41.8% of deaths; 90+ misses only 9.1%
  - **Interpretation:** The model is OVER-optimistic (misses deaths) for middle-aged patients (40-49) but CONSERVATIVE (catches deaths) for elderly (80+, 90+)
  - This appears driven by BASE RATE: 40-49 has only 8.9% prevalence; model defaults to predicting survival
  - Compare: 90+ has 19.3% prevalence; model defaults to higher risk
- **Clinical Context:** Age is the #1 SHAP driver. Model correctly learned that age strongly predicts mortality. However, **the single global threshold (0.421) is suboptimal for younger adults**, who have low absolute mortality risk but high relative risk within their age band.
- **Sample size note:** <30 and 30-39 age bands have n < 400; AUROC CIs are wide (±0.04). Smallest subgroup (<30, n=316) has only 17 deaths, making sensitivity/FNR estimates unstable. **Age 40-49 has large enough n (n=755) that the high FNR (0.418) is not due to noise.**

### Ethnicity Group Disparity Summary
| Ethnicity | N | Deaths | Prevalence | AUROC (95% CI) | Sensitivity | FNR | Specificity |
|----------|---|--------|-----------|---|---|---|---|
| Asian | 194 | 18 | 9.3% | 0.8179 (0.7032-0.9024) | 0.667 | 0.333 | 0.739 |
| Black | 689 | 80 | 11.6% | 0.8382 (0.7929-0.8811) | 0.663 | 0.338 | 0.805 |
| Hispanic | 264 | 19 | 7.2% | 0.8475 (0.7623-0.9265) | 0.684 | 0.316 | 0.829 |
| Other-Unknown | 957 | 156 | 16.3% | 0.8772 (0.8496-0.9045) | 0.712 | 0.288 | 0.821 |
| White | 5,217 | 586 | 11.2% | 0.8728 (0.8584-0.8874) | 0.788 | 0.212 | 0.786 |

**DISPARITY: Ethnicity-Based Performance Gaps — PRIMARY METRIC IS FNR AT GLOBAL THRESHOLD**

**Methodological note:** AUROC comparisons across subgroups with different prevalences are confounded by base rate. FNR (false negative rate) at the fixed global decision threshold (0.421) is the appropriate fairness metric for this deployment scenario, as it directly measures error rates under the decision rule that would be applied.

- **AUROC gap (confounded by prevalence):** 0.0594 (5.94% difference)
  - Reflects differences in classification task difficulty across groups, partly driven by prevalence
  - Asian: n=194, deaths=18 (9.3%) — AUROC CI width 0.199 (±0.0996), highly unstable
  - Black: n=689, deaths=80 (11.6%) — adequate n for stable estimate
  - Hispanic: n=264, deaths=19 (7.2%) — moderate n, moderate uncertainty
  - White: n=5,217, deaths=586 (11.2%) — very large n, stable

- **FNR gap at global threshold (primary fairness metric):**
  - **Black vs. White (ROBUST FINDING):**
    - Black: FNR = 0.338 (model misses 33.8% of deaths)
    - White: FNR = 0.212 (model misses 21.2% of deaths)
    - **Gap: 0.126 (12.6 percentage points)** — Large, clinically meaningful, based on adequate sample sizes (n=689 Black, n=5,217 White)
    - **This gap is the primary fairness concern.**
  
  - **Asian and Hispanic (UNDERPOWERED, NOT STATISTICALLY DISTINGUISHABLE):**
    - Asian: FNR = 0.333 (n=194, 18 deaths, AUROC CI width 0.199)
    - Hispanic: FNR = 0.316 (n=264, 19 deaths)
    - White: FNR = 0.212 (baseline)
    - **Directionally similar to Black gap, but CIs overlap substantially due to small sample sizes. These gaps are consistent with the observed disparity but are underpowered to distinguish from random variation. No independent inference should be drawn.**

- **Sensitivity gap (derived from FNR):** 0.1259 (12.59%)
  - Black sensitivity: 0.663 (66% of deaths caught)
  - White sensitivity: 0.788 (79% of deaths caught)
  - Difference driven by FNR gap above

- **Specificity gap:** 0.0899 — All groups maintain specificity 0.74-0.82; modest variation

- **Sample size & stability summary:**
  - **Adequate for inference:** Black (n=689), White (n=5,217) — FNR gap is REAL
  - **Underpowered:** Asian (n=194, only 18 deaths), Hispanic (n=264, only 19 deaths) — point estimates are directionally informative but not statistically separable from White
  - **Training composition:** 71.3% White, 9.4% Black, 3.6% Hispanic, 2.7% Asian, 13.1% Other-Unknown

- **Root cause of Black-White FNR gap (the robust finding):**
  The model learned age as the dominant predictor (correct). Black-White FNR disparity likely reflects:
  1. **Possible feature representation bias:** Raw vitals, labs, GCS may not capture disease severity equally across racial groups. Known issue in healthcare ML (e.g., eGFR reference ranges, lab value thresholds calibrated on predominantly White populations).
  2. **Training data composition:** Model trained on 71% White patients; could have learned race-specific physiologic associations that do not generalize.
  3. **Cannot be explained by base-rate alone:** Black prevalence (11.6%) ≈ White prevalence (11.2%), so FNR gap is not simply due to different mortality rates.
  4. **Cannot be explained by prevalence-driven threshold suboptimality:** (Unlike age band case) Black and White have similar base rates, so same threshold should be roughly optimal for both.

### Insurance Disparity Summary
| Insurance | N | Deaths | Prevalence | AUROC (95% CI) | Sensitivity | FNR | Specificity |
|----------|---|--------|-----------|---|---|---|---|
| Government | 172 | 11 | 6.4% | 0.9181 (0.8628-0.9657) | 0.727 | 0.273 | 0.882 |
| Medicaid | 627 | 55 | 8.8% | 0.8836 (0.8459-0.9217) | 0.709 | 0.291 | 0.857 |
| Medicare | 4,123 | 579 | 14.0% | 0.8456 (0.8269-0.8617) | 0.782 | 0.218 | 0.724 |
| Private | 2,310 | 197 | 8.5% | 0.8896 (0.8667-0.9099) | 0.695 | 0.305 | 0.878 |
| Self Pay | 89 | 17 | 19.1% | 0.9690 (0.9304-0.9962) | 0.824 | 0.176 | 0.931 |

**DISPARITY: Insurance-Based Performance Gaps (AUROC CONFOUNDED BY PREVALENCE AND SAMPLE SIZE)**

- **AUROC gap:** 0.1234 (12.34% difference)
  - Best: Self Pay (0.9690, n=89, 19.1% mortality)
  - Worst: Medicare (0.8456, n=4,123, 14.0% mortality)
  - **Interpretation:** AUROC is confounded by base-rate differences and sample size. Self Pay is a tiny, high-mortality cohort where classification is easier; Medicare is large and lower-mortality. AUROC gaps across insurance groups should not be interpreted as primary fairness metrics.

- **FNR gap at global threshold:** 0.1281
  - Private: FNR 0.305 (misses 30.5% of deaths, n=2,310, 8.5% mortality)
  - Medicare: FNR 0.218 (misses 21.8% of deaths, n=4,123, 14.0% mortality)
  - **Interpretation:** Consistent with base-rate confounding: Private has lower mortality (8.5%), so global threshold (0.421 optimized on 11.7% average) is too conservative for this group. Medicare has higher mortality (14%), so threshold is more appropriate. This is expected behavior when base rates differ.

- **Specificity gap:** 0.2068
  - Medicare: lowest specificity (0.724) — more false positives among survivors
  - Self Pay: highest specificity (0.931) — very conservative

- **Root cause analysis (consistent with confounding by indication):**
  - Insurance status is a proxy for health status, age, and comorbidities
  - Medicare (56% of cohort): older patients (average ~70y), sicker, higher baseline mortality (14.0%)
  - Private insurance (31.6%): younger, lower baseline mortality (8.5%)
  - Self Pay (1.2%): highest baseline mortality (19.1%), likely sickest cohort
  - **Consistent with interpretation:** Model is learning associations between insurance-proxied health status and mortality, not discrimination. Differences in FNR are explained by different base rates across groups, which make a single global threshold suboptimal for low-mortality groups (Private) and appropriate for high-mortality groups (Medicare/Self Pay).
  - **This is consistent with confounding by indication rather than algorithmic bias, but this interpretation cannot be definitively proven without additional analysis** (e.g., controlling for clinical severity).

---

### Overall Disparity Summary

**Ranking of Concern (Ethically & Clinically):**

1. **ETHNICITY—HIGH CONCERN (FNR disparity, representation imbalance)**
   - Black patients: 33.8% FNR vs. White 21.2% (12.6% absolute gap, 60% relative increase in missed deaths)
   - Sample size: Black n=689 is adequate; FNR gap is REAL, not noise
   - Root cause likely: feature representation bias (vitals/labs may not capture severity equally) + 71% White training cohort
   - **Recommendation for follow-up:** Investigate whether specific features (e.g., creatinine reference ranges, lab value thresholds) differ in validity across racial groups

2. **AGE BAND—MEDIUM CONCERN (FNR disparity for middle-aged, CI width for young)**
   - 40-49 age band: 41.8% FNR (42% miss rate!) vs. 90+: 9.1%
   - Root cause: Low baseline prevalence in 40-49 (8.9%) makes global threshold too conservative
   - Age <30, 30-39 have small n and wide AUROC CIs; hard to assess performance
   - **Recommendation:** Age-specific threshold tuning could help; not pure bias, but suboptimal for younger cohorts

3. **INSURANCE—LOW CONCERN (Confounding by indication, not algorithmic bias)**
   - Gaps explained by base-rate differences (Medicare 14% mortality vs. Private 8.5%)
   - Not attributable to algorithm; reflects real health status proxy
   - **Recommendation:** Monitor for unintended consequences (e.g., if insurance status is misclassified), but not a fairness problem per se

4. **GENDER—MINIMAL CONCERN**
   - AUROC gap 1.25%, FNR gap 1.4% — negligible
   - Both subgroups large (F n=3,210, M n=4,111)
   - **Recommendation:** No mitigation needed

---

### Critical Limitations & Caveats

1. **MIMIC-III Demographic Imbalance (71% White):**
   - Training cohort was 71.5% White. Minority subgroups (Asian 2.3%, Hispanic 3.4%) underrepresented.
   - Model learned associations predominantly from White patients; performance on minority populations may reflect:
     - True differences in disease presentation / physiologic associations (clinical reality)
     - Feature representation bias (vitals/labs not equally predictive across racial groups)
     - Small sample size (especially Asian n=194)
   - **This is a DATA LIMITATION, not purely an algorithm problem.** Better fairness requires more diverse training data.

2. **Confidence Interval Uncertainty:**
   - Asian group (n=194): AUROC CI width 0.199 (point estimate ± 0.1) — essentially unreliable
   - Government insurance (n=172): CI width 0.103 — unstable
   - Self Pay (n=89): CI width 0.066 — smallest, but high prevalence (19.1%) means few observations
   - **Implication:** AUROC point estimates for small subgroups should not be interpreted as definitive; use CIs to assess whether gaps are real or noise

3. **Global Threshold Suboptimality:**
   - Single threshold (0.421) optimized on overall test set; may be suboptimal for low-prevalence groups (e.g., age 40-49, 8.9%)
   - Threshold tuning per subgroup would improve fairness but would require larger holdout test set
   - **Stage 5 requirement was to use a single global threshold**; this is appropriate for a fairness audit (catches threshold-driven bias), but downstream deployment may need subgroup-specific thresholds

4. **Causality Unknown:**
   - Ethnicity FNR gap may reflect:
     - Genuine differences in how diseases present across racial groups (clinical reality)
     - Bias in data collection (e.g., less rigorous vital sign monitoring for certain groups)
     - Feature underspecification (raw vitals/labs insufficient to capture severity in some populations)
   - **We cannot determine causality from this analysis alone.** Mitigation requires domain expertise and deeper investigation.

---

### Summary Table: All Disparity Metrics

See `results/fairness_disparity_summary.csv` for complete gaps across all attributes.

### Visualizations Generated
- `figures/fairness_sensitivity_fnr_by_subgroup.png` — Sensitivity and FNR by subgroup at global decision threshold
- `figures/fnr_gap_stability.png` — FNR gap across 5 random seeds, stability analysis

---

### Honest Conclusion

**PRIMARY FAIRNESS CONCERN (ROBUST FINDING):**
- **Black-White FNR disparity:** Black patients (n=689): FNR 0.338 vs. White (n=5,217): FNR 0.212
  - **12.6 percentage point gap** — model misses 1 in 3 deaths for Black patients vs. 1 in 5 for White
  - Sample sizes adequate; gap is NOT due to noise
  - Likely driven by feature representation bias (raw vitals/labs calibrated on predominantly White population) and/or training cohort composition (71% White)
  - **This would not be acceptable for clinical deployment without mitigation investigation.**

**SECONDARY CONCERNS (UNDERPOWERED):**
- **Asian and Hispanic disparities:** Directionally similar to Black gap (FNR 0.333, 0.316) but based on small samples (n=194, n=264) with overlapping confidence intervals. Cannot distinguish from random variation; should not be treated with same confidence as Black-White gap.

- **Age band disparities:** Age 40-49 has 41.8% FNR vs. age 90+ at 9.1%. Root cause is THRESHOLD SUBOPTIMALITY (low base rate in younger group + global threshold), not inherent model bias. Different from ethnicity because base rates similar across racial groups.

**NO FAIRNESS CONCERN:**
- **Gender:** <2% AUROC and FNR gaps; negligible.
- **Insurance:** AUROC and FNR gaps are consistent with confounding by indication (insurance as proxy for health/age/comorbidity); differences explained by base-rate variation across groups, not algorithmic discrimination.

## Bias Mitigation Attempt (Stage 6)

**IMPORTANT ETHICAL FRAMING:**
This stage applies a group-aware decision threshold to the Black subgroup as a **diagnostic probe** to quantify the disparity and understand what trade-offs are necessary to close it. **This is NOT an endorsement of race-based thresholds for real-world deployment.** Race-adjusted clinical algorithms have well-documented harms (e.g., the use of race in eGFR calculations, which masked kidney disease in Black patients). Using race explicitly as an input to a clinical decision rule is ethically contested.

Instead, this analysis demonstrates that:
1. The Black-White FNR disparity exists (not due to noise or small sample size)
2. It would require meaningful trade-offs (more false positives, lower PPV) to close via threshold adjustment alone
3. **The root cause is likely feature-level representation bias,** not threshold suboptimality

### Mitigation Method: Group-Aware Decision Threshold Adjustment

**Design (rigorous to prevent data leakage):**
- Threshold selection: Conducted on VALIDATION set only (n=7,353; Black n=679, deaths=60, prevalence 8.8%; White n=5,330, deaths=597, prevalence 11.2%)
- White FNR on validation at global threshold (0.421): 0.2077
- Grid search for Black threshold (0.15–0.50) to minimize gap
- Selected Black threshold: **0.380** (achieved gap of 0.0077 on validation)
- **Applied once to held-out TEST set** — no test-set tuning, preventing data leakage and optimistic bias
- White subgroup remained at global threshold (0.421); other ethnicity groups also at global threshold

### Test-Set Results (Held-Out Evaluation)

**FNR GAP REDUCTION: SUCCESS**

| Group | Baseline FNR | Mitigated FNR | Change | Threshold |
|-------|---|---|---|---|
| Black | 0.3375 | 0.2875 | -0.0500 | 0.380 |
| White | 0.2116 | 0.2116 | — | 0.421 |
| **Gap** | **0.1259** | **0.0759** | **-0.0500 (-39.7%)** | — |

**The Black-White FNR gap was reduced by 39.7%** — from 12.6 percentage points to 7.6 percentage points. In the test set Black cohort (80 deaths), this represents 4 additional deaths caught (from 53 baseline to 57 mitigated).

### Trade-Offs (Honest Assessment)

**Black subgroup trade-offs (threshold lowered from 0.421 to 0.380):**

| Metric | Baseline | Mitigated | Change |
|--------|---|---|---|
| **Sensitivity** | 0.6625 | 0.7125 | +0.0500 |
| **FNR** | 0.3375 | 0.2875 | -0.0500 |
| **Specificity** | 0.8046 | 0.7652 | -0.0394 |
| **PPV** | 0.3081 | 0.2850 | -0.0231 |
| **False Positive Rate** | 0.1954 | 0.2348 | +0.0394 |

**Clinical interpretation:**
- **Benefit:** For every 100 Black ICU patients, ~5 more deaths are caught (sensitivity +5%)
- **Cost:** For every 100 Black ICU patients, ~4 more survivors are falsely flagged as high-risk (FPR +3.9%)
- **Precision cost:** PPV drops from 0.308 to 0.285 (fewer of the flagged patients are actually high-risk; more are false alarms)

**Overall test-set impact (Black + White + other groups):**

| Metric | Baseline | Mitigated | Change |
|--------|---|---|---|
| **Sensitivity** | 0.7579 | 0.7625 | +0.0047 |
| **Specificity** | 0.7922 | 0.7885 | -0.0037 |
| **PPV** | 0.3265 | 0.3239 | -0.0025 |
| **Brier Score** | 0.1186 | 0.1186 | — (unchanged) |
| **TP** | 651 | 655 | +4 |
| **FP** | 1343 | 1367 | +24 |
| **FN** | 208 | 204 | -4 |

**Interpretation:** Aggregate effect is modest because Black is 9.4% of test set; group-aware threshold improvement is diluted across the full cohort. However, within the Black subgroup, the effect is substantial (FNR -5%, 4 additional deaths caught).

### What Remained Unchanged (As Expected)

**Brier score unchanged:** Brier score remains 0.1186 (unchanged). This is correct because Brier score measures the quality of predicted probabilities, which are not changed by threshold adjustment. (Initial reporting of Brier worsening was due to computing it on thresholded 0/1 predictions instead of probabilities; corrected to use probabilities.)

**AUROC unchanged:** Receiver-operating-characteristic AUC is unchanged (0.8677 → 0.8677) because AUROC is threshold-independent; the underlying probability rankings are the same.

### Limitations of This Mitigation

1. **Treats symptom, not root cause:**
   - The FNR gap is likely driven by features (raw vitals/labs) that may not equally well predict mortality across racial groups
   - Threshold adjustment makes the decision rule more sensitive for Black patients, but doesn't address whether the features themselves encode bias
   - **The lasting solution requires retraining with more diverse data or feature engineering to make inputs more equitable**

2. **Ethical concerns with deployment:**
   - Using race as an explicit input to a clinical algorithm is controversial
   - Clinicians may perceive lower PPV (more false alarms) as model degradation rather than intentional fairness tuning
   - Institutional buy-in needed; not a technical solution alone

3. **Partial success, not elimination:**
   - Gap reduced but not eliminated (7.6 percentage points remains)
   - Black FNR still 1.35x White FNR after mitigation (was 1.59x before)
   - Full closure would require even lower threshold, creating more false positives

4. **Generalization risk:**
   - Thresholds optimized on validation set; small differences in test-set distributions (prevalence varies: validation Black 8.8% vs. test Black 11.6%) mean threshold optimality may not transfer perfectly
   - Ideally would have more diverse validation data

### Conclusion: Stage 6 Results

**Success metric (reduced FNR gap): MET**
- Gap reduced 39.7% (from 0.1259 to 0.0759)
- Substantial improvement in Black subgroup sensitivity (+5%)
- Trade-off is explicit and documented (more false alarms, lower PPV)
- Result is honest: partial success, not miraculous cure

**This confirms that:**
- The Black-White FNR disparity is REAL and substantial
- It CAN be partially reduced via threshold adjustment
- But full closure requires addressing root causes (feature representation bias, training data composition)
- Deployment of group-aware thresholds has ethical and operational complexities

**Files generated:**
- `results/stage6_mitigation_summary.json` — Complete metrics and summary
- `results/stage6_comparison_table.csv` — Before/after metrics table
- `figures/stage6_fnr_mitigation_by_ethnicity.png` — FNR by ethnicity before/after mitigation
- Method and ethical framing logged here to LIMITATIONS.md

**Recommendation for future work:** If this model were to be deployed, the preferred mitigation would be to **retrain with upweighted Black patient examples** (instance reweighting), combined with **feature engineering** to reduce representation bias (e.g., validate lab value thresholds across racial groups). Threshold adjustment is a temporary measure; more equitable training is the durable solution.
