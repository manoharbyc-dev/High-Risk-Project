-- MIMIC-III Cohort Extraction: Adult First ICU Stays, First 24h, Session 1
-- Purpose: Extract cohort for in-hospital mortality prediction
-- Rows: One per unique ICU stay (icustay_id)
-- Filters: age >= 18, first_icu_stay = TRUE
-- Features: raw vitals, labs, GCS, blood gas from first 24h
-- Demographics: gender, age_band (with 90+ cap), ethnicity, insurance
-- Outcome: in-hospital mortality (hospital_expire_flag from icustay_detail)

WITH cohort_base AS (
  -- Base cohort: adult first ICU stays
  SELECT
    i.subject_id,
    i.hadm_id,
    i.icustay_id,
    i.gender,
    -- Age: cap > 89 into "90+" band per MIMIC-III convention
    CASE
      WHEN i.admission_age >= 90 THEN '90+'
      WHEN i.admission_age >= 80 THEN '80-89'
      WHEN i.admission_age >= 70 THEN '70-79'
      WHEN i.admission_age >= 60 THEN '60-69'
      WHEN i.admission_age >= 50 THEN '50-59'
      WHEN i.admission_age >= 40 THEN '40-49'
      WHEN i.admission_age >= 30 THEN '30-39'
      ELSE '<30'
    END AS age_band,
    i.admission_age AS admission_age_raw,
    i.ethnicity,
    i.los_hospital,
    i.los_icu,
    i.hospital_expire_flag  -- in-hospital mortality (outcome)
  FROM `physionet-data.mimiciii_derived.icustay_detail` i
  WHERE i.first_icu_stay = TRUE
    AND i.admission_age >= 18
),

with_insurance AS (
  -- Add insurance from ADMISSIONS
  SELECT
    c.*,
    a.insurance
  FROM cohort_base c
  LEFT JOIN `physionet-data.mimiciii_clinical.admissions` a
    ON c.hadm_id = a.hadm_id
),

with_vitals AS (
  -- Join first-24h vitals
  SELECT
    c.*,
    v.heartrate_mean,
    v.heartrate_min,
    v.heartrate_max,
    v.sysbp_mean,
    v.sysbp_min,
    v.sysbp_max,
    v.diasbp_mean,
    v.diasbp_min,
    v.diasbp_max,
    v.meanbp_mean,
    v.meanbp_min,
    v.meanbp_max,
    v.resprate_mean,
    v.resprate_min,
    v.resprate_max,
    v.tempc_mean,
    v.tempc_min,
    v.tempc_max,
    v.spo2_mean,
    v.spo2_min,
    v.spo2_max
  FROM with_insurance c
  LEFT JOIN `physionet-data.mimiciii_derived.vitals_first_day` v
    ON c.icustay_id = v.icustay_id
),

with_labs AS (
  -- Join first-24h labs
  SELECT
    c.*,
    l.albumin_min,
    l.albumin_max,
    l.bilirubin_min,
    l.bilirubin_max,
    l.creatinine_min,
    l.creatinine_max,
    l.chloride_min,
    l.chloride_max,
    l.glucose_min,
    l.glucose_max,
    l.hematocrit_min,
    l.hematocrit_max,
    l.hemoglobin_min,
    l.hemoglobin_max,
    l.lactate_min,
    l.lactate_max,
    l.platelet_min,
    l.platelet_max,
    l.potassium_min,
    l.potassium_max,
    l.sodium_min,
    l.sodium_max,
    l.wbc_min,
    l.wbc_max
  FROM with_vitals c
  LEFT JOIN `physionet-data.mimiciii_derived.labs_first_day` l
    ON c.icustay_id = l.icustay_id
),

with_gcs AS (
  -- Join first-24h GCS
  SELECT
    c.*,
    g.mingcs,
    g.gcsmotor,
    g.gcsverbal,
    g.gcseyes
  FROM with_labs c
  LEFT JOIN `physionet-data.mimiciii_derived.gcs_first_day` g
    ON c.icustay_id = g.icustay_id
)

-- MAIN: Return final cohort
SELECT
  subject_id,
  hadm_id,
  icustay_id,
  gender,
  age_band,
  admission_age_raw,
  ethnicity,
  insurance,
  los_hospital,
  los_icu,
  hospital_expire_flag,
  -- Vitals
  heartrate_mean,
  heartrate_min,
  heartrate_max,
  sysbp_mean,
  sysbp_min,
  sysbp_max,
  diasbp_mean,
  diasbp_min,
  diasbp_max,
  meanbp_mean,
  meanbp_min,
  meanbp_max,
  resprate_mean,
  resprate_min,
  resprate_max,
  tempc_mean,
  tempc_min,
  tempc_max,
  spo2_mean,
  spo2_min,
  spo2_max,
  -- Labs
  albumin_min,
  albumin_max,
  bilirubin_min,
  bilirubin_max,
  creatinine_min,
  creatinine_max,
  chloride_min,
  chloride_max,
  glucose_min,
  glucose_max,
  hematocrit_min,
  hematocrit_max,
  hemoglobin_min,
  hemoglobin_max,
  lactate_min,
  lactate_max,
  platelet_min,
  platelet_max,
  potassium_min,
  potassium_max,
  sodium_min,
  sodium_max,
  wbc_min,
  wbc_max,
  -- GCS
  mingcs,
  gcsmotor,
  gcsverbal,
  gcseyes
FROM with_gcs
ORDER BY icustay_id
;
