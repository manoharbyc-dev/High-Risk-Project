"""
Stage 2: Preprocessing, Feature Engineering, Train/Val/Test Split (Corrected)

Loads raw cohort, applies ethnicity mapping, feature selection, and patient-level
stratified split (70/15/15) with seed 42. All stays from a given patient go to
exactly ONE partition (no leakage across splits).

Output: data/train.csv, data/val.csv, data/test.csv, data/feature_info.json
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

print("=" * 100)
print("STAGE 2: PREPROCESSING & TRAIN/VAL/TEST SPLIT (PATIENT-LEVEL)")
print("=" * 100)

# Load raw cohort
cohort = pd.read_csv('data/cohort.csv')
print(f"\n✓ Loaded cohort.csv: {len(cohort):,} rows")

# Load ethnicity mapping
with open('docs/ethnicity_mapping.json') as f:
    ethnicity_map_data = json.load(f)
    ethnicity_mapping = ethnicity_map_data['mapping']

# Apply ethnicity mapping
cohort['ethnicity_group'] = cohort['ethnicity'].map(ethnicity_mapping).fillna('Other-Unknown')
print(f"✓ Applied ethnicity mapping: {cohort['ethnicity_group'].nunique()} groups")

# Define 46 features (drop albumin/bilirubin)
features_to_use = [
    # Vitals (21)
    'heartrate_mean', 'heartrate_min', 'heartrate_max',
    'sysbp_mean', 'sysbp_min', 'sysbp_max',
    'diasbp_mean', 'diasbp_min', 'diasbp_max',
    'meanbp_mean', 'meanbp_min', 'meanbp_max',
    'resprate_mean', 'resprate_min', 'resprate_max',
    'tempc_mean', 'tempc_min', 'tempc_max',
    'spo2_mean', 'spo2_min', 'spo2_max',
    # Labs (20: excluding albumin/bilirubin)
    'creatinine_min', 'creatinine_max',
    'glucose_min', 'glucose_max',
    'hematocrit_min', 'hematocrit_max',
    'hemoglobin_min', 'hemoglobin_max',
    'platelet_min', 'platelet_max',
    'potassium_min', 'potassium_max',
    'sodium_min', 'sodium_max',
    'chloride_min', 'chloride_max',
    'wbc_min', 'wbc_max',
    'lactate_min', 'lactate_max',
    # GCS (4)
    'mingcs', 'gcsmotor', 'gcsverbal', 'gcseyes',
    # Age (1)
    'admission_age_raw'
]

print(f"✓ Using {len(features_to_use)} features")

# ============================================================================
# PATIENT-LEVEL SPLIT (70/15/15) — CRITICAL TO AVOID LEAKAGE
# ============================================================================

# Get unique patients with their outcome
# Use ANY outcome (since a patient should have one outcome per first ICU stay)
patient_outcomes = cohort.groupby('subject_id')['hospital_expire_flag'].first().reset_index()
patient_outcomes.columns = ['subject_id', 'outcome']

print(f"\n✓ Unique patients: {len(patient_outcomes):,}")
print(f"  Patient mortality rate: {patient_outcomes['outcome'].mean():.4f}")

# Split patients into train (70%) and temp (30%)
train_pat, temp_pat = train_test_split(
    patient_outcomes, test_size=0.30, stratify=patient_outcomes['outcome'],
    random_state=SEED
)

# Split temp into val (50% of temp = 15% of total) and test (50% of temp = 15% of total)
val_pat, test_pat = train_test_split(
    temp_pat, test_size=0.50, stratify=temp_pat['outcome'],
    random_state=SEED
)

train_patient_ids = set(train_pat['subject_id'])
val_patient_ids = set(val_pat['subject_id'])
test_patient_ids = set(test_pat['subject_id'])

print(f"\n✓ Patient-level split:")
print(f"  Train patients: {len(train_pat):,} ({train_pat['outcome'].mean():.4f} mortality)")
print(f"  Val patients: {len(val_pat):,} ({val_pat['outcome'].mean():.4f} mortality)")
print(f"  Test patients: {len(test_pat):,} ({test_pat['outcome'].mean():.4f} mortality)")

# Map stays to their patient's partition
cohort['partition'] = cohort['subject_id'].apply(
    lambda pid: 'train' if pid in train_patient_ids else
                ('val' if pid in val_patient_ids else
                 ('test' if pid in test_patient_ids else 'ERROR'))
)

# Verify no stays were unmapped
assert 'ERROR' not in cohort['partition'].values, "ERROR: Some stays not mapped to partition"

# Split into train/val/test
cohort_train = cohort[cohort['partition'] == 'train'].copy()
cohort_val = cohort[cohort['partition'] == 'val'].copy()
cohort_test = cohort[cohort['partition'] == 'test'].copy()

# ============================================================================
# ASSERTIONS: No leakage, correct totals
# ============================================================================

print(f"\n✓ Partition checks:")

# Check 1: Total rows = original
total_rows = len(cohort_train) + len(cohort_val) + len(cohort_test)
assert total_rows == len(cohort), f"ERROR: Total rows {total_rows} != {len(cohort)}"
print(f"  ✓ Total rows: {total_rows} == {len(cohort)}")

# Check 2: Zero patient overlap
overlap_train_val = train_patient_ids & val_patient_ids
overlap_train_test = train_patient_ids & test_patient_ids
overlap_val_test = val_patient_ids & test_patient_ids
assert len(overlap_train_val) == 0, f"ERROR: {len(overlap_train_val)} patients in both train and val"
assert len(overlap_train_test) == 0, f"ERROR: {len(overlap_train_test)} patients in both train and test"
assert len(overlap_val_test) == 0, f"ERROR: {len(overlap_val_test)} patients in both val and test"
print(f"  ✓ Zero patient overlap between all partitions")

# Check 3: Mortality rates ~11.6% in each partition
train_mort = cohort_train['hospital_expire_flag'].mean()
val_mort = cohort_val['hospital_expire_flag'].mean()
test_mort = cohort_test['hospital_expire_flag'].mean()
print(f"  ✓ Mortality rates:")
print(f"    Train: {train_mort:.4f} ({cohort_train['hospital_expire_flag'].sum():,} deaths)")
print(f"    Val: {val_mort:.4f} ({cohort_val['hospital_expire_flag'].sum():,} deaths)")
print(f"    Test: {test_mort:.4f} ({cohort_test['hospital_expire_flag'].sum():,} deaths)")

print(f"\n✓ ICU stays per partition:")
print(f"  Train: {len(cohort_train):,} rows")
print(f"  Val: {len(cohort_val):,} rows")
print(f"  Test: {len(cohort_test):,} rows")
print(f"  Total: {total_rows:,} rows")

# ============================================================================
# FEATURE ENGINEERING & IMPUTATION
# ============================================================================

# Extract features (with NaNs)
X_train = cohort_train[features_to_use].copy()
X_val = cohort_val[features_to_use].copy()
X_test = cohort_test[features_to_use].copy()

# Impute with median FIT ON TRAIN ONLY (no leakage)
print(f"\n✓ Imputing with median (fit on TRAIN only):")
medians = X_train.median()
X_train = X_train.fillna(medians)
X_val = X_val.fillna(medians)
X_test = X_test.fillna(medians)

# Verify no missing values
assert X_train.isna().sum().sum() == 0, "ERROR: NaNs in train after impute"
assert X_val.isna().sum().sum() == 0, "ERROR: NaNs in val after impute"
assert X_test.isna().sum().sum() == 0, "ERROR: NaNs in test after impute"
print(f"  ✓ All missing values imputed, no NaNs remain")

# ============================================================================
# PREPARE OUTPUT
# ============================================================================

# Combine features + sensitive attributes + outcome + subject_id
def prepare_output(X_data, cohort_data):
    out = X_data.copy()
    out['gender'] = cohort_data['gender'].values
    out['age_band'] = cohort_data['age_band'].values
    out['insurance'] = cohort_data['insurance'].values
    out['ethnicity_group'] = cohort_data['ethnicity_group'].values
    out['hospital_expire_flag'] = cohort_data['hospital_expire_flag'].values
    out['subject_id'] = cohort_data['subject_id'].values
    return out

train_out = prepare_output(X_train, cohort_train)
val_out = prepare_output(X_val, cohort_val)
test_out = prepare_output(X_test, cohort_test)

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

train_out.to_csv('data/train.csv', index=False)
val_out.to_csv('data/val.csv', index=False)
test_out.to_csv('data/test.csv', index=False)

feature_info = {
    'features_in_X': features_to_use,
    'sensitive_attributes': ['gender', 'age_band', 'insurance', 'ethnicity_group'],
    'outcome_column': 'hospital_expire_flag',
    'features_dropped': ['albumin_min', 'albumin_max', 'bilirubin_min', 'bilirubin_max'],
    'reason_dropped': 'Albumin/bilirubin >50% missing; high imputation risk',
    'imputation_method': 'Median (fit on train, applied to val/test)',
    'stratification': 'Patient-level stratified split, 70/15/15, seed=42'
}

with open('data/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f"\n✓ Saved outputs:")
print(f"  data/train.csv ({len(train_out):,} rows)")
print(f"  data/val.csv ({len(val_out):,} rows)")
print(f"  data/test.csv ({len(test_out):,} rows)")
print(f"  data/feature_info.json")

print("\n" + "=" * 100)
print("STAGE 2 COMPLETE: Clean patient-level split with assertions passed")
print("=" * 100)
