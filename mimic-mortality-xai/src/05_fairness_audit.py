"""
Stage 5: Fairness & Subgroup Disparity Analysis

Computes per-subgroup metrics (AUROC with 95% bootstrap CI, FNR, sensitivity, etc.)
for gender, age_band, ethnicity_group, and insurance.

Output: results/fairness_subgroup_metrics.csv, results/fairness_disparity_summary.csv
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("=" * 100)
print("STAGE 5: FAIRNESS & SUBGROUP DISPARITY ANALYSIS")
print("=" * 100)

# Load data
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')

with open('data/feature_info.json') as f:
    feature_cols = json.load(f)['features_in_X']

X_train = train_df[feature_cols].values
y_train = train_df['hospital_expire_flag'].values

X_val = val_df[feature_cols].values
y_val = val_df['hospital_expire_flag'].values

X_test = test_df[feature_cols].values
y_test = test_df['hospital_expire_flag'].values

print(f"Loaded: Train {len(X_train):,}, Val {len(X_val):,}, Test {len(X_test):,}")

# Train XGBoost
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, random_state=SEED,
    use_label_encoder=False, eval_metric='logloss'
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              early_stopping_rounds=10, verbose=False)

# Get predictions
y_proba_test = xgb_model.predict_proba(X_test)[:, 1]
global_threshold = 0.421
y_pred_test = (y_proba_test >= global_threshold).astype(int)

print(f"✓ Model trained, predictions on test set")

# ============================================================================
# BOOTSTRAP CI FOR AUROC
# ============================================================================

def bootstrap_auroc_ci(y_true, y_proba, n_bootstrap=1000, ci=95):
    if len(np.unique(y_true)) < 2:
        return np.nan, (np.nan, np.nan)

    auc_scores = []
    np.random.seed(SEED)
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_scores.append(roc_auc_score(y_true[idx], y_proba[idx]))

    auc_scores = np.array(auc_scores)
    point_estimate = roc_auc_score(y_true, y_proba)
    lower = np.percentile(auc_scores, (100 - ci) / 2)
    upper = np.percentile(auc_scores, 100 - (100 - ci) / 2)
    return point_estimate, (lower, upper)

# ============================================================================
# COMPUTE SUBGROUP METRICS
# ============================================================================

def compute_subgroup_metrics(y_true, y_pred, y_proba, attr_values, attr_name):
    results = []

    for attr_val in sorted(attr_values.unique()):
        mask = attr_values == attr_val
        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]
        y_proba_sub = y_proba[mask]

        n = len(y_true_sub)
        n_pos = y_true_sub.sum()
        prevalence = n_pos / n if n > 0 else np.nan

        if n < 1 or len(np.unique(y_true_sub)) < 2:
            results.append({
                'Attribute': attr_name, 'Subgroup': str(attr_val), 'N': n, 'N_Deaths': n_pos,
                'Prevalence': prevalence, 'AUROC': np.nan, 'AUROC_CI_Lower': np.nan,
                'AUROC_CI_Upper': np.nan, 'AUPRC': np.nan, 'Sensitivity': np.nan,
                'FNR': np.nan, 'Specificity': np.nan, 'PPV': np.nan, 'Brier': np.nan
            })
            continue

        # AUROC with bootstrap CI
        auroc_point, (auroc_lower, auroc_upper) = bootstrap_auroc_ci(y_true_sub, y_proba_sub)

        # AUPRC
        try:
            auprc = average_precision_score(y_true_sub, y_proba_sub)
        except:
            auprc = np.nan

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred_sub).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan

        # Calibration
        brier = brier_score_loss(y_true_sub, y_proba_sub)

        results.append({
            'Attribute': attr_name, 'Subgroup': str(attr_val), 'N': n, 'N_Deaths': n_pos,
            'Prevalence': prevalence, 'AUROC': auroc_point, 'AUROC_CI_Lower': auroc_lower,
            'AUROC_CI_Upper': auroc_upper, 'AUPRC': auprc, 'Sensitivity': sensitivity,
            'FNR': fnr, 'Specificity': specificity, 'PPV': ppv, 'Brier': brier
        })

    return pd.DataFrame(results)

# Compute for each attribute
print("\nComputing subgroup metrics...")

all_results = []
for attr_name in ['gender', 'age_band', 'ethnicity_group', 'insurance']:
    df_attr = compute_subgroup_metrics(y_test, y_pred_test, y_proba_test,
                                       test_df[attr_name], attr_name)
    all_results.append(df_attr)

df_all_subgroups = pd.concat(all_results, ignore_index=True)

print("\nSubgroup counts:")
for attr in ['gender', 'age_band', 'ethnicity_group', 'insurance']:
    subset = df_all_subgroups[df_all_subgroups['Attribute'] == attr]
    print(f"\n{attr.upper()}:")
    for _, row in subset.iterrows():
        print(f"  {row['Subgroup']:20s}: N={row['N']:5d}, Deaths={row['N_Deaths']:3d}, FNR={row['FNR']:.4f}")

# ============================================================================
# DISPARITY GAPS
# ============================================================================

print("\nDisparity gaps (best - worst)...")

disparity_summary = []

for attr_name in ['gender', 'age_band', 'ethnicity_group', 'insurance']:
    df_attr_subset = df_all_subgroups[df_all_subgroups['Attribute'] == attr_name].copy()
    df_valid = df_attr_subset.dropna(subset=['AUROC', 'Sensitivity', 'FNR'])

    if len(df_valid) < 2:
        continue

    auroc_gap = df_valid['AUROC'].max() - df_valid['AUROC'].min()
    sens_gap = df_valid['Sensitivity'].max() - df_valid['Sensitivity'].min()
    fnr_gap = df_valid['FNR'].max() - df_valid['FNR'].min()

    best_auroc_group = df_valid.loc[df_valid['AUROC'].idxmax(), 'Subgroup']
    worst_auroc_group = df_valid.loc[df_valid['AUROC'].idxmin(), 'Subgroup']
    worst_fnr_group = df_valid.loc[df_valid['FNR'].idxmax(), 'Subgroup']
    worst_fnr_val = df_valid['FNR'].max()

    disparity_summary.append({
        'Attribute': attr_name,
        'AUROC_Gap': auroc_gap,
        'Best_AUROC_Group': best_auroc_group,
        'Worst_AUROC_Group': worst_auroc_group,
        'Sensitivity_Gap': sens_gap,
        'FNR_Gap': fnr_gap,
        'Worst_FNR_Group': worst_fnr_group,
        'Worst_FNR_Value': worst_fnr_val
    })

df_disparity = pd.DataFrame(disparity_summary)

# ============================================================================
# SAVE RESULTS
# ============================================================================

df_all_subgroups.to_csv('results/fairness_subgroup_metrics.csv', index=False)
df_disparity.to_csv('results/fairness_disparity_summary.csv', index=False)

print(f"\n✓ Saved results/fairness_subgroup_metrics.csv")
print(f"✓ Saved results/fairness_disparity_summary.csv")

print("\n" + "=" * 100)
print("STAGE 5 COMPLETE")
print("=" * 100)
