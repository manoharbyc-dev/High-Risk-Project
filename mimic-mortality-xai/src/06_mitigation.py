"""
Stage 6: Bias Mitigation — Group-Aware Threshold Adjustment

Selects Black-group threshold on validation set to minimize FNR gap with White.
Applies both thresholds to test set and reports FNR gap reduction.

Output: results/stage6_comparison_table.csv, results/stage6_mitigation_summary.json,
        figures/stage6_fnr_mitigation_by_ethnicity.png
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import confusion_matrix, brier_score_loss, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("=" * 100)
print("STAGE 6: BIAS MITIGATION — GROUP-AWARE THRESHOLD")
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
ethnicity_val = val_df['ethnicity_group'].values

X_test = test_df[feature_cols].values
y_test = test_df['hospital_expire_flag'].values
ethnicity_test = test_df['ethnicity_group'].values

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

print("✓ Model trained")

# ============================================================================
# STEP 1: THRESHOLD TUNING ON VALIDATION SET
# ============================================================================

print("\n" + "-" * 100)
print("STEP 1: Threshold tuning on validation set (no test-set involvement)")
print("-" * 100)

y_proba_val = xgb_model.predict_proba(X_val)[:, 1]

val_black_mask = ethnicity_val == 'Black'
val_white_mask = ethnicity_val == 'White'

y_true_val_black = y_val[val_black_mask]
y_proba_val_black = y_proba_val[val_black_mask]

y_true_val_white = y_val[val_white_mask]
y_proba_val_white = y_proba_val[val_white_mask]

print(f"\nValidation composition:")
print(f"  Black: n={len(y_true_val_black)}, deaths={y_true_val_black.sum()}")
print(f"  White: n={len(y_true_val_white)}, deaths={y_true_val_white.sum()}")

# White FNR at global threshold
global_threshold = 0.421
y_pred_val_white = (y_proba_val_white >= global_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true_val_white, y_pred_val_white).ravel()
fnr_white_val = fn / (fn + tp)
print(f"\nWhite FNR at global threshold ({global_threshold}) on validation: {fnr_white_val:.4f}")

# Grid search for Black threshold
thresholds_to_try = np.arange(0.15, 0.50, 0.01)
best_black_threshold = None
best_gap = np.inf

for threshold in thresholds_to_try:
    y_pred_val_black = (y_proba_val_black >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_val_black, y_pred_val_black).ravel()
    fnr_black = fn / (fn + tp)
    gap = abs(fnr_black - fnr_white_val)

    if gap < best_gap:
        best_gap = gap
        best_black_threshold = threshold

print(f"\nSelected Black threshold on validation: {best_black_threshold:.2f}")
print(f"  Achieved gap: {best_gap:.4f}")

# ============================================================================
# STEP 2: APPLY TO TEST SET
# ============================================================================

print("\n" + "-" * 100)
print("STEP 2: Apply thresholds to held-out test set")
print("-" * 100)

y_proba_test = xgb_model.predict_proba(X_test)[:, 1]

test_black_mask = ethnicity_test == 'Black'
test_white_mask = ethnicity_test == 'White'

y_true_test_black = y_test[test_black_mask]
y_proba_test_black = y_proba_test[test_black_mask]

y_true_test_white = y_test[test_white_mask]
y_proba_test_white = y_proba_test[test_white_mask]

print(f"\nTest composition:")
print(f"  Black: n={len(y_true_test_black)}, deaths={y_true_test_black.sum()}")
print(f"  White: n={len(y_true_test_white)}, deaths={y_true_test_white.sum()}")

# ============================================================================
# BASELINE (GLOBAL THRESHOLD)
# ============================================================================

y_pred_test_black_baseline = (y_proba_test_black >= global_threshold).astype(int)
y_pred_test_white_baseline = (y_proba_test_white >= global_threshold).astype(int)

# Black baseline
tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_test_black, y_pred_test_black_baseline).ravel()
fnr_black_baseline = fn_b / (fn_b + tp_b)
sens_black_baseline = tp_b / (tp_b + fn_b)
spec_black_baseline = tn_b / (tn_b + fp_b)
ppv_black_baseline = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
fpr_black_baseline = fp_b / (fp_b + tn_b)

# White baseline
tn_w, fp_w, fn_w, tp_w = confusion_matrix(y_true_test_white, y_pred_test_white_baseline).ravel()
fnr_white_baseline = fn_w / (fn_w + tp_w)

fnr_gap_baseline = abs(fnr_black_baseline - fnr_white_baseline)

print(f"\nBASELINE (global threshold {global_threshold}):")
print(f"  Black FNR: {fnr_black_baseline:.4f}, White FNR: {fnr_white_baseline:.4f}")
print(f"  FNR Gap: {fnr_gap_baseline:.4f}")

# ============================================================================
# MITIGATION (GROUP-AWARE THRESHOLD)
# ============================================================================

y_pred_test_black_mitigated = (y_proba_test_black >= best_black_threshold).astype(int)

# Black mitigated
tn_b_m, fp_b_m, fn_b_m, tp_b_m = confusion_matrix(y_true_test_black, y_pred_test_black_mitigated).ravel()
fnr_black_mitigated = fn_b_m / (fn_b_m + tp_b_m)
sens_black_mitigated = tp_b_m / (tp_b_m + fn_b_m)
spec_black_mitigated = tn_b_m / (tn_b_m + fp_b_m)
ppv_black_mitigated = tp_b_m / (tp_b_m + fp_b_m) if (tp_b_m + fp_b_m) > 0 else 0
fpr_black_mitigated = fp_b_m / (fp_b_m + tn_b_m)

fnr_gap_mitigated = abs(fnr_black_mitigated - fnr_white_baseline)

print(f"\nMITIGATED (Black threshold {best_black_threshold:.3f}, White {global_threshold:.3f}):")
print(f"  Black FNR: {fnr_black_mitigated:.4f}, White FNR: {fnr_white_baseline:.4f}")
print(f"  FNR Gap: {fnr_gap_mitigated:.4f}")
print(f"  Gap reduction: {fnr_gap_baseline - fnr_gap_mitigated:.4f} ({100*(fnr_gap_baseline-fnr_gap_mitigated)/fnr_gap_baseline:.1f}%)")

# ============================================================================
# OVERALL IMPACT
# ============================================================================

y_pred_test_baseline = (y_proba_test >= global_threshold).astype(int)
y_pred_test_mitigated = np.copy(y_pred_test_baseline)
y_pred_test_mitigated[test_black_mask] = y_pred_test_black_mitigated
y_pred_test_mitigated[test_white_mask] = y_pred_test_white_baseline

auroc_baseline = roc_auc_score(y_test, y_proba_test)
auprc_baseline = average_precision_score(y_test, y_proba_test)
brier_baseline = brier_score_loss(y_test, y_proba_test)

tn_all_b, fp_all_b, fn_all_b, tp_all_b = confusion_matrix(y_test, y_pred_test_baseline).ravel()
ppv_all_baseline = tp_all_b / (tp_all_b + fp_all_b) if (tp_all_b + fp_all_b) > 0 else 0

tn_all_m, fp_all_m, fn_all_m, tp_all_m = confusion_matrix(y_test, y_pred_test_mitigated).ravel()
ppv_all_mitigated = tp_all_m / (tp_all_m + fp_all_m) if (tp_all_m + fp_all_m) > 0 else 0

print(f"\nOVERALL IMPACT (all subgroups):")
print(f"  Baseline PPV: {ppv_all_baseline:.4f}, Mitigated: {ppv_all_mitigated:.4f}")
print(f"  TP: {tp_all_b} → {tp_all_m}, FN: {fn_all_b} → {fn_all_m}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

mitigation_summary = {
    'method': 'Group-aware decision threshold adjustment',
    'black_threshold': float(best_black_threshold),
    'white_threshold': float(global_threshold),
    'baseline_metrics': {
        'black_fnr': float(fnr_black_baseline),
        'white_fnr': float(fnr_white_baseline),
        'fnr_gap': float(fnr_gap_baseline)
    },
    'mitigated_metrics': {
        'black_fnr': float(fnr_black_mitigated),
        'fnr_gap': float(fnr_gap_mitigated),
        'gap_reduction': float(fnr_gap_baseline - fnr_gap_mitigated),
        'gap_reduction_pct': float(100 * (fnr_gap_baseline - fnr_gap_mitigated) / fnr_gap_baseline)
    }
}

with open('results/stage6_mitigation_summary.json', 'w') as f:
    json.dump(mitigation_summary, f, indent=2)

comparison_data = {
    'Metric': [
        'Black FNR', 'White FNR', 'FNR Gap', 'Black Sensitivity', 'Black PPV', 'Black Specificity',
        'Overall PPV'
    ],
    'Baseline': [
        f"{fnr_black_baseline:.4f}", f"{fnr_white_baseline:.4f}", f"{fnr_gap_baseline:.4f}",
        f"{sens_black_baseline:.4f}", f"{ppv_black_baseline:.4f}", f"{spec_black_baseline:.4f}",
        f"{ppv_all_baseline:.4f}"
    ],
    'Mitigated': [
        f"{fnr_black_mitigated:.4f}", f"{fnr_white_baseline:.4f}", f"{fnr_gap_mitigated:.4f}",
        f"{sens_black_mitigated:.4f}", f"{ppv_black_mitigated:.4f}", f"{spec_black_mitigated:.4f}",
        f"{ppv_all_mitigated:.4f}"
    ]
}

df_comparison = pd.DataFrame(comparison_data)
df_comparison.to_csv('results/stage6_comparison_table.csv', index=False)

print(f"\n✓ Saved results/stage6_mitigation_summary.json")
print(f"✓ Saved results/stage6_comparison_table.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

ethnicity_groups = ['Asian', 'Black', 'Hispanic', 'Other-Unknown', 'White']
fnr_baseline_vals = [0.333, fnr_black_baseline, 0.316, 0.288, fnr_white_baseline]
fnr_mitigated_vals = [0.333, fnr_black_mitigated, 0.316, 0.288, fnr_white_baseline]

x = np.arange(len(ethnicity_groups))
width = 0.35

bars1 = ax.bar(x - width/2, fnr_baseline_vals, width, label='Baseline', color='#d62728', alpha=0.8)
bars2 = ax.bar(x + width/2, fnr_mitigated_vals, width, label='Mitigated', color='#2ca02c', alpha=0.8)

ax.set_xlabel('Ethnicity Group', fontsize=12, fontweight='bold')
ax.set_ylabel('False Negative Rate (FNR)', fontsize=12, fontweight='bold')
ax.set_title('FNR by Ethnicity: Baseline vs. Mitigation', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(ethnicity_groups)
ax.set_ylim([0, 0.40])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/stage6_fnr_mitigation_by_ethnicity.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved figures/stage6_fnr_mitigation_by_ethnicity.png")

print("\n" + "=" * 100)
print("STAGE 6 COMPLETE")
print("=" * 100)
