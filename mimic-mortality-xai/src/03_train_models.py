"""
Stage 3: Model Training (Logistic Regression + XGBoost)

Trains LR baseline and XGBoost on train set with validation-based threshold tuning.
Evaluates on held-out test set using F1-max on validation.

Output: results/model_metrics.json, results/model_comparison.csv,
        results/feature_importance_full.csv, results/feature_importance_top10.csv
"""

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, f1_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("=" * 100)
print("STAGE 3: MODEL TRAINING (LR + XGBoost)")
print("=" * 100)

# Load splits
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')

# Load feature list
with open('data/feature_info.json') as f:
    feature_cols = json.load(f)['features_in_X']

X_train = train_df[feature_cols].values
y_train = train_df['hospital_expire_flag'].values

X_val = val_df[feature_cols].values
y_val = val_df['hospital_expire_flag'].values

X_test = test_df[feature_cols].values
y_test = test_df['hospital_expire_flag'].values

print(f"\nLoaded splits: Train {len(X_train):,}, Val {len(X_val):,}, Test {len(X_test):,}")
print(f"Test mortality: {y_test.mean():.4f} ({y_test.sum():,} deaths)")

# ============================================================================
# LOGISTIC REGRESSION (Baseline)
# ============================================================================

print("\n" + "-" * 100)
print("LOGISTIC REGRESSION (Baseline)")
print("-" * 100)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(
    max_iter=1000, solver='lbfgs', random_state=SEED,
    class_weight='balanced'
)
lr_model.fit(X_train_scaled, y_train)

# Validation threshold tuning (F1-max)
y_val_proba_lr = lr_model.predict_proba(X_val_scaled)[:, 1]
best_f1_lr = 0
best_threshold_lr = 0.5
for threshold in np.arange(0.1, 0.9, 0.01):
    y_val_pred = (y_val_proba_lr >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1_lr:
        best_f1_lr = f1
        best_threshold_lr = threshold

print(f"Validation F1-max: {best_f1_lr:.4f} at threshold {best_threshold_lr:.3f}")

# Test evaluation
y_test_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred_lr = (y_test_proba_lr >= best_threshold_lr).astype(int)

auroc_lr = roc_auc_score(y_test, y_test_proba_lr)
auprc_lr = average_precision_score(y_test, y_test_proba_lr)
brier_lr = brier_score_loss(y_test, y_test_proba_lr)
tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y_test, y_test_pred_lr).ravel()
sens_lr = tp_lr / (tp_lr + fn_lr)
spec_lr = tn_lr / (tn_lr + fp_lr)
ppv_lr = tp_lr / (tp_lr + fp_lr) if (tp_lr + fp_lr) > 0 else 0
npv_lr = tn_lr / (tn_lr + fn_lr) if (tn_lr + fn_lr) > 0 else 0

print(f"\nTest Performance (LR):")
print(f"  AUROC: {auroc_lr:.4f}")
print(f"  AUPRC: {auprc_lr:.4f}")
print(f"  Brier: {brier_lr:.4f}")
print(f"  Sensitivity: {sens_lr:.4f}, Specificity: {spec_lr:.4f}")
print(f"  PPV: {ppv_lr:.4f}, NPV: {npv_lr:.4f}")
print(f"  Confusion: TP={tp_lr}, FN={fn_lr}, FP={fp_lr}, TN={tn_lr}")

# ============================================================================
# XGBOOST
# ============================================================================

print("\n" + "-" * 100)
print("XGBOOST")
print("-" * 100)

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"Class imbalance (scale_pos_weight): {scale_pos_weight:.4f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, random_state=SEED,
    use_label_encoder=False, eval_metric='logloss'
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              early_stopping_rounds=10, verbose=False)

print(f"✓ Trained XGBoost (best iteration: {xgb_model.best_iteration})")

# Validation threshold tuning
y_val_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
best_f1_xgb = 0
best_threshold_xgb = 0.5
for threshold in np.arange(0.1, 0.9, 0.01):
    y_val_pred = (y_val_proba_xgb >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1_xgb:
        best_f1_xgb = f1
        best_threshold_xgb = threshold

print(f"Validation F1-max: {best_f1_xgb:.4f} at threshold {best_threshold_xgb:.3f}")

# Test evaluation
y_test_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_test_pred_xgb = (y_test_proba_xgb >= best_threshold_xgb).astype(int)

auroc_xgb = roc_auc_score(y_test, y_test_proba_xgb)
auprc_xgb = average_precision_score(y_test, y_test_proba_xgb)
brier_xgb = brier_score_loss(y_test, y_test_proba_xgb)
tn_xgb, fp_xgb, fn_xgb, tp_xgb = confusion_matrix(y_test, y_test_pred_xgb).ravel()
sens_xgb = tp_xgb / (tp_xgb + fn_xgb)
spec_xgb = tn_xgb / (tn_xgb + fp_xgb)
ppv_xgb = tp_xgb / (tp_xgb + fp_xgb) if (tp_xgb + fp_xgb) > 0 else 0
npv_xgb = tn_xgb / (tn_xgb + fn_xgb) if (tn_xgb + fn_xgb) > 0 else 0

print(f"\nTest Performance (XGBoost):")
print(f"  AUROC: {auroc_xgb:.4f}")
print(f"  AUPRC: {auprc_xgb:.4f}")
print(f"  Brier: {brier_xgb:.4f}")
print(f"  Sensitivity: {sens_xgb:.4f}, Specificity: {spec_xgb:.4f}")
print(f"  PPV: {ppv_xgb:.4f}, NPV: {npv_xgb:.4f}")
print(f"  Confusion: TP={tp_xgb}, FN={fn_xgb}, FP={fp_xgb}, TN={tn_xgb}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

model_metrics = {
    'test_size': len(X_test),
    'test_mortality': float(y_test.mean()),
    'logistic_regression': {
        'threshold': float(best_threshold_lr),
        'auroc': float(auroc_lr),
        'auprc': float(auprc_lr),
        'brier': float(brier_lr),
        'sensitivity': float(sens_lr),
        'specificity': float(spec_lr),
        'ppv': float(ppv_lr),
        'npv': float(npv_lr),
        'confusion': {'tp': int(tp_lr), 'fn': int(fn_lr), 'fp': int(fp_lr), 'tn': int(tn_lr)}
    },
    'xgboost': {
        'threshold': float(best_threshold_xgb),
        'auroc': float(auroc_xgb),
        'auprc': float(auprc_xgb),
        'brier': float(brier_xgb),
        'sensitivity': float(sens_xgb),
        'specificity': float(spec_xgb),
        'ppv': float(ppv_xgb),
        'npv': float(npv_xgb),
        'confusion': {'tp': int(tp_xgb), 'fn': int(fn_xgb), 'fp': int(fp_xgb), 'tn': int(tn_xgb)}
    }
}

with open('results/model_metrics.json', 'w') as f:
    json.dump(model_metrics, f, indent=2)

# Feature importance (XGBoost)
feature_importance = xgb_model.get_booster().get_score(importance_type='gain')
fi_df = pd.DataFrame([
    {'Feature': feat, 'Importance': imp}
    for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
])
fi_df.to_csv('results/feature_importance_full.csv', index=False)
fi_df.head(10).to_csv('results/feature_importance_top10.csv', index=False)

print(f"\n✓ Saved results/model_metrics.json")
print(f"✓ Saved results/feature_importance_full.csv")
print(f"✓ Saved results/feature_importance_top10.csv")

print("\n" + "=" * 100)
print("STAGE 3 COMPLETE")
print("=" * 100)
