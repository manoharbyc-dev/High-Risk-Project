"""
Stage 4: SHAP Explainability (TreeSHAP on XGBoost)

Computes TreeSHAP values on 2,000 test samples, generates summary plots,
and reports top 10 features by mean |SHAP|.

Output: figures/shap_summary_beeswarm.png, figures/shap_mean_abs_bar.png,
        results/shap_feature_importance.csv, results/shap_vs_gain_comparison.csv
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("=" * 100)
print("STAGE 4: SHAP EXPLAINABILITY (TreeSHAP)")
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

# Train XGBoost model
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, random_state=SEED,
    use_label_encoder=False, eval_metric='logloss'
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              early_stopping_rounds=10, verbose=False)

print(f"✓ Model trained (best iteration: {xgb_model.best_iteration})")

# ============================================================================
# TREESHAP COMPUTATION
# ============================================================================

print("\nComputing TreeSHAP on test set...")

# Sample for speed
sample_size = min(2000, len(X_test))
if sample_size < len(X_test):
    sample_indices = np.random.RandomState(SEED).choice(len(X_test), sample_size, replace=False)
    X_test_shap = X_test[sample_indices]
else:
    X_test_shap = X_test

print(f"Using {sample_size} test samples")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_shap)

# Handle binary classification
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(f"✓ TreeSHAP computed")

# ============================================================================
# FEATURE IMPORTANCE BY MEAN |SHAP|
# ============================================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values('Mean_Abs_SHAP', ascending=False)

print("\nTop 10 features by mean |SHAP|:")
for rank, (idx, row) in enumerate(shap_importance_df.head(10).iterrows(), 1):
    print(f"  {rank:2d}. {row['Feature']:30s}: {row['Mean_Abs_SHAP']:.6f}")

top10_shap = shap_importance_df.head(10)['Feature'].tolist()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nGenerating visualizations...")

# SHAP summary plot (beeswarm)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_shap, feature_names=feature_cols,
                  plot_type='dot', show=False, max_display=15)
plt.title('SHAP Summary Plot (Beeswarm) — Top 15 Features', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

# SHAP bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_shap, feature_names=feature_cols,
                  plot_type='bar', show=False, max_display=15)
plt.title('SHAP Mean Absolute Importance (Bar Plot) — Top 15 Features', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/shap_mean_abs_bar.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved figures/shap_summary_beeswarm.png")
print(f"✓ Saved figures/shap_mean_abs_bar.png")

# ============================================================================
# COMPARISON WITH GAIN IMPORTANCE
# ============================================================================

gain_importance = pd.read_csv('results/feature_importance_full.csv')
top10_gain = gain_importance.head(10)['Feature'].tolist()

comparison_list = []
for rank, feature in enumerate(top10_shap, 1):
    shap_val = shap_importance_df[shap_importance_df['Feature'] == feature]['Mean_Abs_SHAP'].values[0]
    try:
        gain_rank = list(gain_importance['Feature']).index(feature) + 1
    except ValueError:
        gain_rank = np.nan

    comparison_list.append({
        'SHAP_Rank': rank,
        'Feature': feature,
        'Mean_Abs_SHAP': shap_val,
        'Gain_Rank': gain_rank,
        'Agreement': 'Yes' if (gain_rank <= 10 if not pd.isna(gain_rank) else False) else 'No'
    })

comparison_df = pd.DataFrame(comparison_list)

print("\nTop 10 by SHAP vs Gain importance:")
print(comparison_df.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================

shap_importance_df.to_csv('results/shap_feature_importance.csv', index=False)
comparison_df.to_csv('results/shap_vs_gain_comparison.csv', index=False)

print(f"\n✓ Saved results/shap_feature_importance.csv")
print(f"✓ Saved results/shap_vs_gain_comparison.csv")

print("\n" + "=" * 100)
print("STAGE 4 COMPLETE")
print("=" * 100)
