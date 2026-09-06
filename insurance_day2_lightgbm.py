"""
Florida Hurricane Claim Severity Predictor - Day 2: LightGBM Modeling
Trains and evaluates a LightGBM regressor on the Day 1 dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("\n" + "=" * 70)
print("FLORIDA HURRICANE CLAIM SEVERITY PREDICTOR - DAY 2: LIGHTGBM MODELING")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\nSTEP 1: Loading Clean Dataset")
print("-" * 70)

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'insurance_day1_clean.csv'))
print(f"Dataset loaded: {len(df):,} rows × {len(df.columns)} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

# ============================================================================
# STEP 2: FEATURE SELECTION & PREPARATION
# ============================================================================
print("\nSTEP 2: Feature Selection & Preparation")
print("-" * 70)

# Define feature set
feature_cols = [
    'building_age_years',
    'claim_lag_days',
    'is_elevated',
    'flood_zone_encoded',
    'occupancy_encoded',
    'distance_bin',
    'storm_category_encoded',
    'historical_storm_freq',
    'latitude',
    'longitude'
]

# Verify all features exist
missing_features = [f for f in feature_cols if f not in df.columns]
if missing_features:
    print(f"WARNING: Missing features: {missing_features}")
    feature_cols = [f for f in feature_cols if f not in missing_features]

target = 'log_amountpaid'

# Check target variable
print(f"Features: {len(feature_cols)}")
print(f"Target: {target}")
print(f"Target non-null: {df[target].notna().sum():,} ({100*df[target].notna().sum()/len(df):.1f}%)")

# Create feature matrix and target
X = df[feature_cols].copy()
y = df[target].copy()

# Remove rows with NaN in target or features
mask = y.notna() & X.notna().all(axis=1)
X = X[mask]
y = y[mask]

print(f"After removing NaN: {len(X):,} rows")

# Feature statistics
print(f"\nFeature Statistics:")
print(X.describe().loc[['mean', 'min', 'max']].T)

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
print("\nSTEP 3: Train-Test Split")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train):,} rows ({100*len(X_train)/len(X):.1f}%)")
print(f"Test set: {len(X_test):,} rows ({100*len(X_test)/len(X):.1f}%)")
print(f"\nTarget distribution:")
print(f"  Train - Mean: {y_train.mean():.3f}, Std: {y_train.std():.3f}")
print(f"  Test  - Mean: {y_test.mean():.3f}, Std: {y_test.std():.3f}")

# ============================================================================
# STEP 4: TRAIN LIGHTGBM MODEL
# ============================================================================
print("\nSTEP 4: Training LightGBM Model")
print("-" * 70)

# Model parameters - optimized for synthetic data
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 50,
    'max_depth': 10,
    'min_data_in_leaf': 10,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

print(f"Model parameters:")
for k, v in params.items():
    if k not in ['verbose', 'random_state']:
        print(f"  {k}: {v}")

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train model
print(f"\nTraining...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[lgb.log_evaluation(period=100), lgb.early_stopping(50)]
)

print(f"Model trained successfully")
print(f"Number of boosting rounds: {model.num_trees()}")

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================
print("\nSTEP 5: Model Evaluation")
print("-" * 70)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Performance Metrics:")
print(f"\n  RMSE (Root Mean Squared Error):")
print(f"    Train: {train_rmse:.4f}")
print(f"    Test:  {test_rmse:.4f}")

print(f"\n  MAE (Mean Absolute Error):")
print(f"    Train: {train_mae:.4f}")
print(f"    Test:  {test_mae:.4f}")

print(f"\n  R² Score:")
print(f"    Train: {train_r2:.4f}")
print(f"    Test:  {test_r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(
    lgb.LGBMRegressor(**params),
    X_train, y_train,
    cv=5,
    scoring='r2'
)

print(f"\n  5-Fold Cross-Validation R² Scores:")
print(f"    Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"    Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Prediction statistics
print(f"\nPrediction Statistics (Test Set):")
print(f"  Actual   - Mean: {y_test.mean():.3f}, Std: {y_test.std():.3f}")
print(f"  Predicted - Mean: {y_test_pred.mean():.3f}, Std: {y_test_pred.std():.3f}")

# ============================================================================
# STEP 6: FEATURE IMPORTANCE
# ============================================================================
print("\nSTEP 6: Feature Importance Analysis")
print("-" * 70)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(),
    'importance_gain': model.feature_importance(importance_type='gain')
})

feature_importance = feature_importance.sort_values('importance', ascending=False)

print(f"Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:6.0f} (gain: {row['importance_gain']:8.0f})")

# ============================================================================
# STEP 7: VISUALIZATION: FEATURE IMPORTANCE
# ============================================================================
print("\nSTEP 7: Creating Visualizations")
print("-" * 70)

# Plot 1: Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(feature_importance['feature'], feature_importance['importance'], color='steelblue', edgecolor='black')
axes[0].set_xlabel('Importance (Split Count)')
axes[0].set_title('LightGBM Feature Importance (Split-based)')
axes[0].invert_yaxis()

axes[1].barh(feature_importance['feature'], feature_importance['importance_gain'], color='darkgreen', edgecolor='black')
axes[1].set_xlabel('Importance Gain (Sum of Loss Reduction)')
axes[1].set_title('LightGBM Feature Importance (Gain-based)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_feature_importance.png'), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Plot 1: 08_feature_importance.png")

# Plot 2: Predicted vs Actual
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Train set
axes[0].scatter(y_train, y_train_pred, alpha=0.3, s=10)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r-', lw=2)
axes[0].set_xlabel('Actual Log Amount Paid')
axes[0].set_ylabel('Predicted Log Amount Paid')
axes[0].set_title(f'Train Set (R² = {train_r2:.4f})')
axes[0].grid(True, alpha=0.3)

# Test set
axes[1].scatter(y_test, y_test_pred, alpha=0.3, s=10, color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2)
axes[1].set_xlabel('Actual Log Amount Paid')
axes[1].set_ylabel('Predicted Log Amount Paid')
axes[1].set_title(f'Test Set (R² = {test_r2:.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_predictions_vs_actual.png'), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Plot 2: 09_predictions_vs_actual.png")

# Plot 3: Residuals
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_train_pred, residuals_train, alpha=0.3, s=10)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Log Amount Paid')
axes[0].set_ylabel('Residuals')
axes[0].set_title(f'Train Set Residuals (Std: {residuals_train.std():.4f})')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test_pred, residuals_test, alpha=0.3, s=10, color='orange')
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Log Amount Paid')
axes[1].set_ylabel('Residuals')
axes[1].set_title(f'Test Set Residuals (Std: {residuals_test.std():.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_residuals.png'), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Plot 3: 10_residuals.png")

# Plot 4: Residual Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(residuals_train, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Train Residual Distribution (Mean: {residuals_train.mean():.4f})')
axes[0].axvline(x=0, color='r', linestyle='--', lw=2)

axes[1].hist(residuals_test, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Test Residual Distribution (Mean: {residuals_test.mean():.4f})')
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '11_residual_distribution.png'), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Plot 4: 11_residual_distribution.png")

# Plot 5: Learning Curve (approximate via cross-validation)
print("✓ Plot 5: [Skipped - requires additional CV runs]")

# ============================================================================
# STEP 8: PREDICTIONS ON FULL DATASET
# ============================================================================
print("\nSTEP 8: Generating Predictions")
print("-" * 70)

# Predict on all data
X_full = df[feature_cols].copy()
y_full = df[target].copy()
mask = y_full.notna() & X_full.notna().all(axis=1)
X_full = X_full[mask]
y_full = y_full[mask]

predictions_log = model.predict(X_full)
predictions_actual = np.expm1(predictions_log)  # Convert back to original scale

# Add predictions to results
results_df = df[mask].copy()
results_df['prediction_log'] = predictions_log
results_df['prediction_amount'] = predictions_actual
results_df['residual'] = y_full.values - predictions_log
results_df['abs_error_pct'] = 100 * np.abs(results_df['prediction_amount'] - results_df['amountPaid']) / results_df['amountPaid']

print(f"Predictions generated for {len(results_df):,} claims")
print(f"Mean absolute error (original scale): ${results_df['prediction_amount'].mean():,.0f}")
print(f"Mean absolute percent error: {results_df['abs_error_pct'].mean():.1f}%")

# Export predictions
predictions_path = os.path.join(OUTPUT_DIR, 'insurance_day2_predictions.csv')
results_df.to_csv(predictions_path, index=False)
print(f"Predictions saved: insurance_day2_predictions.csv")

# ============================================================================
# STEP 9: MODEL SUMMARY REPORT
# ============================================================================
print("\nSTEP 9: Model Summary Report")
print("=" * 70)

summary_stats = f"""
MODEL: LightGBM Regression
Dataset: insurance_day1_clean.csv (synthetic data)

TRAINING CONFIGURATION:
  Learning Rate: 0.05
  Max Depth: 8
  Num Leaves: 31
  Num Boosting Rounds: {model.num_trees()}
  Early Stopping Patience: 10

PERFORMANCE METRICS:
  Train RMSE: {train_rmse:.4f}
  Test RMSE:  {test_rmse:.4f}

  Train MAE: {train_mae:.4f}
  Test MAE:  {test_mae:.4f}

  Train R²: {train_r2:.4f}
  Test R²:  {test_r2:.4f}

  5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

TOP 5 IMPORTANT FEATURES:
  1. {feature_importance.iloc[0]['feature']:30s} - {feature_importance.iloc[0]['importance']:6.0f}
  2. {feature_importance.iloc[1]['feature']:30s} - {feature_importance.iloc[1]['importance']:6.0f}
  3. {feature_importance.iloc[2]['feature']:30s} - {feature_importance.iloc[2]['importance']:6.0f}
  4. {feature_importance.iloc[3]['feature']:30s} - {feature_importance.iloc[3]['importance']:6.0f}
  5. {feature_importance.iloc[4]['feature']:30s} - {feature_importance.iloc[4]['importance']:6.0f}

PREDICTIONS:
  Total Claims Predicted: {len(results_df):,}
  Mean Predicted Amount: ${predictions_actual.mean():,.0f}
  Std Dev Predicted Amount: ${predictions_actual.std():,.0f}
  Mean Absolute Percent Error: {results_df['abs_error_pct'].mean():.1f}%

INTERPRETATION:
  - Model explains {100*test_r2:.1f}% of variance in test data
  - Average prediction error: {test_mae:.4f} log points (~{np.expm1(test_mae)*100:.0f}% in original scale)
  - Top feature: {feature_importance.iloc[0]['feature']} (accounts for {100*feature_importance.iloc[0]['importance']/feature_importance['importance'].sum():.1f}% of importance)

FILES GENERATED:
  [OK] 08_feature_importance.png
  [OK] 09_predictions_vs_actual.png
  [OK] 10_residuals.png
  [OK] 11_residual_distribution.png
  [OK] insurance_day2_predictions.csv
  [OK] insurance_day2_lightgbm.py (this script)
"""

print(summary_stats)

# Save summary to file
summary_path = os.path.join(OUTPUT_DIR, 'DAY2_MODEL_SUMMARY.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary_stats)

print("\n" + "=" * 70)
print("DAY 2 MODELING COMPLETE")
print("=" * 70)
print(f"\nNext Steps:")
print(f"  1. Review model performance and predictions")
print(f"  2. Analyze feature importance patterns")
print(f"  3. Evaluate predictions on new data")
print(f"  4. Consider hyperparameter tuning for production model")
print()
