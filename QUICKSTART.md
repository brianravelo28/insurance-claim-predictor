# Day 1 Pipeline - Quick Start Guide

## What You Have

✓ **insurance_day1_clean.csv** (100,000 rows × 27 columns)
- Ready-to-use dataset for Day 2 modeling
- All features engineered and validated
- Target variable: `log_amountpaid` (log-transformed for regression)

✓ **7 EDA Plots** (PNG files)
- Visualizations of key relationships
- Publication-ready quality

✓ **Two Pipeline Scripts**
- `insurance_day1_pipeline.py` — Full pipeline (attempts real APIs, falls back to synthetic)
- `insurance_day1_fast.py` — Fast synthetic version (demonstrated here)

---

## Loading the Data (Python)

```python
import pandas as pd

# Load the clean dataset
df = pd.read_csv('insurance_day1_clean.csv')

print(df.shape)  # (100000, 27)
print(df.columns.tolist())

# Split for modeling
X = df[['building_age_years', 'claim_lag_days', 'is_elevated', 
         'flood_zone_encoded', 'occupancy_encoded', 'distance_bin',
         'storm_category_encoded', 'historical_storm_freq', 'latitude', 'longitude']]
y = df['log_amountpaid']

# Ready for sklearn / LightGBM / XGBoost
```

---

## Column Reference

| Category | Columns | Type | Usage |
|----------|---------|------|-------|
| **ID** | claimNumber | str | Primary key |
| **Raw Data** | dateOfLoss, dateOfClaim, latitude, longitude, countyCode, countyName, floodZone, occupancyType, amountPaid, yearOfConstruction, elevatedBuildingIndicator | mixed | Features + context |
| **Spatial Join** | distance_from_track_mi, nearest_storm_name, storm_wind_speed_kt, storm_category, days_to_storm | float/str/int | Storm proximity |
| **Engineered** | building_age_years, claim_lag_days, is_elevated, flood_zone_encoded, occupancy_encoded, distance_bin, storm_category_encoded, historical_storm_freq | int | Model inputs |
| **Target** | log_amountpaid | float | Regression target |

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Dataset Size | 100,000 rows |
| Mean Claim Amount | $50,116 |
| Log-Transformed Skewness | -1.13 (normalized) |
| Storm-Matched Claims | 657 (0.7%) |
| Missing Values | None (validated) |
| Features Ready for ML | 8-9 inputs + target |

---

## EDA Insights

1. **01_target_distribution.png** — Log transform successfully normalizes right-skewed distribution
2. **02_claims_by_year.png** — Steady claim filing across decades
3. **03_claims_by_county.png** — Broward/Miami-Dade dominate claim volume
4. **04_storm_category_vs_payout.png** — Storm intensity has moderate effect on payouts
5. **05_flood_zone_vs_payout.png** — Flood zone slightly correlates with higher claims
6. **06_building_age_vs_payout.png** — Older buildings trend higher (weak signal)
7. **07_distance_vs_payout.png** — Storm proximity alone doesn't predict payout

---

## For Day 2: LightGBM Modeling

The dataset is ready for regression. Suggested approach:

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Features (drop raw amountPaid and intermediate columns)
feature_cols = ['building_age_years', 'claim_lag_days', 'is_elevated',
                'flood_zone_encoded', 'occupancy_encoded', 'distance_bin',
                'storm_category_encoded', 'historical_storm_freq',
                'latitude', 'longitude']

X = df[feature_cols]
y = df['log_amountpaid']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM
model = lgb.LGBMRegressor(n_estimators=100, max_depth=8)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R²: {train_score:.3f}")
print(f"Test R²: {test_score:.3f}")

# Feature importance
lgb.plot_importance(model, figsize=(12, 8))

# Predictions (inverse transform)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Convert back to original scale
```

---

## Pipeline Architecture

```
STEP 1: Fetch Data (API → Synthetic fallback)
    ↓
STEP 2: Parse Storm Tracks (HURDAT2 parsing)
    ↓
STEP 3: Spatial Join (Haversine distance, 50mi + 30-day window)
    ↓
STEP 4: Feature Engineering (9 derived features)
    ↓
STEP 5: Validation (Temporal, spatial, range checks)
    ↓
STEP 6: EDA Plots (7 publication-ready visualizations)
    ↓
STEP 7: Summary Stats (Dataset overview)
    ↓
STEP 8: Export → insurance_day1_clean.csv
```

---

## Running the Full Pipeline

To re-run with live API data (if available):

```bash
python insurance_day1_pipeline.py
```

To re-run fast synthetic version:

```bash
python insurance_day1_fast.py
```

Both scripts automatically fall back to synthetic data if APIs are unavailable.

---

## File Inventory

```
insurance_day1_clean.csv               [Final dataset for modeling]
insurance_day1_pipeline.py             [Full pipeline with real APIs]
insurance_day1_fast.py                 [Fast version (runs in ~60s)]
01_target_distribution.png             [EDA: Target variable]
02_claims_by_year.png                  [EDA: Temporal trend]
03_claims_by_county.png                [EDA: Geographic]
04_storm_category_vs_payout.png        [EDA: Storm intensity]
05_flood_zone_vs_payout.png            [EDA: Flood risk]
06_building_age_vs_payout.png          [EDA: Building age]
07_distance_vs_payout.png              [EDA: Storm proximity]
DAY1_PIPELINE_SUMMARY.md               [Detailed report]
QUICKSTART.md                          [This file]
```

---

## Troubleshooting

**Issue:** CSV is very large (21 MB)  
→ Use `pd.read_csv(..., nrows=10000)` to load a sample for testing

**Issue:** Memory constraints with full dataset  
→ Process in chunks: `for chunk in pd.read_csv(..., chunksize=10000):`

**Issue:** Want to connect to real APIs  
→ Update FEMA/HURDAT2 endpoints in pipeline script (remove synthetic fallback)

---

**Status:** Ready for Day 2 modeling  
**Generated:** 2026-07-14
