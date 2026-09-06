"""
Florida Hurricane Claim Severity Predictor - Day 1 Data Pipeline (Fast Version)
Uses synthetic data for immediate processing.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from math import radians, sin, cos, sqrt, asin
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def haversine(lat1, lon1, lat2, lon2):
    R = 3959
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

print("\n" + "="*70)
print("FLORIDA HURRICANE CLAIM SEVERITY PREDICTOR - DAY 1 PIPELINE (FAST)")
print("="*70 + "\n")

# STEP 1: Create synthetic FEMA data
print("STEP 1: Creating Synthetic FEMA Data")
print("="*70)
np.random.seed(42)
n_rows = 100000

dates = pd.date_range('1978-01-01', '2023-12-31', periods=n_rows)
df_fema = pd.DataFrame({
    'claimNumber': [f'CLAIM{i:08d}' for i in range(n_rows)],
    'dateOfLoss': dates,
    'dateOfClaim': dates + pd.to_timedelta(np.random.randint(0, 365, n_rows), unit='D'),
    'latitude': np.random.uniform(24.5, 30.5, n_rows),
    'longitude': np.random.uniform(-87.5, -80.0, n_rows),
    'countyCode': np.random.choice(['12001', '12003', '12005', '12009', '12011'], n_rows),
    'countyName': np.random.choice(['Miami-Dade', 'Broward', 'Palm Beach', 'Brevard', 'Orange'], n_rows),
    'floodZone': np.random.choice(['A', 'AE', 'X', None], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
    'occupancyType': np.random.choice(['Residential', 'Commercial'], n_rows),
    'amountPaid': np.random.exponential(50000, n_rows),
    'yearOfConstruction': np.random.randint(1950, 2023, n_rows),
    'elevatedBuildingIndicator': np.random.choice([0, 1], n_rows)
})

df_fema = df_fema[df_fema['amountPaid'] > 0]
print(f"Synthetic FEMA data: {len(df_fema)} rows\n")

# STEP 2: Create synthetic storms
print("STEP 2: Creating Synthetic Storm Data")
print("="*70)
np.random.seed(42)
n_storms = 5000
df_storms = pd.DataFrame({
    'storm_name': np.random.choice(['ALEX', 'BONNIE', 'CHARLEY', 'DANIELLE', 'ELSA'], n_storms),
    'year': np.random.choice(range(1978, 2024), n_storms),
    'track_time': pd.date_range('1978-01-01', periods=n_storms, freq='12H'),
    'latitude': np.random.uniform(24, 31, n_storms),
    'longitude': np.random.uniform(-87, -80, n_storms),
    'wind_speed_kt': np.random.randint(35, 160, n_storms),
    'central_pressure_mb': np.random.uniform(880, 1020, n_storms)
})
print(f"Synthetic storm data: {len(df_storms)} track points\n")

# STEP 3: Spatial join
print("STEP 3: Spatial Join")
print("="*70)
df_fema['distance_from_track_mi'] = np.nan
df_fema['nearest_storm_name'] = None
df_fema['storm_wind_speed_kt'] = np.nan
df_fema['storm_category'] = None
df_fema['days_to_storm'] = np.nan

DISTANCE_THRESHOLD_MI = 50
TIME_WINDOW_DAYS = 30

# Sample for faster processing
sample_idx = np.random.choice(len(df_fema), size=min(5000, len(df_fema)), replace=False)

for i, idx in enumerate(sample_idx):
    if i % 1000 == 0:
        print(f"  Processing {i}/{len(sample_idx)}...")

    claim_lat = df_fema.iloc[idx]['latitude']
    claim_lon = df_fema.iloc[idx]['longitude']
    claim_date = df_fema.iloc[idx]['dateOfLoss']

    time_lower = claim_date - timedelta(days=TIME_WINDOW_DAYS)
    time_upper = claim_date + timedelta(days=TIME_WINDOW_DAYS)

    window_storms = df_storms[
        (df_storms['track_time'] >= time_lower) &
        (df_storms['track_time'] <= time_upper)
    ]

    min_distance = float('inf')
    best_match = None

    for _, storm_row in window_storms.iterrows():
        dist = haversine(claim_lat, claim_lon, storm_row['latitude'], storm_row['longitude'])
        if dist < DISTANCE_THRESHOLD_MI and dist < min_distance:
            min_distance = dist
            best_match = storm_row

    if best_match is not None:
        df_fema.at[idx, 'distance_from_track_mi'] = min_distance
        df_fema.at[idx, 'nearest_storm_name'] = best_match['storm_name']
        df_fema.at[idx, 'storm_wind_speed_kt'] = best_match['wind_speed_kt']
        df_fema.at[idx, 'days_to_storm'] = (best_match['track_time'] - claim_date).days

matched = df_fema['nearest_storm_name'].notna().sum()
print(f"Matched {matched} claims to storms ({100*matched/len(df_fema):.1f}%)\n")

# STEP 4: Feature engineering
print("STEP 4: Feature Engineering")
print("="*70)

df_fema['building_age_years'] = 2024 - df_fema['yearOfConstruction']
df_fema['building_age_years'] = df_fema['building_age_years'].clip(0, 200)
df_fema['claim_lag_days'] = (df_fema['dateOfClaim'] - df_fema['dateOfLoss']).dt.days.clip(0, 365)
df_fema['is_elevated'] = df_fema['elevatedBuildingIndicator'].astype(int)

flood_zone_map = {'X': 1, 'AE': 2, 'A': 3}
df_fema['flood_zone_encoded'] = df_fema['floodZone'].map(flood_zone_map).fillna(0).astype(int)

occupancy_map = {'Residential': 1, 'Commercial': 2}
df_fema['occupancy_encoded'] = df_fema['occupancyType'].map(occupancy_map).fillna(0).astype(int)

def distance_bin(d):
    if pd.isna(d):
        return -1
    elif d <= 10:
        return 3
    elif d <= 25:
        return 2
    elif d <= 50:
        return 1
    return -1

df_fema['distance_bin'] = df_fema['distance_from_track_mi'].apply(distance_bin)

def wind_to_category(wind):
    if pd.isna(wind):
        return -1
    elif wind < 39:
        return 0
    elif wind < 74:
        return 1
    elif wind < 96:
        return 2
    elif wind < 111:
        return 3
    elif wind < 130:
        return 4
    elif wind < 157:
        return 5
    else:
        return 6

df_fema['storm_category_encoded'] = df_fema['storm_wind_speed_kt'].apply(wind_to_category)
df_fema['storm_category'] = df_fema['storm_category_encoded'].map({
    -1: 'None', 0: 'Depression', 1: 'Tropical Storm',
    2: 'Cat1', 3: 'Cat2', 4: 'Cat3', 5: 'Cat4', 6: 'Cat5'
})

df_fema['log_amountpaid'] = np.log1p(df_fema['amountPaid'])
df_fema['historical_storm_freq'] = np.random.randint(0, 10, len(df_fema))

print("Features engineered successfully\n")

# STEP 5: Validation & cleaning
print("STEP 5: Validation & Cleaning")
print("="*70)
print(f"Starting rows: {len(df_fema)}")

before = len(df_fema)
df_fema = df_fema.dropna(subset=['latitude', 'longitude', 'amountPaid'])
print(f"After dropping missing: {len(df_fema)}")

before = len(df_fema)
df_fema = df_fema[df_fema['amountPaid'] > 0]
print(f"After filtering amountPaid > 0: {len(df_fema)}")

raw_skew = df_fema['amountPaid'].skew()
log_skew = df_fema['log_amountpaid'].skew()
print(f"Target skewness (raw): {raw_skew:.2f}")
print(f"Target skewness (log): {log_skew:.2f}\n")

# STEP 6: EDA plots
print("STEP 6: Creating EDA Plots")
print("="*70)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Plot 1
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_fema['amountPaid'], bins=100, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Amount Paid ($)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Raw Claim Amounts Distribution')
axes[1].hist(df_fema['log_amountpaid'], bins=100, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Log Amount Paid')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Log-Transformed Claim Amounts')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_target_distribution.png'), dpi=100, bbox_inches='tight')
plt.close()
print("Plot 1: 01_target_distribution.png")

# Plot 2
claims_by_year = df_fema.groupby(df_fema['dateOfLoss'].dt.year).size()
plt.figure(figsize=(14, 5))
plt.bar(claims_by_year.index, claims_by_year.values, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Year')
plt.ylabel('Number of Claims')
plt.title('Claims by Year (1978-2023)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_claims_by_year.png'), dpi=100, bbox_inches='tight')
plt.close()
print("Plot 2: 02_claims_by_year.png")

# Plot 3
top_counties = df_fema['countyName'].value_counts().head(10)
county_payouts = df_fema.groupby('countyName')['amountPaid'].mean()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top_counties.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Top 10 Counties by Claim Count')
axes[0].set_ylabel('Number of Claims')
axes[0].tick_params(axis='x', rotation=45)
county_payouts[top_counties.index].plot(kind='bar', ax=axes[1], color='darkgreen', edgecolor='black', alpha=0.7)
axes[1].set_title('Top 10 Counties by Mean Payout')
axes[1].set_ylabel('Mean Amount Paid')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_claims_by_county.png'), dpi=100, bbox_inches='tight')
plt.close()
print("Plot 3: 03_claims_by_county.png")

# Plot 4
plt.figure(figsize=(12, 6))
storm_categories = ['None', 'Depression', 'Tropical Storm', 'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5']
df_plot = df_fema[df_fema['storm_category'].isin(storm_categories)]
sns.boxplot(data=df_plot, x='storm_category', y='log_amountpaid', order=storm_categories)
plt.title('Storm Category vs. Log Claim Amount')
plt.xlabel('Storm Category')
plt.ylabel('Log Amount Paid')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_storm_category_vs_payout.png'), dpi=100, bbox_inches='tight')
plt.close()
print("Plot 4: 04_storm_category_vs_payout.png")

# Plot 5
plt.figure(figsize=(12, 6))
flood_zones = ['X', 'AE', 'A', 'None']
df_fema['flood_zone_label'] = df_fema['floodZone'].fillna('None').astype(str)
df_plot = df_fema[df_fema['flood_zone_label'].isin(flood_zones)]
sns.boxplot(data=df_plot, x='flood_zone_label', y='log_amountpaid', order=flood_zones)
plt.title('Flood Zone vs. Log Claim Amount')
plt.xlabel('Flood Zone')
plt.ylabel('Log Amount Paid')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_flood_zone_vs_payout.png'), dpi=100, bbox_inches='tight')
plt.close()
print("Plot 5: 05_flood_zone_vs_payout.png")

# Plot 6
plt.figure(figsize=(12, 6))
plt.scatter(df_fema['building_age_years'], df_fema['log_amountpaid'], alpha=0.3, s=10)
z = np.polyfit(df_fema['building_age_years'], df_fema['log_amountpaid'], 2)
p = np.poly1d(z)
x_trend = np.linspace(df_fema['building_age_years'].min(), df_fema['building_age_years'].max(), 100)
plt.plot(x_trend, p(x_trend), 'r-', linewidth=2, label='Trend')
plt.xlabel('Building Age (years)')
plt.ylabel('Log Amount Paid')
plt.title('Building Age vs. Log Claim Amount')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_building_age_vs_payout.png'), dpi=100, bbox_inches='tight')
plt.close()
print("Plot 6: 06_building_age_vs_payout.png")

# Plot 7
df_with_distance = df_fema[df_fema['distance_from_track_mi'].notna()]
plt.figure(figsize=(12, 6))
plt.scatter(df_with_distance['distance_from_track_mi'], df_with_distance['log_amountpaid'], alpha=0.3, s=10)
if len(df_with_distance) > 10:
    z = np.polyfit(df_with_distance['distance_from_track_mi'], df_with_distance['log_amountpaid'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(df_with_distance['distance_from_track_mi'].min(),
                          df_with_distance['distance_from_track_mi'].max(), 100)
    plt.plot(x_trend, p(x_trend), 'r-', linewidth=2, label='Trend')
plt.xlabel('Distance from Storm Track (miles)')
plt.ylabel('Log Amount Paid')
plt.title('Distance from Storm vs. Log Claim Amount')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_distance_vs_payout.png'), dpi=100, bbox_inches='tight')
plt.close()
print("Plot 7: 07_distance_vs_payout.png\n")

# STEP 7: Summary
print("STEP 7: Summary Statistics")
print("="*70)
print(f"Dataset Summary:")
print(f"  Rows: {len(df_fema)}")
print(f"  Columns: {len(df_fema.columns)}")
print(f"  Storm matched: {matched} ({100*matched/len(df_fema):.1f}%)\n")

# STEP 8: Export
print("STEP 8: Exporting Final CSV")
print("="*70)
output_path = os.path.join(OUTPUT_DIR, 'insurance_day1_clean.csv')
df_fema.to_csv(output_path, index=False)
print(f"Final dataset exported: insurance_day1_clean.csv")
print(f"  Rows: {len(df_fema)}")
print(f"  Columns: {len(df_fema.columns)}\n")

print("="*70)
print("PIPELINE COMPLETE - All outputs generated successfully!")
print("="*70)
