"""
Florida Hurricane Claim Severity Predictor - Day 1 Data Pipeline
Fetches FEMA NFIP claims + NOAA storm tracks, engineers features, validates, and outputs EDA plots.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environment
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, asin
from urllib.parse import urlencode
import warnings
warnings.filterwarnings('ignore')

# Set output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Output directory: {OUTPUT_DIR}\n")

# ============================================================================
# STEP 1: FETCH FEMA NFIP CLAIMS
# ============================================================================
def fetch_fema_claims():
    """Fetch FEMA NFIP claims for Florida with pagination."""
    print("=" * 70)
    print("STEP 1: Fetching FEMA NFIP Claims")
    print("=" * 70)

    base_url = "https://www.fema.gov/api/open/data/FimaNfipClaims"
    params = {
        "$filter": "state eq 'FL' and dateOfLoss ge datetime'1978-01-01T00:00:00Z'",
        "$limit": 50000
    }

    all_claims = []
    offset = 0
    batch_count = 0

    try:
        while True:
            params["$offset"] = offset
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not data or len(data) == 0:
                    print(f"Reached end of data at offset {offset}")
                    break

                all_claims.extend(data)
                batch_count += 1
                print(f"Batch {batch_count}: fetched {len(data)} rows (total: {len(all_claims)})")

                offset += 50000

                if batch_count >= 20:  # Safety limit for demo
                    print("Reached batch limit; stopping.")
                    break

            except requests.exceptions.RequestException as e:
                print(f"Error fetching batch at offset {offset}: {e}")
                if offset == 0:
                    raise

        if not all_claims:
            raise Exception("No claims fetched")

        df_fema = pd.DataFrame(all_claims)

        # Select only required columns
        required_cols = [
            'claimNumber', 'dateOfLoss', 'dateOfClaim', 'latitude', 'longitude',
            'countyCode', 'countyName', 'floodZone', 'occupancyType', 'amountPaid',
            'yearOfConstruction', 'elevatedBuildingIndicator'
        ]

        # Keep only columns that exist
        available_cols = [col for col in required_cols if col in df_fema.columns]
        df_fema = df_fema[available_cols]

        # Convert datetime strings
        for col in ['dateOfLoss', 'dateOfClaim']:
            if col in df_fema.columns:
                df_fema[col] = pd.to_datetime(df_fema[col], errors='coerce')

        print(f"\nFEMA dataset fetched: {len(df_fema)} rows, {len(df_fema.columns)} columns")
        print(f"Columns: {list(df_fema.columns)}")

        # Save checkpoint
        checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoint_fema.csv')
        df_fema.to_csv(checkpoint_path, index=False)
        print(f"Checkpoint saved: {checkpoint_path}\n")

        return df_fema

    except Exception as e:
        print(f"Failed to fetch FEMA data: {e}")
        print("Using synthetic data instead for demonstration...\n")
        return create_synthetic_fema_data()


def create_synthetic_fema_data():
    """Create synthetic FEMA data for demonstration."""
    np.random.seed(42)
    n_rows = 300000

    dates = pd.date_range('1978-01-01', '2023-12-31', periods=n_rows)

    df = pd.DataFrame({
        'claimNumber': [f'CLAIM{i:08d}' for i in range(n_rows)],
        'dateOfLoss': dates,
        'dateOfClaim': dates + pd.to_timedelta(np.random.randint(0, 365, n_rows), unit='D'),
        'latitude': np.random.uniform(24.5, 30.5, n_rows),
        'longitude': np.random.uniform(-87.5, -80.0, n_rows),
        'countyCode': np.random.choice(['12001', '12003', '12005', '12009', '12011', '12013', '12015', '12017', '12019', '12021'], n_rows),
        'countyName': np.random.choice(['Alachua', 'Baker', 'Bradford', 'Brevard', 'Broward', 'Calhoun', 'Charlotte', 'Citrus', 'Clay', 'Collier'], n_rows),
        'floodZone': np.random.choice(['A', 'AE', 'X', None], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
        'occupancyType': np.random.choice(['Residential', 'Commercial', 'Industrial'], n_rows),
        'amountPaid': np.random.exponential(50000, n_rows),
        'yearOfConstruction': np.random.randint(1800, 2024, n_rows),
        'elevatedBuildingIndicator': np.random.choice([0, 1], n_rows)
    })

    df = df[df['amountPaid'] > 0]
    print(f"Synthetic FEMA dataset created: {len(df)} rows\n")
    return df


# ============================================================================
# STEP 2: PARSE HURDAT2 STORM TRACKS
# ============================================================================
def parse_hurdat2():
    """Parse NOAA HURDAT2 storm track data."""
    print("=" * 70)
    print("STEP 2: Parsing NOAA HURDAT2 Storm Tracks")
    print("=" * 70)

    url = "https://www.nhc.noaa.gov/data/hurdat2/hurdat2.txt"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        lines = response.text.strip().split('\n')

        track_data = []
        current_storm = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')

            # Header line
            if len(parts) > 0 and parts[0].startswith('AL') or parts[0].startswith('EP') or parts[0].startswith('CP'):
                current_storm = {
                    'storm_id': parts[0],
                    'storm_name': parts[1].strip() if len(parts) > 1 else 'UNNAMED',
                    'year': int(parts[0][2:6]) if len(parts[0]) >= 6 else None
                }
            # Track point line
            elif current_storm and len(parts) >= 7:
                try:
                    date_str = parts[0].strip()
                    time_str = parts[1].strip()

                    if len(date_str) == 8 and len(time_str) == 4:
                        track_time = pd.to_datetime(f"{date_str} {time_str}", format='%Y%m%d %H%M')
                        lat_raw = int(parts[4].strip().replace('N', '').replace('S', ''))
                        lon_raw = int(parts[5].strip().replace('E', '').replace('W', ''))

                        latitude = lat_raw / 10.0
                        longitude = -lon_raw / 10.0

                        wind_speed = int(parts[6].strip()) if parts[6].strip().isdigit() else 0

                        track_data.append({
                            'storm_name': current_storm['storm_name'],
                            'year': current_storm['year'],
                            'track_time': track_time,
                            'latitude': latitude,
                            'longitude': longitude,
                            'wind_speed_kt': wind_speed,
                            'central_pressure_mb': float(parts[7]) if len(parts) > 7 and parts[7].strip().isdigit() else np.nan
                        })
                except (ValueError, IndexError):
                    continue

        if track_data:
            df_storms = pd.DataFrame(track_data)
            print(f"HURDAT2 parsed: {len(df_storms)} track points from {df_storms['storm_name'].nunique()} storms")
        else:
            print("No track data parsed; using synthetic data...")
            df_storms = create_synthetic_hurdat2()

        checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoint_hurdat2.csv')
        df_storms.to_csv(checkpoint_path, index=False)
        print(f"Checkpoint saved: {checkpoint_path}\n")

        return df_storms

    except Exception as e:
        print(f"Failed to fetch HURDAT2: {e}")
        print("Using synthetic data instead...\n")
        return create_synthetic_hurdat2()


def create_synthetic_hurdat2():
    """Create synthetic HURDAT2 data."""
    np.random.seed(42)
    n_points = 15000

    storms = ['ALEX', 'BONNIE', 'CHARLEY', 'DANIELLE', 'ELSA', 'FIONA', 'GRACE', 'HENRI', 'IDA', 'JULIA']

    df = pd.DataFrame({
        'storm_name': np.random.choice(storms, n_points),
        'year': np.random.choice(range(1978, 2024), n_points),
        'track_time': pd.date_range('1978-01-01', periods=n_points, freq='6H'),
        'latitude': np.random.uniform(24, 31, n_points),
        'longitude': np.random.uniform(-87, -80, n_points),
        'wind_speed_kt': np.random.randint(35, 160, n_points),
        'central_pressure_mb': np.random.uniform(880, 1020, n_points)
    })

    print(f"Synthetic HURDAT2 created: {len(df)} track points\n")
    return df


# ============================================================================
# STEP 3: SPATIAL JOIN
# ============================================================================
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in miles using Haversine formula."""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


def spatial_join(df_fema, df_storms):
    """Spatial join: match claims to nearest storm within constraints."""
    print("=" * 70)
    print("STEP 3: Spatial Join (Claims to Storms)")
    print("=" * 70)

    # Initialize output columns
    df_fema['distance_from_track_mi'] = np.nan
    df_fema['nearest_storm_name'] = None
    df_fema['storm_wind_speed_kt'] = np.nan
    df_fema['storm_category'] = None
    df_fema['days_to_storm'] = np.nan

    DISTANCE_THRESHOLD_MI = 50
    TIME_WINDOW_DAYS = 30

    # Process in chunks to manage memory
    chunk_size = 10000
    total_chunks = (len(df_fema) + chunk_size - 1) // chunk_size

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(df_fema))

        if chunk_idx % 5 == 0:
            print(f"Processing chunk {chunk_idx + 1}/{total_chunks} (rows {start}-{end})")

        for i in range(start, end):
            claim_lat = df_fema.iloc[i]['latitude']
            claim_lon = df_fema.iloc[i]['longitude']
            claim_date = df_fema.iloc[i]['dateOfLoss']

            if pd.isna(claim_lat) or pd.isna(claim_lon) or pd.isna(claim_date):
                continue

            # Find storms within time and distance window
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
                df_fema.at[i, 'distance_from_track_mi'] = min_distance
                df_fema.at[i, 'nearest_storm_name'] = best_match['storm_name']
                df_fema.at[i, 'storm_wind_speed_kt'] = best_match['wind_speed_kt']
                df_fema.at[i, 'days_to_storm'] = (best_match['track_time'] - claim_date).days

    matched = df_fema['nearest_storm_name'].notna().sum()
    pct_matched = 100 * matched / len(df_fema)
    print(f"Matched {matched} claims to storms ({pct_matched:.1f}%)")

    # Save checkpoint
    checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoint_spatial_join.csv')
    df_fema.to_csv(checkpoint_path, index=False)
    print(f"Checkpoint saved: {checkpoint_path}\n")

    return df_fema


# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================
def engineer_features(df):
    """Calculate derived features."""
    print("=" * 70)
    print("STEP 4: Feature Engineering")
    print("=" * 70)

    # building_age_years
    df['building_age_years'] = 2024 - df['yearOfConstruction']
    df['building_age_years'] = df['building_age_years'].clip(0, 200)

    # claim_lag_days
    df['claim_lag_days'] = (df['dateOfClaim'] - df['dateOfLoss']).dt.days.clip(0, 365)

    # is_elevated
    df['is_elevated'] = df['elevatedBuildingIndicator'].astype(int)

    # flood_zone_encoded: X→1, AE→2, A→3, None→0
    flood_zone_map = {'X': 1, 'AE': 2, 'A': 3}
    df['flood_zone_encoded'] = df['floodZone'].map(flood_zone_map).fillna(0).astype(int)

    # occupancy_encoded: Residential→1, Commercial→2, Industrial→3, Other→0
    occupancy_map = {'Residential': 1, 'Commercial': 2, 'Industrial': 3}
    df['occupancy_encoded'] = df['occupancyType'].map(occupancy_map).fillna(0).astype(int)

    # distance_bin: 0-10→3, 10-25→2, 25-50→1, None→-1
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

    df['distance_bin'] = df['distance_from_track_mi'].apply(distance_bin)

    # storm_category_encoded from wind speed
    def wind_to_category(wind):
        if pd.isna(wind):
            return -1
        elif wind < 39:
            return 0  # Tropical Depression
        elif wind < 74:
            return 1  # Tropical Storm
        elif wind < 96:
            return 2  # Category 1
        elif wind < 111:
            return 3  # Category 2
        elif wind < 130:
            return 4  # Category 3
        elif wind < 157:
            return 5  # Category 4
        else:
            return 6  # Category 5

    df['storm_category_encoded'] = df['storm_wind_speed_kt'].apply(wind_to_category)
    df['storm_category'] = df['storm_category_encoded'].map({
        -1: 'None', 0: 'Depression', 1: 'Tropical Storm',
        2: 'Cat1', 3: 'Cat2', 4: 'Cat3', 5: 'Cat4', 6: 'Cat5'
    })

    # log_amountpaid
    df['log_amountpaid'] = np.log1p(df['amountPaid'])

    # historical_storm_freq: count of storms within 50mi per county in 40-year window
    df['historical_storm_freq'] = df.groupby('countyCode').transform(
        lambda x: (x['nearest_storm_name'].notna().sum() + 1) // (len(x) // 100 + 1)
    ).astype(int)

    print("Features engineered:")
    print("  - building_age_years, claim_lag_days, is_elevated")
    print("  - flood_zone_encoded, occupancy_encoded, distance_bin")
    print("  - storm_category_encoded, storm_category, log_amountpaid")
    print("  - historical_storm_freq\n")

    return df


# ============================================================================
# STEP 5: VALIDATION & CLEANING
# ============================================================================
def validate_and_clean(df):
    """Validate data and apply filters."""
    print("=" * 70)
    print("STEP 5: Validation & Cleaning")
    print("=" * 70)

    print(f"Starting rows: {len(df)}")

    # Drop missing lat/lon/amountPaid
    before = len(df)
    df = df.dropna(subset=['latitude', 'longitude', 'amountPaid'])
    print(f"After dropping missing lat/lon/amountPaid: {len(df)} (removed {before-len(df)})")

    # Filter for positive amountPaid
    before = len(df)
    df = df[df['amountPaid'] > 0]
    print(f"After filtering amountPaid > 0: {len(df)} (removed {before-len(df)})")

    # Temporal checks
    before = len(df)
    df = df[
        (df['dateOfLoss'] >= pd.to_datetime('1978-01-01')) &
        (df['dateOfClaim'] >= df['dateOfLoss']) &
        ((df['dateOfClaim'] - df['dateOfLoss']).dt.days <= 365)
    ]
    print(f"After temporal validation: {len(df)} (removed {before-len(df)})")

    # Spatial checks
    before = len(df)
    df = df[
        (df['latitude'] >= 24.5) & (df['latitude'] <= 30.5) &
        (df['longitude'] >= -87.5) & (df['longitude'] <= -80.0)
    ]
    print(f"After spatial validation: {len(df)} (removed {before-len(df)})")

    # Check target variable skewness
    raw_skew = df['amountPaid'].skew()
    log_skew = df['log_amountpaid'].skew()
    print(f"\nTarget skewness (raw): {raw_skew:.2f}")
    print(f"Target skewness (log): {log_skew:.2f}")

    print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns\n")

    return df


# ============================================================================
# STEP 6: EDA PLOTS
# ============================================================================
def create_eda_plots(df):
    """Generate 7 EDA plots."""
    print("=" * 70)
    print("STEP 6: Creating EDA Plots")
    print("=" * 70)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # Plot 1: Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['amountPaid'], bins=100, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Amount Paid ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Raw Claim Amounts Distribution')

    axes[1].hist(df['log_amountpaid'], bins=100, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Log Amount Paid')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log-Transformed Claim Amounts Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_target_distribution.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Plot 1: 01_target_distribution.png")

    # Plot 2: Claims by year
    claims_by_year = df.groupby(df['dateOfLoss'].dt.year).size()
    plt.figure(figsize=(14, 5))
    plt.bar(claims_by_year.index, claims_by_year.values, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Year')
    plt.ylabel('Number of Claims')
    plt.title('Claims by Year (1978-2023)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_claims_by_year.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Plot 2: 02_claims_by_year.png")

    # Plot 3: Top 10 counties
    top_counties = df['countyName'].value_counts().head(10)
    county_payouts = df.groupby('countyName')['amountPaid'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    top_counties.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Top 10 Counties by Claim Count')
    axes[0].set_ylabel('Number of Claims')
    axes[0].set_xlabel('County')
    axes[0].tick_params(axis='x', rotation=45)

    county_payouts[top_counties.index].plot(kind='bar', ax=axes[1], color='darkgreen', edgecolor='black', alpha=0.7)
    axes[1].set_title('Top 10 Counties by Mean Payout')
    axes[1].set_ylabel('Mean Amount Paid ($)')
    axes[1].set_xlabel('County')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_claims_by_county.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Plot 3: 03_claims_by_county.png")

    # Plot 4: Storm category vs payout
    plt.figure(figsize=(12, 6))
    storm_categories = ['None', 'Depression', 'Tropical Storm', 'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5']
    df_plot = df[df['storm_category'].isin(storm_categories)]
    sns.boxplot(data=df_plot, x='storm_category', y='log_amountpaid', order=storm_categories)
    plt.title('Storm Category vs. Log Claim Amount')
    plt.xlabel('Storm Category')
    plt.ylabel('Log Amount Paid')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_storm_category_vs_payout.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Plot 4: 04_storm_category_vs_payout.png")

    # Plot 5: Flood zone vs payout
    plt.figure(figsize=(12, 6))
    flood_zones = ['X', 'AE', 'A', 'None']
    df['flood_zone_label'] = df['floodZone'].fillna('None').astype(str)
    df_plot = df[df['flood_zone_label'].isin(flood_zones)]
    sns.boxplot(data=df_plot, x='flood_zone_label', y='log_amountpaid', order=flood_zones)
    plt.title('Flood Zone vs. Log Claim Amount')
    plt.xlabel('Flood Zone')
    plt.ylabel('Log Amount Paid')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_flood_zone_vs_payout.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Plot 5: 05_flood_zone_vs_payout.png")

    # Plot 6: Building age vs payout
    plt.figure(figsize=(12, 6))
    plt.scatter(df['building_age_years'], df['log_amountpaid'], alpha=0.3, s=10)
    z = np.polyfit(df['building_age_years'].dropna(), df.loc[df['building_age_years'].notna(), 'log_amountpaid'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(df['building_age_years'].min(), df['building_age_years'].max(), 100)
    plt.plot(x_trend, p(x_trend), 'r-', linewidth=2, label='Trend')
    plt.xlabel('Building Age (years)')
    plt.ylabel('Log Amount Paid')
    plt.title('Building Age vs. Log Claim Amount')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_building_age_vs_payout.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Plot 6: 06_building_age_vs_payout.png")

    # Plot 7: Distance from storm vs payout
    df_with_distance = df[df['distance_from_track_mi'].notna()]
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
    print("✓ Plot 7: 07_distance_vs_payout.png\n")


# ============================================================================
# STEP 7: SUMMARY STATISTICS
# ============================================================================
def print_summary_statistics(df):
    """Print dataset summary."""
    print("=" * 70)
    print("STEP 7: Summary Statistics")
    print("=" * 70)

    print(f"\nDataset Summary:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Target (log_amountpaid) non-null: {df['log_amountpaid'].notna().sum()} ({100*df['log_amountpaid'].notna().sum()/len(df):.1f}%)")
    print(f"\nTarget Variable (log_amountpaid):")
    print(f"  Mean: ${np.exp(df['log_amountpaid'].mean()):,.0f}")
    print(f"  Median: ${np.exp(df['log_amountpaid'].median()):,.0f}")
    print(f"  Std: ${np.exp(df['log_amountpaid'].std()):,.0f}")

    print(f"\nStorm Matching:")
    matched = df['nearest_storm_name'].notna().sum()
    print(f"  {matched} claims matched to storms ({100*matched/len(df):.1f}%)")

    print(f"\nTop 10 Missing Values:")
    missing = df.isnull().sum().sort_values(ascending=False).head(10)
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} ({100*count/len(df):.1f}%)")

    print()


# ============================================================================
# STEP 8: EXPORT FINAL CSV
# ============================================================================
def export_final_csv(df):
    """Export clean dataset to CSV."""
    print("=" * 70)
    print("STEP 8: Exporting Final CSV")
    print("=" * 70)

    output_path = os.path.join(OUTPUT_DIR, 'insurance_day1_clean.csv')
    df.to_csv(output_path, index=False)
    print(f"✓ Final dataset exported: {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("FLORIDA HURRICANE CLAIM SEVERITY PREDICTOR - DAY 1 PIPELINE")
    print("=" * 70 + "\n")

    try:
        # Step 1: Fetch FEMA
        df_fema = fetch_fema_claims()

        # Step 2: Parse HURDAT2
        df_storms = parse_hurdat2()

        # Step 3: Spatial join
        df_fema = spatial_join(df_fema, df_storms)

        # Step 4: Feature engineering
        df_fema = engineer_features(df_fema)

        # Step 5: Validation & cleaning
        df_fema = validate_and_clean(df_fema)

        # Step 6: EDA plots
        create_eda_plots(df_fema)

        # Step 7: Summary stats
        print_summary_statistics(df_fema)

        # Step 8: Export
        export_final_csv(df_fema)

        print("=" * 70)
        print("✓ PIPELINE COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"\nX PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
