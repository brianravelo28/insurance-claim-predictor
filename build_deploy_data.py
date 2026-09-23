"""Build the slim deploy_data/ bundle the hosted dashboard loads (columns trimmed and downcast, parquet + snappy).

Run after build_dataset.py and train_model.py:   python build_deploy_data.py

  claims.parquet       one row per paid claim: attributes, storm match, actual and out-of-sample predicted payout
  model.txt            LightGBM booster trained on all claims (predicts log of the payout in 2025 dollars)
  model_metrics.json   honest evaluation numbers, feature importance, per-year results
  build_report.json    data-cleaning counts and checks
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "deploy_data"

KEEP = [
    "claimId", "yearOfLoss", "countyName", "latitude", "longitude", "occupancy_group", "occupancy_group_encoded",
    "flood_zone_group", "flood_zone_encoded", "building_age_years", "building_age_missing", "is_elevated",
    "storm_category", "storm_category_encoded", "distance_from_track_mi", "storm_wind_speed_kt", "days_to_storm",
    "distance_bin", "historical_storm_freq", "amountPaid", "amountPaid_real", "cpi_factor", "storm_id",
    "nearest_storm_name",
]


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "insurance_clean.csv.gz", usecols=KEEP, low_memory=False)
    pred = pd.read_csv(DATA / "predictions.csv.gz", usecols=["claimId", "prediction_amount"])
    df = df.merge(pred, on="claimId", how="left", validate="one_to_one")
    assert df["prediction_amount"].notna().all()

    year = pd.to_numeric(df["storm_id"].str[4:8], errors="coerce")
    df["storm_label"] = (df["nearest_storm_name"] + " (" + year.astype("Int64").astype("string") + ")").astype("category")
    df = df.drop(columns=["claimId", "storm_id", "nearest_storm_name"])

    small = {"yearOfLoss": "int16", "building_age_years": "int16", "building_age_missing": "int8", "is_elevated": "int8",
             "occupancy_group_encoded": "int8", "flood_zone_encoded": "int8", "storm_category_encoded": "int8",
             "distance_bin": "int8", "historical_storm_freq": "int8"}
    for col, dt in small.items():
        df[col] = df[col].astype(dt)
    for col in ["latitude", "longitude", "distance_from_track_mi", "storm_wind_speed_kt", "days_to_storm",
                "amountPaid", "amountPaid_real", "cpi_factor", "prediction_amount"]:
        df[col] = df[col].astype("float32")
    for col in ["countyName", "occupancy_group", "flood_zone_group", "storm_category"]:
        df[col] = df[col].astype("category")

    df.to_parquet(OUT / "claims.parquet", compression="snappy", index=False)
    for name in ("model.txt", "model_metrics.json", "build_report.json"):
        shutil.copy(DATA / name, OUT / name)

    total = 0.0
    for f in sorted(OUT.iterdir()):
        mb = f.stat().st_size / 1e6
        total += mb
        print(f"{f.name:22s} {mb:6.2f} MB")
    print(f"{'TOTAL':22s} {total:6.2f} MB | {len(df):,} rows x {df.shape[1]} columns | "
          f"in-memory ~{df.memory_usage(deep=True).sum() / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
