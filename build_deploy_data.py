"""Build the slim deploy_data/ bundle the hosted dashboard loads.

Run after build_dataset.py and train_model.py:   python build_deploy_data.py

The free Render tier has 512 MB of RAM, so every static aggregate is computed here, once, instead of at app startup:

  claims.parquet       ALL paid claims, one compact row each (categoricals + float32); every tab filters and aggregates it live
  aggregates.json      county centroids, feature codes and residual quantiles (small, static)
  model.txt            LightGBM booster trained on all claims (predicts log of the payout in 2025 dollars)
  model_metrics.json   honest evaluation numbers, feature importance, per-year results
  build_report.json    data-cleaning counts and checks
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (AGE_BINS, AGE_LABELS, DIST_LABELS, ELEVATED_LABELS, FLOOD_ORDER, OCC_ORDER, STORM_ORDER)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "deploy_data"

USECOLS = [
    "claimId", "yearOfLoss", "countyName", "latitude", "longitude", "occupancy_group", "occupancy_group_encoded",
    "flood_zone_group", "flood_zone_encoded", "building_age_years", "is_elevated", "storm_category",
    "distance_from_track_mi", "historical_storm_freq", "amountPaid_real", "storm_id", "nearest_storm_name",
]
QUANTILES = {0.1: "q10", 0.25: "q25", 0.5: "q50", 0.75: "q75", 0.9: "q90"}


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "insurance_clean.csv.gz", usecols=USECOLS, low_memory=False)
    pred = pd.read_csv(DATA / "predictions.csv.gz", usecols=["claimId", "prediction_amount"])
    df = df.merge(pred, on="claimId", how="left", validate="one_to_one")
    assert df["prediction_amount"].notna().all()

    df["elevated"] = np.where(df["is_elevated"] == 1, "Elevated", "Not elevated")
    df["age_band"] = pd.cut(df["building_age_years"], AGE_BINS, labels=AGE_LABELS).astype(str)
    d = df["distance_from_track_mi"]
    df["distance_band"] = np.select([d.isna(), d <= 25, d <= 50, d <= 100], DIST_LABELS[:4], default=DIST_LABELS[4])
    df["matched"] = df["storm_id"].notna()
    year = pd.to_numeric(df["storm_id"].str[4:8], errors="coerce")
    df["storm_label"] = df["nearest_storm_name"] + " (" + year.astype("Int64").astype("string") + ")"

    agg = {"n_claims": int(len(df)), "year_min": int(df["yearOfLoss"].min()), "year_max": int(df["yearOfLoss"].max())}

    cty = df.groupby("countyName").agg(lat=("latitude", "mean"), lon=("longitude", "mean"), freq=("historical_storm_freq", "median"))
    agg["counties"] = {k: {c: float(v) for c, v in row.items()} for k, row in cty.iterrows()}
    agg["occupancy_code"] = {k: int(v) for k, v in dict(zip(df["occupancy_group"], df["occupancy_group_encoded"])).items()}
    agg["flood_code"] = {k: int(v) for k, v in dict(zip(df["flood_zone_group"], df["flood_zone_encoded"])).items()}

    resid = np.log1p(df["amountPaid_real"]) - np.log1p(df["prediction_amount"])
    q10, q50, q90 = resid.quantile([0.10, 0.50, 0.90]).tolist()
    agg["residual_quantiles"] = {"q10": q10, "q50": q50, "q90": q90}

    (OUT / "aggregates.json").write_text(json.dumps(agg))

    # One compact row per paid claim: every categorical is a category, every float is float32.
    table = pd.DataFrame({
        "yearOfLoss": df["yearOfLoss"].astype("int16"),
        "countyName": df["countyName"].astype("category"),
        "occupancy_group": pd.Categorical(df["occupancy_group"], categories=OCC_ORDER),
        "flood_zone_group": pd.Categorical(df["flood_zone_group"], categories=FLOOD_ORDER),
        "storm_category": pd.Categorical(df["storm_category"], categories=STORM_ORDER),
        "elevated": pd.Categorical(df["elevated"], categories=ELEVATED_LABELS),
        "age_band": pd.Categorical(df["age_band"], categories=AGE_LABELS),
        "distance_band": pd.Categorical(df["distance_band"], categories=DIST_LABELS),
        "storm_label": df["storm_label"].astype("category"),
        "amountPaid_real": df["amountPaid_real"].astype("float32"),
        "prediction_amount": df["prediction_amount"].astype("float32"),
    })
    for col in ("occupancy_group", "flood_zone_group", "storm_category", "elevated", "age_band", "distance_band"):
        assert table[col].notna().all(), f"unmapped label in {col}"
    table.to_parquet(OUT / "claims.parquet", compression="snappy", index=False)

    for name in ("model.txt", "model_metrics.json", "build_report.json"):
        shutil.copy(DATA / name, OUT / name)

    total = 0.0
    for f in sorted(OUT.iterdir()):
        mb = f.stat().st_size / 1e6
        total += mb
        print(f"{f.name:22s} {mb:6.2f} MB")
    print(f"{'TOTAL':22s} {total:6.2f} MB | claims table {len(table):,} rows x {table.shape[1]} cols, "
          f"~{table.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory")


if __name__ == "__main__":
    main()
