"""Build the slim deploy_data/ bundle the hosted dashboard loads.

Run after build_dataset.py and train_model.py:   python build_deploy_data.py

The free Render tier has 512 MB of RAM, so every static aggregate is computed here, once, instead of at app startup:

  claims_slim.parquet  4 columns per claim (year, county, real-dollar payout, storm-matched) for the year-range filter
  aggregates.json      box-plot stats, storm table, county centroids, calibration bins, residual quantiles, feature codes
  model.txt            LightGBM booster trained on all claims (predicts log of the payout in 2025 dollars)
  model_metrics.json   honest evaluation numbers, feature importance, per-year results
  build_report.json    data-cleaning counts and checks
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from constants import AGE_BINS, AGE_LABELS, DIST_LABELS, FACTOR_COLUMNS

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

    box = {}
    for name, (col, order) in FACTOR_COLUMNS.items():
        g = df.groupby(col)["amountPaid_real"]
        q = g.quantile(list(QUANTILES)).unstack().rename(columns=QUANTILES)
        q["n"] = g.size()
        q = q.reindex([o for o in order if o in q.index])
        box[name] = [{"label": label, **{k: float(v) for k, v in row.items()}} for label, row in q.iterrows()]
    agg["box"] = box

    g = df[df["matched"]].groupby("storm_label")["amountPaid_real"]
    agg["storms"] = [{"label": k, "claims": int(r["size"]), "total": float(r["sum"]), "median": float(r["median"])}
                     for k, r in g.agg(["size", "sum", "median"]).iterrows()]

    top = df[df["matched"]].groupby(["yearOfLoss", "storm_label"]).size().reset_index(name="n")
    agg["year_top_storm"] = top.sort_values("n").groupby("yearOfLoss").tail(1).set_index("yearOfLoss")["storm_label"].to_dict()

    cty = df.groupby("countyName").agg(lat=("latitude", "mean"), lon=("longitude", "mean"), freq=("historical_storm_freq", "median"))
    agg["counties"] = {k: {c: float(v) for c, v in row.items()} for k, row in cty.iterrows()}
    agg["occupancy_code"] = {k: int(v) for k, v in dict(zip(df["occupancy_group"], df["occupancy_group_encoded"])).items()}
    agg["flood_code"] = {k: int(v) for k, v in dict(zip(df["flood_zone_group"], df["flood_zone_encoded"])).items()}

    bins = pd.qcut(df["prediction_amount"], 20, duplicates="drop")
    cal = df.groupby(bins, observed=True).agg(pred=("prediction_amount", "median"), actual=("amountPaid_real", "median"))
    agg["calibration"] = {"pred": cal["pred"].tolist(), "actual": cal["actual"].tolist()}

    resid = np.log1p(df["amountPaid_real"]) - np.log1p(df["prediction_amount"])
    q10, q50, q90 = resid.quantile([0.10, 0.50, 0.90]).tolist()
    agg["residual_quantiles"] = {"q10": q10, "q50": q50, "q90": q90}

    (OUT / "aggregates.json").write_text(json.dumps(agg))

    slim = df[["yearOfLoss", "countyName", "amountPaid_real", "matched"]].copy()
    slim["yearOfLoss"] = slim["yearOfLoss"].astype("int16")
    slim["amountPaid_real"] = slim["amountPaid_real"].astype("float32")
    slim["countyName"] = slim["countyName"].astype("category")
    slim.to_parquet(OUT / "claims_slim.parquet", compression="snappy", index=False)

    for name in ("model.txt", "model_metrics.json", "build_report.json"):
        shutil.copy(DATA / name, OUT / name)

    total = 0.0
    for f in sorted(OUT.iterdir()):
        mb = f.stat().st_size / 1e6
        total += mb
        print(f"{f.name:22s} {mb:6.2f} MB")
    print(f"{'TOTAL':22s} {total:6.2f} MB | slim table {len(slim):,} rows, ~{slim.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory")


if __name__ == "__main__":
    main()
