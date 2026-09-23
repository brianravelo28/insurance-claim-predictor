"""Is building age doing real work, or is it a stand-in for the loss year?

Building age (age at time of loss) is the model's strongest feature, but old buildings appear mostly in recent, larger
claims, so age and loss year are entangled. This scores the same model with and without a loss-year control, and with
building age removed, using the same year-grouped cross-validation as train_model.py. If age mostly carries the era,
adding the year should absorb its importance and removing age should cost little once year is present.

Run after build_dataset.py:   python ablation_year_control.py   ->  data/ablation_year_control.json
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import train_model as tm

ROOT = Path(__file__).parent
AGE = ["building_age_years"]
BASE = tm.FEATURES
NO_AGE = [f for f in BASE if f not in AGE + ["building_age_missing"]]

VARIANTS = {
    "Current model": BASE,
    "Add loss year (year control)": BASE + ["yearOfLoss"],
    "Remove building age": NO_AGE,
    "Loss year, no building age": NO_AGE + ["yearOfLoss"],
    "Loss year + construction year instead of age": [f for f in BASE if f not in AGE] + ["yearOfLoss", "yearOfConstruction"],
}


def fit(X, y, features):
    cat = [c for c in tm.CATEGORICAL if c in features]
    return lgb.train(tm.PARAMS, lgb.Dataset(X[features], y, categorical_feature=cat), num_boost_round=tm.N_ROUNDS)


def main():
    df = pd.read_csv(tm.DATA, low_memory=False)
    y, year = df[tm.TARGET], df["yearOfLoss"]
    folds = list(GroupKFold(n_splits=5).split(df, y, groups=year))
    results = {}
    for name, feats in VARIANTS.items():
        oof = np.zeros(len(df))
        for tr, te in folds:
            oof[te] = fit(df.iloc[tr], y.iloc[tr], feats).predict(df.iloc[te][feats])
        actual = np.expm1(y)
        ape = float(np.median(np.abs(np.expm1(oof) - actual) / actual) * 100)
        final = fit(df, y, feats)
        sample = df[feats].sample(20000, random_state=42)
        imp = np.abs(final.predict(sample, pred_contrib=True)[:, :-1]).mean(axis=0)
        share = dict(zip(feats, (100 * imp / imp.sum()).round(2).tolist()))
        results[name] = {
            "r2_log": round(float(r2_score(y, oof)), 4), "median_abs_pct_error": round(ape, 1),
            "fold_r2": [round(float(r2_score(y.iloc[te], oof[te])), 3) for _, te in folds],
            "importance_pct": dict(sorted(share.items(), key=lambda kv: -kv[1])[:6]),
            "age_share_pct": share.get("building_age_years", 0.0),
            "year_share_pct": round(share.get("yearOfLoss", 0.0), 2),
            "construction_year_share_pct": round(share.get("yearOfConstruction", 0.0), 2),
        }
        r = results[name]
        print(f"{name:46s} R2 {r['r2_log']:.3f} | err {r['median_abs_pct_error']:.0f}% | age {r['age_share_pct']:5.1f}% | "
              f"year {r['year_share_pct']:5.1f}% | constr.year {r['construction_year_share_pct']:5.1f}%", flush=True)

    (ROOT / "data" / "ablation_year_control.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
