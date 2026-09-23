"""Train the year-control variant used by the claim estimator (the production features plus yearOfLoss).

Building age is mostly a stand-in for the calendar (see ablation_year_control.py). Giving the estimator the loss year
lets it separate "how old is the building" from "when did the loss happen". The dashboard's headline model and Model
tab stay on the original feature set; only the estimator uses this one. It gets its own out-of-sample residuals so its
80% range and its correction for under-predicting unseen years come from the model that actually produces the estimate.

Run after train_model.py:   python train_year_model.py   ->  data/model_year.txt, data/year_control_model.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import train_model as tm

ROOT = Path(__file__).parent
FEATURES = tm.FEATURES + ["yearOfLoss"]


def main():
    df = pd.read_csv(tm.DATA, low_memory=False)
    y, year = df[tm.TARGET], df["yearOfLoss"]
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(df, y, groups=year):
        oof[te] = tm.fit(df.iloc[tr][FEATURES], y.iloc[tr]).predict(df.iloc[te][FEATURES])

    actual = np.expm1(y)
    q10, q50, q90 = (y - oof).quantile([0.10, 0.50, 0.90]).tolist()
    out = {
        "features": FEATURES,
        "r2_log": float(r2_score(y, oof)),
        "median_abs_pct_error": float(np.median(np.abs(np.expm1(oof) - actual) / actual) * 100),
        "residual_quantiles": {"q10": q10, "q50": q50, "q90": q90},
        "year_min": int(year.min()), "year_max": int(year.max()),
        "claims_by_year": {int(k): int(v) for k, v in year.value_counts().sort_index().items()},
    }
    tm.fit(df[FEATURES], y).save_model(str(ROOT / "data" / "model_year.txt"))
    (ROOT / "data" / "year_control_model.json").write_text(json.dumps(out, indent=2))
    print(f"year-control model: R2 {out['r2_log']:.3f} | median abs % error {out['median_abs_pct_error']:.0f}% | "
          f"residual q10/q50/q90 {q10:+.2f} / {q50:+.2f} / {q90:+.2f} (actual/predicted median {np.exp(q50):.2f}x)")


if __name__ == "__main__":
    main()
