"""Can FEMA's post-FIRM construction flag replace (or add to) building age?

The flag says whether construction started before or after the community's initial flood map (FIRM); buildings started
after 1974-12-31 or the initial FIRM, whichever is later, are "post-FIRM" and were built to floodplain standards. Unlike
the construction date it is complete for every claim (no 1492 placeholder; every placeholder building is flagged pre-FIRM).
Scored with the same year-grouped cross-validation and folds as ablation_year_control.py, so results are comparable.

Run after build_dataset.py:   python ablation_postfirm.py   ->  data/ablation_postfirm.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import train_model as tm
from ablation_year_control import AGE, BASE, NO_AGE, fit

ROOT = Path(__file__).parent

VARIANTS = {
    "Current model": BASE,
    "Add post-FIRM flag (age kept)": BASE + ["post_firm"],
    "Post-FIRM flag instead of building age": NO_AGE + ["post_firm"],
    "Loss year + post-FIRM flag (age kept)": BASE + ["yearOfLoss", "post_firm"],
    "Loss year + post-FIRM flag, no building age": NO_AGE + ["yearOfLoss", "post_firm"],
}


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
        sample = df[feats].sample(20000, random_state=42)
        imp = np.abs(fit(df, y, feats).predict(sample, pred_contrib=True)[:, :-1]).mean(axis=0)
        share = dict(zip(feats, (100 * imp / imp.sum()).round(2).tolist()))
        results[name] = {
            "r2_log": round(float(r2_score(y, oof)), 4), "median_abs_pct_error": round(ape, 1),
            "fold_r2": [round(float(r2_score(y.iloc[te], oof[te])), 3) for _, te in folds],
            "age_share_pct": share.get("building_age_years", 0.0), "post_firm_share_pct": share.get("post_firm", 0.0),
            "year_share_pct": share.get("yearOfLoss", 0.0),
        }
        r = results[name]
        print(f"{name:46s} R2 {r['r2_log']:.3f} | err {r['median_abs_pct_error']:.0f}% | age {r['age_share_pct']:5.1f}% | "
              f"post-FIRM {r['post_firm_share_pct']:5.1f}% | year {r['year_share_pct']:5.1f}%", flush=True)
    (ROOT / "data" / "ablation_postfirm.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
