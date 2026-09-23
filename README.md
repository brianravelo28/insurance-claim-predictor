# Florida Flood Claim Severity

How much does a paid NFIP flood claim cost in Florida, and what drives it? Real FEMA claims joined to NOAA hurricane tracks, modeled with LightGBM and served as an interactive Dash dashboard.

**Live dashboard: https://florida-flood-claim-severity.onrender.com/**

- **311,400 paid claims**, 1978-2026, every payout in constant **2025 dollars** (CPI-U).
- **77.8% matched to a storm** (within 150 miles and 7 days of the loss). Checked against FEMA's own storm labels: 95.6% of claims FEMA calls a named hurricane or tropical storm matched the same-named storm.
- **Model R² 0.20 on unseen loss years** (predicting the average scores -0.05), about 74% median error. It is a modest signal, not a precise predictor.

The dashboard has five tabs: Overview, Storms, Risk factors, Model, and a Claim estimator.

## Data

| Source | What | Notes |
|---|---|---|
| [OpenFEMA NFIP claims v2](https://www.fema.gov/api/open/v2/FimaNfipClaims) | 448,425 Florida claims | **Deprecated; removed Oct 15, 2026.** Data frozen as of 2026-06-01. |
| [NOAA HURDAT2](https://www.nhc.noaa.gov/data/) | Atlantic best-track storm positions | Filename changes each release; the fetcher looks it up. |
| [FRED CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | CPI-U, for inflation adjustment | October 2025 has no published value; interpolated. |

Payout = building + contents + increased-cost-of-compliance. About 30% of claims were closed without payment and are excluded, so this describes severity given a paid claim.

## Method notes

- **Storm join:** tracks are interpolated to hourly positions; each claim takes the nearest track point within 150 miles and +/-7 days. The spec's original 50 miles / +/-30 days was tested against FEMA's `floodEvent` labels and matched the wrong storm a third of the time.
- **Evaluation:** 5-fold cross-validation grouped by loss year, so every claim is predicted by a model that never saw its year. A random split scores higher (0.41) only because claims from the same storm leak across train and test.
- **Known limits:** the model under-predicts unseen years by about 1.4x (the estimator corrects for it). Building age, its strongest signal, is mostly a stand-in for loss year: a loss-year control alone recovers essentially all of age's predictive value (R² 0.206 vs 0.204). The claim estimator therefore uses the year-control model and lets you pick the loss year. FEMA's post-FIRM flag, tested as a cleaner alternative to age, doesn't help. Locations are blurred by FEMA to 0.1 degree (~7 miles).
- **[DATA_QUIRKS.md](DATA_QUIRKS.md)** catalogs the odd things found in the data (a `1492-10-12` construction-date placeholder on 4% of rows, two coexisting occupancy code schemes, malformed HURDAT2 lines, and more).

## Run it

```bash
pip install -r requirements.txt

python fetch_data.py          # download and cache raw data into data/raw/ (gitignored)
python build_dataset.py       # clean, storm-join, inflation-adjust, engineer features
python train_model.py         # year-grouped CV, metrics, out-of-sample predictions
python ablation_year_control.py  # is building age just a stand-in for the loss year? (~10 min)
python ablation_postfirm.py   # does FEMA's post-FIRM flag replace or add to building age? (~10 min)
python train_year_model.py    # the year-control model behind the claim estimator
python build_deploy_data.py   # compact bundle for the app -> deploy_data/
python app.py                 # dashboard at http://localhost:8058
```

`deploy_data/` is committed, so the dashboard runs without any of the steps above. Because the FEMA v2 endpoint is being retired, keep your local `data/raw/` cache if you want to rebuild from scratch after October 2026.

## Deploy (Render)

`render.yaml` is a Render Blueprint (New + > Blueprint > pick this repo). It installs `requirements-render.txt` and serves `wsgi.py` with gunicorn; `wsgi.py` answers the health check immediately while the app loads in the background.

## Files

| File | Purpose |
|---|---|
| `fetch_data.py` | Download and cache FEMA claims, HURDAT2, CPI |
| `build_dataset.py` | Cleaning, storm join, features, validation |
| `train_model.py` | LightGBM training and honest evaluation |
| `ablation_year_control.py` | Tests whether building age is a stand-in for the loss year |
| `ablation_postfirm.py` | Tests FEMA's post-FIRM construction flag as a replacement for, or addition to, building age (it doesn't help) |
| `train_year_model.py` | Trains the loss-year-control model the claim estimator uses (R² 0.211, with its own out-of-sample calibration) |
| `build_deploy_data.py` | Package all 311,400 paid claims as one compact table (~4 MB on disk, ~10 MB in memory) plus a few small static aggregates. Render has 512 MB of RAM, so the app keeps data compact and does no heavy work at startup |
| `constants.py` | Category labels shared by the build script and the app |
| `app.py`, `wsgi.py` | Dash dashboard and its hosting entry point |
