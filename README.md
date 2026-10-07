# Florida Flood Claim Severity

How much does a paid NFIP flood claim cost in Florida, and what drives it? Real FEMA claims joined to NOAA hurricane tracks, modeled with LightGBM and served as an interactive Dash dashboard.

**Live dashboard: https://florida-flood-claim-severity.onrender.com/**

- **311,400 paid claims**, 1978-2026, every payout in constant **2025 dollars** (CPI-U).
- **77.8% matched to a storm** (within 150 miles and 7 days of the loss). Checked against FEMA's own storm labels: 95.6% of claims FEMA calls a named hurricane or tropical storm matched the same-named storm.
- **Model R² 0.20 on unseen loss years** (predicting the average scores -0.05), about 74% median error. It is a modest signal, not a precise predictor.

## The dashboard

A shared filter bar above the tabs (**Loss Years, County, Occupancy, Flood Zone, Storm Categories**; empty means all) drives the first four tabs. The Claim Estimator replaces it with its own control bar in the same spot.

| Tab | What it shows |
|---|---|
| **Overview** | Claims, total paid and median claim by year of loss (hover a bar to see the storm behind that year), plus a county map. |
| **Storms** | The top storms by claims, total paid or median claim, with a table. |
| **Risk Factors** | Payout distributions by storm category, flood zone, occupancy type, elevated building, building age and distance from the storm track, with a table of claims, % of claims, median, mean and middle 50%. |
| **Model** | The evaluation table (always all claims); calibration and actual-vs-predicted charts that respond to the filters, including R², error and bias for the filtered slice; feature importance; and two tests of whether building age is just a stand-in for the loss year. |
| **Claim Estimator** | A what-if tool: the typical payout and an 80% range for a hypothetical claim, as storm strength changes and across loss years. |

Every chart hides Plotly's mode bar and shows a one-line hint about zooming and the legend underneath it.

## Data

| Source | What | Notes |
|---|---|---|
| [OpenFEMA NFIP claims v2](https://www.fema.gov/api/open/v2/FimaNfipClaims) | 448,425 Florida claims | **Deprecated; FEMA schedules removal for Oct 15, 2026.** Data frozen as of 2026-06-01. FEMA names `nfip-redacted-claims-v3` as the replacement, but I couldn't find its API path (checked 2026-10-07), so rebuilding after removal needs the cached `data/raw/` or a port to v3. |
| [NOAA HURDAT2](https://www.nhc.noaa.gov/data/) | Atlantic best-track storm positions | Filename changes each release; the fetcher looks it up. |
| [FRED CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | CPI-U, for inflation adjustment | October 2025 has no published value; interpolated. |

Payout = building + contents + increased-cost-of-compliance. About 30% of claims were closed without payment and are excluded, so this describes severity given a paid claim.

## Method notes

- **Storm join:** tracks are interpolated to hourly positions; each claim takes the nearest track point within 150 miles and +/-7 days. The spec's original 50 miles / +/-30 days was tested against FEMA's `floodEvent` labels and matched the wrong storm a third of the time.
- **Model:** LightGBM on the log of the 2025-dollar payout with an L1 (median) objective, 13 features (building age and a missing-age flag, elevated, flood zone, occupancy group, storm category / wind / distance / days to the storm, storms nearby in the prior 40 years, latitude and longitude). Damage amounts and coverage limits are not features; they only exist after a claim does.
- **Evaluation:** 5-fold cross-validation grouped by loss year, so every claim is predicted by a model that never saw its year. A random split scores higher (0.41) only because claims from the same storm leak across train and test. A train-on-the-past, test-on-the-future split scores negative (-0.27 for 2020 onward, versus -0.62 for predicting the average) because payouts keep growing faster than inflation.
- **Known limits:** the model under-predicts unseen years by about 1.4x. Building age, its strongest signal, is mostly a stand-in for loss year: a loss-year control alone recovers essentially all of age's predictive value (R² 0.206 vs 0.204). The Claim Estimator therefore uses a second model that also knows the loss year (R² 0.211) with its own out-of-sample calibration (it under-predicts by 1.53x and the estimator corrects for that). FEMA's post-FIRM flag, tested as a cleaner alternative to age, doesn't help. About 3.8% of claims have an unusable construction date; the model fills them with the median age and a flag, and a year-aware fill made no difference to the model, so it is used only for the Building Age chart. Locations are blurred by FEMA to 0.1 degree (~7 miles). Within narrow filter slices R² is much lower than the headline.

## Run it

```bash
pip install -r requirements.txt

python fetch_data.py             # download and cache raw data into data/raw/ (gitignored)
python build_dataset.py          # clean, storm-join, inflation-adjust, engineer features
python train_model.py            # year-grouped CV, metrics, out-of-sample predictions
python train_year_model.py       # the year-control model behind the Claim Estimator (required)
python ablation_year_control.py  # optional: is building age just a stand-in for the loss year? (~10 min; shown on the Model tab)
python ablation_postfirm.py      # optional: does FEMA's post-FIRM flag replace or add to age? (~10 min; shown on the Model tab)
python ablation_placeholder_age.py  # optional: year-aware fill for unusable construction dates (a few min; not shown in the app)
python build_deploy_data.py      # compact bundle for the app -> deploy_data/
python app.py                    # dashboard at http://localhost:8058
```

`deploy_data/` is committed, so the dashboard runs without any of the steps above. Because the FEMA v2 endpoint is being retired, keep your local `data/raw/` cache if you want to rebuild from scratch after October 2026.

## Deploy (Render)

`render.yaml` is a Render Blueprint (New + > Blueprint > pick this repo). It installs `requirements-render.txt` and serves `wsgi.py` with gunicorn (one worker, four threads) on the Starter plan, which stays awake and has 512 MB of RAM; the app uses about 190 MB at startup and roughly 250 MB under use.

`wsgi.py` answers `/healthz` immediately and loads the app on the first request, in a background thread. (Starting that thread at import time doesn't survive gunicorn's fork: the app finishes loading in the master process while the worker keeps serving the loading page.) `/debugz` reports whether the app loaded, its memory, and any startup error.

## Files

| File | Purpose |
|---|---|
| `fetch_data.py` | Download and cache FEMA claims, HURDAT2 and CPI |
| `build_dataset.py` | Cleaning, storm join, features, inflation adjustment, validation |
| `train_model.py` | LightGBM training and honest evaluation (year-grouped CV) |
| `train_year_model.py` | The loss-year-control model the Claim Estimator uses (R² 0.211, own calibration) |
| `ablation_year_control.py` | Tests whether building age is a stand-in for the loss year |
| `ablation_postfirm.py` | Tests FEMA's post-FIRM construction flag as a replacement for, or addition to, building age (it doesn't help) |
| `ablation_placeholder_age.py` | Tests a year-aware fill for the ~11.7k unusable construction dates vs. the flat median (no meaningful difference) |
| `build_deploy_data.py` | Packages all 311,400 paid claims as one compact table (~4 MB on disk) plus the models, metrics and a few small aggregates into `deploy_data/`. Render has 512 MB of RAM, so the app keeps data compact and does no heavy work at startup |
| `constants.py` | Category labels shared by the build script and the app |
| `app.py`, `wsgi.py` | Dash dashboard and its hosting entry point |
| `render.yaml`, `requirements-render.txt` | Render Blueprint and the hosted app's dependencies |
| `requirements.txt` | Dependencies for the full pipeline and the app |
| `deploy_data/` | What the hosted app loads (committed) |

*Numbers above were last checked against `deploy_data/` on 2026-10-07.*
