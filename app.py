"""
Florida Flood Claim Severity - interactive Dash app (real FEMA NFIP claims + NOAA HURDAT2 storm tracks).

Tabs:
1. Overview          claims by year and county (payouts in constant 2025 dollars)
2. Storms            which storms drove the most claims and dollars
3. Risk factors      how payouts differ by storm strength, flood zone, occupancy, elevation, age, distance
4. Model             honest out-of-sample evaluation of the LightGBM severity model
5. Claim estimator   what-if: estimated payout for a hypothetical claim

Run locally: python app.py   (port from $PORT, default 8058).  Hosted: gunicorn wsgi:application
Data: deploy_data/ (built by build_deploy_data.py).
"""
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from constants import FACTOR_COLUMNS, FLOOD_ORDER, OCC_ORDER, STORM_ORDER

BASE_DIR = Path(__file__).resolve().parent
DEPLOY = BASE_DIR / "deploy_data"
_T0 = time.time()


def _log(stage):
    """Startup progress to stderr (elapsed seconds + resident memory on Linux) for the host's logs."""
    rss = ""
    try:
        with open("/proc/self/status") as f:
            rss = f", {int(next(l.split()[1] for l in f if l.startswith('VmRSS'))) // 1024} MB"
    except (OSError, StopIteration):
        pass
    print(f"[app] {stage} ({time.time() - _T0:.1f}s{rss})", file=sys.stderr, flush=True)


# Design tokens (same light palette as the Rossmann dashboard)
SURFACE, PAGE_BG, BORDER, GRID = "#fcfcfb", "#f4f4f2", "#e5e4df", "#ecebe7"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#6b6a66"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4"]

# ---------------------------------------------------------------- data ----
# One compact row per paid claim (all 311,400; categoricals + float32, ~10 MB in memory). Every tab filters and
# aggregates it live, so nothing is a precomputed summary of a subset.
claims = pd.read_parquet(DEPLOY / "claims.parquet")
claims["matched"] = claims["storm_label"].notna()
claims["log_actual"] = np.log1p(claims["amountPaid_real"]).astype("float32")
claims["log_pred"] = np.log1p(claims["prediction_amount"]).astype("float32")
AGG = json.loads((DEPLOY / "aggregates.json").read_text())
METRICS = json.loads((DEPLOY / "model_metrics.json").read_text())
REPORT = json.loads((DEPLOY / "build_report.json").read_text())
booster = lgb.Booster(model_file=str(DEPLOY / "model.txt"))
FEATURES = METRICS["features"]
_log("data loaded")

YEAR_MIN, YEAR_MAX, N_CLAIMS = AGG["year_min"], AGG["year_max"], AGG["n_claims"]
RESID_Q10, RESID_Q50, RESID_Q90 = (AGG["residual_quantiles"][k] for k in ("q10", "q50", "q90"))
UNDERPREDICT = float(np.exp(RESID_Q50))  # actual median claim / predicted, on years the model never saw
LOG_TICKS = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6]

# Short forms for chart x-axis ticks only; full names stay in the dropdown, table, and hover.
AXIS_SHORT_LABELS = {
    "Tropical Depression": "Trop. Dep.", "Tropical Storm": "Trop. Storm",
    "X / B / C (moderate-minimal)": "X / B / C",
    "A / AH / AO / AR / A99 (no base elevation, shallow, special)": "A / AH / AO...",
    "AE / A1-A30 (base elevation known)": "AE / A1-A30", "V / VE (velocity, coastal)": "V / VE",
    "Condo association": "Condo assoc.", "Unit in multi-unit building": "Multi-unit", "Non-residential": "Non-resid.",
    "No storm match": "No match",
}

FACTOR_NOTES = {
    "Storm category": "Strongest category of the nearest storm (HURDAT2 wind at the closest track point within 150 miles and 7 days). The order is not strictly increasing because a few catastrophic storms dominate individual categories; categories with no claims are omitted.",
    "Flood zone": "FEMA flood zone the property was rated in, grouped by FEMA's zone definitions.",
    "Occupancy type": "FEMA's legacy (1-4, 6) and newer (11-19) occupancy codes describe the same building types, so they are grouped together.",
    "Elevated building": "FEMA's elevatedBuildingIndicator.",
    "Building age": "Age at time of loss. About 4% of buildings have an unknown construction date (FEMA uses a 1492 placeholder) and are set to the median age. Caution: old-building claims are mostly recent claims (median age at loss was about 30 in the 1980s and 50 in the 2020s), and recent claims are larger even after inflation adjustment, so age and loss year are entangled.",
    "Distance from storm track": "Distance from the claim's (blurred) location to the nearest storm track point.",
}
FACTORS = {name: (col, order, FACTOR_NOTES[name]) for name, (col, order) in FACTOR_COLUMNS.items()}

COUNTY_INFO = AGG["counties"]
COUNTIES = sorted(COUNTY_INFO)
OCC_CODE, FLOOD_CODE = AGG["occupancy_code"], AGG["flood_code"]


def _present(col, order=None):
    """Category labels that actually occur in the data (a category with no claims isn't a useful filter option)."""
    have = set(claims[col].dropna().unique())
    return [o for o in (order or sorted(have)) if o in have]


COUNTY_OPTIONS = _present("countyName")
OCC_OPTIONS = _present("occupancy_group", OCC_ORDER)
FLOOD_OPTIONS = _present("flood_zone_group", FLOOD_ORDER)
STORM_OPTIONS = _present("storm_category", STORM_ORDER)


@lru_cache(maxsize=4)
def _subset(years, counties, occs, floods, storms):
    m = claims["yearOfLoss"].between(years[0], years[1]).to_numpy()
    for col, sel in (("countyName", counties), ("occupancy_group", occs), ("flood_zone_group", floods),
                     ("storm_category", storms)):
        if sel:
            m &= claims[col].isin(sel).to_numpy()
    return claims[m]


def filtered(years, counties, occs, floods, storms):
    """The claims matching the global filter bar (an empty multi-select means 'all')."""
    return _subset(tuple(years or (YEAR_MIN, YEAR_MAX)), tuple(counties or ()), tuple(occs or ()), tuple(floods or ()),
                   tuple(storms or ()))


_log("precompute done")

FEATURE_NAMES = {
    "building_age_years": "Building age", "building_age_missing": "Building age unknown", "is_elevated": "Elevated building",
    "flood_zone_encoded": "Flood zone", "occupancy_group_encoded": "Occupancy type", "distance_bin": "Distance band to storm",
    "storm_category_encoded": "Storm category", "historical_storm_freq": "Storms nearby, prior 40 yrs",
    "latitude": "Latitude", "longitude": "Longitude", "distance_from_track_mi": "Distance to storm track",
    "storm_wind_speed_kt": "Storm wind speed", "days_to_storm": "Days from loss to storm",
}


# ------------------------------------------------------------- helpers ----
def money(x):
    x = float(x)
    if abs(x) >= 1e9:
        return f"${x / 1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e4:
        return f"${x / 1e3:.0f}k"
    return f"${x:,.0f}"


def pct(x):
    return f"{x:.1f}%" if isinstance(x, (int, float)) else "-"


def wind_category(w):
    if w is None or np.isnan(w) or w <= 0:
        return -1
    return int(np.select([w < 39, w < 74, w < 96, w < 111, w < 130, w < 157], [0, 1, 2, 3, 4, 5], default=6))


def distance_bin(dist):
    if dist is None or np.isnan(dist):
        return -1
    return 4 if dist <= 25 else 3 if dist <= 50 else 2 if dist <= 100 else 1


def predict_payout(county, occupancy, flood, age, elevated, wind, dist):
    """Estimated payout in constant 2025 dollars for one hypothetical claim."""
    storm = wind is not None and wind > 0
    info = COUNTY_INFO[county]
    row = {
        "building_age_years": age, "building_age_missing": 0, "is_elevated": int(elevated),
        "flood_zone_encoded": FLOOD_CODE[flood], "occupancy_group_encoded": OCC_CODE[occupancy],
        "distance_bin": distance_bin(dist) if storm else -1, "storm_category_encoded": wind_category(wind),
        "historical_storm_freq": info["freq"], "latitude": info["lat"], "longitude": info["lon"],
        "distance_from_track_mi": dist if storm else np.nan, "storm_wind_speed_kt": wind if storm else np.nan,
        "days_to_storm": 0.0 if storm else np.nan,
    }
    x = pd.DataFrame([row])[FEATURES]
    return float(np.expm1(booster.predict(x, num_threads=1))[0])


# ---------------------------------------------------------- app + style ----
CSS = f"""
:root {{ color-scheme: light; }}
body {{ margin: 0; background: {PAGE_BG}; color: {INK};
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.wrap {{ font-size: 15px; max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }}
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px 18px; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
.kpi {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 16px; }}
.kpi .v {{ font-size: 28px; font-weight: 650; letter-spacing: -0.01em; }}
.kpi .l {{ font-size: 15px; color: {INK_2}; margin-bottom: 2px; }}
.kpi .s {{ font-size: 15px; color: {INK_3}; margin-top: 2px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 16px 20px; align-items: flex-end; }}
.controls label {{ display: block; font-size: 15px; color: {INK_2}; margin-bottom: 4px; }}
.note {{ font-size: 15px; color: {INK_2}; line-height: 1.5; margin: 10px 2px 0; }}
details.about {{ margin-top: 12px; }}
details.about summary {{ cursor: pointer; font-weight: 600; color: {INK_2}; }}
details.about ul {{ margin: 8px 0 0; padding-left: 20px; color: {INK_2}; line-height: 1.55; font-size: 15px; }}
.dash-dropdown, .dash-dropdown *, .dash-datepicker-input, .DateInput_input, .Select-value-label, .Select-input input {{ font-size: 15px !important; }}
.dash-options-list-option-checkbox {{ width: 18px; height: 18px; margin: 0; cursor: pointer; }}
.dash-options-list:not(.dash-checklist) .dash-options-list-option {{ display: flex !important; align-items: center; gap: 10px; width: 100%; box-sizing: border-box; padding: 8px 12px; margin: 0; cursor: pointer; font-size: 15px; }}
.dash-options-list:not(.dash-checklist) .dash-options-list-option:hover {{ background: {PAGE_BG}; }}
table.eval {{ border-collapse: collapse; width: 100%; font-size: 15px; }}
table.eval th, table.eval td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid {BORDER}; }}
table.eval th {{ color: {INK_2}; font-weight: 600; }}
"""

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Florida Flood Claim Severity"
server = app.server
app.index_string = f"""<!DOCTYPE html>
<html>
<head>
{{%metas%}}
<title>{{%title%}}</title>
{{%favicon%}}
{{%css%}}
<style>{CSS}</style>
</head>
<body>
{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""


def kpi(label, value, sub=None):
    return html.Div([html.Div(label, className="l"), html.Div(value, className="v")]
                    + ([html.Div(sub, className="s")] if sub else []), className="kpi")


def style_fig(fig, title=None, height=None, unified=True):
    fig.update_layout(
        template="plotly_white", title=dict(text=title, x=0, font=dict(size=17, color=INK)) if title else None,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK_2, size=15), hoverlabel=dict(font_size=15),
        margin=dict(l=76, r=20, t=60 if title else 30, b=80), height=height,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", yanchor="top", font=dict(color=INK_2)),
        hovermode="x unified" if unified else "closest",
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False, automargin=True)
    return fig


def log_dollar_axis(fig, axis="y", title=None, lo=None, hi=None):
    """Log-scale dollar axis with readable ticks ($1k, $3k, $10k...) instead of Plotly's crowded 2..9 minor labels."""
    ticks = [t for t in LOG_TICKS if (lo is None or t >= lo * 0.99) and (hi is None or t <= hi * 1.01)]
    kw = dict(type="log", tickvals=ticks, ticktext=[money(t) for t in ticks], minor=dict(showgrid=False))
    if title:
        kw["title"] = title
    (fig.update_yaxes if axis == "y" else fig.update_xaxes)(**kw)


def field(label, control):
    return html.Div([html.Label(label), control])


def header():
    m = METRICS
    h, base = m["headline_year_grouped_cv"], m["baseline_year_grouped_cv_predict_train_mean"]
    return html.Div(
        [
            html.H1("Florida Flood Claim Severity", style={"margin": "0 0 4px", "fontSize": "28px", "letterSpacing": "-0.02em"}),
            html.Div("How much does a paid NFIP flood claim cost? Real FEMA claims joined to NOAA hurricane tracks, "
                     "with every payout in constant 2025 dollars.", style={"color": INK_2, "marginBottom": "16px"}),
            html.Div(
                [
                    kpi("Paid claims analyzed", f"{N_CLAIMS:,}", f"Florida, {YEAR_MIN}-{YEAR_MAX}"),
                    kpi("Matched to a storm", pct(REPORT["storm_matched_pct"]), "within 150 mi and 7 days of the loss"),
                    kpi("Model R² on unseen years", f"{h['r2_log']:.2f}", f"predicting the average scores {base['r2_log']:.2f}"),
                    kpi("Typical prediction error", f"{h['median_abs_pct_error']:.0f}%", "median absolute % error"),
                ],
                className="kpi-row",
            ),
            html.Details(
                [
                    html.Summary("About this data and model"),
                    html.Ul([
                        html.Li("Claims: FEMA OpenFEMA NFIP claims (v2), a frozen snapshot as of 2026-06-01. Storms: NOAA HURDAT2 Atlantic best-track, interpolated to hourly positions. Prices: CPI-U from FRED, so a 1992 claim and a 2024 claim are comparable."),
                        html.Li("Only claims that were actually paid are included (payout = building + contents + increased-cost-of-compliance). About 30% of claims were closed without payment and are excluded, so this describes severity given a paid claim."),
                        html.Li("A claim is matched to the nearest storm track point within 150 miles and 7 days of the loss date. Checked against FEMA's own storm labels: 95.6% of claims FEMA calls a named hurricane or tropical storm matched the same-named storm."),
                        html.Li(f"The model is LightGBM on log payout. Its score comes from 5-fold cross-validation grouped by loss year, so each claim is predicted by a model that never saw that year: R² {h['r2_log']:.2f}. This is a modest signal, not a precise predictor; storm-to-storm variation is large."),
                        html.Li("Caveat: building age is the model's strongest signal, but it is entangled with loss year (older buildings appear mostly in recent, larger claims), so treat it as a correlation, not a cause."),
                        html.Li("Locations are deliberately blurred by FEMA to 0.1 degree (about 7 miles), and 135 'Florida' claims carried coordinates in other states and were dropped."),
                        html.Li("The project's data-quirk log (sentinel dates, changing occupancy codes, malformed HURDAT2 lines, and more) is in DATA_QUIRKS.md in the repository."),
                    ]),
                ],
                className="about",
            ),
        ],
        className="card", style={"marginBottom": "16px"},
    )


TAB_STYLE = {"padding": "10px 14px", "fontSize": "15px", "border": f"1px solid {BORDER}", "backgroundColor": PAGE_BG,
             "color": INK_2, "fontWeight": "500"}
TAB_SELECTED = {**TAB_STYLE, "backgroundColor": SURFACE, "color": INK, "fontWeight": "650", "borderTop": f"2px solid {BLUE}"}

def filter_bar():
    def multi(id_, options, placeholder, width):
        return dcc.Dropdown(id=id_, options=options, multi=True, placeholder=placeholder, style={"width": width})

    return html.Div(
        [
            html.Div([
                field("Loss years", html.Div(dcc.RangeSlider(
                    id="f-years", min=YEAR_MIN, max=YEAR_MAX, step=1, value=[YEAR_MIN, YEAR_MAX], allowCross=False,
                    marks={y: str(y) for y in range(1980, YEAR_MAX, 10)}), style={"width": "300px", "paddingBottom": "18px"})),
                field("County", multi("f-county", COUNTY_OPTIONS, "All counties", "200px")),
                field("Occupancy", multi("f-occ", OCC_OPTIONS, "All types", "200px")),
                field("Flood zone", multi("f-flood", FLOOD_OPTIONS, "All zones", "200px")),
                field("Storm category", multi("f-storm", STORM_OPTIONS, "All categories", "200px")),
            ], className="controls"),
            html.Div(id="f-count", className="note", style={"marginTop": "8px"}),
        ],
        id="filters", className="card", style={"marginBottom": "12px"},
    )


app.layout = html.Div(
    className="wrap",
    children=[
        header(),
        filter_bar(),
        dcc.Tabs(
            id="tabs", value="tab1", colors={"border": BORDER, "primary": BLUE, "background": PAGE_BG},
            children=[dcc.Tab(label=lbl, value=val, style=TAB_STYLE, selected_style=TAB_SELECTED) for val, lbl in
                      [("tab1", "Overview"), ("tab2", "Storms"), ("tab3", "Risk factors"), ("tab4", "Model"),
                       ("tab5", "Claim estimator")]],
        ),
        html.Div(id="tab-content", style={"marginTop": "16px"}),
    ],
)


# ------------------------------------------------------------ tab layouts ----
MEASURES = {"claims": "Number of claims", "total": "Total paid (2025 $)", "median": "Median claim (2025 $)"}


def tab1_layout():
    return html.Div([
        html.Div([
            field("Measure", dcc.Dropdown(id="t1-measure", options=[{"label": v, "value": k} for k, v in MEASURES.items()],
                                          value="total", clearable=False, style={"width": "240px"})),
        ], className="controls card"),
        html.Div(id="t1-kpis", className="kpi-row", style={"margin": "12px 0"}),
        html.Div(dcc.Graph(id="t1-year-chart"), className="card"),
        html.Div(dcc.Graph(id="t1-map"), className="card", style={"marginTop": "12px"}),
        html.Div("Storm years stand out: hover a bar to see the storm behind most of that year's claims. "
                 "Map bubbles are sized by the selected measure and colored by median claim.", className="note"),
    ])


def tab2_layout():
    return html.Div([
        html.Div([
            field("Rank storms by", dcc.Dropdown(id="t2-metric", options=[{"label": v, "value": k} for k, v in MEASURES.items()],
                                                 value="total", clearable=False, style={"width": "240px"})),
            field("Show top", dcc.Dropdown(id="t2-n", options=[{"label": str(n), "value": n} for n in (10, 15, 25)],
                                           value=15, clearable=False, style={"width": "110px"})),
        ], className="controls card"),
        html.Div(dcc.Graph(id="t2-chart"), className="card", style={"marginTop": "12px"}),
        html.Div(id="t2-table", className="card", style={"marginTop": "12px"}),
        html.Div("Each claim is assigned the nearest storm within 150 miles and 7 days; claims with no storm nearby are not counted here. "
                 "Older, unnamed storms appear as 'Unnamed'.", className="note"),
    ])


def tab3_layout():
    return html.Div([
        html.Div([field("Compare payouts by", dcc.Dropdown(id="t3-factor", options=[{"label": k, "value": k} for k in FACTORS],
                                                           value="Storm category", clearable=False, style={"width": "300px"}))],
                 className="controls card"),
        html.Div(dcc.Graph(id="t3-chart"), className="card", style={"marginTop": "12px"}),
        html.Div(id="t3-table", className="card", style={"marginTop": "12px"}),
        html.Div(id="t3-note", className="note"),
    ])


def tab4_layout():
    m = METRICS
    rows = [
        ("Year-grouped cross-validation (headline)", m["headline_year_grouped_cv"], "Every claim predicted by a model that never saw its loss year."),
        ("  baseline: predict the training average", m["baseline_year_grouped_cv_predict_train_mean"], "What 'no model' scores."),
        (f"Time-based stress test ({m['temporal_stress_test']['test_years']})", m["temporal_stress_test"]["model"], f"Train on {m['temporal_stress_test']['train_years']}, test on later years."),
        ("  baseline: predict the training average", m["temporal_stress_test"]["baseline_predict_train_mean"], "Still negative: recent storm years are larger even in constant dollars."),
        ("Random split (for contrast only)", m["random_split_for_contrast"], "Looks better because claims from the same storm land in both train and test."),
    ]
    table = html.Table([
        html.Thead(html.Tr([html.Th(c) for c in ["Evaluation", "R² (log)", "Typical % error", "Claims tested", "What it means"]])),
        html.Tbody([html.Tr([html.Td(n), html.Td(f"{s['r2_log']:.2f}"), html.Td(f"{s['median_abs_pct_error']:.0f}%"),
                             html.Td(f"{s['n']:,}"), html.Td(txt)]) for n, s, txt in rows]),
    ], className="eval")

    imp = pd.Series(m["shap_importance_pct"]).sort_values()
    f3 = go.Figure(go.Bar(x=imp.values, y=[FEATURE_NAMES.get(k, k) for k in imp.index], orientation="h", marker_color=BLUE,
                          hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    style_fig(f3, "What the model relies on (share of mean |SHAP|, all claims)", 420, unified=False)
    f3.update_xaxes(title="% of total attribution")

    folds = ", ".join(f"{r:.2f}" for r in m["fold_r2"])
    return html.Div([
        html.Div([html.H3("Evaluation", style={"margin": "0 0 8px"}), table], className="card"),
        html.Div(f"R² by fold is {folds}: some storm seasons are far harder to predict than others. "
                 f"On years it has never seen, the model under-predicts: the actual median claim runs about {UNDERPREDICT:.1f}x its prediction "
                 "(payouts in a held-out storm year tend to be larger than in the years it learned from). The calibration chart below shows "
                 "this, and the claim estimator corrects for it. The table above always covers all claims; everything "
                 "below the next line responds to the filters.", className="note"),
        html.Div(id="t4-dyn", style={"marginTop": "12px"}),
        html.Div(dcc.Graph(figure=f3), className="card", style={"marginTop": "12px"}),
        html.Div("R² and error are on the log scale of the payout in 2025 dollars. 'Typical % error' is the median of |predicted - actual| / actual. "
                 "Features do not include damage amounts or coverage limits, which are only known after the claim exists.", className="note"),
    ])


def tab5_layout():
    return html.Div([
        html.Div([
            field("County", dcc.Dropdown(id="t5-county", options=COUNTIES, value="Lee", clearable=False, style={"width": "200px"})),
            field("Occupancy", dcc.Dropdown(id="t5-occ", options=[o for o in OCC_ORDER if o != "Unknown"], value="Single-family",
                                            clearable=False, style={"width": "260px"})),
            field("Flood zone", dcc.Dropdown(id="t5-flood", options=[f for f in FLOOD_ORDER if f in FLOOD_CODE],
                                             value=FLOOD_ORDER[2], clearable=False, style={"width": "330px"})),
            field("Building age (years)", html.Div(dcc.Slider(id="t5-age", min=0, max=100, step=1, value=30,
                                                             marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"}),
                                                   style={"width": "260px", "paddingBottom": "18px"})),
            field("Elevated building", dcc.Checklist(id="t5-elev", options=[{"label": " Elevated", "value": 1}], value=[])),
            field("Storm wind speed (kt, 0 = no storm)", html.Div(dcc.Slider(id="t5-wind", min=0, max=150, step=5, value=90,
                                                                            marks={0: "0", 39: "TS", 74: "Cat 1", 111: "Cat 3", 150: "150"}),
                                                                  style={"width": "300px", "paddingBottom": "18px"})),
            field("Distance from storm track (mi)", html.Div(dcc.Slider(id="t5-dist", min=0, max=150, step=5, value=40,
                                                                        marks={0: "0", 50: "50", 100: "100", 150: "150"}),
                                                              style={"width": "260px", "paddingBottom": "18px"})),
        ], className="controls card"),
        html.Div(id="t5-kpis", className="kpi-row", style={"margin": "12px 0"}),
        html.Div(dcc.Graph(id="t5-chart"), className="card"),
        html.Div("An estimate for a paid claim, in 2025 dollars. The model's raw output is scaled up by the median out-of-sample under-prediction, "
                 "and with R² of about 0.2 the range is wide by design: it covers the middle 80% of out-of-sample errors. Storm inputs are the wind speed at the storm's closest track point and its distance from the claim; "
                 "location uses the county's average claim coordinates.", className="note"),
    ])


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return {"tab1": tab1_layout, "tab2": tab2_layout, "tab3": tab3_layout, "tab4": tab4_layout, "tab5": tab5_layout}[tab]()


# -------------------------------------------------------------- callbacks ----
def _measure_series(frame_group, measure):
    if measure == "claims":
        return frame_group.size()
    col = frame_group["amountPaid_real"]
    return col.sum() if measure == "total" else col.median()


FILTERS = [Input("f-years", "value"), Input("f-county", "value"), Input("f-occ", "value"), Input("f-flood", "value"),
           Input("f-storm", "value")]


def empty_fig(message, height=300):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color=INK_2), xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig, height=height)


NO_CLAIMS = "No claims match these filters"


@app.callback(Output("f-count", "children"), *FILTERS)
def update_filter_count(years, counties, occs, floods, storms):
    sub = filtered(years, counties, occs, floods, storms)
    if len(sub) == N_CLAIMS:
        return f"Showing all {N_CLAIMS:,} paid claims. Filters apply to every tab except the claim estimator."
    return (f"Showing {len(sub):,} of {N_CLAIMS:,} paid claims ({100 * len(sub) / N_CLAIMS:.1f}%), "
            f"{money(sub['amountPaid_real'].sum())} paid in 2025 dollars.")


@app.callback(Output("filters", "style"), Input("tabs", "value"))
def toggle_filters(tab):
    return {"marginBottom": "12px", "display": "none" if tab == "tab5" else "block"}


def _measure_series(frame_group, measure):
    if measure == "claims":
        return frame_group.size()
    col = frame_group["amountPaid_real"]
    return col.sum() if measure == "total" else col.median()


def update_tab1(years, counties, occs, floods, storms, measure):
    sub = filtered(years, counties, occs, floods, storms)
    if sub.empty:
        return [kpi("Claims", "0")], empty_fig(NO_CLAIMS), empty_fig(NO_CLAIMS, 520)
    kpis = [kpi("Claims", f"{len(sub):,}"), kpi("Total paid", money(sub["amountPaid_real"].sum()), "2025 dollars"),
            kpi("Median claim", money(sub["amountPaid_real"].median()), "2025 dollars"),
            kpi("Near a storm", pct(100 * sub["matched"].mean()), "within 150 mi and 7 days")]

    y = _measure_series(sub.groupby("yearOfLoss"), measure)
    top = sub[sub["matched"]].groupby(["yearOfLoss", "storm_label"], observed=True).size().reset_index(name="n")
    top_storm = top.sort_values("n").groupby("yearOfLoss").tail(1).set_index("yearOfLoss")["storm_label"].astype(str).to_dict()
    tops = [top_storm.get(int(i), "none matched") for i in y.index]
    fig = go.Figure(go.Bar(x=y.index, y=y.values, marker_color=BLUE, customdata=tops,
                           hovertemplate="%{x}: %{y:,.0f}<br>Most claims from: %{customdata}<extra></extra>"))
    style_fig(fig, f"{MEASURES[measure]} by year of loss", 380, unified=False)
    fig.update_yaxes(title=MEASURES[measure], tickprefix="" if measure == "claims" else "$")

    g = sub.groupby("countyName", observed=True)
    size = _measure_series(g, measure)
    med = g["amountPaid_real"].median()
    counts = g.size()
    keep = counts[counts > 0].index
    size, med, counts = size[keep], med[keep], counts[keep]
    ci = pd.DataFrame(COUNTY_INFO).T.loc[size.index]
    scale = 46 / np.sqrt(max(size.max(), 1))
    m = go.Figure(go.Scattermapbox(
        lat=ci["lat"], lon=ci["lon"], mode="markers", text=size.index,
        marker=dict(size=np.maximum(np.sqrt(size.values) * scale, 5), color=med.values, colorscale="YlOrRd", showscale=True,
                    colorbar=dict(title="Median<br>claim", tickprefix="$"), opacity=0.75),
        customdata=np.c_[size.values, med.values, counts.values],
        hovertemplate="%{text}<br>" + MEASURES[measure] + ": %{customdata[0]:,.0f}<br>Median claim: %{customdata[1]:$,.0f}"
                      "<br>Claims: %{customdata[2]:,.0f}<extra></extra>"))
    m.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=27.9, lon=-83.2), zoom=5.4), height=520,
                    margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=SURFACE)
    return kpis, fig, m


app.callback(Output("t1-kpis", "children"), Output("t1-year-chart", "figure"), Output("t1-map", "figure"),
             *FILTERS, Input("t1-measure", "value"))(update_tab1)


def update_tab2(years, counties, occs, floods, storms, metric, n):
    sub = filtered(years, counties, occs, floods, storms)
    sub = sub[sub["matched"]]
    if sub.empty:
        return empty_fig("No storm-matched claims match these filters"), html.Div()
    stats = sub.groupby("storm_label", observed=True).agg(
        claims=("amountPaid_real", "size"), total=("amountPaid_real", "sum"), median=("amountPaid_real", "median"))
    top = stats.sort_values(metric, ascending=False).head(n)
    fig = go.Figure(go.Bar(x=top[metric][::-1], y=top.index.astype(str)[::-1], orientation="h", marker_color=ORANGE,
                           customdata=np.c_[top["claims"][::-1], top["total"][::-1], top["median"][::-1]],
                           hovertemplate="%{y}<br>Claims: %{customdata[0]:,.0f}<br>Total paid: %{customdata[1]:$,.0f}"
                                         "<br>Median claim: %{customdata[2]:$,.0f}<extra></extra>"))
    style_fig(fig, f"Top {len(top)} storms by {MEASURES[metric].lower()}", max(360, 26 * len(top) + 120), unified=False)
    fig.update_xaxes(title=MEASURES[metric], tickprefix="" if metric == "claims" else "$")
    tbl = dash_table.DataTable(
        data=[{"Storm": str(i), "Claims": f"{int(r['claims']):,}", "Total paid (2025 $)": money(r["total"]),
               "Median claim (2025 $)": money(r["median"])}
              for i, r in top.iterrows()],
        columns=[{"name": c, "id": c} for c in ["Storm", "Claims", "Total paid (2025 $)", "Median claim (2025 $)"]],
        style_cell={"fontSize": "15px", "padding": "8px 10px", "textAlign": "left", "border": "none",
                    "borderBottom": f"1px solid {BORDER}", "backgroundColor": SURFACE, "color": INK_2},
        style_header={"fontWeight": "600", "color": INK_2, "backgroundColor": SURFACE, "border": "none",
                      "borderBottom": f"1px solid {BORDER}"},
    )
    return fig, tbl


app.callback(Output("t2-chart", "figure"), Output("t2-table", "children"), *FILTERS,
             Input("t2-metric", "value"), Input("t2-n", "value"))(update_tab2)


def update_tab3(years, counties, occs, floods, storms, factor):
    col, order, note = FACTORS[factor]
    sub = filtered(years, counties, occs, floods, storms)
    if sub.empty:
        return empty_fig(NO_CLAIMS, 460), html.Div(), note
    g = sub.groupby(col, observed=True)["amountPaid_real"]
    st = g.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    st.columns = ["q10", "q25", "q50", "q75", "q90"]
    st["n"] = g.size()
    st = st[st["n"] > 0].reindex([o for o in order if o in st.index])
    labels = [AXIS_SHORT_LABELS.get(str(i), str(i)) for i in st.index]
    fig = go.Figure(go.Box(x=labels, q1=st["q25"], median=st["q50"], q3=st["q75"], lowerfence=st["q10"],
                           upperfence=st["q90"], marker_color=BLUE, line=dict(color=BLUE), fillcolor="rgba(42,120,214,0.18)",
                           hoverinfo="y"))
    style_fig(fig, f"Payout by {factor.lower()} (box = 25th-75th percentile, whiskers = 10th-90th)", 460, unified=False)
    log_dollar_axis(fig, "y", "Payout (2025 $, log scale)")
    fig.update_xaxes(tickangle=0, tickfont=dict(size=12))
    tbl = html.Table([
        html.Thead(html.Tr([html.Th(c) for c in [factor, "Claims", "Median payout", "Middle 50% of claims"]])),
        html.Tbody([html.Tr([html.Td(str(i)), html.Td(f"{int(r['n']):,}"), html.Td(money(r["q50"])),
                             html.Td(f"{money(r['q25'])} to {money(r['q75'])}")]) for i, r in st.iterrows()]),
    ], className="eval")
    return fig, tbl, note


app.callback(Output("t3-chart", "figure"), Output("t3-table", "children"), Output("t3-note", "children"), *FILTERS,
             Input("t3-factor", "value"))(update_tab3)


def update_tab4(years, counties, occs, floods, storms):
    sub = filtered(years, counties, occs, floods, storms)
    if len(sub) < 200:
        return html.Div("Too few claims match these filters to evaluate the model (need at least 200).", className="card")
    y, p = sub["log_actual"].to_numpy(dtype="float64"), sub["log_pred"].to_numpy(dtype="float64")
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - p) ** 2).sum()) / sst if sst > 0 else float("nan")
    actual = sub["amountPaid_real"].to_numpy(dtype="float64")
    ape = float(np.median(np.abs(sub["prediction_amount"].to_numpy(dtype="float64") - actual) / actual) * 100)
    kpis = html.Div([kpi("Claims in this view", f"{len(sub):,}"),
                     kpi("R\u00b2 on this slice", f"{r2:.2f}", "out-of-sample predictions"),
                     kpi("Typical error on this slice", f"{ape:.0f}%", "median absolute % error"),
                     kpi("Actual / predicted (median)", f"{np.exp(np.median(y - p)):.2f}x", "1.00x = unbiased")], className="kpi-row")

    bins = pd.qcut(sub["prediction_amount"], 20, duplicates="drop")
    cal = sub.groupby(bins, observed=True).agg(pred=("prediction_amount", "median"), actual=("amountPaid_real", "median"))
    lo, hi = float(cal.min().min()) * 0.8, float(cal.max().max()) * 1.2
    f1 = go.Figure([go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction", line=dict(color=INK_3, dash="dash")),
                    go.Scatter(x=cal["pred"], y=cal["actual"], mode="lines+markers", name="Median actual claim", line=dict(color=BLUE),
                               hovertemplate="Predicted %{x:$,.0f}<br>Actual (median) %{y:$,.0f}<extra></extra>")])
    style_fig(f1, "Calibration: claims grouped by predicted payout (20 groups)", 420, unified=False)
    log_dollar_axis(f1, "x", "Median predicted payout (2025 $)", lo, hi)
    log_dollar_axis(f1, "y", "Median actual payout (2025 $)", lo, hi)

    by = sub.groupby("yearOfLoss").agg(n=("log_actual", "size"), actual=("log_actual", "mean"), pred=("log_pred", "mean"))
    f2 = go.Figure([
        go.Scatter(x=by.index, y=np.expm1(by["actual"]), name="Actual (typical claim)", line=dict(color=BLUE),
                   customdata=by["n"], hovertemplate="%{x}: actual %{y:$,.0f} (%{customdata:,.0f} claims)<extra></extra>"),
        go.Scatter(x=by.index, y=np.expm1(by["pred"]), name="Predicted (typical claim)", line=dict(color=ORANGE),
                   hovertemplate="%{x}: predicted %{y:$,.0f}<extra></extra>")])
    style_fig(f2, "Typical claim by loss year: actual vs predicted (out of sample)", 380)
    log_dollar_axis(f2, "y", "Typical payout (2025 $)")
    note = html.Div("R\u00b2 within a narrow slice is not comparable to the headline: filtering removes the variation the model "
                    "uses to separate claims, so scores usually fall. Predictions are the year-grouped out-of-sample ones.",
                    className="note")
    return html.Div([kpis, html.Div(dcc.Graph(figure=f1), className="card", style={"marginTop": "12px"}),
                     html.Div(dcc.Graph(figure=f2), className="card", style={"marginTop": "12px"}), note])


app.callback(Output("t4-dyn", "children"), *FILTERS)(update_tab4)


@app.callback(Output("t5-kpis", "children"), Output("t5-chart", "figure"),
              Input("t5-county", "value"), Input("t5-occ", "value"), Input("t5-flood", "value"), Input("t5-age", "value"),
              Input("t5-elev", "value"), Input("t5-wind", "value"), Input("t5-dist", "value"))
def update_tab5(county, occ, flood, age, elev, wind, dist):
    raw = predict_payout(county, occ, flood, age, bool(elev), wind, dist)
    est = raw * UNDERPREDICT  # correct the model's under-prediction on unseen years (see Model tab)
    lo, hi = raw * np.exp(RESID_Q10), raw * np.exp(RESID_Q90)
    kpis = [kpi("Typical payout (median)", money(est), "for a claim like this, 2025 $"),
            kpi("80% of claims like this", f"{money(lo)} to {money(hi)}", "from out-of-sample errors"),
            kpi("Storm", "None nearby" if wind == 0 else STORM_ORDER[wind_category(wind) + 1], f"{wind} kt at {dist} mi" if wind else "no storm within range")]
    grid = np.arange(35, 155, 5)
    ys = [predict_payout(county, occ, flood, age, bool(elev), w, dist) * UNDERPREDICT for w in grid]
    no_storm = predict_payout(county, occ, flood, age, bool(elev), 0, dist) * UNDERPREDICT
    traces = [
        go.Scatter(x=grid, y=ys, mode="lines", name="With a storm nearby", line=dict(color=BLUE),
                   hovertemplate="%{x} kt: %{y:$,.0f}<extra></extra>"),
        go.Scatter(x=[grid[0], grid[-1]], y=[no_storm, no_storm], mode="lines", name="No storm nearby",
                   line=dict(color=INK_3, dash="dash"), hovertemplate="No storm: %{y:$,.0f}<extra></extra>"),
    ]
    if wind > 0:
        traces.append(go.Scatter(x=[wind], y=[est], mode="markers", name="Selected", marker=dict(color=ORANGE, size=12),
                                 hovertemplate="Selected: %{y:$,.0f}<extra></extra>"))
    fig = go.Figure(traces)
    style_fig(fig, f"Typical payout as storm strength changes ({dist} mi from the track)", 380)
    fig.update_xaxes(title="Storm wind speed at closest approach (kt)")
    fig.update_yaxes(title="Typical payout (2025 $)", tickprefix="$")
    return kpis, fig


_log("app ready")

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 8058)))
