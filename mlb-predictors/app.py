import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="MLB Prediction Suite", layout="wide")

st.title("MLB Prediction Suite")
st.caption("Hit Probability & Pitch Type Prediction — built on 2023-2024 Statcast data")

tab1, tab2 = st.tabs(["Hit Predictor", "Pitch Predictor"])


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def prob_bar(label_a, prob_a, label_b, prob_b):
    fig = go.Figure(go.Bar(
        x=[prob_a, prob_b],
        y=[label_a, label_b],
        orientation="h",
        marker_color=["#2ecc71" if prob_a >= prob_b else "#e74c3c",
                      "#2ecc71" if prob_b >= prob_a else "#e74c3c"],
        text=[f"{prob_a:.1%}", f"{prob_b:.1%}"],
        textposition="inside",
    ))
    fig.update_layout(xaxis_range=[0, 1], height=200, margin=dict(l=0, r=0, t=10, b=10))
    return fig


with tab1:
    st.header("Will this ball be a hit?")

    bundle = load_model(str(BASE_DIR / "models" / "hit_predictor.pkl"))
    model, columns, metrics = bundle["model"], bundle["columns"], bundle["metrics"]

    col1, col2, col3 = st.columns(3)
    with col1:
        launch_angle = st.slider("Launch angle (degrees)", -50, 60, 15)
    with col2:
        exit_velocity = st.slider("Exit velocity (mph)", 40, 120, 95)
    with col3:
        spray_angle = st.slider("Spray angle (degrees, 0 = center)", -45, 45, 0)

    hit_distance_sc = st.slider("Hit distance (feet)", 0, 500, 250)
    game_type = st.radio("Game type", ["RS", "PS"], horizontal=True)

    if st.button("Predict", key="predict_hit"):
        row = pd.DataFrame([{
            "launch_angle": launch_angle,
            "exit_velocity": exit_velocity,
            "hit_distance_sc": hit_distance_sc,
            "spray_angle": spray_angle,
            "game_type": game_type,
        }])
        row = pd.get_dummies(row, columns=["game_type"], drop_first=True)
        row = row.reindex(columns=columns, fill_value=0)

        hit_prob = model.predict_proba(row)[0][1]

        st.metric("Hit probability", f"{hit_prob:.1%}")
        st.plotly_chart(prob_bar("Out", 1 - hit_prob, "Hit", hit_prob), use_container_width=True)

        if hit_prob > 0.6:
            st.success(f"High exit velocity and launch angle push this toward a hit ({hit_prob:.1%}).")
        elif hit_prob < 0.3:
            st.warning(f"This batted ball profile looks like an out ({hit_prob:.1%} hit chance).")
        else:
            st.info(f"This is a toss-up batted ball ({hit_prob:.1%} hit chance).")

        try:
            examples = pd.read_csv(BASE_DIR / "data" / "hit_predictor_examples.csv")
            examples["dist"] = (
                (examples["launch_angle"] - launch_angle) ** 2
                + (examples["exit_velocity"] - exit_velocity) ** 2
                + (examples["spray_angle"] - spray_angle) ** 2
            )
            similar = examples.nsmallest(3, "dist")
            st.subheader("Similar recent batted balls")
            st.dataframe(
                similar[["launch_angle", "exit_velocity", "hit_distance_sc", "spray_angle", "is_hit"]],
                hide_index=True,
            )
        except FileNotFoundError:
            pass

    with st.expander("How this works"):
        st.write(
            "A LightGBM classifier trained on Statcast batted-ball data predicts hit "
            "probability from launch angle, exit velocity, hit distance, spray angle, "
            "and game type (regular season vs. postseason)."
        )
        st.write(f"Model metrics (test set): {metrics}")
        importances = pd.Series(model.feature_importances_, index=columns).sort_values()
        st.plotly_chart(go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h")),
                         use_container_width=True)


with tab2:
    st.header("What pitch is coming?")

    bundle2 = load_model(str(BASE_DIR / "models" / "pitch_predictor.pkl"))
    model2 = bundle2["model"]
    columns2 = bundle2["columns"]
    metrics2 = bundle2["metrics"]
    fb_pct = bundle2["fastball_pct_by_pitcher"]
    al_west = bundle2["al_west_pitchers"]

    pitcher_options = sorted(fb_pct.keys(), key=lambda n: (n not in al_west, n))
    labels = {name: f"{name} ⭐ AL West" if name in al_west else name for name in pitcher_options}

    pitcher = st.selectbox("Pitcher", pitcher_options, format_func=lambda n: labels[n])
    if pitcher in al_west:
        st.caption("⭐ AL West pitcher — added for this build alongside the original elite-starter list.")

    c1, c2, c3 = st.columns(3)
    with c1:
        balls = st.slider("Balls", 0, 3, 1)
    with c2:
        strikes = st.slider("Strikes", 0, 2, 1)
    with c3:
        inning = st.slider("Inning", 1, 9, 5)

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        outs = st.slider("Outs", 0, 2, 1)
    with c5:
        on_1b = st.checkbox("Runner on 1B")
    with c6:
        on_2b = st.checkbox("Runner on 2B")
    with c7:
        on_3b = st.checkbox("Runner on 3B")

    run_differential = st.slider("Run differential (pitcher's team perspective)", -5, 5, 0)

    if st.button("Predict", key="predict_pitch"):
        row = pd.DataFrame([{
            "balls": balls,
            "strikes": strikes,
            "outs_when_up": outs,
            "on_1b": int(on_1b),
            "on_2b": int(on_2b),
            "on_3b": int(on_3b),
            "inning": inning,
            "run_differential": run_differential,
            "pitcher_fastball_pct": fb_pct[pitcher],
        }])[columns2]

        fb_prob = model2.predict_proba(row)[0][0]  # class 0 = fastball

        st.metric("Fastball probability", f"{fb_prob:.1%}")
        st.plotly_chart(prob_bar("Off-speed", 1 - fb_prob, "Fastball", fb_prob), use_container_width=True)

    with st.expander("How this works"):
        st.write(
            "A LightGBM classifier trained on 2023-2024 pitches from the selected "
            "starters predicts fastball vs. off-speed based on the count, base/out "
            "state, inning, score, and the pitcher's overall fastball rate."
        )
        st.write(f"Model metrics (test set): {metrics2}")
        importances2 = pd.Series(model2.feature_importances_, index=columns2).sort_values()
        st.plotly_chart(go.Figure(go.Bar(x=importances2.values, y=importances2.index, orientation="h")),
                         use_container_width=True)
