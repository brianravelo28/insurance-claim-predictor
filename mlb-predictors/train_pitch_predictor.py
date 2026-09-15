"""
Model 2: Pitch Predictor.

Given game-state features and a pitcher's historical fastball rate, predict
whether the next pitch is a fastball (FF) or off-speed (CH/CU/SL).
"""
import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "inning", "run_differential", "pitcher_fastball_pct",
]
TARGET = "is_fastball"
OFFSPEED_TYPES = ["CH", "CU", "SL"]

# AL West additions get flagged in the UI so users can see which pitchers were
# added specifically for this build vs. the original elite-starter list.
AL_WEST_PITCHERS = {"Framber Valdez", "Luis Castillo", "George Kirby"}


def load_and_clean():
    df = pd.read_csv("data/model2_raw.csv", low_memory=False)

    df = df[df["pitch_type"].isin(["FF"] + OFFSPEED_TYPES)].copy()
    df = df.dropna(subset=["balls", "strikes", "outs_when_up", "inning", "pitch_type"])

    df["on_1b"] = df["on_1b"].notna().astype(int)
    df["on_2b"] = df["on_2b"].notna().astype(int)
    df["on_3b"] = df["on_3b"].notna().astype(int)

    # run_differential from the pitching team's perspective: positive = ahead
    is_pitcher_home = df["inning_topbot"] == "Bot"
    df["run_differential"] = (
        (df["home_score"] - df["away_score"]).where(is_pitcher_home,
         df["away_score"] - df["home_score"])
    )

    fb_pct = df.groupby("pitcher_name")["pitch_type"].apply(
        lambda s: (s == "FF").mean()
    )
    df["pitcher_fastball_pct"] = df["pitcher_name"].map(fb_pct)

    df["is_fastball"] = (df["pitch_type"] == "FF").astype(int)

    df = df.dropna(subset=FEATURES + [TARGET])
    return df[FEATURES + [TARGET, "pitcher_name"]], dict(fb_pct)


def train():
    data, fastball_pct_by_pitcher = load_and_clean()
    print(f"Training on {len(data)} pitches across {data['pitcher_name'].nunique()} pitchers")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print("Pitch Predictor metrics:", metrics)

    with open("models/pitch_predictor.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "columns": FEATURES,
            "metrics": metrics,
            "fastball_pct_by_pitcher": fastball_pct_by_pitcher,
            "al_west_pitchers": AL_WEST_PITCHERS,
        }, f)

    return model, metrics


if __name__ == "__main__":
    train()
