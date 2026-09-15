"""
Model 1: Hit Predictor.

Given batted-ball metrics (launch angle, exit velocity, distance, spray angle,
game type), predict whether the ball became a hit (single/double/triple/HR)
or an out.
"""
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURES = ["launch_angle", "exit_velocity", "hit_distance_sc", "spray_angle", "game_type"]
TARGET = "is_hit"


def load_and_clean():
    df = pd.read_csv("data/model1_raw.csv", low_memory=False)

    bip = df[df["type"] == "X"].copy()
    bip = bip.dropna(subset=["launch_angle", "launch_speed", "hit_distance_sc", "hc_x", "hc_y"])

    bip["exit_velocity"] = bip["launch_speed"]
    # Standard Statcast spray-angle approximation: horizontal angle of the hit
    # location relative to home plate, in degrees (0 = straight up the middle).
    bip["spray_angle"] = np.degrees(np.arctan2(bip["hc_x"] - 125.42, 198.27 - bip["hc_y"])) * 0.75
    bip["game_type"] = bip["game_type"].map(lambda x: "PS" if x != "R" else "RS")
    bip["is_hit"] = bip["events"].isin(["single", "double", "triple", "home_run"]).astype(int)

    bip = bip.dropna(subset=FEATURES + [TARGET])
    return bip[FEATURES + [TARGET]]


def train():
    data = load_and_clean()
    print(f"Training on {len(data)} batted balls")

    X = pd.get_dummies(data[FEATURES], columns=["game_type"], drop_first=True)
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
    print("Hit Predictor metrics:", metrics)

    with open("models/hit_predictor.pkl", "wb") as f:
        pickle.dump({"model": model, "columns": list(X.columns), "metrics": metrics}, f)

    sample = data.sample(min(500, len(data)), random_state=42)
    sample.to_csv("data/hit_predictor_examples.csv", index=False)

    return model, metrics


if __name__ == "__main__":
    train()
