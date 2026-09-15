"""
Fetch Statcast data for both models.

Model 1 (Hit Predictor) needs league-wide batted-ball data. Pulling the full
2023-2024 season via statcast() is millions of rows and too slow for a demo,
so we sample two days per month across both seasons instead -- still tens of
thousands of batted balls, which is plenty to train on.

Model 2 (Pitch Predictor) only needs pitches thrown by our 7 target starters,
so we pull their full 2023-2024 game logs directly via statcast_pitcher --
that's bounded (~5-6k pitches/pitcher/season) and doesn't need sampling.
"""
import pandas as pd
from pybaseball import statcast, statcast_pitcher, cache

cache.enable()

SAMPLE_DATES = [
    f"{year}-{month:02d}-{day:02d}"
    for year in (2023, 2024)
    for month in (4, 5, 6, 7, 8, 9)
    for day in (10, 24)
]

PITCHER_IDS = pd.read_csv("data/pitcher_ids.csv")


def fetch_model1_raw():
    frames = []
    for date in SAMPLE_DATES:
        print(f"Fetching {date} ...")
        try:
            day_df = statcast(start_dt=date, end_dt=date, verbose=False)
        except Exception as e:
            print(f"  skipped {date}: {e}")
            continue
        if day_df is not None and not day_df.empty:
            frames.append(day_df)
    full = pd.concat(frames, ignore_index=True)
    full.to_csv("data/model1_raw.csv", index=False)
    print(f"Model 1 raw: {len(full)} rows -> data/model1_raw.csv")


def fetch_model2_raw():
    frames = []
    for _, row in PITCHER_IDS.iterrows():
        print(f"Fetching pitches for {row['name']} ...")
        pdf = statcast_pitcher("2023-01-01", "2024-12-31", int(row["mlbam_id"]))
        if pdf is not None and not pdf.empty:
            pdf["pitcher_name"] = row["name"]
            frames.append(pdf)
    full = pd.concat(frames, ignore_index=True)
    full.to_csv("data/model2_raw.csv", index=False)
    print(f"Model 2 raw: {len(full)} rows -> data/model2_raw.csv")


if __name__ == "__main__":
    fetch_model1_raw()
    fetch_model2_raw()
