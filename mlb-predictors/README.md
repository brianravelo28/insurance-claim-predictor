# MLB Prediction Suite

A Streamlit app with two LightGBM models built on Statcast (Baseball Savant) data:

1. **Hit Predictor** — given a batted ball's launch angle, exit velocity, distance, and spray angle, predicts the probability it becomes a hit.
2. **Pitch Predictor** — given the count, base/out state, inning, score, and pitcher, predicts fastball vs. off-speed for the next pitch.

## Why these predictions matter

Hit probability from batted-ball metrics is the basis of modern "expected stats" (xBA, xSLG) used across MLB front offices. Pitch-type tendencies by count/situation are core to how hitters and advance scouts prepare for at-bats.

## Data source

[Baseball Savant / Statcast](https://baseballsavant.mlb.com/), pulled via [`pybaseball`](https://github.com/jldbc/pybaseball), 2023-2024 seasons.

- **Hit Predictor**: a sampled set of full game-days across both seasons (league-wide batted balls in play).
- **Pitch Predictor**: full 2023-2024 pitch logs for 7 elite starters — Gerrit Cole, Max Scherzer, Jacob deGrom, Shane Bieber, plus three AL West additions (⭐ Framber Valdez, Luis Castillo, George Kirby), flagged in the UI.

## Model metrics

See the "How this works" expander in each tab for live accuracy/precision/recall/F1 on the held-out test set.

## Running locally

```bash
pip install -r requirements.txt
python fetch_pitcher_ids.py
python fetch_data.py
python train_hit_predictor.py
python train_pitch_predictor.py
streamlit run app.py
```

## Future work

- Phase 2: batter-specific adjustments to the hit predictor
- Pitch-specific breakdown (not just fastball vs. off-speed) for the pitch predictor
- Expand the pitcher list beyond the current 7 elite starters

## Limitations

- Hit Predictor is trained on a sample of game-days, not the full season — some rare batted-ball profiles may be underrepresented.
- Pitch Predictor only generalizes to the 7 pitchers it was trained on; it is not a general league-wide pitch predictor.
- Historical tendencies (e.g. pitcher fastball %) can shift season to season and don't account for injury/velocity changes.
