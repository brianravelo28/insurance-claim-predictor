from pybaseball import playerid_lookup
import pandas as pd

pitchers = [
    ("cole", "gerrit", None),
    ("scherzer", "max", None),
    ("degrom", "jacob", None),
    ("bieber", "shane", None),
    ("valdez", "framber", None),
    # "Luis Castillo" has 3 MLBAM matches (a 1996-2010 player, and two others);
    # pin the Mariners/Reds ace (career 2017-2026) explicitly to avoid picking wrong one.
    ("castillo", "luis", 622491),
    ("kirby", "george", None),
]

rows = []
for last, first, forced_id in pitchers:
    if forced_id is not None:
        rows.append({"name": f"{first.title()} {last.title()}", "mlbam_id": forced_id})
        print(f"{first.title()} {last.title()}: mlbam_id={forced_id} (pinned)")
        continue
    df = playerid_lookup(last, first)
    if df.empty:
        print(f"NOT FOUND: {first} {last}")
        continue
    if len(df) > 1:
        raise ValueError(f"Ambiguous lookup for {first} {last}: {len(df)} matches -- pin the id explicitly")
    row = df.iloc[0]
    rows.append({
        "name": f"{first.title()} {last.title()}",
        "mlbam_id": row["key_mlbam"],
    })
    print(f"{first.title()} {last.title()}: mlbam_id={row['key_mlbam']}")

pd.DataFrame(rows).to_csv("data/pitcher_ids.csv", index=False)
