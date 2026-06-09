"""
fix_csv.py
----------
Fixes all column mappings from the HLTV scrape in one pass.
Run once after scraping: python3 fix_csv.py

The HLTV stats table columns as scraped:
  maps_played = maps played (correct)
  kd_ratio    = total kills (mislabeled)
  adr         = total damage (mislabeled)
  kast        = actual K/D ratio (mislabeled)
  impact      = actual impact score (correct)
  rating      = NaN (not captured)
  kills       = total damage alt (mislabeled)
  deaths      = deaths per map (correct)
"""

import pandas as pd
import numpy as np
from pathlib import Path

FILE = Path(__file__).parent / "data" / "hltv_raw.csv"

print(f"Loading {FILE}...")
df = pd.read_csv(FILE)
print(f"Loaded {len(df)} rows")
print("Columns:", df.columns.tolist())
print(df.head(3).to_string())

# The scraped columns contain:
# kd_ratio  -> actually total kills
# adr       -> actually total damage
# kast      -> actually K/D ratio
# kills     -> actually total damage (duplicate, drop)
# deaths    -> actually deaths per map (correct)

# Step 1: rename to what they actually are
df = df.rename(columns={
    "kd_ratio": "total_kills",
    "adr":      "total_damage",
    "kast":     "kd_ratio",    # this is the real K/D ratio
    "kills":    "_drop",
})

# Step 2: derive kills per map from total kills / maps played
df["kills"] = (df["total_kills"] / df["maps_played"].replace(0, np.nan)).round(2)

# Step 3: derive ADR from total damage / maps / 25 rounds
df["adr"] = (df["total_damage"] / df["maps_played"].replace(0, np.nan) / 25).round(1)

# Step 4: fix deaths using real K/D ratio
df["deaths"] = (df["kills"] / df["kd_ratio"].replace(0, np.nan)).round(2)

# Step 5: estimate rating from kd_ratio and impact
df["rating"] = ((df["kd_ratio"].fillna(1.0) * 0.7) + (df["impact"].fillna(1.0) * 0.3)).round(2)

# Step 6: drop junk columns
df = df.drop(columns=["total_kills", "total_damage", "_drop"], errors="ignore")

# Step 7: clean up
df["kills"]  = df["kills"].clip(0, 40)
df["deaths"] = df["deaths"].clip(0, 40)
df["adr"]    = df["adr"].clip(0, 200)
df["kd_ratio"] = df["kd_ratio"].clip(0.3, 3.0)

# Final column order
cols = ["playername", "team", "maps_played", "kills", "deaths", "hs_pct", "adr", "kd_ratio", "impact", "rating"]
df = df[[c for c in cols if c in df.columns]]

print(f"\nFixed data ({len(df)} rows):")
print(df.head(5).to_string())

df.to_csv(FILE, index=False)
print(f"\nSaved to {FILE}")
