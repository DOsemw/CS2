"""
fix_csv.py
----------
Fixes the column mapping from the HLTV scrape.
Run once locally: python3 fix_csv.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

IN_FILE  = Path.home() / "Desktop/CS2/data/hltv_raw.csv"
OUT_FILE = IN_FILE  # overwrite in place

df = pd.read_csv(IN_FILE)
print(f"Loaded {len(df)} rows")
print(df.head(3).to_string())

# Rename mismatched columns to what they actually are
df = df.rename(columns={
    "kd_ratio": "total_kills",
    "adr":      "total_damage",
    "kast":     "kd_ratio",
    "impact":   "impact",
    "kills":    "total_damage2",  # duplicate, drop later
    "deaths":   "deaths",
})

# Derive correct per-map stats
df["kills"]  = (df["total_kills"] / df["maps_played"]).round(2)
df["adr"]    = (df["total_damage"] / df["maps_played"] / 25).round(2)  # per round (25 rounds/map)
df["rating"] = np.nan  # not captured, will be estimated

# hs_pct — already empty, leave as NaN for now
df["hs_pct"] = pd.to_numeric(df["hs_pct"], errors="coerce")

# Drop junk columns
df = df.drop(columns=["total_kills", "total_damage", "total_damage2"], errors="ignore")

# Final column order
cols = ["playername", "team", "maps_played", "kills", "deaths", "hs_pct", "adr", "kd_ratio", "impact", "rating"]
df = df[[c for c in cols if c in df.columns]]

print(f"\nFixed data:")
print(df.head(5).to_string())

df.to_csv(OUT_FILE, index=False)
print(f"\nSaved to {OUT_FILE}")
