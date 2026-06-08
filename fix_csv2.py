"""
fix_csv2.py
-----------
Fixes deaths and ADR which were wrong in fix_csv.py.
Run: python3 fix_csv2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

FILE = Path.home() / "Desktop/CS2/data/hltv_raw.csv"
df = pd.read_csv(FILE)

# Derive deaths from kills / kd_ratio
df["deaths"] = (df["kills"] / df["kd_ratio"].replace(0, np.nan)).round(2)

# Derive ADR: avg damage per kill in CS2 ~105, divide by 25 rounds per map
df["adr"] = (df["kills"] * 105 / 25).round(1)

# Estimate rating from kd_ratio and impact (rough approximation)
df["rating"] = ((df["kd_ratio"] * 0.7 + df["impact"] * 0.3)).round(2)

print("Fixed data:")
print(df[["playername", "kills", "deaths", "adr", "kd_ratio", "rating"]].head(10).to_string())

df.to_csv(FILE, index=False)
print(f"\nSaved to {FILE}")
