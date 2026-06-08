"""
feature_engineering.py
-----------------------
Builds features from aggregate HLTV player stats (2026).

Since we have one row per player (not per-map rows), features are
derived directly from the aggregate stats rather than rolling averages.
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)
TARGETS = ["kills", "deaths", "hs_pct"]


def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()

    if verbose:
        log.info(f"[features] Building features for {len(df)} players")

    df = _clean(df)
    df = _derive_features(df)
    df = _add_opponent_features(df)

    if verbose:
        log.info(f"[features] Done — {df.shape[1]} columns")

    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["kills", "deaths", "hs_pct", "adr", "rating", "kd_ratio", "kast", "impact", "maps_played"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["kills"]  = df["kills"].clip(0, 40)
    df["deaths"] = df["deaths"].clip(0, 40)
    df["hs_pct"] = df["hs_pct"].clip(0, 100)
    df["adr"]    = df.get("adr", pd.Series(dtype=float)).clip(0, 200)
    df["rating"] = df.get("rating", pd.Series(dtype=float)).clip(0, 3)
    return df


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    # KD ratio if not present
    if "kd_ratio" not in df.columns or df["kd_ratio"].isna().all():
        df["kd_ratio"] = (df["kills"] / df["deaths"].replace(0, np.nan)).clip(0, 5)

    # Normalize maps played
    if "maps_played" in df.columns:
        df["maps_played_norm"] = (df["maps_played"] - df["maps_played"].min()) / (df["maps_played"].max() - df["maps_played"].min() + 1)
    else:
        df["maps_played_norm"] = 0.5

    # Win probability placeholder (will be overridden at predict time)
    df["win_prob"] = 0.5

    # Form score composite
    rating_norm = df["rating"].fillna(1.0) / 2.0
    kd_norm     = df["kd_ratio"].fillna(1.0) / 3.0
    hs_norm     = df["hs_pct"].fillna(50) / 100.0
    df["form_score"] = (0.4 * rating_norm + 0.3 * kd_norm + 0.3 * hs_norm).clip(0, 1)

    # Dummy rolling features expected by model (set to actual values)
    for stat in ["kills", "deaths", "hs_pct"]:
        df[f"{stat}_roll5"]      = df[stat]
        df[f"{stat}_roll10"]     = df[stat]
        df[f"{stat}_roll20"]     = df[stat]
        df[f"{stat}_career_avg"] = df[stat]
        df[f"{stat}_roll10_win"] = df[stat] * 1.1  # approximate win map boost
        df[f"{stat}_roll10_loss"]= df[stat] * 0.9

    df["kills_roll5"]  = df["kills"]
    df["rating_roll5"] = df["rating"].fillna(1.0)
    df["player_winrate_roll5"]  = 0.5
    df["player_winrate_roll10"] = 0.5
    df["team_winrate_roll5"]    = 0.5
    df["team_winrate_roll10"]   = 0.5

    for w in [5, 10]:
        k = df[f"kills_roll{w}"]
        d = df[f"deaths_roll{w}"].replace(0, np.nan)
        df[f"kd_ratio_roll{w}"] = (k / d).clip(0, 10)

    return df


def _add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    # Opponent defensive strength — use league average as default
    avg_kills = df["kills"].mean()
    df["opp_kills_allowed_roll10"] = avg_kills
    df["opp_adr_allowed_roll10"]   = df["adr"].mean() if "adr" in df.columns else 80.0
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "match_id", "date", "event", "team", "opponent", "playername",
        "kills", "deaths", "hs_pct",
        "total_kills", "total_deaths", "kills_per_round", "deaths_per_round",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        and df[c].nunique() > 1
    ]
    log.info(f"[features] {len(feature_cols)} feature columns")
    return feature_cols
