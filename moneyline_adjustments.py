"""
moneyline_adjustments.py
------------------------
Utilities for applying win-probability-based feature blending.

This is the core accuracy mechanism — the same approach used in the LoL model:
  - If a player's team is a 75% favourite, features are blended:
    75% from their historical winning-map stats + 25% from losing-map stats.
  - Opponent defensive strength is overridden with the actual upcoming opponent.
"""

import numpy as np
import pandas as pd


def apply_win_prob_blend(
    X: pd.DataFrame,
    win_prob: float,
    targets: list[str] = ("kills", "deaths", "hs_pct"),
) -> pd.DataFrame:
    """
    Blend win-map and loss-map rolling features based on win probability.
    Directly overrides player_winrate and team_winrate features.
    """
    X = X.copy()

    for stat in targets:
        win_col  = f"{stat}_roll10_win"
        loss_col = f"{stat}_roll10_loss"
        base_col = f"{stat}_roll10"

        if win_col not in X.columns or loss_col not in X.columns:
            continue

        win_val  = float(X[win_col].fillna(X.get(base_col, pd.Series([np.nan])).iloc[0]).iloc[0])
        loss_val = float(X[loss_col].fillna(X.get(base_col, pd.Series([np.nan])).iloc[0]).iloc[0])

        if np.isnan(win_val):
            win_val = float(X[base_col].iloc[0]) if base_col in X.columns else loss_val
        if np.isnan(loss_val):
            loss_val = float(X[base_col].iloc[0]) if base_col in X.columns else win_val

        blended = win_prob * win_val + (1 - win_prob) * loss_val

        if base_col in X.columns:
            X[base_col] = blended
        career_col = f"{stat}_career_avg"
        if career_col in X.columns:
            X[career_col] = blended

    # Override win rate features
    for col in ["player_winrate_roll5", "player_winrate_roll10",
                "team_winrate_roll5", "team_winrate_roll10"]:
        if col in X.columns:
            X[col] = win_prob

    return X


def apply_opponent_adjustment(
    X: pd.DataFrame,
    opponent: str,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Override opponent defensive strength features with the actual upcoming opponent.
    Returns (adjusted X, adjustment_info dict).

    In CS2 there's no position dimension — we just look at all 5 players together.
    """
    opp_mask = df["team"].str.lower() == opponent.strip().lower()
    if not opp_mask.any():
        opp_mask = df["team"].str.lower().str.contains(opponent.strip().lower(), na=False)

    if not opp_mask.any():
        return X, {}

    opp_df = df[opp_mask].sort_values("date", ascending=False)

    if len(opp_df) < 3:
        return X, {}

    # Opponent's recent kills allowed (= kills scored by opponents against this team)
    col = "opp_kills_allowed_roll10"
    if col in opp_df.columns and opp_df[col].notna().any():
        opp_kills_allowed = float(opp_df[col].dropna().iloc[0])
    else:
        opp_kills_allowed = float(opp_df["kills"].tail(10).mean())

    avg_kills = float(df["kills"].mean())
    avg_hs    = float(df["hs_pct"].mean())

    kills_ratio = min(max(opp_kills_allowed / max(avg_kills, 0.1), 0.6), 1.6)

    # Override opponent features in X
    if "opp_kills_allowed_roll10" in X.columns:
        X["opp_kills_allowed_roll10"] = opp_kills_allowed
    if "opp_adr_allowed_roll10" in X.columns:
        opp_adr_allowed = float(opp_df.get("adr", pd.Series(dtype=float)).tail(10).mean())
        X["opp_adr_allowed_roll10"] = opp_adr_allowed

    return X, {
        "kills_ratio": round(kills_ratio, 3),
        "opp_team": opp_df.iloc[0]["team"],
    }
