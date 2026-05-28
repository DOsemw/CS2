"""
feature_engineering.py
-----------------------
Builds ML features from raw HLTV player-map rows.

Key differences from LoL:
  - No "position" concept — all 5 players are the same role
  - CT/T side split replaces Blue/Red side
  - HS% replaces Assists as third target
  - Opponent strength = opponent's avg kills allowed / ADR allowed
  - Series format: Bo1 or Bo3 (no Bo5 in CS2 majors typically)
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

TARGETS = ["kills", "deaths", "hs_pct"]

# ── Core feature builder ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Takes raw player-map rows and engineers all features.
    Returns enriched DataFrame ready for model training.
    """
    df = df.copy()
    df = df.sort_values(["playername", "date", "match_id", "map_number"]).reset_index(drop=True)

    if verbose:
        log.info(f"[features] Building features for {len(df):,} rows, {df['playername'].nunique()} players")

    df = _clean_and_cast(df)
    df = _add_rolling_player_stats(df)
    df = _add_win_loss_splits(df)
    df = _add_opponent_features(df)
    df = _add_team_features(df)
    df = _add_event_tier(df)
    df = _add_recent_form_score(df)

    if verbose:
        log.info(f"[features] Done — {df.shape[1]} columns")

    return df


def _clean_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    df["kills"] = pd.to_numeric(df["kills"], errors="coerce").clip(0, 60)
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce").clip(0, 60)
    df["hs_pct"] = pd.to_numeric(df["hs_pct"], errors="coerce").clip(0, 100)
    df["adr"] = pd.to_numeric(df["adr"], errors="coerce").clip(0, 200)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(0, 3)
    df["result"] = pd.to_numeric(df["result"], errors="coerce").fillna(0).astype(int)
    df["map_score_team"] = pd.to_numeric(df.get("map_score_team", pd.Series(dtype=float)), errors="coerce")
    df["map_score_opp"] = pd.to_numeric(df.get("map_score_opp", pd.Series(dtype=float)), errors="coerce")
    return df


def _rolling_player(df, stat, windows=(5, 10, 20), group_col="playername"):
    """Add rolling mean features for a stat, per player, no-leakage (shift 1)."""
    grp = df.groupby(group_col)[stat]
    for w in windows:
        col = f"{stat}_roll{w}"
        df[col] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=max(1, w//2)).mean())
    # Career average (all prior games)
    df[f"{stat}_career_avg"] = grp.transform(lambda x: x.shift(1).expanding().mean())
    return df


def _add_rolling_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    for stat in ["kills", "deaths", "hs_pct", "adr", "rating"]:
        df = _rolling_player(df, stat)
    # Rolling kill/death ratio
    for w in (5, 10):
        k = df[f"kills_roll{w}"]
        d = df[f"deaths_roll{w}"].replace(0, np.nan)
        df[f"kd_ratio_roll{w}"] = (k / d).clip(0, 10)
    # Player win rate
    df["player_winrate_roll5"] = (
        df.groupby("playername")["result"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["player_winrate_roll10"] = (
        df.groupby("playername")["result"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    return df


def _add_win_loss_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling averages split by win vs loss maps.
    This is the same win-probability blending trick from the LoL model.
    """
    for stat in ["kills", "deaths", "hs_pct"]:
        for outcome, label in [(1, "win"), (0, "loss")]:
            col_out = f"{stat}_roll10_{label}"
            df[col_out] = (
                df.groupby("playername")
                .apply(lambda g: (
                    g[stat]
                    .where(g["result"] == outcome)
                    .shift(1)
                    .rolling(10, min_periods=1)
                    .mean()
                ))
                .reset_index(level=0, drop=True)
            )
    return df


def _add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player-map row, add the opponent team's recent defensive stats:
      - avg kills allowed per map (last 10 maps)
      - avg ADR allowed per map
    This mirrors the LoL opponent adjustment mechanism.
    """
    # Build team-level kills/ADR allowed series
    # "kills allowed" = opponent's kills when facing this team
    # We compute it by flipping: for each map, team A's kills = kills allowed by team B

    # Create opponent rows view
    opp_df = df[["date", "team", "opponent", "match_id", "map_number", "kills", "adr"]].copy()
    opp_df = opp_df.rename(columns={"team": "defending_team", "opponent": "opp_team",
                                     "kills": "opp_kills", "adr": "opp_adr"})

    # Rolling opp defensive strength: kills allowed to opponents per map
    opp_df = opp_df.sort_values(["defending_team", "date"])
    opp_df["opp_kills_allowed_roll10"] = (
        opp_df.groupby("defending_team")["opp_kills"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    opp_df["opp_adr_allowed_roll10"] = (
        opp_df.groupby("defending_team")["opp_adr"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # Merge onto main df by: player's opponent = defending_team
    opp_lookup = opp_df[["defending_team", "match_id", "map_number",
                          "opp_kills_allowed_roll10", "opp_adr_allowed_roll10"]].drop_duplicates()
    df = df.merge(
        opp_lookup,
        left_on=["opponent", "match_id", "map_number"],
        right_on=["defending_team", "match_id", "map_number"],
        how="left",
    ).drop(columns=["defending_team"], errors="ignore")

    return df


def _add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Team-level rolling win rate."""
    df = df.sort_values(["team", "date"])
    df["team_winrate_roll5"] = (
        df.groupby("team")["result"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["team_winrate_roll10"] = (
        df.groupby("team")["result"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    return df


# Rough event tier mapping — major events have higher stakes / better opponents
_EVENT_TIER = {
    "major": 3,
    "iem": 2, "esl pro league": 2, "blast premier": 2,
    "iem cologne": 3, "iem katowice": 3, "pgl major": 3,
    "default": 1,
}

def _add_event_tier(df: pd.DataFrame) -> pd.DataFrame:
    def tier(event: str) -> int:
        ev = str(event).lower()
        for key, val in _EVENT_TIER.items():
            if key in ev:
                return val
        return _EVENT_TIER["default"]

    df["event_tier"] = df["event"].apply(tier)
    return df


def _add_recent_form_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite recent form score:
    form = 0.4 * norm_kills_roll5 + 0.3 * norm_rating_roll5 + 0.3 * winrate_roll5
    """
    for col in ["kills_roll5", "rating_roll5", "player_winrate_roll5"]:
        if col not in df.columns:
            df[col] = np.nan

    df["form_score"] = (
        0.4 * df["kills_roll5"].fillna(df["kills_roll5"].median()) / df["kills_roll5"].max().clip(1)
        + 0.3 * df["rating_roll5"].fillna(1.0) / 2.0
        + 0.3 * df["player_winrate_roll5"].fillna(0.5)
    ).clip(0, 1)

    return df


# ── Feature column selection ──────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns to pass to the model.
    Excludes targets, IDs, and raw string columns.
    """
    exclude = {
        "match_id", "map_name", "map_number", "date", "event",
        "team", "opponent", "playername",
        # targets
        "kills", "deaths", "hs_pct",
        # raw stats not used directly
        "adr", "rating",
        "map_score_team", "map_score_opp",
        "result",
    }
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]
    # Drop near-constant columns
    feature_cols = [c for c in feature_cols if df[c].nunique() > 1]
    log.info(f"[features] {len(feature_cols)} feature columns selected")
    return feature_cols
