"""
main.py — CS2 Props Prediction API
KPR-based prediction with kill share architecture.

Series support:
  Bo1        — 1 map
  M1-2 (Bo3) — 2 maps guaranteed (both maps happen regardless of series outcome)
  M1-3 (Bo5) — 3 maps guaranteed (min score in Bo5 is 3-0)
"""

import os, sys, logging
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

STATE = {
    "df": None,
    "ready": False,
    "n_players": 0,
}

ROUNDS_COMPETITIVE  = 20.5
ROUNDS_STOMP        = 18.5
STOMP_THRESHOLD     = 500
KPR_CALIBRATION     = 0.65
SHRINKAGE_THRESHOLD = 20
GLOBAL_AVG_KPR      = 0.60

# Series format -> number of guaranteed maps
SERIES_MAP_COUNT = {
    "BO1":   1,
    "M1":    1,
    "BO3":   2,
    "M1-2":  2,
    "BO5":   3,
    "M1-3":  3,
}

FALLBACK_KPR = {
    "awp":     0.62,
    "star":    0.62,
    "fragger": 0.62,
    "entry":   0.58,
    "rifler":  0.58,
    "igl":     0.50,
    "support": 0.50,
    "default": 0.55,
}


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _series_map_count(series: str) -> int:
    """Resolve series format string to guaranteed map count. Defaults to 2 (Bo3)."""
    key = str(series).upper().replace(" ", "")
    return SERIES_MAP_COUNT.get(key, 2)


def load_everything():
    from data_ingestion import load_raw, filter_top_teams

    log.info("Loading CS2 data...")
    raw = load_raw()
    raw = filter_top_teams(raw, min_maps=5)

    raw["kpr"] = (raw["kills"] * KPR_CALIBRATION / ROUNDS_COMPETITIVE).round(4)
    raw["kpr"] = raw["kpr"].clip(0.25, 0.90)

    # Recent form KPR — from last 20 maps scraped
    if "recent_kills_avg" in raw.columns:
        raw["recent_kpr"] = (raw["recent_kills_avg"] * KPR_CALIBRATION / ROUNDS_COMPETITIVE).round(4)
        raw["recent_kpr"] = raw["recent_kpr"].clip(0.25, 0.90)
        raw["recent_kpr"] = raw["recent_kpr"].fillna(raw["kpr"])
    else:
        raw["recent_kpr"] = raw["kpr"]

    if "form_trend" not in raw.columns:
        raw["form_trend"] = 0.0
    raw["form_trend"] = raw["form_trend"].fillna(0.0)
    if "recent_maps_count" not in raw.columns:
        raw["recent_maps_count"] = 0
    raw["recent_maps_count"] = raw["recent_maps_count"].fillna(0)

    if "kd_ratio" not in raw.columns or raw["kd_ratio"].isna().all():
        raw["kd_ratio"] = (raw["kills"] / raw["deaths"].replace(0, np.nan)).clip(0.3, 3.0)
    raw["kd_ratio"] = raw["kd_ratio"].clip(0.3, 3.0)

    raw["team"] = raw["team"].fillna("").astype(str)
    team_avg = (
        raw.groupby("team")["kpr"]
        .mean()
        .reset_index()
        .rename(columns={"kpr": "team_avg_kpr"})
    )
    raw = raw.merge(team_avg, on="team", how="left")
    raw["team_avg_kpr"] = raw["team_avg_kpr"].fillna(GLOBAL_AVG_KPR)
    raw["kill_share"] = (raw["kpr"] / raw["team_avg_kpr"]).clip(0.5, 2.0)

    STATE.update({
        "df": raw,
        "ready": True,
        "n_players": raw["playername"].nunique(),
    })
    log.info(f"Ready — {STATE['n_players']} players loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_everything()
    except FileNotFoundError as e:
        log.warning(f"[startup] No data — starting empty.\n{e}")
    yield


app = FastAPI(title="CS2 Props API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _expected_rounds(team_ml=None, opp_ml=None):
    if team_ml is None and opp_ml is None:
        return ROUNDS_COMPETITIVE
    ml = team_ml if team_ml is not None else opp_ml
    return ROUNDS_STOMP if abs(ml) > STOMP_THRESHOLD else ROUNDS_COMPETITIVE


def _win_prob(team_ml, opp_ml):
    if team_ml is None or opp_ml is None:
        return 0.5
    def implied(ml):
        return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)
    p1, p2 = implied(team_ml), implied(opp_ml)
    return round(p1 / (p1 + p2), 3)


def _lookup_player(player: str):
    df = STATE["df"]
    mask = df["playername"].str.lower() == player.strip().lower()
    if not mask.any():
        mask = df["playername"].str.lower().str.contains(player.strip().lower(), na=False)
    if not mask.any():
        return None
    return df[mask].iloc[0].to_dict()


def _lookup_team_avg_kpr(team: str) -> float:
    try:
        df = STATE["df"]
        team_clean = str(team).strip().lower()
        if not team_clean:
            return GLOBAL_AVG_KPR
        mask = df["team"].fillna("").str.lower().str.contains(team_clean, na=False)
        if not mask.any():
            return GLOBAL_AVG_KPR
        result = df[mask]["kpr"].mean()
        if pd.isna(result):
            return GLOBAL_AVG_KPR
        return float(result)
    except Exception as e:
        log.warning(f"[team_lookup] Error looking up '{team}': {e}")
        return GLOBAL_AVG_KPR


def _shrinkage_blend(kpr: float, maps_played: float) -> float:
    if maps_played >= SHRINKAGE_THRESHOLD:
        return kpr
    w = maps_played / SHRINKAGE_THRESHOLD
    return w * kpr + (1 - w) * GLOBAL_AVG_KPR


def _predict(player, team_ml=None, opp_ml=None, opponent=None, role=None, series="Bo3"):
    expected_rounds = _expected_rounds(team_ml, opp_ml)
    win_prob        = _win_prob(team_ml, opp_ml)
    map_count       = _series_map_count(series)

    player_data = _lookup_player(player)
    fallback    = False
    notes       = ""

    if player_data is not None:
        kpr_full      = float(player_data.get("kpr", GLOBAL_AVG_KPR))
        kpr_recent    = float(player_data.get("recent_kpr", kpr_full))
        kd_ratio      = float(player_data.get("kd_ratio", 1.0))
        kill_share    = float(player_data.get("kill_share", 1.0))
        team_avg_kpr  = float(player_data.get("team_avg_kpr", GLOBAL_AVG_KPR))
        maps_played   = float(player_data.get("maps_played", 0))
        recent_count  = float(player_data.get("recent_maps_count", 0) or 0)
        form_trend    = float(player_data.get("form_trend", 0) or 0)
        team          = str(player_data.get("team", ""))
        name          = str(player_data.get("playername", player))

        # Blend recent form with full-year average
        # 10+ recent maps: 60% recent, 40% full year
        # 0 recent maps: 100% full year
        recent_weight = min(recent_count / 10.0, 1.0) * 0.6
        full_weight   = 1.0 - recent_weight
        kpr = recent_weight * kpr_recent + full_weight * kpr_full

        # Shrinkage toward global avg for low-sample players
        kpr = _shrinkage_blend(kpr, maps_played)

        # Form trend adjustment: improving/declining form shifts KPR slightly
        # +1 kill/map trend -> +3% KPR boost, capped at +/-5%
        form_adj = float(np.clip(form_trend * 0.03, -0.05, 0.05))
        kpr = kpr * (1 + form_adj)
        kpr = float(np.clip(kpr, 0.25, 0.90))
    else:
        role_key    = (role or "default").lower()
        kpr         = next((v for k, v in FALLBACK_KPR.items() if k in role_key), FALLBACK_KPR["default"])
        kd_ratio    = 1.0
        kill_share  = 1.0
        team_avg_kpr= GLOBAL_AVG_KPR
        maps_played = 0
        team        = ""
        name        = player
        fallback    = True
        notes       = "Fallback KPR used (player not in database)"

    # Win probability adjustment
    kpr_adj = kpr * (0.9 + 0.2 * win_prob)

    # Opponent adjustment
    if opponent:
        try:
            opp_kpr    = _lookup_team_avg_kpr(opponent)
            opp_factor = GLOBAL_AVG_KPR / max(opp_kpr, 0.3)
            opp_factor = min(max(opp_factor, 0.85), 1.15)
            kpr_adj    = kpr_adj * opp_factor
        except Exception as e:
            log.warning(f"[opponent] Adjustment failed for '{opponent}': {e}")

    # Per-map projection
    kills_map1  = round(kpr_adj * expected_rounds, 1)
    deaths_map1 = round(kills_map1 / max(kd_ratio, 0.3), 1)

    # Series projection — scale by guaranteed map count (2 for Bo3, 3 for Bo5)
    kills_series  = round(kills_map1  * map_count, 1)
    deaths_series = round(deaths_map1 * map_count, 1)

    series_label = "M1-3 (Bo5)" if map_count == 3 else ("M1-2 (Bo3)" if map_count == 2 else "Bo1")

    return _clean({
        "player":          name,
        "team":            team,
        "win_prob":        win_prob,
        "kpr":             round(kpr, 4),
        "kpr_adjusted":    round(kpr_adj, 4),
        "kill_share":      round(kill_share, 4),
        "recent_kpr":      round(kpr_recent, 4),
        "form_trend":       round(form_trend, 2),
        "team_avg_kpr":    round(team_avg_kpr, 4),
        "expected_rounds": expected_rounds,
        "maps_in_sample":  int(maps_played),
        "series_map_count": map_count,
        "fallback":        fallback,
        "notes":           notes,
        "map1": {
            "kills":  {"per_map": kills_map1},
            "deaths": {"per_map": deaths_map1},
        },
        "bo3": {
            "series_format":   series_label,
            "maps_counted":    map_count,
            "kills":  {"series_total": kills_series},
            "deaths": {"series_total": deaths_series},
        },
    })


@app.get("/")
def health():
    return _clean({
        "status":  "ok" if STATE["ready"] else "loading",
        "players": STATE["n_players"],
        "game":    "CS2",
    })


@app.get("/search")
def search_players(q: str = Query(...)):
    if not STATE["ready"]: raise HTTPException(503, detail="Loading")
    df = STATE["df"]
    matches = df[df["playername"].str.lower().str.contains(q.strip().lower(), na=False)]
    players = (
        matches[["playername", "team", "maps_played", "kpr", "kill_share", "kd_ratio"]]
        .drop_duplicates("playername")
        .sort_values("maps_played", ascending=False)
        .head(10)
    )
    return _clean(players.fillna("").to_dict(orient="records"))


@app.get("/players")
def list_players(q: str = Query(None), team: str = Query(None)):
    if not STATE["ready"]: raise HTTPException(503, detail="Loading")
    df = STATE["df"]
    if team: df = df[df["team"].fillna("").str.lower().str.contains(team.lower(), na=False)]
    if q:    df = df[df["playername"].str.lower().str.contains(q.lower(), na=False)]
    players = (
        df[["playername","team","maps_played","kills","deaths","kpr","kill_share","kd_ratio"]]
        .drop_duplicates("playername")
        .sort_values("maps_played", ascending=False)
    )
    return _clean(players.fillna("").to_dict(orient="records"))


@app.get("/predict")
def predict_player(
    player:    str = Query(...),
    moneyline: int = Query(None),
    opp_ml:    int = Query(None),
    opponent:  str = Query(None),
    series:    str = Query("Bo3", description="Bo1, M1-2/Bo3, or M1-3/Bo5"),
    role:      str = Query(None),
):
    if not STATE["ready"]: raise HTTPException(503, detail="Loading")
    return _predict(player, moneyline, opp_ml, opponent, role, series)


@app.get("/refresh")
def refresh():
    STATE["ready"] = False
    try:
        load_everything()
        return {"status": "refreshed", "players": STATE["n_players"]}
    except Exception as e:
        STATE["ready"] = True
        raise HTTPException(500, str(e))

