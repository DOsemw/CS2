"""
main.py — CS2 Props Prediction API
KPR-based prediction engine with individual player lookup.

Logic:
  1. Look up player's KPR from dataset
  2. Adjust expected rounds based on moneyline (stomp vs competitive)
  3. Map 1 kills = KPR × expected_rounds
  4. M1-2 (Bo3) kills = Map 1 kills × 2.0 (both maps guaranteed)
  5. Deaths = kills / player_kd_ratio
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

# ── CS2 round constants (MR12 era) ────────────────────────────
ROUNDS_COMPETITIVE = 20.5   # evenly matched
ROUNDS_STOMP       = 18.5   # heavy favourite (abs(ml) > 500)
STOMP_THRESHOLD    = 500

# ── Calibration factor to correct scraped kill inflation ──────
KPR_CALIBRATION = 0.78

# ── Role fallback KPRs ────────────────────────────────────────
FALLBACK_KPR = {
    "awp":     0.74,
    "star":    0.74,
    "fragger": 0.74,
    "entry":   0.68,
    "rifler":  0.68,
    "igl":     0.58,
    "support": 0.58,
    "default": 0.65,
}


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def load_everything():
    from data_ingestion import load_raw, filter_top_teams

    log.info("Loading CS2 data...")
    raw = load_raw()
    raw = filter_top_teams(raw, min_maps=10)

    # KPR with calibration factor to correct scraper inflation
    raw["kpr"] = (raw["kills"] * KPR_CALIBRATION / ROUNDS_COMPETITIVE).round(4)
    raw["kpr"] = raw["kpr"].clip(0.3, 0.95)

    # KD ratio
    if "kd_ratio" not in raw.columns or raw["kd_ratio"].isna().all():
        raw["kd_ratio"] = (raw["kills"] / raw["deaths"].replace(0, np.nan)).clip(0.3, 3.0)
    raw["kd_ratio"] = raw["kd_ratio"].clip(0.3, 3.0)

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


# ── Core prediction logic ─────────────────────────────────────

def _expected_rounds(team_ml: int = None, opp_ml: int = None) -> float:
    if team_ml is None and opp_ml is None:
        return ROUNDS_COMPETITIVE
    ml = team_ml if team_ml is not None else opp_ml
    if abs(ml) > STOMP_THRESHOLD:
        return ROUNDS_STOMP
    return ROUNDS_COMPETITIVE


def _win_prob(team_ml: int, opp_ml: int) -> float:
    if team_ml is None or opp_ml is None:
        return 0.5
    def implied(ml):
        if ml > 0: return 100 / (ml + 100)
        return abs(ml) / (abs(ml) + 100)
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


def _predict(player: str, team_ml: int = None, opp_ml: int = None,
             role: str = None, series: str = "Bo3") -> dict:
    expected_rounds = _expected_rounds(team_ml, opp_ml)
    win_prob = _win_prob(team_ml, opp_ml)

    player_data = _lookup_player(player)
    fallback = False
    notes = ""

    if player_data is not None:
        kpr      = float(player_data.get("kpr", 0.65))
        kd_ratio = float(player_data.get("kd_ratio", 1.0))
        team     = str(player_data.get("team", ""))
        maps     = float(player_data.get("maps_played", 0))
        name     = str(player_data.get("playername", player))
    else:
        role_key = (role or "default").lower()
        kpr = next((v for k, v in FALLBACK_KPR.items() if k in role_key), FALLBACK_KPR["default"])
        kd_ratio = 1.0
        team = ""
        maps = 0
        name = player
        fallback = True
        notes = "Fallback KPR used (player not in database)"
        log.warning(f"[predict] '{player}' not found — fallback KPR {kpr}")

    # Win probability adjustment — slight boost for favourites
    kpr_adjusted = kpr * (0.9 + 0.2 * win_prob)

    # Map 1
    kills_map1  = round(kpr_adjusted * expected_rounds, 1)
    deaths_map1 = round(kills_map1 / max(kd_ratio, 0.3), 1)

    # M1-2 (Bo3) — always exactly 2 maps
    kills_m12  = round(kills_map1 * 2.0, 1)
    deaths_m12 = round(deaths_map1 * 2.0, 1)

    return _clean({
        "player":          name,
        "team":            team,
        "win_prob":        win_prob,
        "kpr":             round(kpr, 4),
        "kpr_adjusted":    round(kpr_adjusted, 4),
        "expected_rounds": expected_rounds,
        "fallback":        fallback,
        "notes":           notes,
        "map1": {
            "kills":  {"per_map": kills_map1},
            "deaths": {"per_map": deaths_map1},
        },
        "bo3": {
            "series_format":   "M1-2 (Bo3)",
            "maps_counted":    2,
            "expected_rounds": expected_rounds,
            "kills":  {"series_total": kills_m12},
            "deaths": {"series_total": deaths_m12},
        },
    })


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def health():
    return _clean({
        "status":  "ok" if STATE["ready"] else "loading",
        "players": STATE["n_players"],
        "game":    "CS2",
    })


@app.get("/search")
def search_players(q: str = Query(...)):
    if not STATE["ready"]:
        raise HTTPException(503, detail="Loading")
    df = STATE["df"]
    matches = df[df["playername"].str.lower().str.contains(q.strip().lower(), na=False)]
    players = (
        matches[["playername", "team", "maps_played", "kpr", "kd_ratio"]]
        .drop_duplicates("playername")
        .sort_values("maps_played", ascending=False)
        .head(10)
    )
    return _clean(players.fillna("").to_dict(orient="records"))


@app.get("/players")
def list_players(q: str = Query(None), team: str = Query(None)):
    if not STATE["ready"]:
        raise HTTPException(503, detail="Loading")
    df = STATE["df"]
    if team:
        df = df[df["team"].str.lower().str.contains(team.lower(), na=False)]
    if q:
        df = df[df["playername"].str.lower().str.contains(q.lower(), na=False)]
    players = (
        df[["playername", "team", "maps_played", "kills", "deaths", "kpr", "kd_ratio"]]
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
    series:    str = Query("Bo3"),
    role:      str = Query(None),
):
    if not STATE["ready"]:
        raise HTTPException(503, detail="Loading")
    return _predict(player, moneyline, opp_ml, role, series)


@app.get("/refresh")
def refresh():
    STATE["ready"] = False
    try:
        load_everything()
        return {"status": "refreshed", "players": STATE["n_players"]}
    except Exception as e:
        STATE["ready"] = True
        raise HTTPException(500, str(e))
