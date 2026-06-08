"""
main.py — CS2 Props Prediction API
Deploy all files flat (no subfolders) to Railway.

Targets: kills / deaths
Series formats: Bo1 / Bo3
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
    "feature_cols": None,
    "ready": False,
    "n_players": 0,
    "date_range": "",
}

TARGETS = ["kills", "deaths"]
MODEL_DIR = Path("models")


def _clean(obj):
    """Recursively replace NaN/inf with None for JSON compliance."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def load_everything():
    from data_ingestion import load_raw, filter_top_teams
    from feature_engineering import build_features, get_feature_columns
    from model import train_all

    log.info("Loading CS2 data...")
    raw = load_raw()
    raw = filter_top_teams(raw, min_maps=int(os.getenv("MIN_MAPS", "20")))
    feat = build_features(raw, verbose=True)
    fcols = get_feature_columns(feat)

    MODEL_DIR.mkdir(exist_ok=True)
    missing = [t for t in TARGETS if not (MODEL_DIR / f"{t}_model.pkl").exists()]
    if missing:
        log.info(f"Training models for: {missing}")
        train_all(feat, fcols)
    else:
        log.info("All models loaded from disk.")

    STATE.update({
        "df": feat,
        "feature_cols": fcols,
        "ready": True,
        "n_players": feat["playername"].nunique(),
        "date_range": f"{feat['date'].min().date()} → {feat['date'].max().date()}",
    })
    log.info(f"Ready — {STATE['n_players']} players, {STATE['date_range']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_everything()
    except FileNotFoundError as e:
        log.warning(f"[startup] No data found — starting in EMPTY mode.\n{e}")
    yield


app = FastAPI(title="CS2 Props API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_ready():
    if not STATE["ready"]:
        raise HTTPException(503, detail="Model still loading — try again in a moment.")


def _get_player_row(player: str):
    df = STATE["df"]
    fcols = STATE["feature_cols"]
    mask = df["playername"].str.lower() == player.strip().lower()
    if not mask.any():
        suggestions = [p for p in df["playername"].unique()
                       if player.lower() in p.lower()][:6]
        raise HTTPException(404, detail={
            "message": f"Player '{player}' not found.",
            "suggestions": suggestions,
        })
    player_df = df[mask].sort_values("date", ascending=False)
    X = player_df.iloc[[0]][fcols].copy()
    return player_df, X


def _predict_stat(model, X: pd.DataFrame, mae: float) -> dict:
    base = max(0.0, float(model.predict(X)[0]))
    preds = []
    for _ in range(150):
        noisy = X.copy()
        for col in noisy.select_dtypes(include=[np.number]).columns:
            noisy[col] += np.random.normal(
                0, abs(float(noisy[col].iloc[0])) * 0.05 + 0.01
            )
        preds.append(max(0.0, float(model.predict(noisy)[0])))
    return {
        "per_map": round(base, 2),
        "low": round(max(0.0, float(np.quantile(preds, 0.05))), 2),
        "high": round(float(np.quantile(preds, 0.95)), 2),
        "mae": mae,
    }


def _load_model_safe(stat: str):
    from model import load_model
    path = MODEL_DIR / f"{stat}_model.pkl"
    if not path.exists():
        return None, None, None
    return load_model(stat)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return _clean({
        "status": "ok" if STATE["ready"] else "loading",
        "players": STATE["n_players"],
        "date_range": STATE["date_range"],
        "game": "CS2",
    })


@app.get("/search")
def search_players(q: str = Query(...)):
    _check_ready()
    df = STATE["df"]
    matches = df[df["playername"].str.lower().str.contains(q.strip().lower(), na=False)]
    players = (
        matches[["playername", "team", "maps_played"]]
        .drop_duplicates("playername")
        .sort_values("maps_played", ascending=False)
        .head(10)
    )
    return _clean(players.fillna("").to_dict(orient="records"))


@app.get("/players")
def list_players(q: str = Query(None), team: str = Query(None)):
    _check_ready()
    df = STATE["df"]
    if team:
        df = df[df["team"].str.lower().str.contains(team.lower(), na=False)]
    if q:
        df = df[df["playername"].str.lower().str.contains(q.lower(), na=False)]
    players = (
        df[["playername", "team", "maps_played", "kills", "deaths", "kd_ratio", "rating"]]
        .drop_duplicates("playername")
        .sort_values("maps_played", ascending=False)
    )
    return _clean(players.fillna("").to_dict(orient="records"))


@app.get("/teams")
def list_teams(q: str = Query(None)):
    _check_ready()
    df = STATE["df"]
    if q:
        df = df[df["team"].str.lower().str.contains(q.lower(), na=False)]
    teams = (
        df.groupby("team")
        .agg(players=("playername", "nunique"), avg_kills=("kills", "mean"))
        .reset_index()
        .sort_values("players", ascending=False)
    )
    teams["avg_kills"] = teams["avg_kills"].round(2)
    return _clean(teams.fillna("").to_dict(orient="records"))


@app.get("/predict")
def predict_player(
    player: str = Query(..., description="Player name e.g. 'donk'"),
    moneyline: int = Query(None, description="Player's team moneyline e.g. -150"),
    opp_ml: int = Query(None, description="Opponent moneyline e.g. 130"),
    opponent: str = Query(None, description="Upcoming opponent team name"),
    series: str = Query("Bo3", description="Series format: Bo1 or Bo3"),
):
    _check_ready()

    from series_predictor import vig_adjusted_probs, build_series_projection
    from moneyline_adjustments import apply_win_prob_blend, apply_opponent_adjustment

    df = STATE["df"]
    player_df, X = _get_player_row(player)

    # Win probability
    win_prob = 0.5
    if moneyline is not None and opp_ml is not None:
        win_prob, _ = vig_adjusted_probs(moneyline, opp_ml)

    # Win/loss blend
    X = apply_win_prob_blend(X, win_prob, targets=TARGETS)

    # Opponent adjustment
    opp_info = {}
    if opponent:
        X, opp_info = apply_opponent_adjustment(X, opponent, df)

    # Per-map predictions
    pm = {}
    for stat in TARGETS:
        model, _, metrics = _load_model_safe(stat)
        if model is None:
            pm[stat] = {"per_map": None, "low": None, "high": None, "mae": None}
            continue
        pm[stat] = _predict_stat(model, X, metrics["mae"])

    # Opponent kill ratio adjustment
    if opp_info:
        kills_ratio = opp_info.get("kills_ratio", 1.0)
        for stat, ratio in [("kills", kills_ratio), ("deaths", 1 / kills_ratio)]:
            ratio = min(max(ratio, 0.6), 1.6)
            if pm[stat]["per_map"] is not None:
                for key in ("per_map", "low", "high"):
                    pm[stat][key] = round(pm[stat][key] * ratio, 2)

    # Series projections
    pm_valid = {k: v for k, v in pm.items() if v["per_map"] is not None}
    bo3 = build_series_projection(pm_valid, win_prob, series) if pm_valid else {}
    bo1 = {stat: vals for stat, vals in pm.items()}

    # Recent form
    cols = [c for c in ["date", "team", "kills", "deaths", "kd_ratio", "rating"] if c in player_df.columns]
    recent = player_df.head(5)[cols].copy()
    recent["date"] = recent["date"].astype(str)

    return _clean({
        "player": player_df.iloc[0]["playername"],
        "team": str(player_df.iloc[0].get("team", "") or ""),
        "win_prob": round(win_prob, 3),
        "moneyline": moneyline,
        "opponent": opp_info.get("opp_team", opponent),
        "map1": bo1,
        "bo3": bo3,
        "recent_form": recent.fillna("").to_dict(orient="records"),
    })


@app.get("/refresh")
def refresh():
    STATE["ready"] = False
    try:
        load_everything()
        return {"status": "refreshed", "players": STATE["n_players"]}
    except Exception as e:
        STATE["ready"] = True
        raise HTTPException(500, str(e))
