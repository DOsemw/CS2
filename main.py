"""
main.py — CS2 Props Prediction API
Deploy all files flat (no subfolders) to Railway.

Mirrors the LoL props API exactly. Key differences:
  - Targets: kills / deaths / hs_pct  (not kills/deaths/assists)
  - No position param (all CS2 players are same role)
  - Series formats: Bo1 / Bo3  (no Bo5 normally)
  - Side param: CT / T  (not Blue/Red)
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

TARGETS = ["kills", "deaths", "hs_pct"]


def load_everything():
    from data_ingestion import load_raw, filter_top_teams
    from feature_engineering import build_features, get_feature_columns
    from model import train_all, load_model

    log.info("Loading CS2 data...")
    raw = load_raw()
    raw = filter_top_teams(raw, min_maps=int(os.getenv("MIN_MAPS", "20")))
    feat = build_features(raw, verbose=True)
    fcols = get_feature_columns(feat)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    models_exist = all((model_dir / f"{t}_model.pkl").exists() for t in TARGETS)

    if not models_exist:
        log.info("Training models (first run — may take a few minutes)...")
        train_all(feat, fcols)
    else:
        log.info("Loaded existing models from disk.")

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
    load_everything()
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
    """Point prediction + bootstrapped confidence interval."""
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok" if STATE["ready"] else "loading",
        "players": STATE["n_players"],
        "date_range": STATE["date_range"],
        "game": "CS2",
    }


@app.get("/search")
def search_players(q: str = Query(...)):
    """Fuzzy player name search."""
    _check_ready()
    df = STATE["df"]
    matches = df[df["playername"].str.lower().str.contains(q.strip().lower(), na=False)]
    players = (
        matches.groupby("playername")
        .agg(team=("team", "last"), maps=("match_id", "nunique"))
        .reset_index()
        .sort_values("maps", ascending=False)
        .head(10)
    )
    return players.to_dict(orient="records")


@app.get("/players")
def list_players(q: str = Query(None), team: str = Query(None)):
    _check_ready()
    df = STATE["df"]
    if team:
        df = df[df["team"].str.lower().str.contains(team.lower(), na=False)]
    if q:
        df = df[df["playername"].str.lower().str.contains(q.lower(), na=False)]
    players = (
        df.groupby("playername")
        .agg(team=("team", "last"), maps=("match_id", "nunique"),
             avg_kills=("kills", "mean"), avg_rating=("rating", "mean"))
        .reset_index()
        .sort_values("maps", ascending=False)
    )
    players["avg_kills"] = players["avg_kills"].round(2)
    players["avg_rating"] = players["avg_rating"].round(3)
    return players.to_dict(orient="records")


@app.get("/teams")
def list_teams(q: str = Query(None)):
    _check_ready()
    df = STATE["df"]
    if q:
        df = df[df["team"].str.lower().str.contains(q.lower(), na=False)]
    teams = (
        df.groupby("team")
        .agg(maps=("match_id", "nunique"), avg_kills=("kills", "mean"))
        .reset_index()
        .sort_values("maps", ascending=False)
    )
    return teams.to_dict(orient="records")


@app.get("/predict")
def predict_player(
    player: str = Query(..., description="Player name e.g. 's1mple'"),
    moneyline: int = Query(None, description="Player's team moneyline e.g. -150"),
    opp_ml: int = Query(None, description="Opponent moneyline e.g. +130"),
    opponent: str = Query(None, description="Upcoming opponent team name"),
    series: str = Query("Bo3", description="Series format: Bo1 or Bo3"),
):
    """
    Predict kills / deaths / hs_pct for a CS2 player.

    Query params:
      player    — required, player name
      moneyline — your player's team moneyline (American odds)
      opp_ml    — opponent's moneyline
      opponent  — opponent team name (used for defensive adjustment)
      series    — Bo1 or Bo3 (default Bo3)
    """
    _check_ready()

    from model import load_model
    from series_predictor import vig_adjusted_probs, build_series_projection
    from moneyline_adjustments import apply_win_prob_blend, apply_opponent_adjustment

    df = STATE["df"]
    fcols = STATE["feature_cols"]

    player_df, X = _get_player_row(player)

    # Win probability
    if moneyline is not None and opp_ml is not None:
        win_prob, _ = vig_adjusted_probs(moneyline, opp_ml)
    else:
        win_prob = 0.5

    # 1. Blend win/loss features based on win probability
    X = apply_win_prob_blend(X, win_prob, targets=TARGETS)

    # 2. Apply opponent defensive adjustment
    opp_info = {}
    if opponent:
        X, opp_info = apply_opponent_adjustment(X, opponent, df)

    # 3. Per-map predictions
    pm = {}
    for stat in TARGETS:
        model, _, metrics = load_model(stat)
        pm[stat] = _predict_stat(model, X, metrics["mae"])

    # 4. Apply opponent kill ratio to predictions
    if opp_info:
        kills_ratio = opp_info.get("kills_ratio", 1.0)
        for stat, ratio in [("kills", kills_ratio), ("deaths", 1 / kills_ratio)]:
            ratio = min(max(ratio, 0.6), 1.6)
            for key in ("per_map", "low", "high"):
                pm[stat][key] = round(pm[stat][key] * ratio, 2)

    # 5. Series projections
    bo3 = build_series_projection(pm, win_prob, "Bo3")
    bo1 = {stat: {
        "per_map": pm[stat]["per_map"],
        "low": pm[stat]["low"],
        "high": pm[stat]["high"],
        "mae": pm[stat]["mae"],
    } for stat in TARGETS}

    # Recent form
    recent = (
        player_df.head(5)[["date", "map_name", "team", "opponent", "kills", "deaths", "hs_pct", "rating"]]
        .copy()
    )
    recent["date"] = recent["date"].astype(str)

    return {
        "player": player_df.iloc[0]["playername"],
        "team": player_df.iloc[0]["team"],
        "win_prob": round(win_prob, 3),
        "moneyline": moneyline,
        "opponent": opp_info.get("opp_team", opponent),
        "map1": bo1,
        "bo3": bo3,
        "recent_form": recent.to_dict(orient="records"),
    }


@app.get("/refresh")
def refresh():
    """Re-scrape and retrain. Kicks off in background on Railway."""
    STATE["ready"] = False
    try:
        load_everything()
        return {"status": "refreshed", "players": STATE["n_players"]}
    except Exception as e:
        STATE["ready"] = True
        raise HTTPException(500, str(e))
