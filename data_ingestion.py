"""
data_ingestion.py
-----------------
Pulls CS2 player match stats from PandaScore API.

Setup:
  Set env var: PANDASCORE_TOKEN=your_key_here

  Run once to build dataset:
    python data_ingestion.py

  Data saved to data/hltv_raw.csv (same schema as before — rest of code unchanged)

PandaScore free tier: 1000 req/month
  ~500 matches = ~600 requests (match list + per-match player stats)
  Use --max-matches 400 to stay safe on free tier
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://api.pandascore.co"
ROWS_FILE = DATA_DIR / "hltv_raw.csv"


def get_token() -> str:
    token = os.getenv("PANDASCORE_TOKEN", "")
    if not token:
        raise RuntimeError(
            "PANDASCORE_TOKEN env var not set.\n"
            "Set it in Railway Variables or locally: export PANDASCORE_TOKEN=your_key"
        )
    return token


def api_get(path: str, params: dict = None, retries: int = 3) -> list | dict:
    """Make a PandaScore API call with retry logic."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE_URL}{path}"
    params = params or {}

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                log.warning(f"[rate limit] Waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"[api] Attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(5 * (attempt + 1))

    return []


# ── Fetch matches ─────────────────────────────────────────────────────────────

def fetch_matches(max_matches: int = 400, start_date: str = "2023-01-01") -> list[dict]:
    """Fetch CS2 past matches from PandaScore."""
    log.info(f"[matches] Fetching up to {max_matches} CS2 matches since {start_date}...")
    matches = []
    page = 1
    per_page = 100

    while len(matches) < max_matches:
        data = api_get("/csgo/matches/past", params={
            "filter[status]": "finished",
            "range[begin_at]": f"{start_date},",
            "sort": "-begin_at",
            "page[number]": page,
            "page[size]": per_page,
        })

        if not data:
            break

        matches.extend(data)
        log.info(f"[matches] Page {page}: {len(data)} matches | total: {len(matches)}")

        if len(data) < per_page:
            break

        page += 1
        time.sleep(1.2)  # ~50 req/min to stay under rate limit

    log.info(f"[matches] Total matches fetched: {len(matches)}")
    return matches[:max_matches]


# ── Fetch player stats per game ───────────────────────────────────────────────

def fetch_game_stats(game_id: int) -> list[dict]:
    """Fetch per-player stats for a single game (map)."""
    data = api_get(f"/csgo/games/{game_id}/players")
    if not data:
        return []
    return data if isinstance(data, list) else []


# ── Build rows ────────────────────────────────────────────────────────────────

def build_rows(matches: list[dict], resume: bool = True) -> pd.DataFrame:
    """
    For each match → for each game (map) → fetch player stats → build rows.
    Saves checkpoints every 25 matches.
    """
    already_done = set()
    all_rows = []

    # Resume from checkpoint
    if resume and ROWS_FILE.exists() and ROWS_FILE.stat().st_size > 100:
        try:
            existing = pd.read_csv(ROWS_FILE)
            already_done = set(existing["match_id"].astype(str))
            all_rows = existing.to_dict("records")
            log.info(f"[resume] {len(already_done)} matches already done")
        except Exception as e:
            log.warning(f"[resume] Could not read checkpoint: {e} — starting fresh")

    todo = [m for m in matches if str(m["id"]) not in already_done]
    log.info(f"[scrape] {len(todo)} matches to process")

    for i, match in enumerate(todo):
        match_id = str(match["id"])
        team1 = match.get("opponents", [{}])[0].get("opponent", {}).get("name", "") if len(match.get("opponents", [])) > 0 else ""
        team2 = match.get("opponents", [{}])[1].get("opponent", {}).get("name", "") if len(match.get("opponents", [])) > 1 else ""
        event = match.get("league", {}).get("name", "") or match.get("serie", {}).get("full_name", "")
        match_date = (match.get("begin_at") or "")[:10]
        series_type = match.get("match_type", "bo3")  # bo1, bo3, bo5

        # Winner
        winner_id = match.get("winner", {}).get("id") if match.get("winner") else None

        games = match.get("games") or []
        if not games:
            # Fetch games list if not embedded
            games_data = api_get(f"/csgo/matches/{match_id}/games")
            games = games_data if isinstance(games_data, list) else []

        for map_idx, game in enumerate(games):
            game_id = game.get("id")
            map_name = game.get("map", {}).get("name", f"map{map_idx+1}") if game.get("map") else f"map{map_idx+1}"

            if not game_id:
                continue

            # Determine map winner
            game_winner_id = game.get("winner", {}).get("id") if game.get("winner") else None

            player_stats = fetch_game_stats(game_id)
            time.sleep(0.8)

            for ps in player_stats:
                player = ps.get("player", {})
                player_name = player.get("name", "")
                if not player_name:
                    continue

                team_id = ps.get("team_id") or (ps.get("team", {}) or {}).get("id")
                team_name = ""
                if team_id:
                    for opp in match.get("opponents", []):
                        if opp.get("opponent", {}).get("id") == team_id:
                            team_name = opp["opponent"].get("name", "")
                            break

                opponent_name = team2 if team_name == team1 else team1
                map_won = 1 if (game_winner_id and team_id == game_winner_id) else 0

                stats = ps.get("stats", {}) or {}

                kills  = _safe(stats.get("kills"))
                deaths = _safe(stats.get("deaths"))
                hs_pct = _safe(stats.get("headshot_percentage"))
                adr    = _safe(stats.get("average_damage_per_round"))
                rating = _safe(stats.get("rating"))

                # PandaScore sometimes puts stats at top level
                if np.isnan(kills):
                    kills  = _safe(ps.get("kills"))
                    deaths = _safe(ps.get("deaths"))
                    hs_pct = _safe(ps.get("headshot_percentage"))
                    adr    = _safe(ps.get("average_damage_per_round"))
                    rating = _safe(ps.get("rating"))

                all_rows.append({
                    "match_id": match_id,
                    "map_name": map_name,
                    "map_number": map_idx + 1,
                    "date": match_date,
                    "event": event,
                    "team": team_name,
                    "opponent": opponent_name,
                    "result": map_won,
                    "playername": player_name,
                    "kills": kills,
                    "deaths": deaths,
                    "hs_pct": hs_pct,
                    "adr": adr,
                    "rating": rating,
                    "map_score_team": np.nan,
                    "map_score_opp": np.nan,
                })

        already_done.add(match_id)

        # Checkpoint every 25
        if (i + 1) % 25 == 0:
            pd.DataFrame(all_rows).to_csv(ROWS_FILE, index=False)
            log.info(f"[checkpoint] {len(all_rows)} rows saved ({i+1}/{len(todo)} matches)")

        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    df.to_csv(ROWS_FILE, index=False)
    log.info(f"[done] {len(df):,} rows saved to {ROWS_FILE}")
    return df


def _safe(val) -> float:
    try:
        return float(val) if val is not None else np.nan
    except Exception:
        return np.nan


# ── Load from disk ────────────────────────────────────────────────────────────

def load_raw(path: str = None) -> pd.DataFrame:
    path = path or os.getenv("DATA_PATH", str(ROWS_FILE))
    p = Path(path)

    if not p.exists() or p.stat().st_size < 100:
        raise FileNotFoundError(
            f"No data found at {p}.\n"
            "Run: python data_ingestion.py\n"
            "Or set PANDASCORE_TOKEN and run the ingestion script."
        )

    log.info(f"[load] Reading {p}")
    df = pd.read_csv(p, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "playername", "kills", "deaths"])
    df["match_id"] = df["match_id"].astype(str)

    log.info(f"[load] {len(df):,} rows | "
             f"{df['date'].min().date()} → {df['date'].max().date()} | "
             f"{df['playername'].nunique()} players")
    return df


def filter_top_teams(df: pd.DataFrame, min_maps: int = 10) -> pd.DataFrame:
    counts = df.groupby("playername")["match_id"].nunique()
    keep = counts[counts >= min_maps].index
    filtered = df[df["playername"].isin(keep)].copy()
    log.info(f"[filter] {len(filtered):,} rows | {filtered['playername'].nunique()} players (>={min_maps} maps)")
    return filtered


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--max-matches", type=int, default=400)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    matches = fetch_matches(max_matches=args.max_matches, start_date=args.start)
    df = build_rows(matches, resume=not args.no_resume)
    print(f"\nDone! {len(df):,} rows, {df['playername'].nunique()} unique players")
