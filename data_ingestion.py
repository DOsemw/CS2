"""
data_ingestion.py
-----------------
Scrapes HLTV player aggregate stats for 2026 using undetected-chromedriver.
Bypasses Cloudflare by using a real Chrome browser with stealth settings.

Run locally once to generate data/hltv_raw.csv:
  python data_ingestion.py

Then commit the CSV and models/ to your repo so Railway doesn't need to scrape.
"""

import os
import time
import random
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_FILE = DATA_DIR / "hltv_raw.csv"

# Date range — 2026 only
START_DATE = "2026-01-01"
END_DATE   = "2026-12-31"

# HLTV stats URLs to scrape
STATS_URL = (
    f"https://www.hltv.org/stats/players"
    f"?startDate={START_DATE}&endDate={END_DATE}"
    f"&rankingFilter=Top50&minMapCount=10"
)


def make_driver():
    import undetected_chromedriver as uc
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    driver = uc.Chrome(options=options)
    return driver


def fetch_page(driver, url: str, wait: float = 4.0) -> str:
    log.info(f"[fetch] {url}")
    driver.get(url)
    time.sleep(wait + random.uniform(1, 3))
    return driver.page_source


def parse_player_stats_table(html: str) -> list[dict]:
    """Parse the main /stats/players table."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    table = soup.select_one("table.stats-table") or soup.select_one("tbody")
    if not table:
        log.warning("[parse] No stats table found on page")
        return rows

    for tr in soup.select("tbody tr"):
        tds = tr.select("td")
        if len(tds) < 8:
            continue

        try:
            player_el = tr.select_one("td.playerCol a") or tr.select_one("a.player-name")
            team_el   = tr.select_one("td.teamCol a") or tr.select_one("a.team-name")

            player_name = player_el.get_text(strip=True) if player_el else ""
            team_name   = team_el.get_text(strip=True) if team_el else ""

            if not player_name:
                continue

            def safe(idx):
                try:
                    return float(tds[idx].get_text(strip=True).replace("%", "").replace("+", "").replace("−", "-").replace("N/A", ""))
                except Exception:
                    return np.nan

            # HLTV stats table columns:
            # Player | Team | Maps | K/D | Dmg/R | KAST | Impact | Rating
            maps_played = safe(2)
            kd_ratio    = safe(3)
            adr         = safe(4)
            kast        = safe(5)
            impact      = safe(6)
            rating      = safe(7)

            rows.append({
                "playername":   player_name,
                "team":         team_name,
                "maps_played":  maps_played,
                "kd_ratio":     kd_ratio,
                "adr":          adr,
                "kast":         kast,
                "impact":       impact,
                "rating":       rating,
            })

        except Exception as e:
            log.warning(f"[parse] Row error: {e}")
            continue

    return rows


def fetch_player_extended(driver, player_el_href: str) -> dict:
    """
    Fetch extended stats (kills, deaths, HS%) from a player's individual stats page.
    Returns dict with kills_per_round, deaths_per_round, hs_pct.
    """
    if not player_el_href:
        return {}

    url = f"https://www.hltv.org{player_el_href}?startDate={START_DATE}&endDate={END_DATE}"
    html = fetch_page(driver, url, wait=3.0)
    soup = BeautifulSoup(html, "html.parser")

    result = {}
    stat_rows = soup.select("div.stats-row")
    for row in stat_rows:
        spans = row.select("span")
        if len(spans) < 2:
            continue
        label = spans[0].get_text(strip=True).lower()
        value = spans[1].get_text(strip=True).replace("%", "").replace(",", "")
        try:
            v = float(value)
            if "headshot" in label or "hs" in label:
                result["hs_pct"] = v
            elif "kills / round" in label or "kills/round" in label:
                result["kills_per_round"] = v
            elif "deaths / round" in label or "deaths/round" in label:
                result["deaths_per_round"] = v
            elif "total kills" in label:
                result["total_kills"] = v
            elif "total deaths" in label:
                result["total_deaths"] = v
        except Exception:
            continue

    return result


def scrape_all_players() -> pd.DataFrame:
    """Main scraping function — fetches player list then extended stats per player."""
    driver = make_driver()
    all_rows = []

    try:
        # ── Step 1: get player list ───────────────────────────────────────────
        log.info(f"[scrape] Fetching player rankings for {START_DATE} to {END_DATE}")
        html = fetch_page(driver, STATS_URL, wait=5.0)
        players_base = parse_player_stats_table(html)
        log.info(f"[scrape] Found {len(players_base)} players in table")

        if not players_base:
            log.error("[scrape] No players found — Cloudflare may have blocked the request")
            return pd.DataFrame()

        # ── Step 2: get extended stats per player ─────────────────────────────
        soup = BeautifulSoup(html, "html.parser")
        player_links = []
        for tr in soup.select("tbody tr"):
            a = tr.select_one("td.playerCol a") or tr.select_one("a.player-name")
            player_links.append(a["href"] if a and a.get("href") else None)

        for i, (base, href) in enumerate(zip(players_base, player_links)):
            log.info(f"[player] {i+1}/{len(players_base)}: {base['playername']}")
            extended = {}
            if href:
                try:
                    extended = fetch_player_extended(driver, href)
                except Exception as e:
                    log.warning(f"[player] Extended stats failed for {base['playername']}: {e}")

            row = {**base, **extended}

            # Derive kills/deaths per map from per-round if available
            # Avg CS2 map = ~25 rounds
            rounds_per_map = 25
            if "kills_per_round" in row:
                row["kills"] = round(row["kills_per_round"] * rounds_per_map, 2)
            elif "total_kills" in row and row.get("maps_played", 0) > 0:
                row["kills"] = round(row["total_kills"] / row["maps_played"], 2)
            else:
                row["kills"] = np.nan

            if "deaths_per_round" in row:
                row["deaths"] = round(row["deaths_per_round"] * rounds_per_map, 2)
            elif "total_deaths" in row and row.get("maps_played", 0) > 0:
                row["deaths"] = round(row["total_deaths"] / row["maps_played"], 2)
            else:
                row["deaths"] = np.nan

            row.setdefault("hs_pct", np.nan)

            all_rows.append(row)

            # Polite delay between players
            time.sleep(random.uniform(3, 6))

    finally:
        driver.quit()

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_FILE, index=False)
    log.info(f"[done] {len(df)} players saved to {OUT_FILE}")
    return df


# ── Load from disk ────────────────────────────────────────────────────────────

def load_raw(path: str = None) -> pd.DataFrame:
    path = path or str(OUT_FILE)
    p = Path(path)

    if not p.exists() or p.stat().st_size < 100:
        raise FileNotFoundError(
            f"No data found at {p}.\n"
            "Run locally: python data_ingestion.py\n"
            "Then commit data/hltv_raw.csv to your repo."
        )

    log.info(f"[load] Reading {p}")
    df = pd.read_csv(p, low_memory=False)
    df["match_id"] = df.index.astype(str)
    df["date"] = pd.Timestamp("2026-01-01")

    # Fill missing values with medians
    for col in ["kills", "deaths", "hs_pct", "adr", "rating", "kd_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    log.info(f"[load] {len(df)} players loaded")
    return df


def filter_top_teams(df: pd.DataFrame, min_maps: int = 10) -> pd.DataFrame:
    if "maps_played" not in df.columns:
        return df
    filtered = df[df["maps_played"] >= min_maps].copy()
    log.info(f"[filter] {len(filtered)} players with >= {min_maps} maps")
    return filtered


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = scrape_all_players()
    if df.empty:
        print("\nNo data collected. Check if Chrome opened and HLTV loaded.")
    else:
        print(f"\nDone! {len(df)} players scraped.")
        print(df[["playername", "team", "kills", "deaths", "hs_pct", "rating"]].head(10))
