"""
data_ingestion.py
-----------------
Scrapes HLTV player aggregate stats for 2026 using undetected-chromedriver.
Bypasses Cloudflare by using a real Chrome browser with stealth settings.

Run locally once to generate data/hltv_raw.csv:
  python3 data_ingestion.py

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

START_DATE = "2026-01-01"
END_DATE   = "2026-12-31"

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


def fetch_page(driver, url: str, wait: float = 12.0) -> str:
    log.info(f"[fetch] {url}")
    driver.get(url)
    time.sleep(wait + random.uniform(1, 3))
    return driver.page_source


def parse_player_stats_table(html: str) -> tuple[list[dict], list[str]]:
    """
    Aggressively parse any table on the page regardless of class names.
    Returns (rows, player_hrefs).
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    hrefs = []

    # Try every tbody on the page
    all_trs = soup.select("tbody tr")
    log.info(f"[parse] Found {len(all_trs)} tbody rows")

    for tr in all_trs:
        tds = tr.select("td")
        if len(tds) < 4:
            continue

        # Get all text values from the row
        texts = [td.get_text(strip=True) for td in tds]

        # Player name — first anchor tag in the row
        player_el = tr.select_one("a")
        player_name = player_el.get_text(strip=True) if player_el else texts[0]
        href = player_el.get("href", "") if player_el else ""

        if not player_name or player_name.isdigit():
            continue

        # Try to find team name — second anchor or second td
        team_anchors = tr.select("a")
        team_name = team_anchors[1].get_text(strip=True) if len(team_anchors) > 1 else ""

        # Extract numeric values from remaining tds
        numeric_vals = []
        for td in tds:
            txt = td.get_text(strip=True).replace("%", "").replace("+", "").replace(",", "")
            try:
                numeric_vals.append(float(txt))
            except Exception:
                numeric_vals.append(np.nan)

        # HLTV stats table order: maps, kd, dmg/r, kast, impact, rating
        # Find them by position — skip non-numeric leading columns
        nums = [v for v in numeric_vals if not np.isnan(v)]

        maps_played = nums[0] if len(nums) > 0 else np.nan
        kd_ratio    = nums[1] if len(nums) > 1 else np.nan
        adr         = nums[2] if len(nums) > 2 else np.nan
        kast        = nums[3] if len(nums) > 3 else np.nan
        impact      = nums[4] if len(nums) > 4 else np.nan
        rating      = nums[5] if len(nums) > 5 else np.nan

        log.info(f"[parse] {player_name} | {team_name} | maps={maps_played} kd={kd_ratio} adr={adr} rating={rating}")

        rows.append({
            "playername":  player_name,
            "team":        team_name,
            "maps_played": maps_played,
            "kd_ratio":    kd_ratio,
            "adr":         adr,
            "kast":        kast,
            "impact":      impact,
            "rating":      rating,
        })
        hrefs.append(href)

    return rows, hrefs


def fetch_player_extended(driver, href: str) -> dict:
    """Fetch HS%, kills/round, deaths/round from individual player stats page."""
    if not href:
        return {}

    url = f"https://www.hltv.org{href}?startDate={START_DATE}&endDate={END_DATE}"
    html = fetch_page(driver, url, wait=4.0)
    soup = BeautifulSoup(html, "html.parser")
    result = {}

    # Try stats-row divs
    for row in soup.select("div.stats-row"):
        spans = row.select("span")
        if len(spans) < 2:
            continue
        label = spans[0].get_text(strip=True).lower()
        value = spans[1].get_text(strip=True).replace("%", "").replace(",", "").strip()
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

    # Fallback — scan all text on page for HS%
    if "hs_pct" not in result:
        for el in soup.select("div.col"):
            txt = el.get_text(strip=True)
            if "Headshot" in txt or "HS%" in txt:
                import re
                m = re.search(r"(\d+\.?\d*)%", txt)
                if m:
                    result["hs_pct"] = float(m.group(1))
                    break

    return result


def scrape_all_players() -> pd.DataFrame:
    driver = make_driver()
    all_rows = []

    try:
        log.info(f"[scrape] Fetching player stats for {START_DATE} to {END_DATE}")
        html = fetch_page(driver, STATS_URL, wait=12.0)
        players_base, hrefs = parse_player_stats_table(html)
        log.info(f"[scrape] Parsed {len(players_base)} players")

        if not players_base:
            log.error("[scrape] No players found — check if HLTV loaded correctly")
            return pd.DataFrame()

        for i, (base, href) in enumerate(zip(players_base, hrefs)):
            log.info(f"[player] {i+1}/{len(players_base)}: {base['playername']}")
            extended = {}
            if href:
                try:
                    extended = fetch_player_extended(driver, href)
                except Exception as e:
                    log.warning(f"[player] Extended stats failed: {e}")

            row = {**base, **extended}

            rounds_per_map = 25
            if "kills_per_round" in row:
                row["kills"] = round(row["kills_per_round"] * rounds_per_map, 2)
            elif "total_kills" in row and row.get("maps_played", 0) > 0:
                row["kills"] = round(row["total_kills"] / row["maps_played"], 2)
            else:
                # Estimate from KD and deaths
                row["kills"] = round((row.get("kd_ratio", 1.0) or 1.0) * 14, 2)

            if "deaths_per_round" in row:
                row["deaths"] = round(row["deaths_per_round"] * rounds_per_map, 2)
            elif "total_deaths" in row and row.get("maps_played", 0) > 0:
                row["deaths"] = round(row["total_deaths"] / row["maps_played"], 2)
            else:
                row["deaths"] = round(row["kills"] / max(row.get("kd_ratio", 1.0), 0.1), 2)

            row.setdefault("hs_pct", np.nan)
            all_rows.append(row)

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
            "Run locally: python3 data_ingestion.py\n"
            "Then commit data/hltv_raw.csv to your repo."
        )

    log.info(f"[load] Reading {p}")
    df = pd.read_csv(p, low_memory=False)
    df["match_id"] = df.index.astype(str)
    df["date"] = pd.Timestamp("2026-01-01")

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
        print("\nNo data collected.")
    else:
        print(f"\nDone! {len(df)} players scraped.")
        print(df[["playername", "team", "kills", "deaths", "hs_pct", "rating"]].head(10).to_string())
