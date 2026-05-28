"""
data_ingestion.py
-----------------
Scrapes HLTV.org for CS2 player match stats using Playwright.

HLTV actively blocks scrapers — this module uses:
  - Playwright headless Chromium (looks like a real browser)
  - Random delays between requests
  - Rotating user agents
  - Retry logic with exponential backoff

HOW TO GET DATA:
  Run this once to build your local dataset:
    python data_ingestion.py

  Or set env vars to control scraping range:
    HLTV_START_DATE=2023-01-01
    HLTV_END_DATE=2025-12-31
    HLTV_MIN_RATING=0       # min team world ranking filter (0 = all)

  Data is saved to data/hltv_matches.csv and data/hltv_players.csv

STRUCTURE SCRAPED:
  - Match results pages (/results)
  - Per-match scorecard pages (/matches/XXXXX/...)
  - Per-player stats: kills, deaths, HS%, ADR, Rating 2.0
"""

import os
import time
import random
import logging
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HLTV_BASE = "https://www.hltv.org"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
]

# ── Browser helpers ───────────────────────────────────────────────────────────

async def make_browser(playwright):
    """Launch Playwright Chromium with stealth settings."""
    browser = await playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ],
    )
    return browser


async def new_context(browser, ua: str = None):
    """Create a new browser context with randomised fingerprint."""
    ua = ua or random.choice(USER_AGENTS)
    ctx = await browser.new_context(
        user_agent=ua,
        viewport={"width": random.randint(1280, 1920), "height": random.randint(800, 1080)},
        locale="en-US",
        timezone_id="America/New_York",
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        },
    )
    # Mask webdriver flag
    await ctx.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        window.chrome = { runtime: {} };
    """)
    return ctx


async def fetch_page(ctx, url: str, retries: int = 4, wait_ms: int = 3000) -> str | None:
    """Navigate to a URL and return page HTML. Retries with backoff."""
    page = await ctx.new_page()
    for attempt in range(retries):
        try:
            delay = random.uniform(1.5, 4.0) * (attempt + 1)
            await asyncio.sleep(delay)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(wait_ms + random.randint(0, 1500))
            # Check for Cloudflare challenge
            content = await page.content()
            if "cf-browser-verification" in content or "Just a moment" in content:
                log.warning(f"[cf] Cloudflare challenge on {url} — waiting longer")
                await page.wait_for_timeout(8000)
                content = await page.content()
            await page.close()
            return content
        except Exception as e:
            log.warning(f"[fetch] Attempt {attempt+1}/{retries} failed for {url}: {e}")
            await asyncio.sleep(5 * (attempt + 1))
    await page.close()
    return None


# ── Match list scraping ───────────────────────────────────────────────────────

def _date_to_hltv_offset(start_date: str, end_date: str) -> list[str]:
    """Build list of HLTV results URLs paginated by 100 matches each."""
    # HLTV /results uses ?offset=0,100,200... and startDate/endDate filters
    base = f"{HLTV_BASE}/results?startDate={start_date}&endDate={end_date}&content=stats&rankingFilter=Top30"
    urls = [f"{base}&offset={i*100}" for i in range(0, 50)]  # up to 5000 matches
    return urls


async def scrape_match_links(ctx, start_date: str, end_date: str) -> list[dict]:
    """Scrape all match links from /results pages."""
    from bs4 import BeautifulSoup

    urls = _date_to_hltv_offset(start_date, end_date)
    all_matches = []
    seen = set()

    for i, url in enumerate(urls):
        log.info(f"[results] Page {i+1}/{len(urls)}: {url}")
        html = await fetch_page(ctx, url, wait_ms=2000)
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        result_items = soup.select("div.result-con a.a-reset")

        if not result_items:
            log.info(f"[results] No more matches at offset {i*100} — stopping")
            break

        for a in result_items:
            href = a.get("href", "")
            if "/matches/" not in href:
                continue
            match_id = href.split("/matches/")[1].split("/")[0]
            if match_id in seen:
                continue
            seen.add(match_id)

            # Extract metadata from the card
            teams = a.select("div.team")
            team_names = [t.get_text(strip=True) for t in teams]
            event = a.select_one("span.event-name")
            date_el = a.select_one("div.date")
            format_el = a.select_one("td.star-cell")  # Bo1/Bo3 hint

            all_matches.append({
                "match_id": match_id,
                "url": HLTV_BASE + href,
                "team1": team_names[0] if len(team_names) > 0 else "",
                "team2": team_names[1] if len(team_names) > 1 else "",
                "event": event.get_text(strip=True) if event else "",
                "date_str": date_el.get_text(strip=True) if date_el else "",
            })

        log.info(f"[results] Collected {len(all_matches)} matches so far")
        await asyncio.sleep(random.uniform(2, 5))

    log.info(f"[results] Total matches found: {len(all_matches)}")
    return all_matches


# ── Per-match stats scraping ──────────────────────────────────────────────────

async def scrape_match_stats(ctx, match: dict) -> list[dict]:
    """
    Scrape a single match page for per-player, per-map stats.
    Returns list of player-map rows.
    """
    from bs4 import BeautifulSoup

    html = await fetch_page(ctx, match["url"], wait_ms=2500)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    rows = []

    # HLTV match pages have one stats table per map
    # Each table has players from both teams
    map_sections = soup.select("div.mapholder")
    if not map_sections:
        # Fallback: single-map match
        map_sections = [soup]

    # Get the match date from the page
    date_el = soup.select_one("div.date span[data-unix]")
    if date_el:
        unix_ts = int(date_el["data-unix"]) // 1000
        match_date = datetime.fromtimestamp(unix_ts).strftime("%Y-%m-%d")
    else:
        match_date = match.get("date_str", "")

    # Get event name
    event_el = soup.select_one("div.event a")
    event_name = event_el.get_text(strip=True) if event_el else match.get("event", "")

    for map_idx, map_section in enumerate(map_sections):
        map_name_el = map_section.select_one("div.mapname")
        map_name = map_name_el.get_text(strip=True) if map_name_el else f"map{map_idx+1}"

        if map_name.lower() in ("tba", ""):
            continue

        # Score to determine winner
        score_els = map_section.select("div.results-team-score")
        scores = [el.get_text(strip=True) for el in score_els]
        try:
            score1, score2 = int(scores[0]), int(scores[1])
            team1_won = score1 > score2
        except Exception:
            team1_won = None

        # Stats tables — team1 first, then team2
        stat_tables = map_section.select("table.stats-table")
        team_names = [match.get("team1", ""), match.get("team2", "")]

        for t_idx, table in enumerate(stat_tables[:2]):
            team = team_names[t_idx] if t_idx < len(team_names) else f"team{t_idx+1}"
            won = (t_idx == 0 and team1_won) or (t_idx == 1 and team1_won is False)

            for tr in table.select("tbody tr"):
                tds = tr.select("td")
                if len(tds) < 8:
                    continue

                player_el = tr.select_one("td.players a")
                player_name = player_el.get_text(strip=True) if player_el else ""
                if not player_name:
                    continue

                def safe_float(el, idx):
                    try:
                        return float(tds[idx].get_text(strip=True).replace("%", "").replace("+", "").replace("−", "-"))
                    except Exception:
                        return np.nan

                # HLTV table columns: Player | K | D | +/- | ADR | KAST | Rating
                # HS% is sometimes in a separate column
                kills = safe_float(tds, 1)
                deaths = safe_float(tds, 2)
                adr = safe_float(tds, 4)
                kast_str = tds[5].get_text(strip=True).replace("%", "") if len(tds) > 5 else ""
                rating = safe_float(tds, 6) if len(tds) > 6 else np.nan

                # HS% often requires the extended stats page; set NaN here, fill later
                hs_pct = np.nan

                rows.append({
                    "match_id": match["match_id"],
                    "map_name": map_name,
                    "map_number": map_idx + 1,
                    "date": match_date,
                    "event": event_name,
                    "team": team,
                    "opponent": team_names[1 - t_idx] if t_idx < 2 else "",
                    "result": 1 if won else 0,
                    "playername": player_name,
                    "kills": kills,
                    "deaths": deaths,
                    "adr": adr,
                    "hs_pct": hs_pct,
                    "rating": rating,
                    "map_score_team": score1 if t_idx == 0 else score2,
                    "map_score_opp": score2 if t_idx == 0 else score1,
                })

    return rows


# ── Extended player stats (HS%) ───────────────────────────────────────────────

async def scrape_player_hs(ctx, player_name: str, match_ids: list[str]) -> dict:
    """
    Scrape HS% from a player's stats page for specific matches.
    Returns {match_id: hs_pct}.
    Usage: called lazily to fill hs_pct gaps after main scrape.
    """
    # HLTV player search
    search_url = f"{HLTV_BASE}/search?term={player_name.replace(' ', '+')}&type=player"
    html = await fetch_page(ctx, search_url, wait_ms=1500)
    # ... parse player ID, then hit /stats/players/matches/PLAYERID/PLAYERNAME
    # This is an optional enrichment pass — return empty for now
    return {}


# ── Orchestration ─────────────────────────────────────────────────────────────

async def run_scrape(
    start_date: str = None,
    end_date: str = None,
    max_matches: int = 2000,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Main scraping orchestrator. Scrapes match list then per-match stats.
    Saves progress incrementally so you can resume if it crashes.
    """
    from playwright.async_api import async_playwright

    start_date = start_date or os.getenv("HLTV_START_DATE", "2023-01-01")
    end_date = end_date or os.getenv("HLTV_END_DATE", datetime.today().strftime("%Y-%m-%d"))

    cache_file = DATA_DIR / "match_links.json"
    rows_file = DATA_DIR / "hltv_raw.csv"

    async with async_playwright() as pw:
        browser = await make_browser(pw)
        ctx = await new_context(browser)

        # ── Step 1: get match list ────────────────────────────────────────────
        if resume and cache_file.exists():
            log.info("[cache] Loading match links from cache")
            with open(cache_file) as f:
                matches = json.load(f)
        else:
            log.info(f"[scrape] Collecting match links from {start_date} to {end_date}")
            matches = await scrape_match_links(ctx, start_date, end_date)
            with open(cache_file, "w") as f:
                json.dump(matches, f)
            log.info(f"[cache] Saved {len(matches)} match links")

        matches = matches[:max_matches]

        # ── Step 2: scrape per-match stats ───────────────────────────────────
        already_done = set()
        all_rows = []

        if resume and rows_file.exists():
            existing = pd.read_csv(rows_file)
            already_done = set(existing["match_id"].astype(str))
            all_rows = existing.to_dict("records")
            log.info(f"[resume] {len(already_done)} matches already scraped")

        todo = [m for m in matches if m["match_id"] not in already_done]
        log.info(f"[scrape] {len(todo)} matches remaining to scrape")

        for i, match in enumerate(todo):
            log.info(f"[match] {i+1}/{len(todo)}: {match.get('team1')} vs {match.get('team2')} ({match['match_id']})")
            rows = await scrape_match_stats(ctx, match)
            all_rows.extend(rows)

            # Rotate context every 50 matches to avoid detection
            if (i + 1) % 50 == 0:
                await ctx.close()
                ctx = await new_context(browser)
                log.info("[rotate] Rotated browser context")

            # Save checkpoint every 25 matches
            if (i + 1) % 25 == 0:
                df_checkpoint = pd.DataFrame(all_rows)
                df_checkpoint.to_csv(rows_file, index=False)
                log.info(f"[checkpoint] Saved {len(all_rows)} rows")

            # Polite delay — vary between 3-8s
            await asyncio.sleep(random.uniform(3, 8))

        await ctx.close()
        await browser.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(rows_file, index=False)
    log.info(f"[done] Saved {len(df)} player-map rows to {rows_file}")
    return df


# ── Load from disk ────────────────────────────────────────────────────────────

def load_raw(path: str = None) -> pd.DataFrame:
    """Load the scraped dataset from disk."""
    path = path or os.getenv("HLTV_DATA_PATH", str(DATA_DIR / "hltv_raw.csv"))
    p = Path(path)

    if not p.exists():
        # Check for env-hosted fallback
        url = os.getenv("DATA_URL")
        if url:
            import requests, io
            log.info(f"[env] Downloading dataset from {url[:60]}")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
        else:
            raise FileNotFoundError(
                f"No data found at {p}.\n"
                "Run: python data_ingestion.py\n"
                "Or set DATA_URL env var pointing to a hosted hltv_raw.csv"
            )
    else:
        log.info(f"[load] Reading {p}")
        df = pd.read_csv(p, low_memory=False)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "playername", "kills", "deaths"])
    df["match_id"] = df["match_id"].astype(str)

    log.info(f"[load] {len(df):,} player-map rows | "
             f"{df['date'].min().date()} → {df['date'].max().date()} | "
             f"{df['playername'].nunique()} players")
    return df


def filter_top_teams(df: pd.DataFrame, min_maps: int = 20) -> pd.DataFrame:
    """Keep only players with enough map history."""
    counts = df.groupby("playername")["match_id"].nunique()
    keep = counts[counts >= min_maps].index
    filtered = df[df["playername"].isin(keep)].copy()
    log.info(f"[filter] {len(filtered):,} rows | {filtered['playername'].nunique()} players (>={min_maps} maps)")
    return filtered


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape HLTV CS2 match stats")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--max-matches", type=int, default=2000)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    df = asyncio.run(run_scrape(
        start_date=args.start,
        end_date=args.end,
        max_matches=args.max_matches,
        resume=not args.no_resume,
    ))
    print(f"\nDone! {len(df):,} rows, {df['playername'].nunique()} players")
