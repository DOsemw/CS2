/**
 * scraper.js
 * ----------
 * Fetches CS2 pro match player stats from HLTV.
 * Outputs data/hltv_raw.csv in the same schema the Python model expects.
 *
 * Usage:
 *   npm install hltv
 *   node scraper.js
 *   node scraper.js --matches 500 --start 2024-01-01
 */

const { HLTV } = require("hltv");
const fs = require("fs");
const path = require("path");

// ── Args ──────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const getArg = (name, def) => {
  const i = args.indexOf(name);
  return i !== -1 ? args[i + 1] : def;
};

const MAX_MATCHES = parseInt(getArg("--matches", "500"));
const START_DATE  = new Date(getArg("--start", "2024-01-01"));
const OUT_FILE    = path.join(__dirname, "data", "hltv_raw.csv");
const DELAY_MS    = 8000; // delay between page requests to avoid Cloudflare

// ── Setup ─────────────────────────────────────────────────────────────────────

if (!fs.existsSync(path.join(__dirname, "data"))) {
  fs.mkdirSync(path.join(__dirname, "data"));
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const CSV_HEADERS = [
  "match_id", "map_name", "map_number", "date", "event",
  "team", "opponent", "result", "playername",
  "kills", "deaths", "hs_pct", "adr", "rating",
  "map_score_team", "map_score_opp",
];

function toCSVRow(obj) {
  return CSV_HEADERS.map((h) => {
    const v = obj[h] ?? "";
    return String(v).includes(",") ? `"${v}"` : v;
  }).join(",");
}

// ── Resume support ────────────────────────────────────────────────────────────

function loadDone() {
  if (!fs.existsSync(OUT_FILE) || fs.statSync(OUT_FILE).size < 10) return new Set();
  const lines = fs.readFileSync(OUT_FILE, "utf8").trim().split("\n");
  const done = new Set();
  for (const line of lines.slice(1)) {
    const id = line.split(",")[0];
    if (id) done.add(id);
  }
  return done;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`[hltv] Fetching up to ${MAX_MATCHES} matches since ${START_DATE.toISOString().slice(0, 10)}`);
  console.log(`[hltv] Using ${DELAY_MS}ms delay between requests to avoid Cloudflare`);

  const doneIds = loadDone();
  console.log(`[resume] ${doneIds.size} matches already in CSV`);

  const fileExists = fs.existsSync(OUT_FILE) && fs.statSync(OUT_FILE).size > 10;
  const outStream = fs.createWriteStream(OUT_FILE, { flags: fileExists ? "a" : "w" });
  if (!fileExists) {
    outStream.write(CSV_HEADERS.join(",") + "\n");
  }

  let fetched = 0;
  let offset  = 0;
  const PER_PAGE = 100;

  while (fetched < MAX_MATCHES) {
    console.log(`\n[matches] Fetching page offset=${offset}...`);

    let results;
    try {
      results = await HLTV.getResults({
        startDate: START_DATE.toISOString().slice(0, 10),
        count: PER_PAGE,
        offset,
        delayBetweenPageRequests: DELAY_MS,
      });
    } catch (e) {
      console.warn(`[matches] Error: ${e.message} — waiting 30s then retrying...`);
      await sleep(30000);
      continue;
    }

    if (!results || results.length === 0) {
      console.log("[matches] No more results — done.");
      break;
    }

    console.log(`[matches] Got ${results.length} matches`);

    for (const result of results) {
      if (fetched >= MAX_MATCHES) break;

      const matchId = String(result.id);
      if (doneIds.has(matchId)) {
        console.log(`[skip] ${matchId} already done`);
        continue;
      }

      const matchDate = result.date ? new Date(result.date) : null;
      if (matchDate && matchDate < START_DATE) continue;

      const team1   = result.team1?.name || "";
      const team2   = result.team2?.name || "";
      const dateStr = matchDate ? matchDate.toISOString().slice(0, 10) : "";

      console.log(`[match] ${fetched + 1}/${MAX_MATCHES}: ${team1} vs ${team2} (id: ${matchId})`);

      // Fetch full match details
      let match;
      try {
        match = await HLTV.getMatch({ id: parseInt(matchId) });
      } catch (e) {
        console.warn(`[match] Failed ${matchId}: ${e.message} — skipping`);
        await sleep(10000);
        continue;
      }

      const event = match.event?.name || result.event?.name || "";
      const maps  = match.maps || [];

      for (let mapIdx = 0; mapIdx < maps.length; mapIdx++) {
        const mapData = maps[mapIdx];
        const mapName = mapData.name || `map${mapIdx + 1}`;
        if (!mapData.statsId) continue;

        const scoreTeam1 = mapData.result?.team1 ?? NaN;
        const scoreTeam2 = mapData.result?.team2 ?? NaN;
        const team1Won   = scoreTeam1 > scoreTeam2;

        // Fetch per-player map stats
        let mapStats;
        try {
          mapStats = await HLTV.getMatchMapStats({ id: mapData.statsId });
          await sleep(DELAY_MS);
        } catch (e) {
          console.warn(`[mapstats] Failed statsId ${mapData.statsId}: ${e.message}`);
          await sleep(10000);
          continue;
        }

        const playerStats = mapStats?.playerStats || [];
        if (playerStats.length === 0) continue;

        for (const ps of playerStats) {
          const playerName = ps.player?.name || ps.playerName || "";
          if (!playerName) continue;

          const isTeam1 = ps.team === 1;
          const teamName = isTeam1 ? team1 : team2;
          const oppName  = isTeam1 ? team2 : team1;
          const won      = isTeam1 ? (team1Won ? 1 : 0) : (team1Won ? 0 : 1);

          outStream.write(toCSVRow({
            match_id:       matchId,
            map_name:       mapName,
            map_number:     mapIdx + 1,
            date:           dateStr,
            event:          event,
            team:           teamName,
            opponent:       oppName,
            result:         won,
            playername:     playerName,
            kills:          ps.kills  ?? "",
            deaths:         ps.deaths ?? "",
            hs_pct:         ps.hsPercent ?? "",
            adr:            ps.adr    ?? "",
            rating:         ps.rating ?? "",
            map_score_team: isTeam1 ? scoreTeam1 : scoreTeam2,
            map_score_opp:  isTeam1 ? scoreTeam2 : scoreTeam1,
          }) + "\n");
        }

        console.log(`  [map] ${mapName} — ${playerStats.length} players scraped`);
      }

      doneIds.add(matchId);
      fetched++;

      // Delay between matches
      await sleep(DELAY_MS + Math.random() * 3000);
    }

    offset += PER_PAGE;
    await sleep(DELAY_MS);
  }

  outStream.end();
  console.log(`\n[done] ${fetched} matches processed → ${OUT_FILE}`);
}

main().catch((e) => {
  console.error("[fatal]", e);
  process.exit(1);
});
