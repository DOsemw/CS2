/**
 * scraper.js
 * ----------
 * Fetches CS2 pro match player stats from HLTV using the hltv npm package.
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
  if (!fs.existsSync(OUT_FILE)) return new Set();
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

  const doneIds = loadDone();
  console.log(`[resume] ${doneIds.size} matches already in CSV`);

  // Write header if file doesn't exist
  const fileExists = fs.existsSync(OUT_FILE) && fs.statSync(OUT_FILE).size > 10;
  const outStream = fs.createWriteStream(OUT_FILE, { flags: fileExists ? "a" : "w" });
  if (!fileExists) {
    outStream.write(CSV_HEADERS.join(",") + "\n");
  }

  let fetched = 0;
  let offset  = 0;
  const PER_PAGE = 100;

  while (fetched < MAX_MATCHES) {
    console.log(`[matches] Fetching page offset=${offset}...`);

    let results;
    try {
      results = await HLTV.getResults({
        startDate: START_DATE.toISOString().slice(0, 10),
        count: PER_PAGE,
        offset,
      });
    } catch (e) {
      console.warn(`[matches] Error fetching results: ${e.message} — retrying in 15s`);
      await sleep(15000);
      continue;
    }

    if (!results || results.length === 0) {
      console.log("[matches] No more results.");
      break;
    }

    for (const result of results) {
      if (fetched >= MAX_MATCHES) break;

      const matchId = String(result.id);
      if (doneIds.has(matchId)) continue;

      // Filter by date
      const matchDate = result.date ? new Date(result.date) : null;
      if (matchDate && matchDate < START_DATE) continue;

      const team1 = result.team1?.name || "";
      const team2 = result.team2?.name || "";
      const dateStr = matchDate ? matchDate.toISOString().slice(0, 10) : "";

      console.log(`[match] ${fetched + 1}/${MAX_MATCHES}: ${team1} vs ${team2} (${matchId})`);

      // Fetch full match details
      let match;
      try {
        match = await HLTV.getMatch({ id: parseInt(matchId) });
        await sleep(2000 + Math.random() * 2000);
      } catch (e) {
        console.warn(`[match] Failed to fetch match ${matchId}: ${e.message}`);
        await sleep(5000);
        continue;
      }

      const event = match.event?.name || result.event?.name || "";
      const maps  = match.maps || [];

      for (let mapIdx = 0; mapIdx < maps.length; mapIdx++) {
        const mapData = maps[mapIdx];
        const mapName = mapData.name || `map${mapIdx + 1}`;

        if (!mapData.statsId && !mapData.result) continue;

        // Map result
        const scoreTeam1 = mapData.result?.team1 ?? NaN;
        const scoreTeam2 = mapData.result?.team2 ?? NaN;
        const team1Won   = scoreTeam1 > scoreTeam2;

        // Fetch map stats
        let mapStats = null;
        if (mapData.statsId) {
          try {
            mapStats = await HLTV.getMatchMapStats({ id: mapData.statsId });
            await sleep(1500 + Math.random() * 1500);
          } catch (e) {
            console.warn(`[mapstats] Failed for statsId ${mapData.statsId}: ${e.message}`);
          }
        }

        const playerStats = mapStats?.playerStats || [];

        if (playerStats.length === 0) {
          // No per-player stats available for this map
          continue;
        }

        for (const ps of playerStats) {
          const playerName = ps.player?.name || ps.playerName || "";
          if (!playerName) continue;

          const isTeam1 = ps.team === 1 || ps.teamId === match.team1?.id;
          const teamName = isTeam1 ? team1 : team2;
          const oppName  = isTeam1 ? team2 : team1;
          const won      = isTeam1 ? (team1Won ? 1 : 0) : (team1Won ? 0 : 1);

          const kills  = ps.kills  ?? ps.K  ?? "";
          const deaths = ps.deaths ?? ps.D  ?? "";
          const hsPct  = ps.hsPercent !== undefined ? ps.hsPercent : (ps.hs ?? "");
          const adr    = ps.adr    ?? ps.ADR ?? "";
          const rating = ps.rating ?? ps.rating2 ?? "";

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
            kills:          kills,
            deaths:         deaths,
            hs_pct:         hsPct,
            adr:            adr,
            rating:         rating,
            map_score_team: isTeam1 ? scoreTeam1 : scoreTeam2,
            map_score_opp:  isTeam1 ? scoreTeam2 : scoreTeam1,
          }) + "\n");
        }
      }

      doneIds.add(matchId);
      fetched++;

      // Polite delay between matches
      await sleep(3000 + Math.random() * 3000);
    }

    offset += PER_PAGE;
    await sleep(5000);
  }

  outStream.end();
  console.log(`\n[done] Finished. ${fetched} matches processed → ${OUT_FILE}`);
}

main().catch((e) => {
  console.error("[fatal]", e);
  process.exit(1);
});
