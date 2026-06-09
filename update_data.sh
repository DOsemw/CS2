#!/bin/bash
# ============================================================
# CS2 Data Update Script
# Run every Monday to keep your dataset fresh
# Usage: bash update_data.sh
# ============================================================

set -e
cd "$(dirname "$0")"

echo "============================================"
echo " CS2 Data Update — $(date '+%Y-%m-%d %H:%M')"
echo "============================================"

# Step 1: Scrape fresh HLTV data
echo ""
echo "[1/4] Scraping HLTV player stats..."
python3 data_ingestion.py

# Step 2: Fix column mappings
echo ""
echo "[2/4] Fixing column mappings..."
python3 fix_csv.py

# Step 3: Fix deaths and ADR
echo ""
echo "[3/4] Fixing deaths and ADR..."
python3 fix_csv2.py

# Step 4: Commit and push to GitHub
echo ""
echo "[4/4] Committing and pushing to GitHub..."
git add data/hltv_raw.csv
git commit -m "data update $(date '+%Y-%m-%d')"
git push

echo ""
echo "============================================"
echo " Done! Railway will auto-redeploy."
echo " Check: https://cs2-production-cfcf.up.railway.app/"
echo "============================================"
