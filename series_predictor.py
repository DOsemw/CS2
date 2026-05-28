"""
series_predictor.py
-------------------
Converts moneyline odds → win probabilities and projects
per-map predictions to Bo1 / Bo3 series totals.

CS2 formats:
  Bo1 — single map, no series projection needed
  Bo3 — first to 2 maps. Expected maps: 2.0 to 3.0
  Bo5 — rare (used at some majors). First to 3.
"""

import math


# ── Moneyline → probability ───────────────────────────────────────────────────

def american_to_implied(ml: int) -> float:
    """Convert American moneyline to implied probability (with vig)."""
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


def vig_adjusted_probs(ml1: int, ml2: int) -> tuple[float, float]:
    """
    Remove the bookmaker's vig and return fair win probabilities.
    Returns (prob_team1, prob_team2) summing to 1.0.
    """
    p1 = american_to_implied(ml1)
    p2 = american_to_implied(ml2)
    total = p1 + p2
    return p1 / total, p2 / total


# ── Expected maps in a series ─────────────────────────────────────────────────

def expected_maps(series_format: str, win_prob: float) -> float:
    """
    Calculate expected number of maps played given a win probability.

    Bo1:  always 1 map
    Bo3:  E[maps] = 2 + 2*p*(1-p)  where p = win prob of favorite per map
    Bo5:  E[maps] = 3 + 6p²(1-p) + 6p(1-p)²  (simplified)
    """
    p = win_prob
    q = 1 - p
    fmt = series_format.lower().replace("-", "").replace(" ", "")

    if fmt == "bo1":
        return 1.0

    elif fmt == "bo3":
        # P(series ends in 2) = p² + q²
        # P(series goes to 3) = 2pq
        p2 = p**2 + q**2
        p3 = 2 * p * q
        return 2 * p2 + 3 * p3

    elif fmt == "bo5":
        # P(ends in 3) = p³ + q³
        # P(ends in 4) = C(3,1)*p³*q + C(3,1)*q³*p  = 3p³q + 3q³p
        # P(ends in 5) = 1 - above two
        p3 = p**3 + q**3
        p4 = 3 * (p**3 * q + q**3 * p)
        p5 = 1 - p3 - p4
        return 3 * p3 + 4 * p4 + 5 * p5

    return 2.0  # fallback


def series_win_prob(win_prob: float, series_format: str) -> float:
    """
    Probability of winning the series given per-map win probability.
    """
    p = win_prob
    q = 1 - p
    fmt = series_format.lower().replace("-", "").replace(" ", "")

    if fmt == "bo1":
        return p

    elif fmt == "bo3":
        # Win in 2: p²
        # Win in 3: C(2,1) * p² * q = 2p²q
        return p**2 + 2 * p**2 * q

    elif fmt == "bo5":
        # Win in 3: p³
        # Win in 4: 3p³q
        # Win in 5: 6p³q²
        return p**3 + 3 * p**3 * q + 6 * p**3 * q**2

    return p


# ── Series projection builder ─────────────────────────────────────────────────

def build_series_projection(per_map: dict, win_prob: float, series_format: str) -> dict:
    """
    Scale per-map predictions to series totals.

    per_map: {"kills": {"per_map": X, "low": Y, "high": Z, "mae": M}, ...}
    Returns same structure with series_total / series_low / series_high added.
    """
    exp_maps = expected_maps(series_format, win_prob)
    s_win_prob = series_win_prob(win_prob, series_format)

    result = {
        "series_format": series_format,
        "expected_maps": round(exp_maps, 2),
        "series_win_prob": round(s_win_prob, 3),
    }

    for stat, vals in per_map.items():
        result[stat] = {
            "per_map": vals["per_map"],
            "series_total": round(vals["per_map"] * exp_maps, 1),
            "series_low": round(vals["low"] * exp_maps, 1),
            "series_high": round(vals["high"] * exp_maps, 1),
            "mae": round(vals["mae"] * exp_maps, 2),
        }

    return result
