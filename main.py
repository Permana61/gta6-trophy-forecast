"""Forecast Exophase earn rates for GTA VI trophies at a future target.

Model, in one paragraph: tracked players P(t) grow roughly like a power
law in time since release, and each trophy's earners E(t) grow like a
power law in P(t).  The exponent beta is the trophy's elasticity: easy
early trophies keep pace with new players (beta near 1, rate flat or
rising) while hard late trophies barely gain earners (beta near 0, rate
diluted as P explodes).  Both exponents are fitted from the case's own
history when it actually moves, and shrunk toward a difficulty prior
built from the trophy's tier, points and description when it does not.
Quantiles are logit-normal around the projected median, with a spread
that combines fit residuals, horizon length, small-count noise and an
extra term when the history is flat (Exophase refresh jumps).

After the medians are computed, the story-chain medians are pooled to
be non-increasing (#2 intro > #3 Ch1 > #4 Ch2 > #5 Ch3 > #6 Ch4 >= #7
final), #14 (100%) is pooled up to at least #1 (platinum), and #25
(8 stranger strands) is pooled down to at most the lowest strand
final, all on the logit scale with each trophy's quantiles shifting
along with its median so spreads are preserved.

Standard library only, no network, fully deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZVALS = (-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722)
RELEASE_HOUR = 0.0


def parse_time(text):
    if text is None:
        return None
    txt = str(text).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def hours_since(moment, origin):
    if moment is None or origin is None:
        return 0.0
    delta = (moment - origin).total_seconds() / 3600.0
    return delta if delta > 0.0 else 0.0


def clamp_pct(value):
    if value < 0.01:
        return 0.01
    if value > 99.99:
        return 99.99
    return value


def logit_of(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(logit_value):
    if logit_value > 20.0:
        return 99.99
    if logit_value < -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-logit_value))


def round_four(value):
    quantised = Decimal(repr(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP)
    number = float(quantised)
    return int(number) if float(number).is_integer() else number


def ols(xs, ys):
    """Least squares slope/intercept; returns (slope, intercept, resid_sd)."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        var = sum(r * r for r in resid) / (n - 2)
        sd = math.sqrt(var) if var > 0.0 else 0.0
    else:
        sd = 0.0
    return slope, intercept, sd


def difficulty_prior(trophy):
    """Beta prior (elasticity of earners w.r.t. players) from metadata.

    Returns a value in [0.08, 0.98]: near 1 for things a new player
    does in the first hours, near 0 for endgame / heavy grinds.
    """
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    text = desc + " " + name
    tier = (trophy.get("tier") or "bronze").lower()
    base = {"platinum": 0.12, "gold": 0.28, "silver": 0.48,
            "bronze": 0.68}.get(tier, 0.65)
    beta = base
    # Story chapters: earlier chapters are reached by far more players.
    if "chapter 1" in text:
        beta = 0.96
    elif "chapter 2" in text:
        beta = 0.90
    elif "chapter 3" in text:
        beta = 0.78
    elif "chapter 4" in text:
        beta = 0.66
    elif "final mission for" in text:
        beta = 0.55
    elif "final mission" in text and "complete the final" in text:
        beta = 0.30
    elif "100% completion" in text or "100% completion" in desc:
        beta = 0.10
    # Named story / side missions (one specific mission each).
    elif "complete" in text and ("mission" in text or "heist" in text
                                 or "takedown" in text or "operation" in text):
        beta = 0.72 if "final mission for" not in text else 0.55
    if "final mission for" in text:
        beta = min(beta, 0.55)
    # Heavy grinds and collections: earners accumulate slowly.
    grind_words = ("250 stunts", "100 locations", "100 vehicles",
                   "50 challenges", "collectible card", "gold medal",
                   "20 different races", "earn $250,000",
                   "spend $100,000", "purchase all", "register 10",
                   "meditate at every", "explore all of leonida",
                   "8 mission strands", "rob 5 bank")
    if any(w in text for w in grind_words):
        beta = min(beta, 0.38)
    if "complete 8" in text:
        beta = min(beta, 0.42)
    # Trivial one-off activities a fresh player can do immediately.
    easy_words = ("try each type", "complete an arcade", "enjoy a cocktail",
                  "gone fishing", "bowling alley", "follow-through",
                  "partnership with mass appeal", "choices made",
                  "second time", "what the hell just happened")
    if any(w in text for w in easy_words):
        beta = max(beta, 0.88)
    if "complete gone fishing" in text:
        beta = 0.88
    if "congratulations" in text:
        beta = 0.12
    return max(0.08, min(0.98, beta))


def forecast_one(trophy, idx, reads, p_last, p_target, alpha,
                 horizon_days, flat_history, logp_range):
    """reads: list of (logP, logE_adj, rate, log_hours) for this trophy."""
    beta0 = difficulty_prior(trophy)
    last_rate = None
    for _lp, _le, rate, _lh in reversed(reads):
        if rate is not None:
            last_rate = rate
            break
    if last_rate is None:
        last_rate = 0.5
    earners_last = last_rate / 100.0 * p_last
    # Continuity-corrected earners: a printed 0% still means <0.5 earners.
    earners_adj = earners_last + 0.5 if earners_last < 1.0 else earners_last
    if last_rate >= 99.99:
        earners_adj = float(p_last)

    beta = beta0
    resid_sd = 0.35
    identifiable = False
    if len(reads) >= 3 and logp_range > math.log(1.25):
        xs = [r[0] for r in reads]
        ys = [r[1] for r in reads]
        slope, _icept, sd = ols(xs, ys)
        if -0.5 < slope < 1.6:
            identifiable = True
            resid_sd = sd
            # Shrink the fitted slope toward the prior; three effective
            # observations worth of prior weight.
            weight = len(reads) / (len(reads) + 3.0)
            beta = weight * slope + (1.0 - weight) * beta0
    beta = max(0.0, min(1.15, beta))

    growth = p_target / p_last if p_last > 0 else 1.0
    earners_target = earners_adj * (growth ** beta)
    if last_rate >= 99.99:
        earners_target = min(earners_target, p_target)
    rate_med = 100.0 * earners_target / p_target if p_target > 0 else last_rate

    # Cross-check: direct logit trend in log(1+hours), shrunk halfway to
    # the last logit, blended in at 25% when the history moves.
    if identifiable and len(reads) >= 3:
        lh = [r[3] for r in reads]
        lg = [logit_of(r[2]) if r[2] is not None else logit_of(0.5)
              for r in reads]
        slope_l, icept_l, _sd = ols(lh, lg)
        lh_target = math.log1p(hours_since_target)
        trend_logit = icept_l + slope_l * lh_target
        last_logit = logit_of(last_rate)
        trend_logit = 0.5 * trend_logit + 0.5 * last_logit
        med_logit = logit_of(rate_med)
        blended = 0.75 * med_logit + 0.25 * trend_logit
        rate_med = inv_logit(blended)

    # Spread on the logit scale.
    count_term = 0.7 / math.sqrt(earners_last + 1.0)
    fit_term = resid_sd if identifiable else 0.55
    horizon_term = 0.20 * math.sqrt(horizon_days)
    refresh_term = 0.60 if flat_history else 0.15
    sigma = math.sqrt(fit_term ** 2 + horizon_term ** 2
                      + count_term ** 2 + refresh_term ** 2)
    sigma = max(0.28, min(1.35, sigma))
    # Very rare trophies (a handful of earners) deserve extra width.
    if earners_last < 10.0:
        sigma = min(1.45, sigma + 0.12)

    centre = logit_of(rate_med)
    quantiles = []
    previous = None
    for z in ZVALS:
        value = round_four(inv_logit(centre + z * sigma))
        if value < 0.01:
            value = 0.01
        if value > 99.99:
            value = 99.99
        if previous is not None and value < previous:
            value = previous
        quantiles.append(value)
        previous = value
    return {"id": trophy["id"], "q": quantiles}


# Set inside forecast_case so forecast_one can use the target hour.
hours_since_target = 0.0


# --- Post-forecast monotonicity (revisi A2) ---
#
# NOTE: a wide research-prior guardrail (median clamped into 2x the
# research ranges for horizons >= 3 days) was implemented and tested
# in this same revision, then REVERTED: on the synthetic long-horizon
# cases it bound on 5-6 trophies (b00: #1,#3,#4,#5,#6,#7; c00:
# #3,#4,#5,#6,#7) and worsened the pinball loss by +0.30 / +0.24,
# because the synthetic world is dilution-only (true Chapter 1 rate
# ~0.1% at target) while the research prior expects ~36%. The
# guardrail fought the data instead of catching wild extrapolation.
# See work/model-notes.md ("Revisi A2") for the numbers. Monotonic
# pooling below is kept: it never fights the data, it only removes
# logically impossible orderings.

# Logical order the medians must respect (research.md §6/§8):
# intro > Chapter 1 > Chapter 2 > Chapter 3 > Chapter 4 >= final mission.
STORY_CHAIN = (2, 3, 4, 5, 6, 7)
# Strand finals; #25 (complete all 8 strands) can never exceed the
# least-completed strand.
STRAND_FINALS = (22, 23, 24, 26, 27)


def median_logit(row):
    return logit_of(row["q"][2])


def shift_row(row, delta_logit):
    """Shift a whole quantile row by delta on the logit scale, so the
    row's spread is preserved; keep the row non-decreasing and inside
    0.01..99.99."""
    shifted = []
    previous = None
    for value in row["q"]:
        new_value = round_four(inv_logit(logit_of(value) + delta_logit))
        if new_value < 0.01:
            new_value = 0.01
        if new_value > 99.99:
            new_value = 99.99
        if previous is not None and new_value < previous:
            new_value = previous
        shifted.append(new_value)
        previous = new_value
    row["q"] = shifted


def pool_to(row, target_logit):
    """Move a row's median to target_logit, quantiles travelling with it."""
    shift_row(row, target_logit - median_logit(row))


def isotonic_decreasing(values):
    """Pool-adjacent-violators for a non-increasing sequence (equal
    weights): returns the pooled values, same length, non-increasing."""
    blocks = []  # each [sum, count]
    for value in values:
        blocks.append([value, 1])
        while (len(blocks) >= 2
               and blocks[-2][0] / blocks[-2][1]
               < blocks[-1][0] / blocks[-1][1]):
            total = blocks[-2][0] + blocks[-1][0]
            count = blocks[-2][1] + blocks[-1][1]
            blocks[-2:] = [[total, count]]
    pooled = []
    for total, count in blocks:
        pooled.extend([total / count] * count)
    return pooled


def apply_monotonic(rows):
    """Enforce logical monotonicity on the medians (logit scale):
    story chain non-increasing, #14 >= #1, #25 <= min strand final.
    Each moved trophy's quantiles travel with its median (same logit
    shift), so spreads are preserved and every row stays non-decreasing
    in 0.01..99.99. Pooling only removes logically impossible orderings;
    it never moves a median toward any prior value.
    """
    by_id = {row["id"]: row for row in rows}
    chain = [by_id[tid] for tid in STORY_CHAIN if tid in by_id]
    if len(chain) >= 2:
        pooled = isotonic_decreasing([median_logit(row) for row in chain])
        for row, target in zip(chain, pooled):
            if target != median_logit(row):
                pool_to(row, target)
    row1 = by_id.get(1)
    row14 = by_id.get(14)
    if row1 is not None and row14 is not None:
        if median_logit(row14) < median_logit(row1):
            pooled = (median_logit(row1) + median_logit(row14)) / 2.0
            pool_to(row1, pooled)
            pool_to(row14, pooled)
    row25 = by_id.get(25)
    strands = [by_id[tid] for tid in STRAND_FINALS if tid in by_id]
    if row25 is not None and strands:
        violators = [row for row in strands
                     if median_logit(row) < median_logit(row25)]
        if violators:
            pool = [row25] + violators
            pooled = sum(median_logit(row) for row in pool) / len(pool)
            for row in pool:
                pool_to(row, pooled)


def forecast_case(case):
    global hours_since_target
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    target = parse_time(case.get("target"))
    checkpoint = parse_time(case.get("at"))
    release_text = (case.get("game") or {}).get("release_date")
    origin = parse_time(release_text + "T00:00:00Z") if release_text else None
    if origin is None:
        origin = parse_time(history[0].get("at"))
    hours_since_target = hours_since(target, origin)

    # Clean reads: (time, players, rates) with sane player counts.
    clean = []
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates")
        moment = parse_time(read.get("at")) or parse_time(read.get("slot"))
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        if not isinstance(rates, list):
            continue
        clean.append((moment, float(players), rates))
    if not clean:
        raise ValueError("case history has no usable read")
    clean.sort(key=lambda item: item[0] or datetime.min.replace(tzinfo=timezone.utc))
    p_last = clean[-1][1]
    last_moment = clean[-1][0]

    # Player growth: log P against log(1 + hours since release), anchored
    # at the last read so a level error cannot shift the forecast.
    distinct_p = []
    seen = set()
    for _moment, players, _rates in clean:
        key = round(players)
        if key not in seen:
            seen.add(key)
            distinct_p.append((_moment, players))
    flat_history = len(distinct_p) <= 1
    alpha = 0.70
    if len(distinct_p) >= 3:
        xs = [math.log1p(hours_since(m, origin)) for m, _p in distinct_p]
        ys = [math.log(p) for _m, p in distinct_p]
        slope, _icept, _sd = ols(xs, ys)
        if 0.05 < slope < 2.0:
            alpha = slope
    alpha = max(0.15, min(1.4, alpha))
    logp_target = (math.log(p_last)
                   + alpha * (math.log1p(hours_since_target)
                              - math.log1p(hours_since(last_moment, origin))))
    p_target = math.exp(logp_target)
    # Sanity caps: no more than 8x the last count, no shrinkage.
    if p_target > 8.0 * p_last:
        p_target = 8.0 * p_last
    if p_target < p_last:
        p_target = p_last
    logp_range = 0.0
    if len(distinct_p) >= 2:
        logs = [math.log(p) for _m, p in distinct_p]
        logp_range = max(logs) - min(logs)

    rows = []
    for idx, trophy in enumerate(trophies):
        reads = []
        prev_logp = None
        for moment, players, rates in clean:
            if idx >= len(rates):
                continue
            rate = rates[idx]
            if rate is None:
                continue
            earners = rate / 100.0 * players
            earners_adj = earners + 0.5 if earners < 1.0 else earners
            if rate >= 99.99:
                earners_adj = players
            logp = math.log(players)
            # Keep one point per distinct player count for the fit.
            if prev_logp is not None and abs(logp - prev_logp) < 1e-9:
                reads[-1] = (logp, math.log(earners_adj), rate,
                             math.log1p(hours_since(moment, origin)))
            else:
                reads.append((logp, math.log(earners_adj), rate,
                              math.log1p(hours_since(moment, origin))))
            prev_logp = logp
        rows.append(forecast_one(trophy, idx, reads, p_last, p_target,
                                 alpha, horizon_days, flat_history,
                                 logp_range))
    apply_monotonic(rows)
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
