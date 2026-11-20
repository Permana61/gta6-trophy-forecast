"""GTA VI trophy earn-rate forecast for the Exophase page at the target.

Round 2.  The core idea: Exophase refreshes a game's page on its own
clock, hours apart, and between refreshes the page is frozen -- the
counts shown are the true counts as of the last refresh, not as of the
moment you look.  A case is scored on the page at its target, so the
job is to predict the true tracked-player total and the true earners
of every trophy at the target, and then to read off what the page
will show after its last refresh before that moment.

True players follow a launch curve: from release, the total grows as a
power of (1 + hours since release).  The house's own recording pins
the exponent -- the launch-night page sits at 638 tracked players and
its first refresh, 4.5 hours after release, shows 3,868, a factor of
6.06, which is 5.5 ** 1.06.  A case whose page never refreshes gets
that exponent as a prior; a case whose page does refresh gets its
exponent fitted from the distinct counts the page itself shows.

Earners arrive in two streams.  Newly synced players each bring about
one trophy: the opening mission, finished by everyone within minutes
of starting, plus around 0.03 other trophies, landing mostly on the
earliest trophies -- the ones the launch-night page already shows
earned most.  The first refresh calibrates this: 3,230 fresh players
brought 3,333 trophies, 3,230 of them the opening mission.  Players
already tracked add trophies as they keep playing; over a short
horizon that stream is small next to a refresh jump, and over days it
is exactly what a moving history measures, trophy by trophy, through
the elasticity of earners against players.

So the forecaster has two regimes.  When the history is flat -- the
page never refreshed inside the case -- it applies the refresh jump:
players to the launch curve, earners to the fresh-syncer profile.
When the history moves, it fits log earners on log players per trophy,
shrinks the fit toward a difficulty prior built from tier, points and
description text, and blends in a logit trend as a cross-check.  The
final case lives in the moving regime: two days of reads and a
four-and-a-half-day horizon.

Uncertainty is logit-normal around each median.  The band is tightest
where the refresh structure is pinned down (a flat history whose jump
the launch curve fixes) and widens with horizon, fit residual and
small earner counts.  Medians are pooled so the story chain, the
100%-completion trophy above the platinum, and the all-strands trophy
below every strand final keep their logical order, with a moved
trophy's whole quantile row travelling on the logit scale.

Standard library only, no network, fully deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Power-law exponent for the true tracked-player total on a history
# that has not refreshed yet: 638 -> 3,868 over the 4.5 hours to the
# first refresh is a factor 6.06, and 6.06 = 5.5 ** 1.06, as the
# competition rules record it.
LAUNCH_GAMMA = 1.06
# Non-intro trophies each fresh syncer brings on average: the first
# refresh added 3,230 players and 3,333 trophies, 3,230 of them the
# opening mission, leaving ~103 others: 103 / 3230.
FRESH_OTHER = 0.032
# The fresh-syncer trophies concentrate on the earliest trophies.
# Half the stream follows launch-earner shares over every trophy,
# half follows them over the EARLY_N trophies with the most launch
# earners -- a hedge between the two shapes the launch-night refresh
# is consistent with, both of which beat pure dilution.
EARLY_N = 20


def read_moment(raw):
    if raw is None:
        return None
    stamp = str(raw).strip()
    if stamp.endswith("Z"):
        stamp = stamp[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def hours_from(moment, origin):
    if moment is None or origin is None:
        return 0.0
    span = (moment - origin).total_seconds() / 3600.0
    return span if span > 0.0 else 0.0


def clamp_rate(value):
    if value < 0.01:
        return 0.01
    if value > 99.99:
        return 99.99
    return value


def to_logit(pct):
    clipped = clamp_rate(pct)
    return math.log(clipped / (100.0 - clipped))


def from_logit(value):
    if value > 20.0:
        return 99.99
    if value < -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-value))


def round4(value):
    stepped = Decimal(repr(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP)
    as_float = float(stepped)
    return int(as_float) if float(as_float).is_integer() else as_float


def linfit(xs, ys):
    """Least-squares line through (xs, ys): (slope, intercept, resid_sd)."""
    k = len(xs)
    if k < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    xbar = sum(xs) / k
    ybar = sum(ys) / k
    vxx = sum((x - xbar) ** 2 for x in xs)
    if vxx <= 1e-12:
        return 0.0, ybar, 0.0
    vxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = vxy / vxx
    icept = ybar - slope * xbar
    if k > 2:
        errs = [y - (icept + slope * x) for x, y in zip(xs, ys)]
        spread = sum(e * e for e in errs) / (k - 2)
        sd = math.sqrt(spread) if spread > 0.0 else 0.0
    else:
        sd = 0.0
    return slope, icept, sd


# Story beats, checked in order; the first phrase that matches sets the
# prior elasticity.  Later rules only tighten or loosen what is set here.
STORY_RULES = (
    ("chapter 1", 0.96),
    ("chapter 2", 0.90),
    ("chapter 3", 0.78),
    ("chapter 4", 0.66),
    ("final mission for", 0.55),
)
GRIND_PHRASES = (
    "250 stunts",
    "100 locations",
    "100 vehicles",
    "50 challenges",
    "collectible card",
    "gold medal",
    "20 different races",
    "earn $250,000",
    "spend $100,000",
    "purchase all",
    "register 10",
    "meditate at every",
    "explore all of leonida",
    "8 mission strands",
    "rob 5 bank",
)
EASY_PHRASES = (
    "try each type",
    "complete an arcade",
    "enjoy a cocktail",
    "gone fishing",
    "bowling alley",
    "follow-through",
    "partnership with mass appeal",
    "choices made",
    "second time",
    "what the hell just happened",
)


def prior_elastic(trophy):
    """Prior elasticity of earners with respect to players, from metadata.

    In [0.08, 0.98]: near 1 for things a fresh player does in the first
    hours, near 0 for endgame and heavy grinds.
    """
    blurb = (trophy.get("description") or "").lower()
    label = (trophy.get("name") or "").lower()
    text = blurb + " " + label
    tier = (trophy.get("tier") or "bronze").lower()
    value = {"platinum": 0.12, "gold": 0.28, "silver": 0.48,
             "bronze": 0.68}.get(tier, 0.65)
    for phrase, elastic in STORY_RULES:
        if phrase in text:
            value = elastic
            break
    else:
        if "final mission" in text and "complete the final" in text:
            value = 0.30
        elif "100% completion" in text:
            value = 0.10
        elif "complete" in text and any(
                word in text
                for word in ("mission", "heist", "takedown", "operation")):
            value = 0.72
    if "final mission for" in text:
        value = min(value, 0.55)
    # Heavy grinds and collections: earners accumulate slowly.
    if any(phrase in text for phrase in GRIND_PHRASES):
        value = min(value, 0.38)
    if "complete 8" in text:
        value = min(value, 0.42)
    # Trivial one-off activities a fresh player can do immediately.
    if any(phrase in text for phrase in EASY_PHRASES):
        value = max(value, 0.88)
    if "complete gone fishing" in text:
        value = 0.88
    if "congratulations" in text:
        value = 0.12
    return max(0.08, min(0.98, value))


def quantile_row(median_rate, sigma):
    """Five logit-normal quantiles around a median rate: non-decreasing
    in 0.01..99.99, rounded to 4 decimals."""
    centre = to_logit(median_rate)
    row = []
    floor = None
    for z in Z:
        point = round4(from_logit(centre + z * sigma))
        if point < 0.01:
            point = 0.01
        if point > 99.99:
            point = 99.99
        if floor is not None and point < floor:
            point = floor
        row.append(point)
        floor = point
    return row


# --- True player total at the target ------------------------------------
#
# The page's count is stale between refreshes: it is the true total as
# of the last refresh, not as of the read.  The target's page will show
# the true total as of the last refresh before the target, so the
# forecast needs the TRUE total at (about) the target, not the page's.

def target_players(marks, origin, h_target):
    """marks: [(moment, players)] at distinct player totals, in time
    order.  Returns (players_target, flat), where flat means the page
    never refreshed inside the case."""
    if not marks:
        return 1.0, True
    flat = len(marks) <= 1
    first_n = marks[0][1]
    first_h = hours_from(marks[0][0], origin)
    if flat:
        # No refresh to fit from: the launch curve, anchored at the
        # first read (the pre-release seed).  For launch night this is
        # 638 * (1 + 4.5) ** 1.06 = 3,868 at the 04:30Z target.
        total = first_n * ((1.0 + h_target) / (1.0 + first_h)) ** LAUNCH_GAMMA
        return max(first_n, total), True
    # Moving history: fit the exponent to the page's own totals.
    gamma = 0.70
    if len(marks) >= 3:
        xs = [math.log1p(hours_from(m, origin)) for m, _n in marks]
        ys = [math.log(n) for _m, n in marks]
        slope, _icept, _sd = linfit(xs, ys)
        if 0.05 < slope < 2.0:
            gamma = slope
    gamma = max(0.15, min(1.4, gamma))
    last_n = marks[-1][1]
    last_h = hours_from(marks[-1][0], origin)
    log_target = (math.log(last_n)
                  + gamma * (math.log1p(h_target) - math.log1p(last_h)))
    total = math.exp(log_target)
    if total > 8.0 * last_n:
        total = 8.0 * last_n
    if total < last_n:
        total = last_n
    return total, False


# --- Flat history: the refresh jump -------------------------------------
#
# The page has not refreshed in the case, so every rate on it is the
# launch-night snapshot and the target's page will be the first refresh:
# the player total jumps to the launch curve and the rates dilute by
# that jump, except the opening mission (every fresh syncer finishes
# it) and a thin stream of other early trophies the fresh syncers bring.

def jump_forecast(trophies, latest_rates, last_n, target_n, span_days):
    """Returns {id: (median_rate, sigma)} for the refresh-jump regime."""
    base = []
    for rate in latest_rates:
        if rate is None:
            base.append(None)
            continue
        earners = rate / 100.0 * last_n
        if rate >= 99.99:
            earners = float(last_n)
        base.append(earners)
    # Fresh-syncer profile: the opening mission for everyone, the rest
    # split half by launch-earner shares over all trophies and half by
    # the same shares over the EARLY_N trophies with most earners.
    others = [(i, e) for i, e in enumerate(base)
              if e is not None and trophies[i]["id"] != 2]  # id 2: intro
    total_base = sum(e for _i, e in others)
    ranked = sorted(others, key=lambda pair: (-pair[1], pair[0]))
    early_set = {i for i, _e in ranked[:EARLY_N]}
    early_total = sum(e for i, e in others if i in early_set)
    newcomers = max(0.0, target_n - last_n)
    out = {}
    for idx, trophy in enumerate(trophies):
        rate = latest_rates[idx] if idx < len(latest_rates) else None
        if rate is None:
            rate = 0.5
        e_base = base[idx] if idx < len(base) else None
        if e_base is None:
            e_base = rate / 100.0 * last_n
        # The intro trophy: every tracked player holds it.
        if trophy["id"] == 2 or rate >= 99.99:
            out[trophy["id"]] = (100.0, 0.10)
            continue
        share_wide = (e_base / total_base) if total_base > 0 else 0.0
        share_early = (e_base / early_total) if (idx in early_set
                                                 and early_total > 0) else 0.0
        fresh_rate = FRESH_OTHER * (0.5 * share_wide + 0.5 * share_early)
        e_target = e_base + newcomers * fresh_rate
        if e_target > target_n:
            e_target = target_n
        median = 100.0 * e_target / target_n if target_n > 0 else rate
        # Spread: the jump's size and the fresh profile are the
        # uncertainty; small earner counts widen it further.
        count_w = 0.7 / math.sqrt(e_base + 1.0)
        horizon_w = 0.10 * math.sqrt(span_days)
        sigma = math.sqrt(0.15 ** 2 + 0.18 ** 2
                          + count_w ** 2 + horizon_w ** 2)
        sigma = max(0.20, min(0.60, sigma))
        out[trophy["id"]] = (median, sigma)
    return out


# --- Moving history: elasticity fit -------------------------------------
#
# The history shows refreshes, so each trophy's earners are fitted
# against players (log E on log P) and shrunk toward the difficulty
# prior; a logit trend in log hours is blended in at a quarter weight
# as a cross-check.  This is the regime the final case lives in: two
# days of reads, hundreds of refreshes, a 4.5-day horizon.

def trend_forecast(trophy, reads, last_n, target_n, span_days, h_target):
    """reads: [(logP, logE_adj, rate, log_hours)] for this trophy.
    Returns (median_rate, sigma)."""
    prior = prior_elastic(trophy)
    latest = None
    for _lp, _le, rate, _lh in reversed(reads):
        if rate is not None:
            latest = rate
            break
    if latest is None:
        latest = 0.5
    base_e = latest / 100.0 * last_n
    adj_e = base_e + 0.5 if base_e < 1.0 else base_e
    if latest >= 99.99:
        adj_e = float(last_n)

    elastic = prior
    fit_sd = 0.35
    fitted = False
    span = 0.0
    if reads:
        logs = [r[0] for r in reads]
        span = max(logs) - min(logs)
    if len(reads) >= 3 and span > math.log(1.25):
        xs = [r[0] for r in reads]
        ys = [r[1] for r in reads]
        slope, _icept, sd = linfit(xs, ys)
        if -0.5 < slope < 1.6:
            fitted = True
            fit_sd = sd
            weight = len(reads) / (len(reads) + 3.0)
            elastic = weight * slope + (1.0 - weight) * prior
    elastic = max(0.0, min(1.15, elastic))

    ratio = target_n / last_n if last_n > 0 else 1.0
    target_e = adj_e * (ratio ** elastic)
    if latest >= 99.99:
        target_e = min(target_e, target_n)
    median_rate = (100.0 * target_e / target_n
                   if target_n > 0 else latest)

    if fitted and len(reads) >= 3:
        lh = [r[3] for r in reads]
        lg = [to_logit(r[2]) if r[2] is not None else to_logit(0.5)
              for r in reads]
        slope_l, icept_l, _sd = linfit(lh, lg)
        trend = icept_l + slope_l * math.log1p(h_target)
        trend = 0.5 * trend + 0.5 * to_logit(latest)
        median_rate = from_logit(0.75 * to_logit(median_rate)
                                 + 0.25 * trend)

    count_w = 0.7 / math.sqrt(base_e + 1.0)
    fit_w = fit_sd if fitted else 0.55
    horizon_w = 0.20 * math.sqrt(span_days)
    refresh_w = 0.15
    sigma = math.sqrt(fit_w ** 2 + horizon_w ** 2
                      + count_w ** 2 + refresh_w ** 2)
    sigma = max(0.28, min(1.35, sigma))
    if base_e < 10.0:
        sigma = min(1.45, sigma + 0.12)
    return median_rate, sigma


def trophy_reads(idx, clean, origin):
    """Per-trophy read list for the moving regime, collapsing repeats
    of the same player total into one entry (the latest rate wins)."""
    rows = []
    prev_logp = None
    for moment, players, rates in clean:
        if idx >= len(rates):
            continue
        rate = rates[idx]
        if rate is None:
            continue
        earners = rate / 100.0 * players
        adj = earners + 0.5 if earners < 1.0 else earners
        if rate >= 99.99:
            adj = players
        logp = math.log(players)
        entry = (logp, math.log(adj), rate,
                 math.log1p(hours_from(moment, origin)))
        if prev_logp is not None and abs(logp - prev_logp) < 1e-9:
            rows[-1] = entry
        else:
            rows.append(entry)
        prev_logp = logp
    return rows


# --- Logical monotonicity (PAVA pooling) --------------------------------
#
# The medians must respect: intro > Chapter 1 > Chapter 2 > Chapter 3 >
# Chapter 4 >= final mission; 100%-completion >= platinum; all-8-strands
# <= the least-completed strand final.  Pooling only removes logically
# impossible orderings; a moved trophy's whole quantile row travels with
# its median on the logit scale, so spreads are preserved.

CHAIN = (2, 3, 4, 5, 6, 7)
STRANDS = (22, 23, 24, 26, 27)


def row_logit(row):
    return to_logit(row["q"][2])


def move_row(row, delta):
    shifted = []
    floor = None
    for point in row["q"]:
        moved = round4(from_logit(to_logit(point) + delta))
        if moved < 0.01:
            moved = 0.01
        if moved > 99.99:
            moved = 99.99
        if floor is not None and moved < floor:
            moved = floor
        shifted.append(moved)
        floor = moved
    row["q"] = shifted


def set_row_logit(row, target):
    move_row(row, target - row_logit(row))


def pool_decreasing(values):
    """Pool-adjacent-violators for a non-increasing sequence."""
    blocks = []
    for value in values:
        blocks.append((value, 1))
        while (len(blocks) >= 2
               and blocks[-2][0] / blocks[-2][1]
               < blocks[-1][0] / blocks[-1][1]):
            merged_total = blocks[-2][0] + blocks[-1][0]
            merged_n = blocks[-2][1] + blocks[-1][1]
            blocks[-2:] = [(merged_total, merged_n)]
    pooled = []
    for total, count in blocks:
        pooled.extend([total / count] * count)
    return pooled


def enforce_order(rows):
    by_id = {row["id"]: row for row in rows}
    chain = [by_id[tid] for tid in CHAIN if tid in by_id]
    if len(chain) >= 2:
        pooled = pool_decreasing([row_logit(row) for row in chain])
        for row, target in zip(chain, pooled):
            if target != row_logit(row):
                set_row_logit(row, target)
    first = by_id.get(1)
    full = by_id.get(14)
    if first is not None and full is not None:
        if row_logit(full) < row_logit(first):
            pooled = (row_logit(first) + row_logit(full)) / 2.0
            set_row_logit(first, pooled)
            set_row_logit(full, pooled)
    all_strands = by_id.get(25)
    finals = [by_id[tid] for tid in STRANDS if tid in by_id]
    if all_strands is not None and finals:
        violators = [row for row in finals
                     if row_logit(row) < row_logit(all_strands)]
        if violators:
            pool = [all_strands] + violators
            pooled = sum(row_logit(row) for row in pool) / len(pool)
            for row in pool:
                set_row_logit(row, pooled)


def run_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    span_days = float(case.get("horizon_minutes", 0)) / 1440.0
    target = read_moment(case.get("target"))
    release_text = (case.get("game") or {}).get("release_date")
    origin = read_moment(release_text + "T00:00:00Z") if release_text else None
    if origin is None:
        origin = read_moment(history[0].get("at"))
    h_target = hours_from(target, origin)

    clean = []
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates")
        moment = read_moment(read.get("at")) or read_moment(read.get("slot"))
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        if not isinstance(rates, list):
            continue
        clean.append((moment, float(players), rates))
    if not clean:
        raise ValueError("case history has no usable read")
    clean.sort(key=lambda item: item[0] or datetime.min.replace(tzinfo=timezone.utc))
    last_n = clean[-1][1]

    marks = []
    seen = set()
    for moment, players, _rates in clean:
        key = round(players)
        if key not in seen:
            seen.add(key)
            marks.append((moment, players))

    target_n, flat = target_players(marks, origin, h_target)
    latest_rates = clean[-1][2]

    out_rows = []
    if flat:
        medians = jump_forecast(trophies, latest_rates, last_n, target_n,
                                span_days)
        for trophy in trophies:
            median, sigma = medians[trophy["id"]]
            out_rows.append({"id": trophy["id"],
                             "q": quantile_row(median, sigma)})
    else:
        for idx, trophy in enumerate(trophies):
            reads = trophy_reads(idx, clean, origin)
            median, sigma = trend_forecast(trophy, reads, last_n, target_n,
                                           span_days, h_target)
            out_rows.append({"id": trophy["id"],
                             "q": quantile_row(median, sigma)})
    enforce_order(out_rows)
    out_rows.sort(key=lambda row: row["id"])
    return {"trophies": out_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for infile in sorted(INPUT_DIR.glob("*.in")):
        payload = json.loads(infile.read_text(encoding="utf-8"))
        result = run_case(payload)
        blob = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (infile.stem + ".out")).write_text(blob, encoding="utf-8")


if __name__ == "__main__":
    main()
