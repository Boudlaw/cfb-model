#!/usr/bin/env python3
"""
Phase 2 — play-level features: success rate and EPA.

WHY THIS FILE IS SEPARATE FROM EVERYTHING ELSE
----------------------------------------------
Every number downstream is a function of two per-play quantities: `success` (0/1)
and `epa` (float). If the definitions here are wrong, every rating, every backtest
metric and every prediction is wrong in a way that looks completely plausible.
So this file is pure functions over plain dicts/DataFrames with no network, no
model and no state, and test_model_offline.py hand-checks each rule against
worked examples.

THE DEFINITIONS
---------------
Success (Bill Connelly's, the first of the "Five Factors", also the anchor of SP+):

    1st down : gain >= 50% of distance to go
    2nd down : gain >= 70% of distance to go
    3rd/4th  : gain >= 100% of distance to go

Success rate = successful plays / total plays. FBS average is ~40%, and that is
the correctness gate in `sanity_report()` — if a fresh ingest does not land near
40% something upstream is broken and nothing downstream should be trusted.

Why success rate and not yards per play: The Power Rank's work found early-season
success rate correlates strongly with late-season success rate, while
*explosiveness* (yards on successful plays) has essentially zero early-to-late
correlation. Yards per play blends the two and so inherits the noise. Success
rate isolates the repeatable half. Its blind spot is magnitude — a 4-yard gain on
3rd-and-4 scores identically to a 70-yard touchdown — which is exactly why EPA is
carried alongside it rather than instead of it.

Garbage time (Connelly's thresholds) — plays are dropped when the score margin
exceeds:

    Q1: 43    Q2: 37    Q3: 27    Q4: 22

Blowout snaps are real football but they are not evidence about team strength:
the winning side is running clock against backups. Leaving them in inflates the
ratings of teams that beat bad opponents badly, which is the single most common
way a homegrown rating ends up over-ranking one-sided schedules.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Play types that represent a scrimmage snap whose yardage is a measure of
# offensive efficiency. Everything else (kicks, timeouts, administrative rows,
# penalty-only rows) is excluded, because a punt is a decision, not an attempt.
#
# Kept deliberately as a whitelist rather than a blacklist: an unfamiliar play
# type from a future CFBD change should be *dropped and reported*, never silently
# folded into the denominator. drop_report() surfaces what was excluded.
SCRIMMAGE_PLAY_TYPES = frozenset(
    {
        "Rush",
        "Rushing Touchdown",
        "Pass",
        "Pass Reception",
        "Pass Completion",
        "Pass Incompletion",
        "Passing Touchdown",
        "Sack",
        "Interception",
        "Interception Return",
        "Interception Return Touchdown",
        "Pass Interception",
        "Pass Interception Return",
        "Fumble Recovery (Own)",
        "Fumble Recovery (Opponent)",
        "Fumble Return Touchdown",
        "Fumble",
        "Safety",
        "Rush Touchdown",
    }
)

# Plays on which the offense surrendered possession.
#
# THIS IS A SIGN FIX, NOT A TUNING CHOICE. On a turnover, CFBD's `yards_gained`
# is the DEFENSE's return yardage. The raw Connelly rule reads that large
# positive number and scores the play a success for the offense that just threw
# the pick. Measured on the 2022-2025 pull before the fix:
#
#     Interception Return Touchdown   success 0.957   mean EPA -8.02
#     Fumble Return Touchdown         success 0.594   mean EPA -6.42
#     Fumble Recovery (Opponent)      success 0.397   mean EPA -1.62
#     Pass Interception Return        success 0.225   mean EPA -1.34
#
# 3.4% of kept plays had the sign inverted: a defense returning a pick 65 yards
# for a touchdown was being credited to the opposing offense's success rate.
#
# `success` is forced to 0 by definition — surrendering the ball is not a
# successful play at any down and distance. EPA is left alone; it was already
# correctly negative, which is what made the inconsistency visible.
#
# "Fumble Recovery (Own)" is deliberately NOT here. The offense keeps the ball,
# so the ordinary yardage rule applies.
TURNOVER_PLAY_TYPES = frozenset(
    {
        "Interception",
        "Interception Return",
        "Interception Return Touchdown",
        "Pass Interception",
        "Pass Interception Return",
        "Fumble Recovery (Opponent)",
        "Fumble Return Touchdown",
    }
)

# Anything matching these is definitively not a scrimmage snap. Used only to
# classify *why* a play was dropped, so drop_report() can distinguish "expected
# exclusion" from "play type we have never seen before".
NON_SCRIMMAGE_HINTS = (
    "kickoff",
    "punt",
    "field goal",
    "extra point",
    "timeout",
    "end of",
    "end period",
    "penalty",
    "kick",
    "two point",
    "2pt",  # "Defensive 2pt Conversion" — no ppa, not a scrimmage snap
    "uncategorized",
    "placeholder",
)

GARBAGE_TIME_MARGIN = {1: 43, 2: 37, 3: 27, 4: 22}

SUCCESS_THRESHOLD = {1: 0.50, 2: 0.70, 3: 1.00, 4: 1.00}

FBS_SUCCESS_RATE_EXPECTED = 0.40
FBS_SUCCESS_RATE_TOLERANCE = 0.04  # so 36%-44% passes the gate

# Mean EPA per play does NOT sit near zero, and a gate demanding that it does
# will fail on a correct ingest. The reasoning that says it should — one side's
# gain is the other's loss — is false for CFBD's PPA column, because the
# drive-ending negatives that would balance the ledger carry no ppa at all:
# punts (31,365), kickoffs (29,972), made field goals (8,423) and missed ones
# (2,426) are null across the board. Measured mean on the four-season pull is
# +0.169 before filtering and +0.173 after.
#
# So the moment is checked loosely, as a smoke test against a scale error or a
# flipped sign, and the real correctness evidence is EPA_ORDERING below.
EPA_MEAN_BAND = (0.00, 0.35)
EPA_SD_BAND = (1.00, 1.80)  # observed 1.326; the plan's variance terms imply ~1.39

# The strongest cheap evidence that the EPA column means what we think it means.
# Each pair must come out strictly in this order, best to worst. A sign flip, a
# column mix-up or a units error breaks several of these at once; a uniform
# shift (the +0.17 above) breaks none of them, which is precisely the
# discrimination the mean-near-zero gate failed to make.
EPA_ORDERING = (
    "Passing Touchdown",
    "Rushing Touchdown",
    "Pass Reception",
    "Rush",
    "Pass Incompletion",
    "Sack",
    "Interception",
)

# Below this many of the EPA_ORDERING types present, the gate reports
# "not applicable" rather than passing or failing. Three gives at least two
# pairs, which is enough for a flipped sign to show up.
MIN_ORDERING_TYPES = 3


def is_success(down: Any, distance: Any, yards_gained: Any) -> bool | None:
    """
    Connelly success for one play. Returns None when the play cannot be judged
    (missing down/distance, non-positive distance), so callers can drop rather
    than silently score it 0 — a None counted as a failure would bias every
    rating downward.
    """
    try:
        d = int(down)
        dist = float(distance)
        gain = float(yards_gained)
    except (TypeError, ValueError):
        return None
    if d not in SUCCESS_THRESHOLD:
        return None
    if not math.isfinite(dist) or not math.isfinite(gain):
        return None
    if dist <= 0:
        # 1st-and-0 / goal-to-go artifacts. No meaningful threshold exists.
        return None
    return gain >= SUCCESS_THRESHOLD[d] * dist


def is_garbage_time(period: Any, offense_score: Any, defense_score: Any) -> bool | None:
    """
    True when the play falls outside Connelly's competitive-margin window.
    Overtime (period > 4) is never garbage time. Returns None if unjudgeable.
    """
    try:
        p = int(period)
        margin = abs(float(offense_score) - float(defense_score))
    except (TypeError, ValueError):
        return None
    if p > 4:
        return False
    if p not in GARBAGE_TIME_MARGIN:
        return None
    return margin > GARBAGE_TIME_MARGIN[p]


def _looks_non_scrimmage(play_type: str) -> bool:
    pt = (play_type or "").strip().lower()
    return any(h in pt for h in NON_SCRIMMAGE_HINTS)


def prepare_plays(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a raw plays table into the modeling table.

    Adds `success` (0/1) and `epa` (float) and keeps only rows that are a
    judgeable, competitive, FBS-vs-FBS scrimmage snap. Returns a copy; never
    mutates the input.

    The returned frame carries exactly the columns the ratings layer needs, so a
    schema change upstream fails loudly here instead of producing a rating built
    on a column that quietly became all-NaN.
    """
    required = {
        "season",
        "week",
        "game_id",
        "offense",
        "defense",
        "home",
        "away",
        "down",
        "distance",
        "yards_gained",
        "play_type",
        "period",
        "offense_score",
        "defense_score",
        "ppa",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"prepare_plays is missing required column(s): {sorted(missing)}. "
            "Fix the ingest mapping rather than defaulting them — a silently "
            "absent column produces a plausible, wrong rating."
        )

    df = raw.copy()

    df["is_scrimmage"] = df["play_type"].isin(SCRIMMAGE_PLAY_TYPES)
    df["success_raw"] = [
        is_success(d, dist, y)
        for d, dist, y in zip(df["down"], df["distance"], df["yards_gained"])
    ]
    df["garbage_raw"] = [
        is_garbage_time(p, o, d)
        for p, o, d in zip(df["period"], df["offense_score"], df["defense_score"])
    ]
    df["epa"] = pd.to_numeric(df["ppa"], errors="coerce")

    keep = (
        df["is_scrimmage"]
        & df["success_raw"].notna()
        & (df["garbage_raw"] == False)  # noqa: E712 — None must not pass
        & df["epa"].notna()
        & df["offense"].notna()
        & df["defense"].notna()
    )

    out = df.loc[keep].copy()
    out["success"] = out["success_raw"].astype(int)

    # Sign fix — see TURNOVER_PLAY_TYPES. Applied AFTER the astype so it
    # overwrites whatever the yardage rule concluded, and before anything
    # downstream reads the column.
    out.loc[out["play_type"].isin(TURNOVER_PLAY_TYPES), "success"] = 0

    out["is_home_offense"] = (out["offense"] == out["home"]).astype(int)

    cols = [
        "season",
        "week",
        "game_id",
        "offense",
        "defense",
        "home",
        "away",
        "is_home_offense",
        "down",
        "distance",
        "yards_gained",
        "period",
        "play_type",  # kept for the semantic-ordering gate in sanity_report
        "success",
        "epa",
    ]
    return out[cols].reset_index(drop=True)


def restrict_to_teams(plays: pd.DataFrame, teams: Iterable[str]) -> pd.DataFrame:
    """
    Keep only plays where BOTH sides are in `teams`.

    Necessary because CFBD's `classification=fbs` on /plays returns every play of
    every game *involving* an FBS team, so the FBS-vs-FCS games come back whole.
    On the 2022-2025 pull that put **238 distinct teams into a design built for
    136 FBS programs**: 102 FCS teams each picking up an offense and a defense
    coefficient from a game or two.

    That is not a cosmetic problem. Those coefficients are noisy by construction
    — one or two games of evidence apiece — and they are exactly what the
    opponent adjustment divides by when it decides how much credit an FBS team
    earned for beating them.
    """
    keep = set(teams)
    before = len(plays)
    out = plays.loc[plays["offense"].isin(keep) & plays["defense"].isin(keep)].copy()
    out.attrs["restrict_dropped"] = before - len(out)
    return out.reset_index(drop=True)


def drop_report(raw: pd.DataFrame) -> dict[str, Any]:
    """
    Explain what prepare_plays threw away and why. Called by the ingest and by
    the backtest so an unexpected schema change shows up as a number a human
    reads, not as a quietly smaller dataset.

    `unknown_play_types` is the field that matters: play types that are neither
    whitelisted nor recognisably a kick/administrative row. A non-empty list
    means CFBD added or renamed something and SCRIMMAGE_PLAY_TYPES needs review.
    """
    total = len(raw)
    if total == 0:
        return {"total": 0, "kept": 0, "unknown_play_types": []}

    scrim = raw["play_type"].isin(SCRIMMAGE_PLAY_TYPES)
    unknown = sorted(
        {
            str(pt)
            for pt in raw.loc[~scrim, "play_type"].dropna().unique()
            if not _looks_non_scrimmage(str(pt))
        }
    )
    success_raw = [
        is_success(d, dist, y)
        for d, dist, y in zip(raw["down"], raw["distance"], raw["yards_gained"])
    ]
    garbage_raw = [
        is_garbage_time(p, o, d)
        for p, o, d in zip(raw["period"], raw["offense_score"], raw["defense_score"])
    ]
    epa = pd.to_numeric(raw["ppa"], errors="coerce")

    return {
        "total": total,
        "kept": len(prepare_plays(raw)),
        "dropped_non_scrimmage": int((~scrim).sum()),
        "dropped_unjudgeable_down": int(sum(s is None for s in success_raw)),
        "dropped_garbage_time": int(sum(g is True for g in garbage_raw)),
        "dropped_missing_epa": int(epa.isna().sum()),
        "unknown_play_types": unknown,
    }


def team_week_aggregate(plays: pd.DataFrame) -> pd.DataFrame:
    """
    Per (season, week, team, side) success rate, EPA/play and play count.

    Not used to fit the ratings — the ridge fits on individual plays — but it is
    the human-readable view, and it is what the sanity gate reads.
    """
    frames = []
    for side, key in (("offense", "offense"), ("defense", "defense")):
        g = (
            plays.groupby(["season", "week", key], observed=True)
            .agg(
                plays_n=("success", "size"),
                success_rate=("success", "mean"),
                epa_per_play=("epa", "mean"),
            )
            .reset_index()
            .rename(columns={key: "team"})
        )
        g["side"] = side
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def sanity_report(plays: pd.DataFrame) -> dict[str, Any]:
    """
    The Phase 2 correctness gate. Reproduce known FBS aggregates before trusting
    anything downstream. `passes` False means stop and fix the ingest.
    """
    if len(plays) == 0:
        return {"passes": False, "reason": "no plays survived preparation"}

    sr = float(plays["success"].mean())
    epa = float(plays["epa"].mean())
    sd = float(plays["epa"].std())
    off_shares = plays.groupby("offense", observed=True).size()

    checks = {
        "success_rate": sr,
        "success_rate_expected": FBS_SUCCESS_RATE_EXPECTED,
        "success_rate_ok": abs(sr - FBS_SUCCESS_RATE_EXPECTED)
        <= FBS_SUCCESS_RATE_TOLERANCE,
        "mean_epa": epa,
        "mean_epa_ok": EPA_MEAN_BAND[0] <= epa <= EPA_MEAN_BAND[1],
        "sd_epa": sd,
        "sd_epa_ok": EPA_SD_BAND[0] <= sd <= EPA_SD_BAND[1],
        "n_plays": int(len(plays)),
        "n_teams": int(plays["offense"].nunique()),
        # Guards against a partial ingest: a season of FBS-vs-FBS football has
        # ~130+ teams, and no single team should own a large share of all plays.
        "max_team_share": float(off_shares.max() / len(plays)) if len(off_shares) else 1.0,
    }
    checks["max_team_share_ok"] = checks["max_team_share"] < 0.05

    # --- EPA semantic ordering -------------------------------------------------
    # Needs at least MIN_ORDERING_TYPES of the named play types to be present.
    # A synthetic fixture with only "Pass" and "Rush" cannot exercise this, and
    # the honest answer there is "not applicable", not a silent pass — so the
    # applicability flag is reported separately and real-data callers are
    # expected to assert it. run_model.py does.
    order_ok = True
    order_pairs: list[dict[str, Any]] = []
    present: list[str] = []
    if "play_type" in plays.columns:
        means = plays.groupby("play_type", observed=True)["epa"].mean()
        present = [pt for pt in EPA_ORDERING if pt in means.index]
        for better, worse in zip(present, present[1:]):
            ok = float(means[better]) > float(means[worse])
            order_ok = order_ok and ok
            order_pairs.append(
                {"better": better, "worse": worse,
                 "epa_better": float(means[better]),
                 "epa_worse": float(means[worse]), "ok": ok}
            )
        checks["epa_by_play_type"] = {k: float(v) for k, v in means.items()}

    checks["epa_ordering_pairs"] = order_pairs
    checks["epa_ordering_applicable"] = bool(len(present) >= MIN_ORDERING_TYPES)
    checks["epa_ordering_ok"] = bool(order_ok) if checks["epa_ordering_applicable"] else None

    checks["passes"] = bool(
        checks["success_rate_ok"]
        and checks["mean_epa_ok"]
        and checks["sd_epa_ok"]
        and (checks["epa_ordering_ok"] is not False)
        and checks["max_team_share_ok"]
    )
    if not checks["passes"]:
        reasons = []
        if not checks["success_rate_ok"]:
            reasons.append(
                f"success rate {sr:.3f} is outside "
                f"{FBS_SUCCESS_RATE_EXPECTED:.2f}+/-{FBS_SUCCESS_RATE_TOLERANCE:.2f}"
            )
        if not checks["mean_epa_ok"]:
            reasons.append(f"mean EPA {epa:+.3f} is outside {EPA_MEAN_BAND}")
        if not checks["sd_epa_ok"]:
            reasons.append(f"SD of EPA {sd:.3f} is outside {EPA_SD_BAND}")
        if checks["epa_ordering_ok"] is False:
            bad = [f"{p['better']}({p['epa_better']:+.2f}) !> {p['worse']}({p['epa_worse']:+.2f})"
                   for p in order_pairs if not p["ok"]]
            reasons.append("EPA ordering violated: " + "; ".join(bad))
        if not checks["max_team_share_ok"]:
            reasons.append(
                f"one team holds {checks['max_team_share']:.1%} of plays "
                "(partial ingest?)"
            )
        checks["reason"] = "; ".join(reasons)
    return checks
