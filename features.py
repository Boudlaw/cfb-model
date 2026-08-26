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
        "Safety",
        "Rush Touchdown",
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
    "uncategorized",
    "placeholder",
)

GARBAGE_TIME_MARGIN = {1: 43, 2: 37, 3: 27, 4: 22}

SUCCESS_THRESHOLD = {1: 0.50, 2: 0.70, 3: 1.00, 4: 1.00}

FBS_SUCCESS_RATE_EXPECTED = 0.40
FBS_SUCCESS_RATE_TOLERANCE = 0.04  # so 36%-44% passes the gate


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
        "success",
        "epa",
    ]
    return out[cols].reset_index(drop=True)


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
    off_shares = plays.groupby("offense", observed=True).size()

    checks = {
        "success_rate": sr,
        "success_rate_expected": FBS_SUCCESS_RATE_EXPECTED,
        "success_rate_ok": abs(sr - FBS_SUCCESS_RATE_EXPECTED)
        <= FBS_SUCCESS_RATE_TOLERANCE,
        # League-wide mean EPA per play sits near zero by construction: one side's
        # gain is the other's loss. A large magnitude means the EPA column is not
        # what we think it is.
        "mean_epa": epa,
        "mean_epa_ok": abs(epa) < 0.15,
        "n_plays": int(len(plays)),
        "n_teams": int(plays["offense"].nunique()),
        # Guards against a partial ingest: a season of FBS-vs-FBS football has
        # ~130+ teams, and no single team should own a large share of all plays.
        "max_team_share": float(off_shares.max() / len(plays)) if len(off_shares) else 1.0,
    }
    checks["max_team_share_ok"] = checks["max_team_share"] < 0.05
    checks["passes"] = bool(
        checks["success_rate_ok"] and checks["mean_epa_ok"] and checks["max_team_share_ok"]
    )
    if not checks["passes"]:
        reasons = []
        if not checks["success_rate_ok"]:
            reasons.append(
                f"success rate {sr:.3f} is outside "
                f"{FBS_SUCCESS_RATE_EXPECTED:.2f}+/-{FBS_SUCCESS_RATE_TOLERANCE:.2f}"
            )
        if not checks["mean_epa_ok"]:
            reasons.append(f"mean EPA {epa:+.3f} is not near zero")
        if not checks["max_team_share_ok"]:
            reasons.append(
                f"one team holds {checks['max_team_share']:.1%} of plays "
                "(partial ingest?)"
            )
        checks["reason"] = "; ".join(reasons)
    return checks
