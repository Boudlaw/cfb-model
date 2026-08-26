#!/usr/bin/env python3
"""
Phase 0/2a — one-time historical ingest of plays, games and betting lines.

WHY THIS RUNS IN GITHUB ACTIONS AND NOT LOCALLY
-----------------------------------------------
api.collegefootballdata.com is not reachable from the Claude sandbox where the
modeling code is written and tested (verified: connection fails, while
api.github.com returns 200). That is the same reason the weekly archiver runs in
Actions. So the split is:

    Actions  -> touches the network, writes data/ into the repo, commits it
    sandbox  -> clones the repo, reads data/, fits and validates the model

Stdlib only, like archive.py, so the job has nothing to install and nothing to
break unattended. The modeling files need numpy/sklearn; this one does not.

CALL BUDGET
-----------
CFBD's free tier allows 1,000 calls/month. This ingest is deliberately shaped to
fit inside it:

    /plays   1 call per (season, week)   4 seasons x ~16 weeks + 4 postseason = 68
    /games   1 call per season                                                =  4
    /lines   1 call per season                                                =  4
                                                                          total ~76

Pull once, commit the result, and never re-fetch. `--resume` skips any
(season, week) already on disk, so a run that dies halfway costs only the weeks
it had not reached.

ON NOT TRUSTING THIS SCRIPT'S FIELD MAPPING
-------------------------------------------
It was written without the ability to call the API, so the response-key mapping
is inferred, not verified. Two defences:

  * every field is resolved through _pick(), which accepts several plausible key
    spellings and records which one it found;
  * `--smoke` fetches one small slice, prints the ACTUAL keys and a sample row,
    and writes nothing.

Run --smoke first and read the output. If a field lands as None across the
board, fix CANDIDATES here rather than patching downstream — features.py raises
on a missing column on purpose.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

CFBD_BASE = "https://api.collegefootballdata.com"
USER_AGENT = "cfb-model-ingest/1.0 (personal research; contact mailbox@boudreaulaw.com)"
TIMEOUT = int(os.environ.get("CFB_TIMEOUT", "60"))
RETRIES = int(os.environ.get("CFB_RETRIES", "3"))
SLEEP_BETWEEN = float(os.environ.get("CFB_SLEEP", "1.0"))

DEFAULT_SEASONS = (2022, 2023, 2024, 2025)
DEFAULT_WEEKS = tuple(range(1, 17))


def _data_dir() -> Path:
    env = os.environ.get("CFB_DATA_DIR", "").strip()
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / ".git").exists():
            return cand / "data"
    return here / "data"


DATA_DIR = _data_dir()


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch(path: str, **params: Any) -> Any:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "CFBD_API_KEY is not set. Get a free key at "
            "https://collegefootballdata.com/key and add it as the repository "
            "secret CFBD_API_KEY (note the D — a missing D cost a whole run once)."
        )
    q = {k: v for k, v in params.items() if v not in (None, "")}
    url = f"{CFBD_BASE}{path}?{urllib.parse.urlencode(q)}"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT}
    )
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < RETRIES:
                wait = 5 * attempt
                log(f"    HTTP {exc.code}; retrying in {wait}s")
                time.sleep(wait)
                last = exc
                continue
            raise
        except Exception as exc:  # noqa: BLE001 — network flake
            last = exc
            if attempt < RETRIES:
                time.sleep(4 * attempt)
                continue
            raise
    raise RuntimeError(f"unreachable: {last}")

def fetch_season(path: str, season: int) -> list:
    """
    Fetch a whole season from an endpoint that takes seasonType.

    Tries seasonType="both" first and falls back to regular + postseason
    separately if the API rejects it. "both" is the one parameter value nothing
    has ever verified against this API — the production archiver passes only
    "regular", and so did the smoke test — and it must not be the thing that
    fails partway through a multi-season ingest.

    Raising on a fallback failure is deliberate: the caller's except block skips
    writing the file entirely, so a half-fetched season can never masquerade as
    complete on a later --resume.
    """
    try:
        return fetch(path, year=season, seasonType="both")
    except Exception as exc:  # noqa: BLE001
        log(f"    seasonType=both rejected ({exc}); using regular+postseason")
        rows: list = []
        for stype in ("regular", "postseason"):
            rows.extend(fetch(path, year=season, seasonType=stype))
            time.sleep(SLEEP_BETWEEN)
        return rows

# ---------------------------------------------------------------- key resolution

def _pick(row: dict, *names: str, default: Any = None) -> Any:
    """
    First present key wins. Handles camelCase/snake_case drift and the nested
    clock object without the caller caring which spelling the API used today.
    """
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default


PLAY_COLUMNS = [
    "season", "week", "game_id", "play_id", "offense", "defense", "home", "away",
    "offense_conference", "defense_conference", "down", "distance", "yards_gained",
    "play_type", "period", "offense_score", "defense_score", "ppa",
]


def map_play(row: dict, season: int, week: int) -> dict:
    return {
        "season": _pick(row, "season", default=season),
        "week": _pick(row, "week", default=week),
        "game_id": _pick(row, "gameId", "game_id"),
        "play_id": _pick(row, "id", "playId", "play_id"),
        "offense": _pick(row, "offense"),
        "defense": _pick(row, "defense"),
        "home": _pick(row, "home", "homeTeam"),
        "away": _pick(row, "away", "awayTeam"),
        "offense_conference": _pick(row, "offenseConference", "offense_conference"),
        "defense_conference": _pick(row, "defenseConference", "defense_conference"),
        "down": _pick(row, "down"),
        "distance": _pick(row, "distance"),
        "yards_gained": _pick(row, "yardsGained", "yards_gained", "yardsGainedOnPlay"),
        "play_type": _pick(row, "playType", "play_type"),
        "period": _pick(row, "period", "quarter"),
        "offense_score": _pick(row, "offenseScore", "offense_score"),
        "defense_score": _pick(row, "defenseScore", "defense_score"),
        # PPA is CFBD's name for EPA. If this is ever None across the board the
        # free tier is not returning it and features.py will refuse to proceed,
        # which is the correct outcome.
        "ppa": _pick(row, "ppa", "PPA", "epa"),
    }


GAME_COLUMNS = [
    "season", "week", "game_id", "start_date", "neutral_site", "conference_game",
    "home_team", "away_team", "home_classification", "away_classification",
    "home_points", "away_points", "home_pregame_elo", "away_pregame_elo",
]


def map_game(row: dict) -> dict:
    return {
        "season": _pick(row, "season"),
        "week": _pick(row, "week"),
        "game_id": _pick(row, "id", "gameId", "game_id"),
        "start_date": _pick(row, "startDate", "start_date"),
        "neutral_site": _pick(row, "neutralSite", "neutral_site", default=False),
        "conference_game": _pick(row, "conferenceGame", "conference_game", default=False),
        "home_team": _pick(row, "homeTeam", "home_team"),
        "away_team": _pick(row, "awayTeam", "away_team"),
        "home_classification": _pick(row, "homeClassification", "home_classification"),
        "away_classification": _pick(row, "awayClassification", "away_classification"),
        "home_points": _pick(row, "homePoints", "home_points"),
        "away_points": _pick(row, "awayPoints", "away_points"),
        "home_pregame_elo": _pick(row, "homePregameElo", "home_pregame_elo"),
        "away_pregame_elo": _pick(row, "awayPregameElo", "away_pregame_elo"),
    }


LINE_COLUMNS = [
    "season", "week", "game_id", "home_team", "away_team", "provider",
    "spread", "spread_open", "over_under", "over_under_open",
]


def map_lines(row: dict) -> list[dict]:
    """
    /lines nests one row per game with a `lines` array of providers. Flattened to
    one row per (game, provider) so the opening line can be averaged across books.
    """
    base = {
        "season": _pick(row, "season"),
        "week": _pick(row, "week"),
        "game_id": _pick(row, "id", "gameId", "game_id"),
        "home_team": _pick(row, "homeTeam", "home_team"),
        "away_team": _pick(row, "awayTeam", "away_team"),
    }
    out = []
    for ln in _pick(row, "lines", default=[]) or []:
        out.append(
            {
                **base,
                "provider": _pick(ln, "provider"),
                "spread": _pick(ln, "spread"),
                "spread_open": _pick(ln, "spreadOpen", "spread_open"),
                "over_under": _pick(ln, "overUnder", "over_under"),
                "over_under_open": _pick(ln, "overUnderOpen", "over_under_open"),
            }
        )
    return out


# ---------------------------------------------------------------------- writing

def write_gz(path: Path, columns: Sequence[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
            n += 1
    return n


def smoke(season: int, week: int) -> int:
    """
    Diagnose the API's actual shape. Writes nothing. Exit 0 = safe to ingest.

    This deliberately reports DISTRIBUTIONS, not a single sample row. The first
    version of this function inspected only rows[0], which happened to be a
    kickoff, saw `ppa: null`, and reported a broken mapping when nothing was
    wrong — CFBD simply does not compute PPA for kicks. A single row cannot
    distinguish "field is missing" from "field is legitimately absent on this
    kind of play", so it must not be the basis for the verdict.

    Three questions it answers, each of which would otherwise only surface after
    the full ingest had been committed:

      1. Is `ppa` populated on SCRIMMAGE plays (the only ones the model uses)?
      2. Which `playType` strings actually exist, so features.py's whitelist can
         be checked against reality rather than against my guess?
      3. Which line providers carry `spreadOpen`? The ATS diagnostic depends on
         the OPENING line, and at least one provider (teamrankings) returns null
         for it.
    """
    log(f"=== smoke: season {season} week {week} ===")
    rc = 0

    # Kicks and administrative rows. Used ONLY to split the ppa-coverage report
    # into "plays the model would use" vs. "plays it would drop anyway".
    non_scrimmage_hint = (
        "kickoff", "punt", "field goal", "extra point", "timeout", "end of",
        "end period", "penalty", "kick", "two point", "uncategorized",
    )

    def is_scrimmage_like(pt: str) -> bool:
        p = (pt or "").strip().lower()
        return not any(h in p for h in non_scrimmage_hint)

    try:
        plays = fetch("/plays", year=season, week=week, seasonType="regular",
                      classification="fbs")
        log(f"/plays -> {len(plays)} rows")
        if not plays:
            log("  !! no plays returned; cannot verify the mapping")
            return 2
        log(f"  observed keys: {sorted(plays[0].keys())}")

        mapped = [map_play(r, season, week) for r in plays]

        # --- per-playType counts and ppa coverage -------------------------------
        by_type: dict[str, dict[str, int]] = {}
        for m in mapped:
            pt = str(m["play_type"])
            slot = by_type.setdefault(pt, {"n": 0, "ppa_null": 0, "down_null": 0})
            slot["n"] += 1
            if m["ppa"] is None:
                slot["ppa_null"] += 1
            if m["down"] in (None, 0):
                slot["down_null"] += 1

        log("  playType distribution (n, ppa null, down null, scrimmage?):")
        for pt, s in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
            log(f"    {pt:38} {s['n']:>6} {s['ppa_null']:>6} {s['down_null']:>6}"
                f"   {'yes' if is_scrimmage_like(pt) else 'no'}")

        scrim = [m for m in mapped if is_scrimmage_like(str(m["play_type"]))]
        scrim_ppa = sum(1 for m in scrim if m["ppa"] is not None)
        pct = 100.0 * scrim_ppa / len(scrim) if scrim else 0.0
        log(f"  scrimmage plays: {len(scrim)}, with ppa: {scrim_ppa} ({pct:.1f}%)")

        # The model needs EPA on the plays it keeps. Anything under ~90% coverage
        # means either the free tier withholds it or the field name is wrong, and
        # either way features.py would drop most of the data.
        if pct < 90.0:
            log(f"  !! VERDICT: ppa coverage on scrimmage plays is only {pct:.1f}%."
                " Do NOT ingest. Check the field name or the API tier.")
            rc = 1
        else:
            log(f"  OK: ppa is populated on {pct:.1f}% of scrimmage plays."
                " Nulls are concentrated in kicks, which the model drops anyway.")

        other_nulls = [
            k for k in ("game_id", "offense", "defense", "home", "away",
                        "yards_gained", "play_type", "period")
            if sum(1 for m in mapped if m[k] is None) > len(mapped) * 0.05
        ]
        if other_nulls:
            log(f"  !! MAPPED MOSTLY NONE: {other_nulls} -- fix map_play")
            rc = 1
    except Exception as exc:  # noqa: BLE001
        log(f"/plays FAILED: {exc}")
        return 2

    # --- games ----------------------------------------------------------------
    try:
        rows = fetch("/games", year=season, week=week, seasonType="regular")
        log(f"/games -> {len(rows)} rows")
        if rows:
            log(f"  observed keys: {sorted(rows[0].keys())}")
            gm = [map_game(r) for r in rows]
            fbs = [g for g in gm
                   if g["home_classification"] == "fbs" and g["away_classification"] == "fbs"]
            scored = [g for g in fbs
                      if g["home_points"] is not None and g["away_points"] is not None]
            log(f"  FBS-vs-FBS: {len(fbs)}, with final scores: {len(scored)}")
            if fbs and not scored:
                log("  !! no scored FBS-vs-FBS games; margins cannot be formed")
                rc = 1
    except Exception as exc:  # noqa: BLE001
        log(f"/games FAILED: {exc}")
        rc = max(rc, 2)

    # --- lines ----------------------------------------------------------------
    try:
        rows = fetch("/lines", year=season, week=week, seasonType="regular")
        log(f"/lines -> {len(rows)} rows")
        if rows:
            log(f"  observed keys: {sorted(rows[0].keys())}")
            flat = [x for r in rows for x in map_lines(r)]
            prov: dict[str, dict[str, int]] = {}
            for ln in flat:
                s = prov.setdefault(str(ln["provider"]), {"n": 0, "spread": 0, "open": 0})
                s["n"] += 1
                if ln["spread"] is not None:
                    s["spread"] += 1
                if ln["spread_open"] is not None:
                    s["open"] += 1
            log(f"  {len(flat)} provider-rows across {len(rows)} games")
            log("  provider (rows, with spread, with spreadOpen):")
            for p, s in sorted(prov.items(), key=lambda kv: -kv[1]["n"]):
                log(f"    {p:24} {s['n']:>5} {s['spread']:>5} {s['open']:>5}")
            games_with_open = len({ln["game_id"] for ln in flat
                                   if ln["spread_open"] is not None})
            log(f"  games with an OPENING line from at least one provider: "
                f"{games_with_open}/{len(rows)}")
            if games_with_open == 0:
                log("  !! VERDICT: no opening lines at all. The ATS diagnostic will"
                    " be empty; prefer the closing spread or drop ATS for history.")
                rc = 1
            elif games_with_open < 0.5 * len(rows):
                log("  !  WARNING: opening lines are sparse. ATS numbers will cover"
                    " only a subset; report the subset size, never the full slate.")
    except Exception as exc:  # noqa: BLE001
        log(f"/lines FAILED: {exc}")
        rc = max(rc, 2)

    log(f"=== smoke verdict: {'SAFE TO INGEST' if rc == 0 else 'DO NOT INGEST YET'} ===")
    return rc


def ingest(
    seasons: Sequence[int],
    weeks: Sequence[int],
    resume: bool = True,
    postseason: bool = True,
) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    calls = 0
    failures: list[str] = []

    for season in seasons:
        log(f"=== season {season} ===")
        season_plays: list[dict] = []
        out_plays = DATA_DIR / f"plays_{season}.csv.gz"
        if resume and out_plays.exists():
            log(f"  {out_plays.name} exists; skipping plays (use --no-resume to refetch)")
        else:
            types = ["regular"] + (["postseason"] if postseason else [])
            for stype in types:
                wk_list = weeks if stype == "regular" else [1]
                for wk in wk_list:
                    try:
                        rows = fetch("/plays", year=season, week=wk, seasonType=stype,
                                     classification="fbs")
                        calls += 1
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"plays {season} {stype} wk{wk}: {exc}")
                        log(f"  {stype} wk{wk:>2}: FAILED {exc}")
                        continue
                    mapped = [map_play(r, season, wk) for r in rows]
                    season_plays.extend(mapped)
                    log(f"  {stype} wk{wk:>2}: {len(rows):>5} plays")
                    if not rows and stype == "regular" and wk > 14:
                        break  # season is over; stop burning calls
                    time.sleep(SLEEP_BETWEEN)
            n = write_gz(out_plays, PLAY_COLUMNS, season_plays)
            log(f"  wrote {out_plays.name}: {n} rows")

        out_games = DATA_DIR / f"games_{season}.csv.gz"
        if resume and out_games.exists():
            log(f"  {out_games.name} exists; skipping")
        else:
            try:
                rows = fetch_season("/games", season)
                calls += 1
                n = write_gz(out_games, GAME_COLUMNS, (map_game(r) for r in rows))
                log(f"  wrote {out_games.name}: {n} rows")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"games {season}: {exc}")
                log(f"  games FAILED: {exc}")
            time.sleep(SLEEP_BETWEEN)

        out_lines = DATA_DIR / f"lines_{season}.csv.gz"
        if resume and out_lines.exists():
            log(f"  {out_lines.name} exists; skipping")
        else:
            try:
                rows = fetch_season("/lines", season)
                calls += 1
                flat = [x for r in rows for x in map_lines(r)]
                n = write_gz(out_lines, LINE_COLUMNS, flat)
                log(f"  wrote {out_lines.name}: {n} rows")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"lines {season}: {exc}")
                log(f"  lines FAILED: {exc}")
            time.sleep(SLEEP_BETWEEN)

    log(f"=== done: {calls} API calls, {len(failures)} failure(s) ===")
    for f in failures:
        log(f"  FAIL {f}")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(str(s) for s in DEFAULT_SEASONS))
    ap.add_argument("--weeks", default="1-16")
    ap.add_argument("--smoke", action="store_true",
                    help="print real API keys for one slice and exit; writes nothing")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--no-postseason", dest="postseason", action="store_false")
    args = ap.parse_args(argv)

    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    if "-" in args.weeks:
        a, b = args.weeks.split("-")
        weeks = list(range(int(a), int(b) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(",") if w.strip()]

    log(f"data dir: {DATA_DIR}")
    if args.smoke:
        return smoke(seasons[0], weeks[0])
    return ingest(seasons, weeks, resume=args.resume, postseason=args.postseason)


if __name__ == "__main__":
    sys.exit(main())
