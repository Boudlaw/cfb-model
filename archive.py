#!/usr/bin/env python3
"""
CFB point-in-time snapshot archiver.

WHY THIS EXISTS
---------------
CollegeFootballData's SP+ and FPI endpoints return ONE rating per season with no
`week` parameter. If you pull 2023 SP+ and use it to "predict" a Week 4 2023 game,
that rating already contains the outcome of the game you are predicting, plus every
game after it. Backtests built that way look excellent and are worthless.

The only fix is to capture the ratings every week, before the games, and keep the
captures forever. That is all this script does. Every week you skip is a week of
point-in-time data that cannot be recovered later.

DESIGN RULES
------------
1. Append-only. Nothing is ever overwritten or recomputed in place.
2. Raw bytes are kept alongside parsed rows. If a parser turns out to be wrong we
   re-parse history from the raw captures instead of losing it.
3. Every row carries captured_at (UTC), season and week. The capture time is the
   fact that makes the row usable; the rating alone is not.
4. Idempotent per (source, season, week, date). Re-running the same day is a no-op,
   so a retry after a partial failure is always safe.
5. Partial failure is fine. One dead source does not stop the others. Failures are
   recorded in the manifest so gaps are visible rather than silent.

USAGE
-----
    export CFBD_API_KEY=...            # from https://collegefootballdata.com/key
    python archive.py --smoke          # verify connectivity + endpoint shapes, write nothing
    python archive.py --dry-run        # fetch and parse, print row counts, write nothing
    python archive.py                  # capture this week
    python archive.py --season 2026 --week 5
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

CFBD_BASE = "https://api.collegefootballdata.com"
SONNY_MOORE_URL = "https://sonnymoorepowerratings.com/col-foot.htm"
PIRATE_URL = "https://piratings.wordpress.com/"

def _default_archive_dir() -> Path:
    """
    Resolve where the archive lives, without caring how the repo is laid out.
    Order: explicit env var -> nearest git repo root -> the script's own directory.
    This has to work whether archive.py sits at the repo root or in a subpackage.
    """
    env = os.environ.get("CFB_ARCHIVE_DIR", "").strip()
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / ".git").exists():
            return cand / "archive"
    if here.name == "cfb_archive":
        return here.parent / "archive"
    return here / "archive"


ARCHIVE_DIR = _default_archive_dir()
RAW_DIR = ARCHIVE_DIR / "raw"
MANIFEST = ARCHIVE_DIR / "manifest.csv"

USER_AGENT = "cfb-archive/1.0 (personal research; contact mailbox@boudreaulaw.com)"
TIMEOUT = 45
RETRIES = 3
RETRY_BACKOFF = 4  # seconds, multiplied by attempt number


# ---------------------------------------------------------------- infrastructure

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp() -> str:
    return now_utc().replace(microsecond=0).isoformat()


def log(msg: str) -> None:
    print(f"[{stamp()}] {msg}", flush=True)


def http_get(url: str, headers: dict[str, str] | None = None) -> bytes:
    """GET with retries. Raises on final failure."""
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf-8", "replace")
            last = RuntimeError(f"HTTP {e.code} for {url}: {body}")
            # Client errors other than rate-limiting will not fix themselves.
            if e.code in (400, 401, 403, 404) and e.code != 429:
                break
        except Exception as e:  # noqa: BLE001 - network layer, anything can happen
            last = e
        if attempt < RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)
    raise RuntimeError(f"GET failed after {RETRIES} attempts: {url}") from last


def cfbd_get(path: str, params: dict[str, Any] | None = None) -> tuple[Any, bytes]:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        raise RuntimeError("CFBD_API_KEY is not set. Get a free key at https://collegefootballdata.com/key")
    qs = ""
    if params:
        clean = {k: v for k, v in params.items() if v is not None}
        if clean:
            qs = "?" + urllib.parse.urlencode(clean)
    raw = http_get(f"{CFBD_BASE}{path}{qs}", headers={"Authorization": f"Bearer {key}",
                                                      "Accept": "application/json"})
    return json.loads(raw.decode("utf-8")), raw


def write_raw(source: str, season: int, week: int, payload: bytes, ext: str) -> Path:
    d = RAW_DIR / f"{season}" / f"week{week:02d}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{source}_{now_utc():%Y%m%dT%H%M%SZ}.{ext}"
    p.write_bytes(payload)
    return p


def append_rows(source: str, rows: list[dict[str, Any]]) -> int:
    """Append rows to archive/<source>.csv, unioning columns across historical schemas."""
    if not rows:
        return 0
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = ARCHIVE_DIR / f"{source}.csv"

    existing_cols: list[str] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing_cols = next(csv.reader(f), [])

    new_cols = [c for c in rows[0] if c not in existing_cols]
    cols = existing_cols + new_cols

    if new_cols and existing_cols:
        # Schema changed. Rewrite the file with the widened header rather than
        # silently dropping the new fields.
        with path.open(newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(old)
        log(f"  {source}: widened schema with {new_cols}")

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols or list(rows[0]), extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def already_captured(source: str, season: int, week: int) -> bool:
    """True if this source was captured for this season/week on this UTC date."""
    if not MANIFEST.exists():
        return False
    today = now_utc().date().isoformat()
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("source") == source and r.get("season") == str(season)
                    and r.get("week") == str(week) and r.get("status") == "ok"
                    and (r.get("captured_at") or "").startswith(today)):
                return True
    return False


def record(source: str, season: int, week: int, status: str, n: int, note: str = "") -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["captured_at", "source", "season", "week",
                                          "status", "rows", "note"])
        if new:
            w.writeheader()
        w.writerow({"captured_at": stamp(), "source": source, "season": season,
                    "week": week, "status": status, "rows": n, "note": note[:500]})


# ------------------------------------------------------------------- CFBD sources

def _tag(rows: list[dict], season: int, week: int, source: str) -> list[dict]:
    """Stamp every row with capture metadata. Flattens one level of nesting."""
    captured = stamp()
    out = []
    for r in rows:
        flat: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}_{k2}"] = v2
            elif isinstance(v, list):
                flat[k] = json.dumps(v, separators=(",", ":"))
            else:
                flat[k] = v
        out.append({"captured_at": captured, "capture_season": season,
                    "capture_week": week, "source": source, **flat})
    return out


def cap_ratings(endpoint: str, source: str, season: int, week: int,
                week_aware: bool) -> list[dict]:
    """
    Capture a CFBD ratings endpoint.

    week_aware=True  -> the endpoint itself accepts `week`, so the value is
                        genuinely point-in-time and can be backfilled later.
    week_aware=False -> season-level only (SP+, FPI). The weekly capture IS the
                        point-in-time record. This cannot be backfilled. Ever.
    """
    params: dict[str, Any] = {"year": season}
    if week_aware:
        params["week"] = week
    data, raw = cfbd_get(endpoint, params)
    write_raw(source, season, week, raw, "json")
    if not isinstance(data, list):
        raise RuntimeError(f"{source}: expected a JSON list, got {type(data).__name__}")
    return _tag(data, season, week, source)


def cap_lines(season: int, week: int) -> list[dict]:
    """
    Betting lines. Opening lines matter most: published research finds the model
    edge exists against the OPENING line and disappears against midweek/closing.
    Capturing early in the week is the whole point.
    """
    data, raw = cfbd_get("/lines", {"year": season, "week": week, "seasonType": "regular"})
    write_raw("lines", season, week, raw, "json")
    rows: list[dict] = []
    captured = stamp()
    for g in data if isinstance(data, list) else []:
        base = {k: v for k, v in g.items() if k != "lines"}
        for ln in (g.get("lines") or [{}]):
            rows.append({"captured_at": captured, "capture_season": season,
                         "capture_week": week, "source": "lines", **base,
                         **{f"line_{k}": v for k, v in ln.items()}})
    return rows


def cap_games(season: int, week: int) -> list[dict]:
    """Schedule and results, for scoring predictions after the fact."""
    data, raw = cfbd_get("/games", {"year": season, "week": week, "seasonType": "regular"})
    write_raw("games", season, week, raw, "json")
    return _tag(data if isinstance(data, list) else [], season, week, "games")


# ---------------------------------------------------------------- scraped sources

SONNY_RE = re.compile(
    r"^\s*(?P<rank>\d{1,3})\s+"
    r"(?P<team>[A-Z][A-Za-z.,'&\-\s]*?)\s+"
    r"(?P<w>\d{1,2})\s+(?P<l>\d{1,2})\s+(?P<t>\d{1,2})\s+"
    r"(?P<sos>-?\d+\.\d+)\s+(?P<pr>-?\d+\.\d+)\s*$"
)


def cap_sonny_moore(season: int, week: int) -> list[dict]:
    """
    Sonny Moore's Computer Power Ratings — running since 1974, static HTML.
    Usage per his own page: compare the two PRs and add ~4 points for home field.
    (We use a 2.5-point league HFA instead; see the build plan.)
    """
    raw = http_get(SONNY_MOORE_URL)
    write_raw("sonny_moore", season, week, raw, "html")
    text = re.sub(r"<[^>]+>", "", raw.decode("utf-8", "replace"))
    compiled = ""
    m = re.search(r"Compiled:\s*([0-9\-]+[^\n<]*)", text)
    if m:
        compiled = m.group(1).strip()

    captured, rows, seen = stamp(), [], set()
    for line in text.splitlines():
        mm = SONNY_RE.match(line.rstrip())
        if not mm:
            continue
        team = " ".join(mm.group("team").split())
        if team in seen:      # the page lists teams twice: by rank, then alphabetically
            continue
        seen.add(team)
        rows.append({"captured_at": captured, "capture_season": season,
                     "capture_week": week, "source": "sonny_moore",
                     "compiled": compiled, "rank": int(mm.group("rank")), "team": team,
                     "wins": int(mm.group("w")), "losses": int(mm.group("l")),
                     "ties": int(mm.group("t")), "sos": float(mm.group("sos")),
                     "rating": float(mm.group("pr"))})
    if len(rows) < 100:
        raise RuntimeError(f"sonny_moore: parsed only {len(rows)} teams, expected ~138. "
                           f"Layout probably changed — inspect the raw capture.")
    return rows


def cap_pirate(season: int, week: int) -> list[dict]:
    """
    Pi-Rate Ratings. 'Pi-Rate Bias' carried the largest single weight (0.281) in
    Coleman's 5-system metamodel, so this one is worth the scraping hassle.
    Layout is a WordPress post table and changes between seasons; we archive the
    raw HTML unconditionally and parse best-effort.
    """
    raw = http_get(PIRATE_URL)
    write_raw("pirate", season, week, raw, "html")
    return []  # parser deliberately deferred until a real in-season page exists


# ------------------------------------------------------------------------- driver

SOURCES: list[tuple[str, Callable[[int, int], list[dict]], str]] = [
    # name          fn                                                        criticality
    ("sp_plus",     lambda s, w: cap_ratings("/ratings/sp", "sp_plus", s, w, False),  "critical"),
    ("fpi",         lambda s, w: cap_ratings("/ratings/fpi", "fpi", s, w, False),     "critical"),
    ("lines",       cap_lines,                                                        "critical"),
    ("elo",         lambda s, w: cap_ratings("/ratings/elo", "elo", s, w, True),      "recoverable"),
    ("srs",         lambda s, w: cap_ratings("/ratings/srs", "srs", s, w, False),     "recoverable"),
    ("core",        lambda s, w: cap_ratings("/ratings/core", "core", s, w, True),    "recoverable"),
    ("games",       cap_games,                                                        "recoverable"),
    ("sonny_moore", cap_sonny_moore,                                                  "critical"),
    ("pirate",      cap_pirate,                                                       "critical"),
]


def current_season_week(today: dt.date | None = None) -> tuple[int, int]:
    """
    Approximate CFB week from the calendar. Week 1 contains the Saturday nearest
    Sept 1; weeks roll Tuesday-to-Monday so a Tuesday capture belongs to the week
    of the games ahead of it. Override with --season/--week for anything that matters.
    """
    today = today or now_utc().date()
    season = today.year if today.month >= 7 else today.year - 1
    kickoff = dt.date(season, 8, 25)
    while kickoff.weekday() != 5:  # Saturday
        kickoff += dt.timedelta(days=1)
    week = max(1, min(16, ((today - (kickoff - dt.timedelta(days=4))).days // 7) + 1))
    return season, week


def smoke() -> int:
    """Verify every endpoint answers and has the shape we expect. Writes nothing."""
    season, week = current_season_week()
    log(f"smoke test — season {season}, week {week}")
    checks = [
        ("CFBD /teams/fbs", lambda: cfbd_get("/teams/fbs", {"year": season})),
        ("CFBD /ratings/sp", lambda: cfbd_get("/ratings/sp", {"year": season})),
        ("CFBD /ratings/fpi", lambda: cfbd_get("/ratings/fpi", {"year": season})),
        ("CFBD /ratings/elo", lambda: cfbd_get("/ratings/elo", {"year": season, "week": week})),
        ("CFBD /ratings/srs", lambda: cfbd_get("/ratings/srs", {"year": season})),
        ("CFBD /ratings/core", lambda: cfbd_get("/ratings/core", {"year": season, "week": week})),
        ("CFBD /lines", lambda: cfbd_get("/lines", {"year": season, "week": week})),
        ("CFBD /games", lambda: cfbd_get("/games", {"year": season, "week": week})),
        ("Sonny Moore page", lambda: (http_get(SONNY_MOORE_URL), b"")),
        ("Pi-Rate page", lambda: (http_get(PIRATE_URL), b"")),
    ]
    bad = 0
    for name, fn in checks:
        try:
            data, _ = fn()
            n = len(data) if isinstance(data, (list, bytes)) else "?"
            log(f"  OK   {name}  ({n} items/bytes)")
        except Exception as e:  # noqa: BLE001
            bad += 1
            log(f"  FAIL {name}: {e}")
    log(f"smoke: {len(checks) - bad}/{len(checks)} passed")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--only", help="comma-separated source names")
    ap.add_argument("--dry-run", action="store_true", help="fetch and parse, write nothing")
    ap.add_argument("--smoke", action="store_true", help="connectivity check only")
    ap.add_argument("--force", action="store_true", help="ignore the same-day idempotency guard")
    args = ap.parse_args()

    if args.smoke:
        return smoke()

    auto_s, auto_w = current_season_week()
    season = args.season or auto_s
    week = args.week or auto_w
    wanted = set(args.only.split(",")) if args.only else None

    log(f"archiving season {season} week {week}"
        f"{' [DRY RUN]' if args.dry_run else ''} -> {ARCHIVE_DIR}")

    total, failures, critical_failures = 0, [], []
    for name, fn, criticality in SOURCES:
        if wanted and name not in wanted:
            continue
        if not args.dry_run and not args.force and already_captured(name, season, week):
            log(f"  skip {name}: already captured today")
            continue
        try:
            rows = fn(season, week)
            if args.dry_run:
                log(f"  {name}: parsed {len(rows)} rows (not written)")
                if rows:
                    log(f"       sample: {json.dumps(rows[0], default=str)[:220]}")
            else:
                n = append_rows(name, rows)
                record(name, season, week, "ok", n)
                total += n
                log(f"  {name}: +{n} rows")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            failures.append(f"{name}: {msg}")
            if criticality == "critical":
                critical_failures.append(name)
            if not args.dry_run:
                record(name, season, week, "fail", 0, msg)
            log(f"  FAIL {name}: {msg}")

    log(f"done: {total} rows, {len(failures)} failure(s)")
    if critical_failures:
        # Critical sources cannot be backfilled. A silent failure here is the one
        # outcome that actually destroys value, so exit non-zero and make noise.
        log(f"CRITICAL — unrecoverable sources failed: {', '.join(critical_failures)}")
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
