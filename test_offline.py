#!/usr/bin/env python3
"""Offline tests: everything that does not need network. Run before every commit."""
import datetime as dt
import os
import sys
import tempfile
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="cfbtest_")
os.environ["CFB_ARCHIVE_DIR"] = TMP
_here = Path(__file__).resolve().parent
for _cand in (_here, _here / "cfb_archive", _here.parent, _here.parent / "cfb_archive"):
    if (_cand / "archive.py").exists():
        sys.path.insert(0, str(_cand))
        break
else:
    sys.exit("could not locate archive.py next to or above this test file")

import archive as A  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + str(detail)}")
    if not cond:
        fails.append(name)


# ---- 1. Sonny Moore parser, against a faithful reproduction of the page layout
SAMPLE = b"""<html><pre>
        SONNY MOORE'S COMPUTER POWER RATINGS
              Compiled: 06-30-2026 - Start of Season

RANK  TEAM                   W  L  T    SOS      PR
  1  INDIANA                 0  0  0   0.00   94.00
  2  OHIO ST.                0  0  0   0.00   90.42
  3  OREGON                  0  0  0   0.00   89.83
  4  NOTRE DAME              0  0  0   0.00   88.50
  5  TEXAS TECH              0  0  0   0.00   87.02
138  KENNESAW ST.            0  0  0   0.00   28.11
</pre></html>"""

rows = A._parse_sonny(SAMPLE) if hasattr(A, "_parse_sonny") else None
if rows is None:
    # parser lives inline in cap_sonny_moore; exercise it via the regex directly
    import re
    text = re.sub(r"<[^>]+>", "", SAMPLE.decode())
    rows = [A.SONNY_RE.match(l.rstrip()) for l in text.splitlines()]
    rows = [m for m in rows if m]

check("sonny parser finds all 6 sample teams", len(rows) == 6, f"got {len(rows)}")
if rows:
    first = rows[0]
    g = first.groupdict() if hasattr(first, "groupdict") else first
    check("sonny parses team name with no trailing digits", g["team"].strip() == "INDIANA", g["team"])
    check("sonny parses rating", float(g["pr"]) == 94.00, g["pr"])
check("sonny parses names containing a period", any(
    (m.groupdict()["team"].strip() == "OHIO ST.") for m in rows), "OHIO ST. not matched")
check("sonny parses multi-word names", any(
    (m.groupdict()["team"].strip() == "NOTRE DAME") for m in rows), "NOTRE DAME not matched")
check("sonny handles 3-digit rank", any(m.groupdict()["rank"] == "138" for m in rows))

# guard: the <100 team tripwire must fire on a truncated page
try:
    A.http_get = lambda *a, **k: SAMPLE          # 6 teams only
    A.write_raw = lambda *a, **k: Path(TMP) / "x"
    A.cap_sonny_moore(2026, 1)
    check("short-page tripwire raises", False, "no exception")
except RuntimeError as e:
    check("short-page tripwire raises", "parsed only" in str(e), str(e))

# ---- 2. season/week inference
cases = [
    (dt.date(2026, 8, 21), 2026, 1),    # before kickoff -> clamped to week 1
    (dt.date(2026, 9, 1),  2026, 2),    # Tue after opening Saturday
    (dt.date(2026, 10, 6), 2026, 7),
    (dt.date(2027, 1, 5),  2026, 16),   # January belongs to the prior season
]
for day, exp_s, exp_w in cases:
    s, w = A.current_season_week(day)
    check(f"week inference {day} -> {exp_s} wk{exp_w}", (s, w) == (exp_s, exp_w), f"got {s} wk{w}")

# ---- 3. append_rows: header, append, schema widening
A.ARCHIVE_DIR = Path(TMP)
A.MANIFEST = Path(TMP) / "manifest.csv"
A.append_rows("t", [{"a": 1, "b": 2}])
A.append_rows("t", [{"a": 3, "b": 4}])
body = (Path(TMP) / "t.csv").read_text().strip().splitlines()
check("append_rows writes one header + 2 rows", len(body) == 3, body)

A.append_rows("t", [{"a": 5, "b": 6, "c": 7}])
body = (Path(TMP) / "t.csv").read_text().strip().splitlines()
check("schema widening keeps old rows", len(body) == 4, body)
check("schema widening adds the new column", body[0].strip().endswith("c"), body[0])
check("widened file preserves earlier values", body[1].startswith("1,2"), body[1])

# ---- 4. idempotency guard
A.record("sp_plus", 2026, 3, "ok", 138)
check("already_captured true for today's ok capture", A.already_captured("sp_plus", 2026, 3))
check("already_captured false for another week", not A.already_captured("sp_plus", 2026, 4))
check("already_captured false for another source", not A.already_captured("fpi", 2026, 3))
A.record("elo", 2026, 3, "fail", 0, "boom")
check("failed capture does not count as captured", not A.already_captured("elo", 2026, 3))

# ---- 5. missing API key is a clear error, not a stack trace
os.environ.pop("CFBD_API_KEY", None)
try:
    A.cfbd_get("/teams/fbs")
    check("missing key raises", False)
except RuntimeError as e:
    check("missing key raises a helpful message", "CFBD_API_KEY" in str(e) and "key" in str(e))

print()
print(f"{'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S): ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
