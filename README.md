# CFB model — Phase 1: the point-in-time archive

The archiver captures the public ratings and betting lines every week, **before** the
games, and keeps every capture forever.

## Why this runs before any modeling

CollegeFootballData's SP+ and FPI endpoints return one rating per season, with no
`week` parameter. Pull 2023 SP+ to "predict" a Week 4 2023 game and the rating
already contains that game's result, plus every game after it. The backtest looks
excellent and means nothing.

So the ratings have to be captured weekly, in advance. Weeks that go uncaptured are
gone — there is no backfill. That is why this is Phase 1 and the modeling is Phase 3.

Sources split into two kinds, and the script labels them:

| Kind | Sources | If a capture fails |
|---|---|---|
| **critical** — cannot be backfilled | SP+, FPI, betting lines, Sonny Moore, Pi-Rate | Exit code 2, run goes red, you get an email |
| **recoverable** — the endpoint accepts `week` | Elo, SRS, CORE, games | Logged, retry later |

## Setup

1. **Get a free CFBD API key** — https://collegefootballdata.com/key (instant, email only).
   The free tier allows 1,000 calls/month; this job uses about 40. If you later pull
   four seasons of play-by-play for the modeling phase, move to Tier 1 ($1/mo) or
   Tier 3 ($10/mo).

2. **Store the key** as a repository secret named `CFBD_API_KEY`
   (Settings → Secrets and variables → Actions → New repository secret).

3. **Verify before trusting it:**
   ```bash
   export CFBD_API_KEY=...
   python cfb_archive/archive.py --smoke      # every endpoint answers?
   python cfb_archive/archive.py --dry-run    # parses correctly, writes nothing
   python cfb_archive/archive.py              # capture for real
   ```

4. The workflow then runs itself every Tuesday at 8:00 AM Eastern, August through
   January, and commits to `archive/`. Trigger it by hand any time from the Actions
   tab (Run workflow).

## Layout

```
archive/
  manifest.csv          every capture attempt: when, what, ok/fail, row count
  sp_plus.csv           append-only, one block of rows per weekly capture
  fpi.csv
  lines.csv             includes opening lines — the ones that matter
  elo.csv  srs.csv  core.csv  games.csv
  sonny_moore.csv
  raw/2026/week03/      untouched JSON and HTML exactly as received
```

Two rules make the archive trustworthy:

- **Append-only.** Nothing is rewritten. A rating row means "this is what the system
  said at this timestamp," which is only true if it is never touched again.
- **Raw bytes kept.** If a parser turns out to be wrong, history is re-parsed from
  `raw/` rather than lost. Parsers will be wrong eventually; the Pi-Rate table layout
  changes between seasons.

Every row carries `captured_at`, `capture_season` and `capture_week`. The capture
timestamp is the part that makes the row usable — the rating alone is not.

## Tests

```bash
python scripts/test_offline.py
```

21 checks covering the Sonny Moore parser (multi-word names, embedded periods,
three-digit ranks, the truncated-page tripwire), season/week inference across the
January rollover, CSV schema widening, and the same-day idempotency guard. No network
needed. Run it before every commit.

## Known gaps

- **Pi-Rate parser is deliberately unwritten.** The page is a WordPress post whose
  table layout changes between seasons, and the 2026 in-season format doesn't exist
  yet. The job archives the raw HTML unconditionally from week 1, so nothing is lost;
  the parser gets written against a real page and applied to history retroactively.
- **Endpoint paths need one live smoke test.** They were written from CFBD's v2 docs,
  not verified against the live API. `--smoke` exists for exactly this and runs as the
  first step of every scheduled run.
- **FEI, Sagarin and Massey are not included.** FEI is robots-disallowed with no API;
  Sagarin is an HTTP-only site that redirects HTTPS to HTTP and breaks most scrapers;
  Massey's tables are JavaScript-rendered. Sonny Moore and Pi-Rate cover the
  "independent closed methodology" slot in the ensemble.

## What comes next

Phase 2 is success rate and EPA from play-by-play; Phase 3 is the opponent-adjusted
ratings. Neither is urgent — play-by-play for 2022–2025 is static and can be pulled
any time. Only the weekly capture is time-sensitive, which is why it exists first.

Full design in the build plan (`claude/cfb-model-build-plan.md` in the CFB project).
