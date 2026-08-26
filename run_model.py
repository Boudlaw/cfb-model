#!/usr/bin/env python3
"""
Driver: load ingested data, run the Phase 2 gate, tune alpha, walk forward, report.

Run this AFTER the historical-ingest workflow has committed data/ to the repo.
It does no networking, so it works anywhere the repo can be cloned.

    python run_model.py                      # full run, default alpha grid
    python run_model.py --alpha 175          # skip tuning
    python run_model.py --first-week 4
    python run_model.py --out report.txt

WHAT TO READ IN THE OUTPUT, IN ORDER
------------------------------------
1. The Phase 2 gate. If success rate is not near 40%, STOP. Nothing below it
   means anything, and the fault is in the ingest mapping, not the model.
2. `unknown_play_types`. Non-empty means CFBD changed something and
   SCRIMMAGE_PLAY_TYPES in features.py needs a look.
3. The WARNINGS block. An MAE under 12 or ATS over 55% is a symptom of a leak,
   not a result. The most likely leak in this project is a season-level rating
   used to predict a game inside that season.
4. Only then the metrics — and read them against the benchmarks printed beside
   them, never alone.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import features as F
import ratings as R


def _data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / ".git").exists() and (cand / "data").exists():
            return cand / "data"
    return here / "data"


def load_plays(data: Path) -> pd.DataFrame:
    files = sorted(data.glob("plays_*.csv.gz"))
    if not files:
        raise SystemExit(
            f"no plays_*.csv.gz in {data}. Run the 'Historical CFB ingest' "
            "workflow (Actions tab) first — the CFBD API is not reachable from "
            "the modeling environment."
        )
    frames = [pd.read_csv(f, compression="gzip", low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(df):,} raw plays from {len(files)} season file(s)")
    return df


def load_games(data: Path) -> pd.DataFrame:
    files = sorted(data.glob("games_*.csv.gz"))
    if not files:
        raise SystemExit(f"no games_*.csv.gz in {data}")
    g = pd.concat(
        [pd.read_csv(f, compression="gzip", low_memory=False) for f in files],
        ignore_index=True,
    )
    # FBS-vs-FBS, completed, with a real score. Anything else cannot train or
    # score a rating: an FCS opponent has no rating to difference against.
    before = len(g)
    g = g.loc[
        (g["home_classification"] == "fbs")
        & (g["away_classification"] == "fbs")
        & g["home_points"].notna()
        & g["away_points"].notna()
    ].copy()
    g["margin"] = g["home_points"].astype(float) - g["away_points"].astype(float)
    g["neutral_site"] = g["neutral_site"].astype(str).str.lower().isin(("true", "1"))
    print(f"loaded {len(g):,} FBS-vs-FBS completed games (from {before:,} rows)")
    return g


def attach_opening_lines(games: pd.DataFrame, data: Path) -> pd.DataFrame:
    """
    Average the OPENING spread across providers and convert to a home margin.

    CFBD's spread convention is negative when the home team is favored, so the
    market's predicted home margin is the negated spread. The opening line is
    used deliberately: published results find whatever edge exists in public
    information lives early in the week and is statistically insignificant
    against midweek and closing lines.
    """
    files = sorted(data.glob("lines_*.csv.gz"))
    if not files:
        print("no lines_*.csv.gz found; ATS diagnostics will be skipped")
        return games
    ln = pd.concat(
        [pd.read_csv(f, compression="gzip", low_memory=False) for f in files],
        ignore_index=True,
    )
    ln["spread_open"] = pd.to_numeric(ln["spread_open"], errors="coerce")
    agg = (
        ln.dropna(subset=["spread_open"])
        .groupby("game_id", as_index=False)["spread_open"]
        .mean()
    )
    agg["opening_line_home"] = -agg["spread_open"]
    out = games.merge(agg[["game_id", "opening_line_home"]], on="game_id", how="left")
    have = out["opening_line_home"].notna().sum()
    print(f"attached opening lines to {have:,}/{len(out):,} games")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir")
    ap.add_argument("--alpha", type=float, default=None,
                    help="skip tuning and use this alpha")
    ap.add_argument("--first-week", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=B.SIGMA_OWN_MODEL)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    data = _data_dir(args.data_dir)
    print(f"data dir: {data}\n")

    raw = load_plays(data)
    games = attach_opening_lines(load_games(data), data)

    print("\n=== Phase 2 gate ===")
    rep = F.drop_report(raw)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    if rep["unknown_play_types"]:
        print("  !! unknown play types present — review SCRIMMAGE_PLAY_TYPES")

    plays = F.prepare_plays(raw)
    gate = F.sanity_report(plays)
    for k, v in gate.items():
        print(f"  {k}: {v}")
    if not gate["passes"]:
        print("\nGATE FAILED. Fix the ingest before reading anything below it.")
        return 2

    alpha = args.alpha
    if alpha is None:
        print("\n=== tuning alpha on held-out game margin ===")
        grid = R.tune_alpha(plays, games)
        print(grid.to_string(index=False))
        if grid.empty:
            print("tuning produced nothing; falling back to the plan default")
            alpha = R.DEFAULT_ALPHA_EPA
        else:
            alpha = float(grid.iloc[0]["alpha"])
        print(f"selected alpha = {alpha:g}")

    print(f"\n=== walk-forward (season-as-fold, alpha={alpha:g}) ===")
    res = B.walk_forward(
        plays, games, alpha=alpha, first_week=args.first_week, sigma=args.sigma
    )

    print("\n=== report ===")
    report = B.format_report(res)
    print(report)

    print("\n=== per season ===")
    if not res.per_season.empty:
        cols = [c for c in ("season", "n", "mae", "rmse", "mean_signed_error",
                            "winner_accuracy", "ats_pct") if c in res.per_season]
        print(res.per_season[cols].to_string(index=False))

    print("\n=== calibration ===")
    if len(res.predictions):
        print(
            B.calibration_table(
                res.predictions["margin"], res.predictions["predicted_margin"], args.sigma
            ).to_string(index=False)
        )

    skipped = res.fold_log.loc[res.fold_log["status"] != "ok"] if not res.fold_log.empty else pd.DataFrame()
    if not skipped.empty:
        print(f"\n=== {len(skipped)} fold(s) not predicted (nothing silently dropped) ===")
        print(skipped.head(25).to_string(index=False))

    if args.out:
        Path(args.out).write_text(report + "\n")
        print(f"\nwrote {args.out}")
    if len(res.predictions):
        outp = data / "walk_forward_predictions.csv"
        keep = [c for c in ("season", "week", "game_id", "home_team", "away_team",
                            "margin", "predicted_margin", "win_prob_home",
                            "opening_line_home", "d_net_epa", "d_net_succ")
                if c in res.predictions]
        res.predictions[keep].to_csv(outp, index=False)
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
