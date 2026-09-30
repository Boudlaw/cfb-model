"""
Build and cache the BoudEff fits the lambda experiment needs.

The expensive object is a ridge + logistic fit on a sparse team design. Tuning
the lambda ramp and the carryover coefficient does NOT require refitting any of
them — blending happens on the ratings tables, downstream. So fit each one once,
pickle it, and let the grid search be cheap.

Two kinds of fit are cached:

  prior[S]         ratings from every play in seasons < S
  in_season[S][w]  ratings from season S plays strictly before week w

Both are point-in-time by construction. The in-season half is the one that could
leak and it is filtered on (season, week); the prior half is filtered on season
alone and cannot.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

from features import prepare_plays, restrict_to_teams, sanity_report, drop_report
from ratings import BoudEff
from run_model import _data_dir, load_games, load_plays, attach_opening_lines

ALPHA = 175.0
FIRST_WEEK = 3
MIN_INSEASON_PLAYS = 500
OUT = Path(__file__).resolve().parent / "cache"


def main() -> int:
    data = _data_dir(None)
    OUT.mkdir(exist_ok=True)

    raw = load_plays(data)
    games = load_games(data)
    games = attach_opening_lines(games, data)

    rep = drop_report(raw)
    print("drop_report:", {k: v for k, v in rep.items() if k != "unknown_play_types"})
    print("unknown_play_types:", rep["unknown_play_types"])
    if rep["unknown_play_types"]:
        raise SystemExit("unknown play types present — review SCRIMMAGE_PLAY_TYPES")

    plays = prepare_plays(raw)
    print(f"prepared {len(plays):,} plays, {plays['offense'].nunique()} distinct offenses")

    # FBS universe comes from the GAMES table, which carries classification.
    fbs = sorted(
        set(games["home_team"]) | set(games["away_team"])
    )
    plays = restrict_to_teams(plays, fbs)
    print(f"restricted to {len(fbs)} FBS teams: {len(plays):,} plays "
          f"({plays.attrs['restrict_dropped']:,} dropped)")

    gate = sanity_report(plays)
    print("\n--- Phase 2 gate ---")
    for k in ("success_rate", "mean_epa", "sd_epa", "n_plays", "n_teams",
              "max_team_share", "epa_ordering_applicable", "epa_ordering_ok", "passes"):
        print(f"  {k:<24} {gate[k]}")
    for p in gate["epa_ordering_pairs"]:
        flag = "ok " if p["ok"] else "BAD"
        print(f"    {flag} {p['better']:<30} {p['epa_better']:+.3f}  >  "
              f"{p['worse']:<22} {p['epa_worse']:+.3f}")
    if not gate["passes"]:
        raise SystemExit("Phase 2 gate FAILED: " + gate.get("reason", ""))
    if not gate["epa_ordering_applicable"]:
        raise SystemExit("ordering gate not applicable on real data — that is a bug")

    all_teams = sorted(set(plays["offense"]) | set(plays["defense"]))
    print(f"\n{len(all_teams)} teams in the design")

    seasons = sorted(plays["season"].unique())
    cache: dict = {
        "all_teams": all_teams, "alpha": ALPHA, "first_week": FIRST_WEEK,
        "prior": {}, "in_season": {}, "gate": gate,
    }

    for season in seasons:
        prior_plays = plays.loc[plays["season"] < season]
        if len(prior_plays) == 0:
            print(f"{season}: no prior season — cannot be predicted, skipping")
            continue
        t0 = time.time()
        m = BoudEff(alpha_epa=ALPHA, alpha_success=ALPHA).fit(prior_plays, teams=all_teams)
        r = m.team_ratings()
        r.attrs["n_plays"] = m.n_plays
        cache["prior"][int(season)] = r
        print(f"  prior[{season}]      {m.n_plays:>7,} plays  ({time.time() - t0:.1f}s)",
              flush=True)

        cache["in_season"][int(season)] = {}
        weeks = sorted(plays.loc[plays["season"] == season, "week"].unique())
        for week in weeks:
            w = int(week) + 1  # ratings available going INTO week w
            cur = plays.loc[(plays["season"] == season) & (plays["week"] < w)]
            if w < FIRST_WEEK or len(cur) < MIN_INSEASON_PLAYS:
                continue
            t0 = time.time()
            mw = BoudEff(alpha_epa=ALPHA, alpha_success=ALPHA).fit(cur, teams=all_teams)
            rw = mw.team_ratings()
            rw.attrs["n_plays"] = mw.n_plays
            cache["in_season"][int(season)][w] = rw
            print(f"  in_season[{season}][{w:>2}] {mw.n_plays:>7,} plays  "
                  f"({time.time() - t0:.1f}s)", flush=True)

    cache["games"] = games
    path = OUT / "fits.pkl"
    with open(path, "wb") as fh:
        pickle.dump(cache, fh)
    print(f"\nwrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
