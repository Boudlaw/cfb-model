"""
Like-for-like baselines, so the lambda result cannot take credit for the
correctness fixes that shipped alongside it.

Three schemes, all on the SAME corrected data and the SAME game set:

  POOLED   the shipped walk_forward — every prior play weighted equally,
           sigma fixed at 16. This is what lambda has to beat.
  BLEND    lambda-weighted current form over a carryover-scaled prior.
  ELO      CFBD's own pregame Elo, difference converted to points by a
           divisor FIT ON TRAINING SEASONS. The archive ops doc records
           -7.1 points of mean signed deviation from the folk diff/25 rule,
           so the divisor is fitted, not assumed.
  SRS      a plain Simple Rating System (margin + strength of schedule,
           solved by least squares) fit point-in-time on the same cutoff.

Build plan Phase 4: if BOUD-EFF cannot beat a plain SRS, it is not earning its
complexity and the ensemble work should not start.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
from features import prepare_plays, restrict_to_teams
from ratings import BoudEff, MarginModel, game_features
from run_model import _data_dir, load_games, load_plays, attach_opening_lines

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
ALPHA = 175.0


# --------------------------------------------------------------------- pooled
def pooled_predictions(plays, games, all_teams, seasons, weeks_by_season,
                       alpha=ALPHA, sigma=16.0):
    """The shipped scheme, evaluated only on (season, week) pairs we also blend."""
    out = []
    for season in seasons:
        prior_plays = plays.loc[plays["season"] < season]
        prior_games = games.loc[games["season"] < season]
        if len(prior_plays) == 0 or len(prior_games) < 50:
            continue
        pm = BoudEff(alpha_epa=alpha, alpha_success=alpha).fit(prior_plays, teams=all_teams)
        pf = game_features(prior_games, pm.team_ratings()).dropna(
            subset=["d_net_epa", "margin"])
        mm = MarginModel().fit(pf)
        for week in weeks_by_season.get(season, []):
            m = BoudEff(alpha_epa=alpha, alpha_success=alpha).fit_through(
                plays, season=season, week=int(week), teams=all_teams)
            wk = games.loc[(games["season"] == season) & (games["week"] == week)]
            f = game_features(wk, m.team_ratings()).dropna(subset=["d_net_epa", "margin"])
            if f.empty:
                continue
            f = f.copy()
            f["predicted_margin"] = mm.predict(f)
            out.append(f)
    allp = pd.concat(out, ignore_index=True)
    allp["win_prob_home"] = B.win_probability(allp["predicted_margin"], sigma)
    return allp


# ------------------------------------------------------------------------ elo
def elo_predictions(games, seasons, weeks_by_season):
    """
    CFBD pregame Elo -> points, divisor fitted on prior seasons.

    diff/25 is folklore. The archive ops doc measured -7.1 points of mean
    signed deviation from it on opening weekend 2026, which is a bias, not noise.
    """
    g = games.dropna(subset=["home_pregame_elo", "away_pregame_elo"]).copy()
    g["d_elo"] = g["home_pregame_elo"].astype(float) - g["away_pregame_elo"].astype(float)
    out = []
    for season in seasons:
        tr = g.loc[g["season"] < season]
        if len(tr) < 100:
            continue
        X = np.column_stack([tr["d_elo"].to_numpy(float),
                             (~tr["neutral_site"].to_numpy(bool)).astype(float)])
        beta, *_ = np.linalg.lstsq(X, tr["margin"].to_numpy(float), rcond=None)
        slope, hfa = float(beta[0]), float(beta[1])
        resid = tr["margin"].to_numpy(float) - X @ beta
        sigma = float(np.std(resid, ddof=1))
        te = g.loc[(g["season"] == season) & (g["week"].isin(weeks_by_season.get(season, [])))]
        if te.empty:
            continue
        te = te.copy()
        te["predicted_margin"] = (te["d_elo"].astype(float) * slope
                                  + (~te["neutral_site"].astype(bool)).astype(float) * hfa)
        te["sigma"] = sigma
        te.attrs["divisor"] = 1.0 / slope if slope else float("nan")
        out.append(te)
        print(f"  elo {season}: 1 point per {1/slope:.1f} Elo, HFA {hfa:+.2f}, "
              f"sigma {sigma:.2f} (fit on {len(tr)} games)")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------------ srs
def srs_ratings(games: pd.DataFrame, teams: list[str]) -> pd.Series:
    """
    Plain SRS: rating_i - rating_j + hfa = margin, least squares, mean-centred.
    No weighting, no priors, no play-by-play. The complexity floor.
    """
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rows, ys = [], []
    for h, a, mg, ns in zip(games["home_team"], games["away_team"],
                            games["margin"], games["neutral_site"]):
        if h not in idx or a not in idx:
            continue
        r = np.zeros(n + 1)
        r[idx[h]] = 1.0
        r[idx[a]] = -1.0
        r[n] = 0.0 if ns else 1.0
        rows.append(r)
        ys.append(float(mg))
    if len(rows) < 20:
        return pd.Series(dtype=float)
    X = np.array(rows)
    y = np.array(ys)
    # Ridge-stabilised: an undefeated team with no common opponents is otherwise
    # unbounded in a pure least-squares solve.
    lam = 1.0
    A = X.T @ X + lam * np.eye(n + 1)
    A[n, n] -= lam
    beta = np.linalg.solve(A, X.T @ y)
    rat = pd.Series(beta[:n], index=teams)
    return rat - rat.mean(), float(beta[n])


def srs_predictions(games, seasons, weeks_by_season, teams):
    out = []
    for season in seasons:
        for week in weeks_by_season.get(season, []):
            tr = games.loc[(games["season"] < season)
                           | ((games["season"] == season) & (games["week"] < week))]
            if len(tr) < 200:
                continue
            rat, hfa = srs_ratings(tr, teams)
            te = games.loc[(games["season"] == season) & (games["week"] == week)].copy()
            if te.empty:
                continue
            te["predicted_margin"] = (
                te["home_team"].map(rat).to_numpy(float)
                - te["away_team"].map(rat).to_numpy(float)
                + (~te["neutral_site"].astype(bool)).astype(float) * hfa
            )
            te = te.dropna(subset=["predicted_margin"])
            out.append(te)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> int:
    with open(CACHE / "fits.pkl", "rb") as fh:
        cache = pickle.load(fh)
    with open(CACHE / "chosen.pkl", "rb") as fh:
        chosen = pickle.load(fh)

    data = _data_dir(None)
    raw = load_plays(data)
    games = attach_opening_lines(load_games(data), data)
    plays = prepare_plays(raw)
    fbs = sorted(set(games["home_team"]) | set(games["away_team"]))
    plays = restrict_to_teams(plays, fbs)
    all_teams = cache["all_teams"]

    seasons = [2024, 2025]
    weeks_by_season = {s: sorted(cache["in_season"].get(s, {})) for s in seasons}

    # The exact game set the blend scored, so nothing differs but the method.
    blends = {s: pd.read_pickle(CACHE / f"blend_{s}.pkl") for s in seasons}
    keep_ids = {s: set(blends[s]["game_id"]) for s in seasons}

    print("=== ELO divisor fits ===")
    elo = elo_predictions(games, seasons, weeks_by_season)
    print("\n=== SRS ===")
    srs = srs_predictions(games, seasons, weeks_by_season, all_teams)
    print("\n=== POOLED (shipped) ===")
    pooled = pooled_predictions(plays, games, all_teams, seasons, weeks_by_season)

    results = {}
    for name, frame, sig in (
        ("POOLED (shipped, sigma=16)", pooled, 16.0),
        ("BLEND (lambda, sigma from train)", pd.concat(blends.values(), ignore_index=True), None),
        ("ELO (fitted divisor)", elo, None),
        ("SRS (plain)", srs, None),
    ):
        if frame is None or frame.empty:
            continue
        rows = []
        for s in seasons:
            sub = frame.loc[(frame["season"] == s) & (frame["game_id"].isin(keep_ids[s]))]
            if sub.empty:
                continue
            sg = sig
            if sg is None:
                sg = (float(sub["sigma"].iloc[0]) if "sigma" in sub
                      else float(np.std(sub["margin"] - sub["predicted_margin"], ddof=1)))
            line = sub["opening_line_home"].to_numpy() if "opening_line_home" in sub else None
            m = B.evaluate(sub["margin"], sub["predicted_margin"], sg, line)
            m["season"] = s
            rows.append(m)
        allsub = frame.loc[frame.apply(
            lambda r: r["season"] in keep_ids and r["game_id"] in keep_ids[r["season"]], axis=1)]
        sg = sig
        if sg is None:
            sg = (float(allsub["sigma"].mean()) if "sigma" in allsub
                  else float(np.std(allsub["margin"] - allsub["predicted_margin"], ddof=1)))
        line = allsub["opening_line_home"].to_numpy() if "opening_line_home" in allsub else None
        tot = B.evaluate(allsub["margin"], allsub["predicted_margin"], sg, line)
        results[name] = {"per_season": pd.DataFrame(rows), "all": tot}

    print("\n" + "=" * 78)
    print("LIKE-FOR-LIKE, 2024+2025, identical games, all on corrected data")
    print("=" * 78)
    hdr = f"{'scheme':<34}{'n':>6}{'MAE':>8}{'win%':>8}{'Brier':>8}{'ATS%':>8}{'sigma':>7}"
    print(hdr)
    print("-" * 78)
    for name, r in results.items():
        m = r["all"]
        print(f"{name:<34}{m['n']:>6}{m['mae']:>8.2f}{m['winner_accuracy']*100:>8.1f}"
              f"{m['brier']:>8.4f}{m.get('ats_pct', float('nan'))*100:>8.1f}"
              f"{m['sigma_used']:>7.2f}")
    m = results["BLEND (lambda, sigma from train)"]["all"]
    print("-" * 78)
    print(f"{'MARKET (same games)':<34}{m['n']:>6}{m['market_mae']:>8.2f}"
          f"{m['market_winner_accuracy']*100:>8.1f}")

    print("\nper season:")
    for name, r in results.items():
        ps = r["per_season"]
        if ps.empty:
            continue
        s = "  ".join(f"{int(row['season'])}: MAE {row['mae']:.2f} / "
                      f"win {row['winner_accuracy']*100:.1f}% / "
                      f"ATS {row.get('ats_pct', float('nan'))*100:.1f}%"
                      for _, row in ps.iterrows())
        print(f"  {name:<34} {s}")

    with open(CACHE / "baselines.pkl", "wb") as fh:
        pickle.dump({k: v["all"] for k, v in results.items()}, fh)
    print("\nchosen blend hyperparameters:", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
