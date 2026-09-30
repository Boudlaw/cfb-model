"""
Phase 6 — the weekly prediction the model never had.

Everything upstream of this file is a backtest. This is the first code that
produces a number for a game that has not been played.

Configuration is NOT chosen here. It is read from what lambda_experiment.py and
ensemble_test.py selected on 2024 and confirmed once on held-out 2025:

    ratings   BOUD-EFF, alpha 175, logistic on success, linear on EPA
    recency   lambda ramp to week 6, carryover 1.0  (lambda = 0.80 at week 5)
    margin    least squares on d_net_epa, d_net_succ, d_elo, HFA
    sigma     from training-fold residuals, never from test output
    training  seasons 2023-2025, blended features, same construction as here

Writes cache/week5_predictions.csv.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import lambda_experiment as LE
from ratings import blend_with_prior

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DATA = HERE / "data"

SEASON, WEEK = 2026, 5
CARRYOVER, RAMP_END = 1.0, 6
FEATURES = ["d_net_epa", "d_net_succ", "d_elo"]


def upcoming_games(season: int, week: int) -> pd.DataFrame:
    g = pd.read_csv(DATA / f"games_{season}.csv.gz", low_memory=False)
    g = g.loc[(g["week"] == week)
              & (g["home_classification"] == "fbs")
              & (g["away_classification"] == "fbs")].copy()
    g["neutral_site"] = g["neutral_site"].astype(str).str.lower().isin(("true", "1"))
    return g.reset_index(drop=True)


def market_lines(season: int, week: int) -> pd.DataFrame:
    ln = pd.read_csv(DATA / f"lines_{season}.csv.gz", low_memory=False)
    ln = ln.loc[ln["week"] == week].copy()
    ln["spread_open"] = pd.to_numeric(ln["spread_open"], errors="coerce")
    ln["spread"] = pd.to_numeric(ln.get("spread"), errors="coerce")
    agg = ln.groupby("game_id").agg(
        spread_open=("spread_open", "mean"),
        spread_current=("spread", "mean"),
        n_books=("spread_open", "count"),
    ).reset_index()
    # CFBD's spread is negative when the HOME team is favoured, so the market's
    # predicted home margin is the negated spread.
    agg["market_margin_open"] = -agg["spread_open"]
    agg["market_margin_now"] = -agg["spread_current"]
    return agg


def main() -> int:
    cache = LE.load()

    # ---- training set: prior seasons, featurised exactly as the target will be
    train_seasons = [s for s in sorted(cache["in_season"])
                     if s < SEASON and s in cache["prior"]]
    frames = []
    for s in train_seasons:
        for w in sorted(cache["in_season"][s]):
            f = LE.blended_features(cache, s, w, CARRYOVER, RAMP_END)
            if f is None:
                continue
            f = f.dropna(subset=["home_pregame_elo", "away_pregame_elo"])
            if f.empty:
                continue
            f = f.copy()
            f["d_elo"] = (f["home_pregame_elo"].astype(float)
                          - f["away_pregame_elo"].astype(float))
            f["hfa_term"] = (~f["neutral_site"].astype(bool)).astype(float)
            frames.append(f)
    train = pd.concat(frames, ignore_index=True)
    print(f"training on {len(train):,} games from seasons {train_seasons}")

    X = np.column_stack([train[c].to_numpy(float) for c in FEATURES]
                        + [train["hfa_term"].to_numpy(float)])
    y = train["margin"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma = float(np.std(resid, ddof=1))
    print("coefficients:")
    for c, b in zip(FEATURES + ["HFA"], beta):
        print(f"  {c:<12} {b:+.4f}")
    print(f"  sigma (training-fold residual SD) = {sigma:.3f}")

    # ---- current-week ratings
    ins = cache["in_season"][SEASON][WEEK]
    prior = cache["prior"][SEASON]
    lam = (WEEK - 1) / (RAMP_END - 1) if WEEK < RAMP_END else 1.0
    ratings = blend_with_prior(ins, prior, week=WEEK, carryover=CARRYOVER,
                               ramp_end=RAMP_END)
    print(f"\nratings: lambda={lam:.3f} on {ins.attrs.get('n_plays', 0):,} plays of "
          f"2026 form, {1-lam:.3f} on a prior fit from "
          f"{prior.attrs.get('n_plays', 0):,} plays of 2022-2025")

    ratings.to_csv(CACHE / f"ratings_{SEASON}w{WEEK}.csv", index=False)
    top = ratings.sort_values("net_epa", ascending=False)
    print("\ntop 12:", ", ".join(f"{r.team} {r.net_epa:+.3f}"
                                 for r in top.head(12).itertuples()))
    print("bottom 5:", ", ".join(f"{r.team} {r.net_epa:+.3f}"
                                 for r in top.tail(5).itertuples()))

    # ---- predict
    g = upcoming_games(SEASON, WEEK)
    r = ratings.set_index("team")
    for col in ("net_epa", "net_succ"):
        g[f"d_{col}"] = (g["home_team"].map(r[col]).to_numpy(float)
                         - g["away_team"].map(r[col]).to_numpy(float))
    g["d_elo"] = (g["home_pregame_elo"].astype(float)
                  - g["away_pregame_elo"].astype(float))
    g["hfa_term"] = (~g["neutral_site"].astype(bool)).astype(float)

    unrated = g.loc[g["d_net_epa"].isna(), ["home_team", "away_team"]]
    if len(unrated):
        print("\nunrated (dropped):")
        print(unrated.to_string(index=False))
    g = g.dropna(subset=["d_net_epa", "d_elo"]).copy()

    Xg = np.column_stack([g[c].to_numpy(float) for c in FEATURES]
                         + [g["hfa_term"].to_numpy(float)])
    g["model_margin"] = Xg @ beta
    g["win_prob_home"] = B.win_probability(g["model_margin"], sigma)

    # component views, for the side-by-side sheet
    b = dict(zip(FEATURES + ["HFA"], beta))
    g["boudeff_only"] = (g["d_net_epa"] * b["d_net_epa"]
                         + g["d_net_succ"] * b["d_net_succ"]
                         + g["hfa_term"] * b["HFA"])
    g["elo_only"] = g["d_elo"] * b["d_elo"] + g["hfa_term"] * b["HFA"]

    mk = market_lines(SEASON, WEEK)
    g = g.merge(mk, on="game_id", how="left")
    g["model_vs_market"] = g["model_margin"] - g["market_margin_now"]

    cols = ["game_id", "start_date", "home_team", "away_team", "neutral_site",
            "d_net_epa", "d_net_succ", "d_elo",
            "boudeff_only", "elo_only", "model_margin", "win_prob_home",
            "market_margin_open", "market_margin_now", "n_books",
            "model_vs_market"]
    out = g[cols].sort_values("model_vs_market", key=abs, ascending=False)
    out.to_csv(CACHE / f"week{WEEK}_predictions.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\n=== {SEASON} week {WEEK}: {len(out)} FBS-vs-FBS games, "
          f"sorted by disagreement with the market ===")
    show = out.copy()
    for c in ("boudeff_only", "elo_only", "model_margin", "market_margin_now",
              "model_vs_market"):
        show[c] = show[c].round(1)
    show["win_prob_home"] = (show["win_prob_home"] * 100).round(1)
    print(show[["home_team", "away_team", "boudeff_only", "elo_only",
                "model_margin", "win_prob_home", "market_margin_now",
                "model_vs_market"]].to_string(index=False))

    with open(CACHE / "week5_model.pkl", "wb") as fh:
        pickle.dump({"beta": beta, "features": FEATURES, "sigma": sigma,
                     "lambda": lam, "carryover": CARRYOVER,
                     "ramp_end": RAMP_END, "n_train": len(train)}, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
