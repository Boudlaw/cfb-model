#!/usr/bin/env python3
"""
Phase 4 — validation. This file decides whether anything else in the project is real.

TWO RULES, BOTH NON-NEGOTIABLE
-----------------------------
1. NEVER random k-fold. Opponent-adjusted ratings fit over a season embed every
   game's outcome into every team's coefficient, so any random split on such
   features is contaminated no matter how it is shuffled. CFBD's own tips post
   suggests "k-fold with shuffling by game ID" while simultaneously warning about
   leakage from team-specific stats; those two pieces of advice contradict each
   other. Follow the warning. Folds here are whole seasons, and within a season
   the ratings are refit week by week from prior weeks only.

2. Report every metric every time. MAE alone hides a systematic
   favorite/underdog tilt, which is why mean SIGNED error is in the table.

BENCHMARKS (Coleman 2025, 5,925 games, 29 systems)
--------------------------------------------------
    closing line            73.93% winners, 12.06 MAE
    best 5-model blend      74.14% winners, 12.26 MAE
    that blend, ATS         53.08% in validation -> 51.47% on the held-out test set
                            (break-even is 52.38% at -110)

Read that degradation twice. Validation-to-test decay is what honest
out-of-sample testing looks like. The blend cleared break-even in testing only
where it disagreed with the opening line by more than 3 points: 55.33% on 291 of
1,509 games.

SANITY THRESHOLDS ARE INVERTED ON PURPOSE
-----------------------------------------
MAE below 12 on a large sample, or ATS above 55% on all games, triggers a
warning, not a celebration. Those are the signatures of a leak, and the most
likely leak in this specific project is using a season-level SP+ or FPI rating
to predict a game inside that season. Believe the warning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ratings import (
    BoudEff,
    MarginModel,
    game_features,
    lambda_schedule,
)

# SD of this model's own residuals, used to turn a margin into a probability.
# ~16 is what real homegrown models actually achieve (CFBD's GBDT: 15.72 test /
# 16.77 live; Ed Feng 16.0; Coleman's metamodel 15.94).
#
# Do NOT use 14 here. 14.1 is the SD of margins around the VEGAS CLOSING SPREAD,
# a sharper predictor than this model; borrowing it would make every probability
# overconfident by roughly 2-4 percentage points. And the commonly quoted "13" is
# an NFL number with no college football source at all.
SIGMA_OWN_MODEL = 16.0
SIGMA_MARKET = 14.1
ATS_BREAKEVEN = 0.5238  # -110 juice
ATS_EDGE_THRESHOLD = 3.0  # points of disagreement with the OPENING line


def _phi(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, vectorised, no scipy dependency."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def win_probability(margin: np.ndarray, sigma: float = SIGMA_OWN_MODEL) -> np.ndarray:
    return _phi(np.asarray(margin, dtype=np.float64) / sigma)


def evaluate(
    actual_margin: np.ndarray,
    predicted_margin: np.ndarray,
    sigma: float = SIGMA_OWN_MODEL,
    opening_line_home: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    The full metric suite for one set of predictions.

    `opening_line_home` is the market's predicted home margin (i.e. the negated
    home spread), used only as a yardstick. Feeding the line in as a model
    FEATURE would produce excellent MAE and zero ability to detect an edge,
    because the result would be a line-adjuster wearing a model's clothes.
    """
    a = np.asarray(actual_margin, dtype=np.float64)
    p = np.asarray(predicted_margin, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    n = len(a)
    if n == 0:
        return {"n": 0}

    err = a - p
    home_won = (a > 0).astype(int)
    prob = win_probability(p, sigma)
    # Clip only for the log-loss term; an exact 0/1 probability would be an
    # infinite penalty and would say more about float precision than about calibration.
    pc = np.clip(prob, 1e-12, 1 - 1e-12)

    out: dict[str, Any] = {
        "n": int(n),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean_signed_error": float(np.mean(err)),
        "winner_accuracy": float(np.mean((p > 0).astype(int) == home_won)),
        "brier": float(np.mean((home_won - prob) ** 2)),
        "log_loss": float(
            -np.mean(home_won * np.log(pc) + (1 - home_won) * np.log(1 - pc))
        ),
        "sigma_used": sigma,
    }

    if opening_line_home is not None:
        m = np.asarray(opening_line_home, dtype=np.float64)[ok]
        has = np.isfinite(m)
        if has.sum() > 0:
            aa, pp, mm = a[has], p[has], m[has]
            disagreement = pp - mm
            # Model covers when the actual margin lands on the side of the spread
            # the model picked. Pushes are excluded from the denominator, which is
            # how a sportsbook settles them.
            picked_home = disagreement > 0
            covered_home = aa > mm
            push = np.isclose(aa, mm)
            live = ~push
            correct = (picked_home == covered_home) & live
            out["ats_n"] = int(live.sum())
            out["ats_pct"] = float(correct.sum() / live.sum()) if live.sum() else float("nan")
            out["ats_breakeven"] = ATS_BREAKEVEN
            out["market_mae"] = float(np.mean(np.abs(aa - mm)))
            out["market_winner_accuracy"] = float(
                np.mean((mm > 0).astype(int) == (aa > 0).astype(int))
            )
            edge = np.abs(disagreement) > ATS_EDGE_THRESHOLD
            sel = edge & live
            out["ats_n_edge"] = int(sel.sum())
            out["ats_pct_edge"] = (
                float((correct & sel).sum() / sel.sum()) if sel.sum() else float("nan")
            )
            out["mean_disagreement"] = float(np.mean(disagreement))

    out["warnings"] = sanity_warnings(out)
    return out


def sanity_warnings(m: dict[str, Any]) -> list[str]:
    """Symptoms of a leak, phrased as such."""
    w = []
    n = m.get("n", 0)
    if n >= 300 and m.get("mae", 99) < 12.0:
        w.append(
            f"MAE {m['mae']:.2f} on {n} games beats the closing line (12.06). "
            "Assume a data leak until proven otherwise."
        )
    if n >= 300 and m.get("winner_accuracy", 0) > 0.76:
        w.append(
            f"Winner accuracy {m['winner_accuracy']:.1%} exceeds the ~74% ceiling "
            "the literature finds. Look for leakage."
        )
    if m.get("ats_n", 0) >= 300 and m.get("ats_pct", 0) > 0.55:
        w.append(
            f"ATS {m['ats_pct']:.1%} on {m['ats_n']} games is above 55%. "
            "That is a symptom, not an achievement."
        )
    if abs(m.get("mean_signed_error", 0.0)) > 1.5 and n >= 100:
        w.append(
            f"Mean signed error {m['mean_signed_error']:+.2f} indicates a "
            "systematic favorite/underdog tilt that MAE is hiding."
        )
    return w


def calibration_table(
    actual_margin: np.ndarray,
    predicted_margin: np.ndarray,
    sigma: float = SIGMA_OWN_MODEL,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Predicted vs. realised home-win rate by probability bucket. The most honest
    single view of the model: a well-calibrated 70% bucket wins ~70% of the time,
    and a model that is accurate but overconfident shows up here and nowhere else.
    """
    p = win_probability(predicted_margin, sigma)
    won = (np.asarray(actual_margin, dtype=np.float64) > 0).astype(int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        rows.append(
            {
                "bucket": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(sel.sum()),
                "mean_predicted": float(p[sel].mean()),
                "realised": float(won[sel].mean()),
                "gap": float(won[sel].mean() - p[sel].mean()),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    per_season: pd.DataFrame
    fold_log: pd.DataFrame


def walk_forward(
    plays: pd.DataFrame,
    games: pd.DataFrame,
    alpha: float = 175.0,
    first_week: int = 3,
    sigma: float = SIGMA_OWN_MODEL,
    ramp_end: int = 7,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Season-as-fold walk-forward. For each (season, week) the ratings are refit on
    plays strictly before that week, the ratings-to-points mapping is fit on
    PRIOR SEASONS ONLY, and that week's games are predicted once and never
    revisited.

    Two separate leakage surfaces get closed here, and it is worth being explicit
    because closing only the first is the common half-fix:

      1. the RATINGS may not see the week being predicted  -> fit_through()
      2. the MarginModel may not see the season being predicted -> trained on
         earlier seasons only

    `first_week` defaults to 3 because weeks 1-2 have too few plays for the
    ratings to mean anything (300 plays ~ 4.3 games before the estimate is even
    the size of the signal). Predicting them needs the preseason prior, which is
    Phase 5 work; until then they are honestly excluded and reported as such.
    """
    seasons = sorted(games["season"].unique())
    all_teams = sorted(set(plays["offense"]) | set(plays["defense"]))
    preds: list[pd.DataFrame] = []
    log: list[dict[str, Any]] = []

    for season in seasons:
        prior_games = games.loc[games["season"] < season]
        prior_plays = plays.loc[plays["season"] < season]
        if len(prior_plays) == 0 or len(prior_games) < 50:
            log.append(
                {
                    "season": season,
                    "week": None,
                    "status": "skipped",
                    "reason": "no prior season to fit the ratings-to-points mapping",
                }
            )
            continue

        # The margin model is fit once per season, on prior seasons only.
        prior_model = BoudEff(alpha_epa=alpha, alpha_success=alpha).fit(
            prior_plays, teams=all_teams
        )
        prior_feat = game_features(prior_games, prior_model.team_ratings()).dropna(
            subset=["d_net_epa", "margin"]
        )
        if len(prior_feat) < 10:
            log.append(
                {"season": season, "week": None, "status": "skipped",
                 "reason": "too few prior games to fit MarginModel"}
            )
            continue
        margin_model = MarginModel().fit(prior_feat)

        for week in sorted(games.loc[games["season"] == season, "week"].unique()):
            if week < first_week:
                log.append({"season": season, "week": int(week), "status": "skipped",
                            "reason": f"week < first_week={first_week}"})
                continue
            wk_games = games.loc[
                (games["season"] == season) & (games["week"] == week)
            ]
            if wk_games.empty:
                continue
            try:
                model = BoudEff(alpha_epa=alpha, alpha_success=alpha).fit_through(
                    plays, season=season, week=int(week), teams=all_teams
                )
            except (ValueError, AssertionError) as exc:
                log.append({"season": season, "week": int(week), "status": "error",
                            "reason": str(exc)[:200]})
                continue

            feat = game_features(wk_games, model.team_ratings())
            feat = feat.dropna(subset=["d_net_epa", "margin"])
            if feat.empty:
                log.append({"season": season, "week": int(week), "status": "skipped",
                            "reason": "no rated games this week"})
                continue
            feat = feat.copy()
            feat["predicted_margin"] = margin_model.predict(feat)
            feat["win_prob_home"] = win_probability(feat["predicted_margin"], sigma)
            feat["lambda_used"] = lambda_schedule(int(week), ramp_end=ramp_end)
            feat["fit_plays"] = model.n_plays
            preds.append(feat)
            log.append({"season": season, "week": int(week), "status": "ok",
                        "reason": f"{len(feat)} games, fit on {model.n_plays} plays"})
            if verbose:
                print(
                    f"  {season} wk{int(week):>2}  games={len(feat):>3}  "
                    f"fit_plays={model.n_plays:>7}",
                    flush=True,
                )

    if not preds:
        return WalkForwardResult(
            predictions=pd.DataFrame(),
            metrics={"n": 0, "warnings": ["walk_forward produced no predictions"]},
            per_season=pd.DataFrame(),
            fold_log=pd.DataFrame(log),
        )

    allp = pd.concat(preds, ignore_index=True)
    line = allp["opening_line_home"].to_numpy() if "opening_line_home" in allp else None
    metrics = evaluate(allp["margin"], allp["predicted_margin"], sigma, line)

    rows = []
    for season, grp in allp.groupby("season"):
        gl = grp["opening_line_home"].to_numpy() if "opening_line_home" in grp else None
        m = evaluate(grp["margin"], grp["predicted_margin"], sigma, gl)
        m["season"] = season
        rows.append(m)
    per_season = pd.DataFrame(rows)

    return WalkForwardResult(
        predictions=allp,
        metrics=metrics,
        per_season=per_season,
        fold_log=pd.DataFrame(log),
    )


def format_report(res: WalkForwardResult) -> str:
    """Human-readable summary, benchmarks inline so a number is never read alone."""
    m = res.metrics
    if m.get("n", 0) == 0:
        return "No predictions produced.\n" + "\n".join(m.get("warnings", []))
    L = []
    L.append(f"Games predicted        : {m['n']}")
    L.append(f"MAE                    : {m['mae']:.2f}   (closing line 12.06, best public blend 12.26)")
    L.append(f"RMSE                   : {m['rmse']:.2f}   (real models 15.7-16.8)")
    L.append(f"Mean signed error      : {m['mean_signed_error']:+.2f}   (Coleman +0.19)")
    L.append(f"Winner accuracy        : {m['winner_accuracy']:.2%}   (~74% ceiling)")
    L.append(f"Brier                  : {m['brier']:.4f}   (0.25 = coinflip, <0.20 good)")
    L.append(f"Log loss               : {m['log_loss']:.4f}   (0.693 = coinflip)")
    if "ats_pct" in m:
        L.append(f"Market MAE (same games): {m['market_mae']:.2f}")
        L.append(f"ATS vs opening line    : {m['ats_pct']:.2%} on {m['ats_n']} "
                 f"(break-even {ATS_BREAKEVEN:.2%})")
        if m.get("ats_n_edge", 0):
            L.append(f"ATS where |disagree|>{ATS_EDGE_THRESHOLD:g} : "
                     f"{m['ats_pct_edge']:.2%} on {m['ats_n_edge']} games")
    if m.get("warnings"):
        L.append("")
        L.append("WARNINGS — read these before believing anything above:")
        for w in m["warnings"]:
            L.append(f"  - {w}")
    return "\n".join(L)
