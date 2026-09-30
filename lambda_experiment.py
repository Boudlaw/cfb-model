"""
Deficiency 1 and 2 from cfb-model-first-results.md, measured rather than assumed.

DEFICIENCY 1 — all history was pooled with EQUAL weight.
`walk_forward` computes `lambda_used`, records it in the output, and never
applies it. A 2025 week-3 rating measured 5.7% current-season data; the other
94% was 2022-2024, with a 2022 play counting exactly as much as last week's.
Against 30%+ annual roster turnover that is a four-year team average wearing a
current-form label.

DEFICIENCY 2 — sigma was hardcoded at 16 while measured residual SD was 17.40.
Here sigma is computed from TRAINING-FOLD residuals only. Taking 17.40 from the
earlier test output and baking it in would have been fitting on the test set.

PROTOCOL, fixed before looking at any result
--------------------------------------------
Blended ratings need a prior fit, so the earliest predictable season under this
scheme is 2024 (2023 has only 2022 behind it, which gives no prior-season
MarginModel training data built the same way). That leaves two seasons:

    tune on   2024
    hold out  2025      <- reported once, at the end, not iterated on

The baseline is re-scored on EXACTLY the same game set, so the comparison is
like-for-like and not a side effect of predicting different games.
"""
from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
from ratings import MarginModel, blend_with_prior, game_features, lambda_schedule

CACHE = Path(__file__).resolve().parent / "cache" / "fits.pkl"

# --- the carryover decision, made once and recorded -------------------------
#
# Grid-searching carryover on 2024 puts the argmin near 2.0 (MAE 12.804). It is
# not taken, for three reasons, and this note exists so nobody re-opens it and
# "discovers" a 0.09-point improvement:
#
#   1. The surface is FLAT. MAE runs 12.89 / 12.81 / 12.80 / 12.83 / 12.90
#      across carryover 1.0 / 1.5 / 2.0 / 2.5 / 3.0 — a 0.1-point basin spanning
#      a threefold change in the parameter. The standard error of MAE on 664
#      games is about 0.39. The whole basin is a quarter of one SE wide.
#   2. Winner accuracy does not move at all across it (71.8%-72.4%).
#   3. Above 1.0 the parameter has no interpretation. It is supposed to be the
#      regression of a season's rating on the prior season's; fit_carryover()
#      measures 0.54 on net_epa (r=0.55), and the prior's rating SD (0.113) is
#      already WIDER than an in-season fit's (0.095), so there is nothing to
#      rescale. What carryover>1 actually does is shift weight from net_succ to
#      net_epa — b_succ falls 21.5 -> 15.2 as carryover goes 0.55 -> 1.5. That
#      is a reparameterisation of the margin model wearing a prior's name.
#
# So: the prior enters at its own scale, no free multiplier, and ramp_end —
# which the data has a real opinion about (MAE 12.81 to 13.43 across its range,
# optimum interior) — is the only thing tuned.
CARRYOVER = 1.0
RAMP_END_GRID = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)


def load() -> dict:
    with open(CACHE, "rb") as fh:
        return pickle.load(fh)


def blended_features(cache: dict, season: int, week: int,
                     carryover: float, ramp_end: int) -> pd.DataFrame | None:
    ins = cache["in_season"].get(season, {}).get(week)
    prior = cache["prior"].get(season)
    if ins is None or prior is None:
        return None
    r = blend_with_prior(ins, prior, week=week, carryover=carryover,
                         ramp_end=ramp_end)
    games = cache["games"]
    wk = games.loc[(games["season"] == season) & (games["week"] == week)]
    if wk.empty:
        return None
    feat = game_features(wk, r).dropna(subset=["d_net_epa", "margin"])
    return feat if not feat.empty else None


def training_frame(cache: dict, target_season: int, carryover: float,
                   ramp_end: int) -> pd.DataFrame:
    """Every prior season's games, featurised the SAME way the target will be."""
    frames = []
    for p in sorted(cache["in_season"]):
        if p >= target_season or p not in cache["prior"]:
            continue
        for w in sorted(cache["in_season"][p]):
            f = blended_features(cache, p, w, carryover, ramp_end)
            if f is not None:
                frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def predict_season(cache: dict, season: int, carryover: float, ramp_end: int
                   ) -> tuple[pd.DataFrame, float, dict]:
    train = training_frame(cache, season, carryover, ramp_end)
    if len(train) < 50:
        return pd.DataFrame(), float("nan"), {}
    mm = MarginModel().fit(train)

    # sigma from TRAINING-fold residuals. Never from the test predictions.
    resid = train["margin"].to_numpy(float) - mm.predict(train)
    sigma = float(np.std(resid, ddof=1))

    out = []
    for w in sorted(cache["in_season"].get(season, {})):
        f = blended_features(cache, season, w, carryover, ramp_end)
        if f is None:
            continue
        f = f.copy()
        f["predicted_margin"] = mm.predict(f)
        f["lambda_used"] = lambda_schedule(w, ramp_end=ramp_end)
        out.append(f)
    if not out:
        return pd.DataFrame(), sigma, {}
    allp = pd.concat(out, ignore_index=True)
    allp["win_prob_home"] = B.win_probability(allp["predicted_margin"], sigma)
    info = {"b_epa": mm.b_epa, "b_succ": mm.b_succ, "n_train": len(train),
            "sigma_train": sigma}
    return allp, sigma, info


def score(allp: pd.DataFrame, sigma: float) -> dict:
    line = allp["opening_line_home"].to_numpy() if "opening_line_home" in allp else None
    return B.evaluate(allp["margin"], allp["predicted_margin"], sigma, line)


def baseline_on(cache: dict, season: int, restrict_ids: set | None = None) -> dict:
    """
    The current shipped behaviour: ratings pooled with equal weight over every
    prior play, sigma fixed at 16. Scored on the same games as the blend.
    """
    from features import prepare_plays, restrict_to_teams
    raise NotImplementedError  # baseline comes from run_baseline.py


def main() -> int:
    cache = load()
    seasons = [s for s in sorted(cache["in_season"]) if s in cache["prior"]]
    predictable = [s for s in seasons
                   if any(p < s and p in cache["prior"] for p in cache["in_season"])]
    print("predictable under the blended scheme:", predictable)

    tune_season = 2024
    test_season = 2025

    rows = []
    for r in RAMP_END_GRID:
        allp, sigma, info = predict_season(cache, tune_season, CARRYOVER, r)
        if allp.empty:
            continue
        m = score(allp, sigma)
        rows.append({
            "ramp_end": r, "lambda_wk5": round(lambda_schedule(5, ramp_end=r), 3),
            "sigma": round(sigma, 3),
            "n": m["n"], "mae": m["mae"], "rmse": m["rmse"],
            "winner_acc": m["winner_accuracy"], "brier": m["brier"],
            "log_loss": m["log_loss"],
            "ats": m.get("ats_pct"), "ats_n": m.get("ats_n"),
        })
    grid = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n=== ramp_end tuning on {tune_season}, carryover pinned at {CARRYOVER} ===")
    print(grid.to_string(index=False))

    best = grid.sort_values("mae").iloc[0]
    c_best, r_best = CARRYOVER, int(best["ramp_end"])
    print(f"\nselected on {tune_season}: carryover={c_best} (pinned), ramp_end={r_best}")
    print(f"  lambda at week 5 = {lambda_schedule(5, ramp_end=r_best):.3f}  "
          f"(the weight on 2026 form for this week's slate)")

    print(f"\n=== HELD OUT: {test_season} ===")
    allp, sigma, info = predict_season(cache, test_season, c_best, r_best)
    m = score(allp, sigma)
    print("info:", {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in info.items()})
    for k in ("n", "mae", "rmse", "market_mae", "winner_accuracy", "market_winner_accuracy", "mean_signed_error",
              "brier", "log_loss", "ats_pct", "ats_n", "ats_pct_edge", "ats_n_edge"):
        if k in m:
            print(f"  {k:<20} {m[k]}")
    for w in B.sanity_warnings(m):
        print("  WARNING:", w)

    allp.to_pickle(Path(__file__).resolve().parent / "cache" / f"blend_{test_season}.pkl")
    grid.to_csv(Path(__file__).resolve().parent / "cache" / "tuning_grid.csv", index=False)

    # Also score the tuned config on 2024 for the record, and save both.
    a24, s24, _ = predict_season(cache, tune_season, c_best, r_best)
    a24.to_pickle(Path(__file__).resolve().parent / "cache" / f"blend_{tune_season}.pkl")
    with open(Path(__file__).resolve().parent / "cache" / "chosen.pkl", "wb") as fh:
        pickle.dump({"carryover": c_best, "ramp_end": r_best}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
