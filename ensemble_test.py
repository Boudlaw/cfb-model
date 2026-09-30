"""
Phase 5, the narrow version: does adding CFBD's pregame Elo to BOUD-EFF beat
either alone?

Worth asking because the like-for-like baseline came back a dead heat — BLEND
12.84 MAE / 71.1% winners against ELO 12.83 / 70.4%, with Elo holding the better
Brier. Two predictors of similar strength that disagree are the textbook case
where a blend beats both; two that agree are not.

Protocol is the same as lambda_experiment: fit on prior seasons only, select on
2024, report 2025 once.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import lambda_experiment as LE
from ratings import blend_with_prior, game_features, lambda_schedule

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CARRYOVER, RAMP_END = 1.0, 6


def frame_for(cache, season, week):
    f = LE.blended_features(cache, season, week, CARRYOVER, RAMP_END)
    if f is None:
        return None
    f = f.dropna(subset=["home_pregame_elo", "away_pregame_elo"])
    if f.empty:
        return None
    f = f.copy()
    f["d_elo"] = (f["home_pregame_elo"].astype(float)
                  - f["away_pregame_elo"].astype(float))
    f["hfa_term"] = (~f["neutral_site"].astype(bool)).astype(float)
    return f


def gather(cache, seasons):
    out = []
    for s in seasons:
        for w in sorted(cache["in_season"].get(s, {})):
            f = frame_for(cache, s, w)
            if f is not None:
                out.append(f)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


SPECS = {
    "BOUD-EFF only":      ["d_net_epa", "d_net_succ"],
    "Elo only":           ["d_elo"],
    "BOUD-EFF + Elo":     ["d_net_epa", "d_net_succ", "d_elo"],
}


def fit_predict(train, test, cols):
    X = np.column_stack([train[c].to_numpy(float) for c in cols]
                        + [train["hfa_term"].to_numpy(float)])
    y = train["margin"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    sigma = float(np.std(y - X @ beta, ddof=1))
    Xt = np.column_stack([test[c].to_numpy(float) for c in cols]
                         + [test["hfa_term"].to_numpy(float)])
    return Xt @ beta, sigma, beta


def main() -> int:
    cache = LE.load()
    rows = []
    store = {}
    for target in (2024, 2025):
        train_seasons = [s for s in sorted(cache["in_season"]) if s < target
                         and s in cache["prior"]]
        train = gather(cache, train_seasons)
        test = gather(cache, [target])
        if train.empty or test.empty:
            continue
        for name, cols in SPECS.items():
            pred, sigma, beta = fit_predict(train, test, cols)
            t = test.copy()
            t["predicted_margin"] = pred
            line = t["opening_line_home"].to_numpy() if "opening_line_home" in t else None
            m = B.evaluate(t["margin"], t["predicted_margin"], sigma, line)
            rows.append({"season": target, "model": name, "n": m["n"],
                         "mae": m["mae"], "win": m["winner_accuracy"],
                         "brier": m["brier"], "ats": m.get("ats_pct"),
                         "sigma": sigma})
            store[(target, name)] = (t, sigma, beta, cols)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== per season ===")
    print(df.to_string(index=False))

    print("\n=== pooled 2024+2025 ===")
    for name in SPECS:
        parts = [store[(s, name)][0] for s in (2024, 2025) if (s, name) in store]
        allp = pd.concat(parts, ignore_index=True)
        sg = float(np.mean([store[(s, name)][1] for s in (2024, 2025) if (s, name) in store]))
        line = allp["opening_line_home"].to_numpy() if "opening_line_home" in allp else None
        m = B.evaluate(allp["margin"], allp["predicted_margin"], sg, line)
        print(f"  {name:<18} n={m['n']}  MAE {m['mae']:.2f}  win {m['winner_accuracy']*100:.1f}%  "
              f"Brier {m['brier']:.4f}  ATS {m.get('ats_pct', float('nan'))*100:.1f}%")
    mk = B.evaluate(allp["margin"], allp["predicted_margin"], sg,
                    allp["opening_line_home"].to_numpy())
    print(f"  {'MARKET':<18} n={mk['n']}  MAE {mk['market_mae']:.2f}  "
          f"win {mk['market_winner_accuracy']*100:.1f}%")

    # How much do the two actually disagree? If they are the same predictor
    # wearing two hats, no blend will help and the dead heat is not a coincidence.
    t, _, _, _ = store[(2025, "BOUD-EFF only")]
    e, _, _, _ = store[(2025, "Elo only")]
    r = float(np.corrcoef(t["predicted_margin"], e["predicted_margin"])[0, 1])
    print(f"\ncorrelation between BOUD-EFF and Elo predicted margins (2025): {r:.3f}")

    print("\ncoefficients of the combined model, fit for 2025:")
    _, _, beta, cols = store[(2025, "BOUD-EFF + Elo")]
    for c, b in zip(cols + ["HFA"], beta):
        print(f"  {c:<14} {b:+.4f}")

    with open(CACHE / "ensemble.pkl", "wb") as fh:
        pickle.dump({"per_season": df}, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
