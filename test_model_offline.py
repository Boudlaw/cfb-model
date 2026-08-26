#!/usr/bin/env python3
"""
Offline test suite for the modeling layer. No network, no real data, no API key.

WHAT THIS IS FOR
----------------
The modeling code was written in a sandbox that cannot reach the CFBD API, so it
could not be developed against real data. That makes a synthetic-truth test suite
the only thing standing between "the code runs" and "the code is right". Every
test here either hand-checks a definition against a worked example or plants a
known truth and asserts the machinery recovers it.

THE TESTS THAT MATTER MOST
--------------------------
test_no_leak_in_fit_through and test_future_perturbation_does_not_move_past.
The second one is the real proof: it corrupts the outcomes of LATE weeks and
asserts that predictions for EARLY weeks do not move by even a float's width. A
model with any backward information flow fails it. Everything else in this file
could pass while the project was still worthless; that one could not.

    python test_model_offline.py            # run all
    python test_model_offline.py -v          # per-check detail
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd

import backtest as B
import features as F
import ratings as R

VERBOSE = "-v" in sys.argv
PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        def run() -> None:
            try:
                fn()
                PASSED.append(name)
                if VERBOSE:
                    print(f"  ok   {name}")
            except Exception as exc:  # noqa: BLE001
                FAILED.append((name, f"{exc}\n{traceback.format_exc()}"))
                print(f"  FAIL {name}: {exc}")
        run.__name__ = fn.__name__
        TESTS.append(run)
        return run
    return deco


TESTS: list[Callable] = []


# ------------------------------------------------------------------- synthetic data

def make_synthetic(
    n_teams: int = 40,
    n_weeks: int = 12,
    n_seasons: int = 3,
    plays_per_game: int = 130,
    seed: int = 12345,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build a fake league with KNOWN team strengths.

    Returns (raw_plays, games, truth). Design choices that make the tests sharp:

      * league mean EPA is centred on zero, so sanity_report's near-zero check is
        a real check and not an accident;
      * success probability is centred on 0.40 to match the FBS average, so the
        Phase 2 correctness gate is exercised;
      * yards_gained is constructed to be exactly consistent with the planted
        success flag, so is_success() is tested against the generator rather than
        against itself;
      * game margin is a deterministic function of planted strength plus noise,
        so a rating that recovers the strengths must also predict margins.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team{i:02d}" for i in range(n_teams)]
    off_true = rng.normal(0, 0.12, n_teams)
    # def_true is defensive QUALITY: plays are generated with
    #     epa = off_true[offense] - def_true[defense] + noise
    # so a higher def_true suppresses the opposing offense, i.e. higher = better.
    #
    # Two consequences the tests below assert, and which an earlier version of
    # this generator got wrong:
    #   * the FITTED defense coefficient estimates -def_true, so def_epa must
    #     correlate NEGATIVELY with def_true;
    #   * true team strength is off_true + def_true (offense plus defensive
    #     quality), which is also what the margin generator below implies:
    #     (off_h - def_a) - (off_a - def_h) == (off_h + def_h) - (off_a + def_a).
    def_true = rng.normal(0, 0.12, n_teams)
    truth = pd.DataFrame(
        {"team": teams, "off_true": off_true, "def_true": def_true,
         "net_true": off_true + def_true}
    )

    play_rows: list[dict] = []
    game_rows: list[dict] = []
    gid = 0
    for season in range(2020, 2020 + n_seasons):
        for week in range(1, n_weeks + 1):
            order = rng.permutation(n_teams)
            for k in range(0, n_teams - 1, 2):
                h, a = int(order[k]), int(order[k + 1])
                gid += 1
                neutral = bool(rng.random() < 0.05)
                hfa = 0.0 if neutral else R.DEFAULT_HFA_POINTS
                strength = (off_true[h] - def_true[a]) - (off_true[a] - def_true[h])
                margin = float(strength * 40.0 + hfa + rng.normal(0, 13.0))
                margin = float(np.round(margin))
                game_rows.append(
                    {
                        "season": season, "week": week, "game_id": gid,
                        "home_team": teams[h], "away_team": teams[a],
                        "neutral_site": neutral,
                        "margin": margin,
                        # Market = truth plus small noise, so it is a genuinely
                        # sharper predictor than any fitted model, as in reality.
                        "opening_line_home": float(
                            np.round((strength * 40.0 + hfa + rng.normal(0, 2.0)) * 2) / 2
                        ),
                    }
                )
                for (o, d, is_home) in ((h, a, 1), (a, h, 0)):
                    edge = off_true[o] - def_true[d]
                    p_succ = float(np.clip(0.40 + edge * 0.6, 0.05, 0.95))
                    for _ in range(plays_per_game // 2):
                        down = int(rng.integers(1, 5))
                        distance = int(rng.integers(1, 16))
                        succ = bool(rng.random() < p_succ)
                        thr = F.SUCCESS_THRESHOLD[down] * distance
                        if succ:
                            yards = int(math.ceil(thr)) + int(rng.integers(0, 6))
                            if yards < thr:
                                yards = int(math.ceil(thr))
                        else:
                            yards = max(-3, int(math.floor(thr)) - 1 - int(rng.integers(0, 4)))
                            if yards >= thr:
                                yards = int(math.floor(thr - 0.5))
                        play_rows.append(
                            {
                                "season": season, "week": week, "game_id": gid,
                                "offense": teams[o], "defense": teams[d],
                                "home": teams[h], "away": teams[a],
                                "down": down, "distance": distance,
                                "yards_gained": yards,
                                "play_type": "Rush" if rng.random() < 0.5 else "Pass",
                                "period": int(rng.integers(1, 5)),
                                "offense_score": 14, "defense_score": 14,
                                "ppa": float(edge + rng.normal(0, 1.35)),
                            }
                        )
    return pd.DataFrame(play_rows), pd.DataFrame(game_rows), truth


_CACHE: dict[str, Any] = {}


def synth() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "synth" not in _CACHE:
        _CACHE["synth"] = make_synthetic()
    return _CACHE["synth"]


def prepared() -> pd.DataFrame:
    if "prep" not in _CACHE:
        raw, _, _ = synth()
        _CACHE["prep"] = F.prepare_plays(raw)
    return _CACHE["prep"]


# ------------------------------------------------------------------ Phase 2 checks

@check("success: 1st down needs 50% of distance")
def _t1() -> None:
    assert F.is_success(1, 10, 5) is True
    assert F.is_success(1, 10, 4) is False
    assert F.is_success(1, 10, 4.9) is False


@check("success: 2nd down needs 70%")
def _t2() -> None:
    assert F.is_success(2, 10, 7) is True
    assert F.is_success(2, 10, 6) is False
    assert F.is_success(2, 5, 3.5) is True


@check("success: 3rd and 4th need the whole distance")
def _t3() -> None:
    assert F.is_success(3, 4, 4) is True
    assert F.is_success(3, 4, 3) is False
    assert F.is_success(4, 1, 1) is True
    assert F.is_success(4, 1, 0) is False


@check("success: unjudgeable plays return None, never 0")
def _t4() -> None:
    # A None silently counted as a failure would bias every rating downward.
    assert F.is_success(None, 10, 5) is None
    assert F.is_success(1, 0, 5) is None
    assert F.is_success(1, -3, 5) is None
    assert F.is_success(5, 10, 5) is None
    assert F.is_success(1, 10, None) is None
    assert F.is_success("x", 10, 5) is None


@check("garbage time: Connelly thresholds by quarter")
def _t5() -> None:
    assert F.is_garbage_time(1, 50, 0) is True    # 50 > 43
    assert F.is_garbage_time(1, 43, 0) is False   # not strictly greater
    assert F.is_garbage_time(2, 38, 0) is True    # 38 > 37
    assert F.is_garbage_time(3, 28, 0) is True    # 28 > 27
    assert F.is_garbage_time(4, 23, 0) is True    # 23 > 22
    assert F.is_garbage_time(4, 22, 0) is False
    assert F.is_garbage_time(5, 99, 0) is False   # overtime is never garbage


@check("prepare_plays drops non-scrimmage, garbage and unjudgeable rows")
def _t6() -> None:
    base = {
        "season": 2024, "week": 1, "game_id": 1, "offense": "A", "defense": "B",
        "home": "A", "away": "B", "down": 1, "distance": 10, "yards_gained": 6,
        "play_type": "Rush", "period": 1, "offense_score": 0, "defense_score": 0,
        "ppa": 0.5,
    }
    rows = [
        dict(base),                                              # keep
        dict(base, play_type="Punt"),                            # non-scrimmage
        dict(base, play_type="Kickoff"),                         # non-scrimmage
        dict(base, down=None),                                   # unjudgeable
        dict(base, period=1, offense_score=60, defense_score=0), # garbage time
        dict(base, ppa=None),                                    # no EPA
        dict(base, distance=0),                                  # unjudgeable
    ]
    out = F.prepare_plays(pd.DataFrame(rows))
    assert len(out) == 1, f"expected 1 surviving play, got {len(out)}"
    assert out.iloc[0]["success"] == 1
    assert out.iloc[0]["is_home_offense"] == 1


@check("prepare_plays raises on a missing column instead of defaulting it")
def _t7() -> None:
    df = pd.DataFrame([{"season": 2024, "week": 1}])
    try:
        F.prepare_plays(df)
    except ValueError as exc:
        assert "missing required column" in str(exc)
        return
    raise AssertionError("prepare_plays silently accepted an incomplete frame")


@check("drop_report flags unknown play types")
def _t8() -> None:
    base = {
        "season": 2024, "week": 1, "game_id": 1, "offense": "A", "defense": "B",
        "home": "A", "away": "B", "down": 1, "distance": 10, "yards_gained": 6,
        "play_type": "Rush", "period": 1, "offense_score": 0, "defense_score": 0,
        "ppa": 0.5,
    }
    rep = F.drop_report(pd.DataFrame([
        dict(base),
        dict(base, play_type="Punt"),                  # recognised exclusion
        dict(base, play_type="Quantum Teleport"),       # genuinely unknown
    ]))
    assert rep["unknown_play_types"] == ["Quantum Teleport"], rep["unknown_play_types"]
    assert rep["kept"] == 1


@check("sanity gate: synthetic league reproduces ~40% success and ~0 mean EPA")
def _t9() -> None:
    rep = F.sanity_report(prepared())
    assert rep["passes"], rep
    assert 0.36 <= rep["success_rate"] <= 0.44, rep["success_rate"]
    assert abs(rep["mean_epa"]) < 0.15, rep["mean_epa"]


@check("sanity gate FAILS loudly on a corrupted success rate")
def _t10() -> None:
    p = prepared().copy()
    p["success"] = 1  # everything succeeds: obviously broken
    rep = F.sanity_report(p)
    assert not rep["passes"]
    assert "success rate" in rep["reason"]


@check("team_week_aggregate covers both sides for every team")
def _t11() -> None:
    agg = F.team_week_aggregate(prepared())
    assert set(agg["side"]) == {"offense", "defense"}
    assert agg["success_rate"].between(0, 1).all()
    assert agg["plays_n"].min() > 0


# ------------------------------------------------------------------ Phase 3 checks

@check("design matrix: shape, 2 non-zeros per play, HFA column")
def _t12() -> None:
    p = prepared().head(500)
    teams = sorted(set(prepared()["offense"]) | set(prepared()["defense"]))
    d = R.build_design(p, teams=teams)
    assert d.X.shape == (500, 2 * len(teams) + 1)
    # Two team dummies per row, plus the home column when the offense is at home.
    nnz_expected = 2 * 500 + int(p["is_home_offense"].sum())
    assert d.X.nnz == nnz_expected, (d.X.nnz, nnz_expected)
    assert d.columns[-1] == "home_offense"


@check("design matrix rejects a team absent from the roster")
def _t13() -> None:
    p = prepared().head(50).copy()
    p.loc[p.index[0], "offense"] = "Nonexistent State"
    try:
        R.build_design(p, teams=sorted(set(prepared()["offense"])))
    except ValueError as exc:
        assert "absent from the roster" in str(exc)
        return
    raise AssertionError("build_design accepted an unknown team")


@check("ridge on EPA recovers the planted team ordering")
def _t14() -> None:
    p = prepared()
    _, _, truth = synth()
    m = R.BoudEff(alpha_epa=1.0, alpha_success=1.0).fit(p)
    got = m.team_ratings().merge(truth, on="team")
    rho = got["net_epa"].corr(got["net_true"], method="spearman")
    assert rho > 0.80, f"rank correlation with planted truth only {rho:.3f}"


@check("logistic ridge on success recovers the planted ordering")
def _t15() -> None:
    p = prepared()
    _, _, truth = synth()
    m = R.BoudEff(alpha_epa=1.0, alpha_success=1.0).fit(p)
    got = m.team_ratings().merge(truth, on="team")
    assert got["net_succ"].notna().all(), "success model did not fit"
    rho = got["net_succ"].corr(got["net_true"], method="spearman")
    assert rho > 0.80, f"rank correlation with planted truth only {rho:.3f}"


@check("offense and defense coefficients are separately recovered")
def _t16() -> None:
    p = prepared()
    _, _, truth = synth()
    got = R.BoudEff(alpha_epa=1.0, alpha_success=1.0).fit(p).team_ratings().merge(
        truth, on="team"
    )
    # def_epa is the RAW fitted coefficient, which estimates -def_true (a good
    # defense suppresses EPA, so its coefficient is negative). This assertion is
    # deliberately signed: an earlier version of this suite planted the truth
    # with the opposite sign, and the resulting "failure" was the test being
    # wrong, not the model. Keep the sign pinned here so that cannot recur
    # silently in either direction.
    assert got["off_epa"].corr(got["off_true"], method="spearman") > 0.7
    assert got["def_epa"].corr(got["def_true"], method="spearman") < -0.7
    # And the composite must track offense PLUS defensive quality.
    assert got["net_epa"].corr(got["net_true"], method="spearman") > 0.7


@check("lambda schedule: 0 at week 1, 1.0 by ramp_end, monotone between")
def _t17() -> None:
    assert R.lambda_schedule(1) == 0.0
    assert R.lambda_schedule(7, ramp_end=7) == 1.0
    assert R.lambda_schedule(12, ramp_end=7) == 1.0
    vals = [R.lambda_schedule(w, ramp_end=7) for w in range(1, 9)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals
    assert 0.4 < R.lambda_schedule(4, ramp_end=7) < 0.6


@check("blend_with_prior: week 1 is pure prior, late season is pure in-season")
def _t18() -> None:
    ins = pd.DataFrame({"team": ["A", "B"], "net_epa": [1.0, -1.0], "net_succ": [1.0, -1.0]})
    pri = pd.DataFrame({"team": ["A", "B"], "net_epa": [-2.0, 2.0], "net_succ": [0.0, 0.0]})
    wk1 = R.blend_with_prior(ins, pri, week=1, carryover=0.7)
    assert np.allclose(wk1["net_epa"], [0.7 * -2.0, 0.7 * 2.0]), wk1["net_epa"].tolist()
    wk9 = R.blend_with_prior(ins, pri, week=9, carryover=0.7)
    assert np.allclose(wk9["net_epa"], [1.0, -1.0])


@check("fit_carryover recovers a planted carryover slope")
def _t19() -> None:
    rng = np.random.default_rng(7)
    teams = [f"T{i}" for i in range(120)]
    r1 = rng.normal(0, 1, 120)
    r2 = 0.65 * r1 + rng.normal(0, 0.3, 120)
    got = R.fit_carryover(
        {
            2023: pd.DataFrame({"team": teams, "net_epa": r1}),
            2024: pd.DataFrame({"team": teams, "net_epa": r2}),
        }
    )
    assert abs(got["carryover"] - 0.65) < 0.12, got


@check("MarginModel holds HFA at the league constant unless told otherwise")
def _t20() -> None:
    raw, games, _ = synth()
    p = F.prepare_plays(raw)
    ratings = R.BoudEff(alpha_epa=1.0, alpha_success=1.0).fit(p).team_ratings()
    feat = R.game_features(games, ratings).dropna(subset=["d_net_epa", "margin"])
    mm = R.MarginModel().fit(feat)
    assert mm.hfa_fitted == R.DEFAULT_HFA_POINTS == 2.5
    # And the free-HFA variant should land near the planted 2.5.
    mm2 = R.MarginModel(fit_hfa=True).fit(feat)
    assert abs(mm2.hfa_fitted - 2.5) < 1.6, mm2.hfa_fitted


@check("MarginModel predicts neutral-site games without home advantage")
def _t21() -> None:
    mm = R.MarginModel()
    mm.b_epa, mm.b_succ, mm.hfa_fitted = 40.0, 0.0, 2.5
    g = pd.DataFrame(
        {"d_net_epa": [0.0, 0.0], "d_net_succ": [0.0, 0.0], "neutral_site": [False, True]}
    )
    pred = mm.predict(g)
    assert np.isclose(pred[0], 2.5) and np.isclose(pred[1], 0.0), pred


@check("game_features leaves unrated teams as NaN rather than zero")
def _t22() -> None:
    ratings = pd.DataFrame({"team": ["A"], "net_epa": [1.0], "net_succ": [0.5]})
    games = pd.DataFrame(
        {"home_team": ["A"], "away_team": ["FCS Directional"], "margin": [21.0]}
    )
    out = R.game_features(games, ratings)
    # A zero difference would be a confident prediction of a coin flip.
    assert np.isnan(out.loc[0, "d_net_epa"])


# ------------------------------------------------------------------ Phase 4 checks

@check("metrics: MAE, RMSE and mean signed error match hand computation")
def _t23() -> None:
    actual = np.array([10.0, -3.0, 7.0, 0.5])
    pred = np.array([7.0, -1.0, 10.0, -0.5])
    m = B.evaluate(actual, pred)
    # errors: +3, -2, -3, +1
    assert np.isclose(m["mae"], (3 + 2 + 3 + 1) / 4)
    assert np.isclose(m["rmse"], math.sqrt((9 + 4 + 9 + 1) / 4))
    assert np.isclose(m["mean_signed_error"], (3 - 2 - 3 + 1) / 4)
    # home won games 1, 3, 4; model picked home in 1, 3 and away in 4 -> 3/4
    assert np.isclose(m["winner_accuracy"], 0.75)


@check("win_probability: 0 margin is 50%, symmetric, monotone")
def _t24() -> None:
    assert np.isclose(B.win_probability(np.array([0.0]))[0], 0.5)
    lo, hi = B.win_probability(np.array([-7.0]))[0], B.win_probability(np.array([7.0]))[0]
    assert np.isclose(lo + hi, 1.0)
    assert hi > 0.5 > lo
    seq = B.win_probability(np.array([-21.0, -7.0, 0.0, 7.0, 21.0]))
    assert all(b > a for a, b in zip(seq, seq[1:]))


@check("win_probability uses sigma 16, not the market's 14")
def _t25() -> None:
    assert B.SIGMA_OWN_MODEL == 16.0
    # A 7-point favourite: sigma 16 must be less confident than sigma 14.1.
    own = B.win_probability(np.array([7.0]), B.SIGMA_OWN_MODEL)[0]
    mkt = B.win_probability(np.array([7.0]), B.SIGMA_MARKET)[0]
    assert own < mkt, (own, mkt)


@check("Brier and log loss match hand computation")
def _t26() -> None:
    # margin 0 -> p = 0.5 for both games; one home win, one home loss.
    m = B.evaluate(np.array([3.0, -3.0]), np.array([0.0, 0.0]))
    assert np.isclose(m["brier"], 0.25)
    assert np.isclose(m["log_loss"], -math.log(0.5))


@check("ATS: pushes excluded, side attribution correct")
def _t27() -> None:
    # line = home -3 everywhere (opening_line_home = +3).
    actual = np.array([10.0, 1.0, 3.0, 10.0])
    pred = np.array([7.0, 7.0, 7.0, -1.0])   # model on home 3x, on away once
    line = np.array([3.0, 3.0, 3.0, 3.0])
    m = B.evaluate(actual, pred, opening_line_home=line)
    # game 3 is a push (actual == line) -> excluded from the denominator
    assert m["ats_n"] == 3, m["ats_n"]
    # g1: picked home, home covered (10>3) -> correct
    # g2: picked home, home did not cover (1<3) -> wrong
    # g4: picked away, home covered -> wrong
    assert np.isclose(m["ats_pct"], 1 / 3), m["ats_pct"]


@check("ATS edge subset only counts disagreements above the threshold")
def _t28() -> None:
    actual = np.array([20.0, 20.0])
    pred = np.array([11.0, 4.0])   # disagreements +8 and +1
    line = np.array([3.0, 3.0])
    m = B.evaluate(actual, pred, opening_line_home=line)
    assert m["ats_n"] == 2
    assert m["ats_n_edge"] == 1, m["ats_n_edge"]


@check("sanity warnings fire on a suspiciously good result")
def _t29() -> None:
    rng = np.random.default_rng(3)
    actual = rng.normal(0, 16, 800)
    m = B.evaluate(actual, actual + rng.normal(0, 1, 800))  # near-perfect: a leak
    assert any("leak" in w.lower() for w in m["warnings"]), m["warnings"]


@check("sanity warnings stay quiet on a realistic result")
def _t30() -> None:
    rng = np.random.default_rng(4)
    actual = rng.normal(0, 20, 900)
    pred = 0.55 * actual + rng.normal(0, 13, 900)
    m = B.evaluate(actual, pred)
    assert m["mae"] > 12.0
    assert not any("leak" in w.lower() for w in m["warnings"]), m["warnings"]


@check("calibration table buckets sum to n and report a realised rate")
def _t31() -> None:
    rng = np.random.default_rng(5)
    pred = rng.normal(0, 14, 2000)
    actual = pred + rng.normal(0, 16, 2000)
    tab = B.calibration_table(actual, pred)
    assert tab["n"].sum() == 2000
    assert tab["realised"].between(0, 1).all()
    # A correctly specified model should be roughly calibrated.
    assert tab["gap"].abs().max() < 0.25, tab


# ------------------------------------------------------- THE LEAKAGE TESTS

@check("LEAK GUARD: fit_through uses only plays strictly before the cutoff")
def _t32() -> None:
    p = prepared()
    m = R.BoudEff(alpha_epa=1.0, alpha_success=1.0)
    season = int(p["season"].max())
    m.fit_through(p, season=season, week=6, teams=sorted(set(p["offense"])))
    # Reproduce the subset the fit was allowed to see and confirm the count.
    allowed = p.loc[(p["season"] < season) | ((p["season"] == season) & (p["week"] < 6))]
    assert m.n_plays == len(allowed), (m.n_plays, len(allowed))
    assert m.n_plays < len(p)


@check("LEAK GUARD: fit_through refuses when there is nothing prior")
def _t33() -> None:
    p = prepared()
    first = int(p["season"].min())
    try:
        R.BoudEff().fit_through(p, season=first, week=1)
    except ValueError as exc:
        assert "no plays available" in str(exc)
        return
    raise AssertionError("fit_through invented data for the first week of history")


@check("LEAK GUARD: corrupting late weeks does not move early-week predictions")
def _t34() -> None:
    """
    The decisive test. Predictions for week 4 are computed twice: once on clean
    data, once after the outcomes of weeks 8+ have been replaced with garbage. If
    any information flows backwards -- ratings fit on the full season, a margin
    model trained on the same season, a stray groupby over all weeks -- the two
    sets of numbers differ. They must be bit-for-bit identical.
    """
    raw, games, _ = synth()
    p = F.prepare_plays(raw)
    teams = sorted(set(p["offense"]) | set(p["defense"]))
    season = int(p["season"].max())

    def predict_week4(plays: pd.DataFrame, gms: pd.DataFrame) -> np.ndarray:
        prior_plays = plays.loc[plays["season"] < season]
        prior_games = gms.loc[gms["season"] < season]
        pm = R.BoudEff(alpha_epa=1.0, alpha_success=1.0).fit(prior_plays, teams=teams)
        pf = R.game_features(prior_games, pm.team_ratings()).dropna(
            subset=["d_net_epa", "margin"]
        )
        mm = R.MarginModel().fit(pf)
        model = R.BoudEff(alpha_epa=1.0, alpha_success=1.0).fit_through(
            plays, season=season, week=4, teams=teams
        )
        wk = gms.loc[(gms["season"] == season) & (gms["week"] == 4)]
        feat = R.game_features(wk, model.team_ratings()).dropna(subset=["d_net_epa"])
        return mm.predict(feat.sort_values("game_id"))

    clean = predict_week4(p, games)

    rng = np.random.default_rng(999)
    p_bad = p.copy()
    late = (p_bad["season"] == season) & (p_bad["week"] >= 8)
    p_bad.loc[late, "epa"] = rng.normal(50, 10, int(late.sum()))
    p_bad.loc[late, "success"] = 1
    g_bad = games.copy()
    late_g = (g_bad["season"] == season) & (g_bad["week"] >= 8)
    g_bad.loc[late_g, "margin"] = 99.0

    dirty = predict_week4(p_bad, g_bad)

    assert clean.shape == dirty.shape, (clean.shape, dirty.shape)
    assert np.array_equal(clean, dirty), (
        "LEAK: week-4 predictions changed when weeks 8+ were corrupted; "
        f"max delta {np.max(np.abs(clean - dirty)):.6g}"
    )


@check("LEAK GUARD: walk_forward margin model never sees the season it predicts")
def _t35() -> None:
    raw, games, _ = synth()
    p = F.prepare_plays(raw)
    res = B.walk_forward(p, games, alpha=1.0, first_week=4, verbose=False)
    assert len(res.predictions) > 0, "walk_forward produced nothing"
    # The earliest season has no prior season, so it must be skipped entirely.
    first = int(games["season"].min())
    assert first not in set(res.predictions["season"]), (
        "the first season was predicted, but its ratings-to-points mapping "
        "could only have come from itself"
    )
    log = res.fold_log
    assert (log["status"] == "ok").any()
    assert (log.loc[log["season"] == first, "status"] == "skipped").all()


@check("walk_forward end to end: predictions are plausible, not perfect")
def _t36() -> None:
    raw, games, _ = synth()
    p = F.prepare_plays(raw)
    res = B.walk_forward(p, games, alpha=1.0, first_week=4, verbose=False)
    m = res.metrics
    assert m["n"] > 50, m["n"]
    # Synthetic noise SD is 13, so MAE must land in a believable band. Anything
    # much under ~9 on this generator would mean the harness is cheating.
    assert 8.0 < m["mae"] < 20.0, m["mae"]
    assert 0.5 < m["winner_accuracy"] < 0.85, m["winner_accuracy"]
    assert "ats_pct" in m
    assert set(["mae", "rmse", "mean_signed_error", "brier", "log_loss"]) <= set(m)
    assert not res.per_season.empty


@check("walk_forward is deterministic across runs")
def _t37() -> None:
    raw, games, _ = synth()
    p = F.prepare_plays(raw)
    a = B.walk_forward(p, games, alpha=1.0, first_week=6, verbose=False)
    b = B.walk_forward(p, games, alpha=1.0, first_week=6, verbose=False)
    assert np.allclose(
        a.predictions.sort_values("game_id")["predicted_margin"].to_numpy(),
        b.predictions.sort_values("game_id")["predicted_margin"].to_numpy(),
    )


@check("format_report renders benchmarks and surfaces warnings")
def _t38() -> None:
    raw, games, _ = synth()
    p = F.prepare_plays(raw)
    res = B.walk_forward(p, games, alpha=1.0, first_week=6, verbose=False)
    txt = B.format_report(res)
    for token in ("MAE", "12.06", "Winner accuracy", "Brier", "Log loss"):
        assert token in txt, token


@check("tune_alpha scores on held-out game margin and returns a sorted grid")
def _t39() -> None:
    raw, games, _ = synth()
    p = F.prepare_plays(raw)
    grid = R.tune_alpha(p, games, alphas=(1.0, 50.0, 500.0))
    assert len(grid) >= 2, grid
    assert list(grid["mae"]) == sorted(grid["mae"])
    assert grid["n_games"].min() > 0


def main() -> int:
    print("offline model test suite — no network, synthetic truth\n")
    for t in TESTS:
        t()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("\n" + "=" * 70)
        for name, tb in FAILED:
            print(f"\nFAILED: {name}\n{tb}")
        return 1
    print("\nAll checks passed. The leakage guards are the ones that matter:")
    for n in PASSED:
        if n.startswith("LEAK GUARD"):
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
