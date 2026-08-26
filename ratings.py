#!/usr/bin/env python3
"""
Phase 3 — BOUD-EFF: opponent-adjusted ratings by ridge regression.

WHAT THIS DOES
--------------
Every play is one row. The design matrix has one dummy per team per unit
(offense_<Team>, defense_<Team>) plus a home-offense indicator, so roughly
2*138 + 1 = 277 columns. Two models are fit on that same matrix:

    epa     ~ offense + defense + home     LINEAR ridge
    success ~ offense + defense + home     LOGISTIC ridge

The offense coefficient for a team is its efficiency *after* removing the quality
of the defenses it happened to face, which is the whole point: an offense that
averaged +0.15 EPA against the best defenses in the country is better than one
that averaged +0.20 against the worst.

WHY LOGISTIC FOR SUCCESS, AND WHY THAT MATTERS
----------------------------------------------
Success is binary. Fitting a linear model to a 0/1 target is a misspecification:
it can predict success probabilities outside [0, 1], and its errors are
heteroskedastic by construction. Every published college football implementation
I could find applies linear ridge to EPA and stops; none does adjusted success
rate with the correct link function. This is the one place in the project where
doing the textbook-correct thing is also doing something new.

The practical consequence: logistic coefficients are in log-odds, not
percentage points, so they cannot be compared to the EPA coefficients directly
and must not be averaged with them. `MarginModel` below learns the conversion
from both to points instead of assuming one.

WHY THE REGULARIZATION IS THE WHOLE BALL GAME
---------------------------------------------
Variance components on comparable data: play-to-play noise SD ~1.39 against
team-ability SD ~0.08. That is a 17:1 noise-to-signal ratio, and it means a
team's mean efficiency needs roughly 300 plays — about 4.3 games — before the
estimate is even the size of the true signal. Published results show the naive
additive adjustment (subtract the opponent's average) buys only ~+0.5 points of
accuracy, while ridge and mixed models buy real gains. The value is in the
shrinkage, not in the adjustment scheme. So: tune alpha, do not fiddle with the
dummy encoding.

Consequence for early-season weeks: with two games played, the honest estimate
of a team is "close to average, with a nudge". `blend_with_prior` enforces that
rather than pretending week 2 ratings mean something.

POINT-IN-TIME DISCIPLINE
------------------------
`fit_through` is the only entry point the backtest is allowed to use. It filters
to plays strictly before the cutoff week and asserts it. Fitting on the full
season and then "predicting" week 4 embeds week 4's result — and every later
week's — in every coefficient. That is the single mechanism by which most
amateur sports models fool their authors, and test_model_offline.py asserts
against it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge

# Defaults from the build plan. Tune with tune_alpha(), do not trust these blind.
DEFAULT_ALPHA_EPA = 175.0
DEFAULT_ALPHA_SUCCESS = 175.0
DEFAULT_HFA_POINTS = 2.5  # league-wide, 2021-2025. Not team-specific: team HFA has
                          # near-zero year-over-year correlation, so fitting it per
                          # team is fitting noise.


def _cutoff_mask(plays: pd.DataFrame, season: int, week: int) -> pd.Series:
    """Plays strictly before (season, week). Prior seasons count in full."""
    return (plays["season"] < season) | (
        (plays["season"] == season) & (plays["week"] < week)
    )


@dataclass
class DesignMatrix:
    X: sparse.csr_matrix
    teams: list[str]
    columns: list[str]

    @property
    def n_teams(self) -> int:
        return len(self.teams)


def build_design(plays: pd.DataFrame, teams: Sequence[str] | None = None) -> DesignMatrix:
    """
    Sparse one-hot design: [offense dummies | defense dummies | home_offense].

    Every row has exactly three non-zeros (or two, for a road offense), so the
    matrix is ~3 non-zeros per play regardless of league size. A season of FBS
    football is ~180k plays; dense would be 180k x 277 floats for no reason.

    All team dummies are included alongside the model intercept. That is
    deliberately collinear, and ridge is what makes it well-posed: the intercept
    absorbs the league mean and the penalty resolves the redundancy by shrinking
    coefficients toward zero. The useful side effect is that a coefficient of 0
    means "league average", which is the interpretation we want.
    """
    if teams is None:
        teams = sorted(set(plays["offense"]) | set(plays["defense"]))
    teams = list(teams)
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    n = len(plays)

    off = plays["offense"].map(idx)
    dfn = plays["defense"].map(idx)
    if off.isna().any() or dfn.isna().any():
        unknown = sorted(
            set(plays.loc[off.isna(), "offense"]) | set(plays.loc[dfn.isna(), "defense"])
        )
        raise ValueError(
            f"plays reference teams absent from the roster: {unknown[:10]}. "
            "Pass the full team list explicitly so ratings stay comparable "
            "across weeks."
        )

    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=np.int64)
    cols[0::2] = off.to_numpy(dtype=np.int64)
    cols[1::2] = dfn.to_numpy(dtype=np.int64) + n_t
    data = np.ones(2 * n, dtype=np.float64)

    base = sparse.csr_matrix((data, (rows, cols)), shape=(n, 2 * n_t))
    home = sparse.csr_matrix(
        plays["is_home_offense"].to_numpy(dtype=np.float64).reshape(-1, 1)
    )
    X = sparse.hstack([base, home], format="csr")

    columns = (
        [f"offense_{t}" for t in teams]
        + [f"defense_{t}" for t in teams]
        + ["home_offense"]
    )
    return DesignMatrix(X=X, teams=teams, columns=columns)


@dataclass
class BoudEff:
    """
    Opponent-adjusted offense/defense ratings on two targets.

    Sign convention, fixed here once so nothing downstream has to guess:
      off_epa  -- higher is a better offense
      def_epa  -- the raw coefficient; LOWER is a better defense (fewer EPA allowed)
      def_epa_quality = -def_epa, so HIGHER is better, matching offense
      net_epa  = off_epa + def_epa_quality, higher is a better team
    Same for the success-model triple.
    """

    alpha_epa: float = DEFAULT_ALPHA_EPA
    alpha_success: float = DEFAULT_ALPHA_SUCCESS
    max_iter: int = 200

    teams: list[str] = field(default_factory=list)
    coef_epa: np.ndarray | None = None
    coef_success: np.ndarray | None = None
    hfa_epa: float = 0.0
    hfa_success: float = 0.0
    n_plays: int = 0
    fit_through_label: str = ""

    def fit(self, plays: pd.DataFrame, teams: Sequence[str] | None = None) -> "BoudEff":
        if len(plays) == 0:
            raise ValueError("BoudEff.fit called with zero plays")
        design = build_design(plays, teams)
        X = design.X
        self.teams = design.teams
        self.n_plays = len(plays)

        y_epa = plays["epa"].to_numpy(dtype=np.float64)
        ridge = Ridge(alpha=self.alpha_epa, fit_intercept=True, solver="sparse_cg")
        ridge.fit(X, y_epa)
        self.coef_epa = ridge.coef_.ravel()
        self.hfa_epa = float(self.coef_epa[-1])

        y_suc = plays["success"].to_numpy(dtype=np.int64)
        # C = 1/alpha keeps the knob in the same direction as Ridge's alpha, but
        # note the two penalties are NOT on the same scale (sklearn's logistic
        # objective scales the loss by n, Ridge's does not). Tune them separately.
        if len(np.unique(y_suc)) < 2:
            # Degenerate slice (can happen in tiny synthetic tests). Leave the
            # success model unfit rather than raising, and let callers see None.
            self.coef_success = None
            self.hfa_success = 0.0
        else:
            # `penalty` is left at its default rather than passed explicitly:
            # the default is L2 in every sklearn version this runs on, and
            # naming it triggers a deprecation warning from 1.8 onward.
            logit = LogisticRegression(
                C=1.0 / self.alpha_success,
                fit_intercept=True,
                solver="lbfgs",
                max_iter=self.max_iter,
            )
            logit.fit(X, y_suc)
            self.coef_success = logit.coef_.ravel()
            self.hfa_success = float(self.coef_success[-1])
        return self

    def fit_through(
        self,
        plays: pd.DataFrame,
        season: int,
        week: int,
        teams: Sequence[str] | None = None,
    ) -> "BoudEff":
        """
        Fit using ONLY plays strictly before (season, week). The assertion below
        is not decoration — it is the guard that makes every backtest number
        believable, and it is cheap.
        """
        mask = _cutoff_mask(plays, season, week)
        subset = plays.loc[mask]
        if len(subset) == 0:
            raise ValueError(
                f"no plays available strictly before {season} week {week}; "
                "week 1 must be predicted from the prior alone"
            )
        leaked = subset.loc[
            (subset["season"] > season)
            | ((subset["season"] == season) & (subset["week"] >= week))
        ]
        assert leaked.empty, (
            f"LEAK: {len(leaked)} plays at or after {season} wk{week} entered a "
            "fit that must be point-in-time"
        )
        self.fit_through_label = f"{season}w{week}"
        return self.fit(subset, teams=teams)

    def _table(self, coef: np.ndarray | None, tag: str) -> pd.DataFrame:
        n = len(self.teams)
        if coef is None:
            return pd.DataFrame(
                {
                    "team": self.teams,
                    f"off_{tag}": np.nan,
                    f"def_{tag}": np.nan,
                    f"net_{tag}": np.nan,
                }
            )
        off = coef[:n]
        dfn = coef[n : 2 * n]
        return pd.DataFrame(
            {
                "team": self.teams,
                f"off_{tag}": off,
                f"def_{tag}": dfn,
                f"net_{tag}": off - dfn,  # -def == defensive quality
            }
        )

    def team_ratings(self) -> pd.DataFrame:
        """One row per team, both models side by side, sorted best-first on EPA."""
        if self.coef_epa is None:
            raise RuntimeError("BoudEff.team_ratings called before fit")
        out = self._table(self.coef_epa, "epa").merge(
            self._table(self.coef_success, "succ"), on="team", how="outer"
        )
        out["n_plays_fit"] = self.n_plays
        out["fit_through"] = self.fit_through_label
        return out.sort_values("net_epa", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ prior blending


def lambda_schedule(week: int, ramp_end: int = 7) -> float:
    """
    Weight on in-season information at a given week.

    lambda(1) = 0 -- week 1 is the prior, full stop, because zero plays have been
    observed. Ramps linearly to 1.0 by `ramp_end`. Two independent practitioner
    sources (TeamRankings, CFBD) put the crossover at weeks 4-6, and the
    300-plays-to-signal arithmetic lands in the same place, so ramp_end=7 is a
    defensible default — but fit it by walk-forward validation, do not trust it.
    """
    if week <= 1:
        return 0.0
    if week >= ramp_end:
        return 1.0
    return (week - 1) / (ramp_end - 1)


def blend_with_prior(
    in_season: pd.DataFrame,
    prior: pd.DataFrame,
    week: int,
    carryover: float,
    ramp_end: int = 7,
    columns: Iterable[str] = ("net_epa", "net_succ"),
) -> pd.DataFrame:
    """
    Rating(w) = lambda(w) * InSeason(w) + (1 - lambda(w)) * carryover * Prior

    `carryover` is the regression of a season's rating on the previous season's,
    fit by fit_carryover(). Prior-season ratings historically explained ~54% of
    next-season variance (r ~ 0.74), but that predates the transfer portal and
    NIL with 30%+ annual roster turnover, so treat 0.74 as a ceiling and expect
    0.60-0.74 from your own fit.
    """
    lam = lambda_schedule(week, ramp_end=ramp_end)
    merged = in_season.merge(prior, on="team", how="left", suffixes=("", "_prior"))
    for col in columns:
        pcol = f"{col}_prior"
        p = merged[pcol] if pcol in merged else pd.Series(0.0, index=merged.index)
        merged[col] = lam * merged[col].fillna(0.0) + (1 - lam) * carryover * p.fillna(0.0)
    merged["lambda"] = lam
    return merged


def fit_carryover(
    ratings_by_season: dict[int, pd.DataFrame], column: str = "net_epa"
) -> dict[str, float]:
    """
    Regress each season's end-of-season rating on the previous season's, pooled
    across consecutive pairs. Returns the slope (the carryover coefficient) and
    the correlation, so a value far above ~0.74 can be treated as a warning
    rather than a triumph.
    """
    xs, ys = [], []
    seasons = sorted(ratings_by_season)
    for prev, cur in zip(seasons, seasons[1:]):
        if cur - prev != 1:
            continue
        merged = ratings_by_season[prev][["team", column]].merge(
            ratings_by_season[cur][["team", column]],
            on="team",
            suffixes=("_prev", "_cur"),
        )
        xs.append(merged[f"{column}_prev"].to_numpy())
        ys.append(merged[f"{column}_cur"].to_numpy())
    if not xs:
        return {"carryover": 0.0, "r": 0.0, "n": 0}
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if len(x) < 3 or np.std(x) == 0:
        return {"carryover": 0.0, "r": 0.0, "n": int(len(x))}
    slope = float(np.polyfit(x, y, 1)[0])
    r = float(np.corrcoef(x, y)[0, 1])
    return {"carryover": slope, "r": r, "n": int(len(x))}


# --------------------------------------------------------- ratings -> points


@dataclass
class MarginModel:
    """
    Converts rating differences into a predicted point margin.

    margin ~ b_epa * (net_epa_home - net_epa_away)
           + b_succ * (net_succ_home - net_succ_away)
           + HFA (fixed, not fitted)

    HFA is held at the league-wide 2.5 rather than fitted, for two reasons: the
    measured value over 2021-2025 is 2.5-2.6 (below the ~3.0 oddsmakers still
    conventionally use, having fallen during the 2020 empty-stadium season and
    settled lower), and letting the regression discover it invites it to absorb
    unrelated bias. Passing `fit_hfa=True` is available for testing that choice,
    not for production.

    The two coefficients exist because EPA coefficients are in points-per-play
    and success coefficients are in log-odds. Averaging them directly would be a
    unit error; this regression is the honest conversion.
    """

    hfa: float = DEFAULT_HFA_POINTS
    fit_hfa: bool = False
    b_epa: float = 0.0
    b_succ: float = 0.0
    hfa_fitted: float = 0.0
    n_games: int = 0

    def _features(self, games: pd.DataFrame) -> np.ndarray:
        cols = [games["d_net_epa"].to_numpy(dtype=np.float64)]
        succ = games["d_net_succ"].to_numpy(dtype=np.float64)
        cols.append(np.nan_to_num(succ, nan=0.0))
        return np.column_stack(cols)

    def fit(self, games: pd.DataFrame) -> "MarginModel":
        """
        `games` needs d_net_epa, d_net_succ, margin, neutral_site.
        `margin` is home points minus away points.
        """
        need = {"d_net_epa", "d_net_succ", "margin", "neutral_site"}
        missing = need - set(games.columns)
        if missing:
            raise ValueError(f"MarginModel.fit missing columns: {sorted(missing)}")
        g = games.dropna(subset=["d_net_epa", "margin"]).copy()
        if len(g) < 10:
            raise ValueError(f"MarginModel.fit needs >=10 games, got {len(g)}")

        X = self._features(g)
        hfa_term = np.where(g["neutral_site"].to_numpy(dtype=bool), 0.0, 1.0)
        y = g["margin"].to_numpy(dtype=np.float64)

        if self.fit_hfa:
            X = np.column_stack([X, hfa_term])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.b_epa, self.b_succ, self.hfa_fitted = (float(b) for b in beta)
        else:
            # Subtract the fixed HFA offset and fit only the rating coefficients.
            beta, *_ = np.linalg.lstsq(X, y - self.hfa * hfa_term, rcond=None)
            self.b_epa, self.b_succ = (float(b) for b in beta)
            self.hfa_fitted = self.hfa
        self.n_games = len(g)
        return self

    def predict(self, games: pd.DataFrame) -> np.ndarray:
        X = self._features(games)
        hfa_term = np.where(games["neutral_site"].to_numpy(dtype=bool), 0.0, 1.0)
        return X @ np.array([self.b_epa, self.b_succ]) + self.hfa_fitted * hfa_term


def game_features(
    games: pd.DataFrame, ratings: pd.DataFrame, columns: Sequence[str] = ("net_epa", "net_succ")
) -> pd.DataFrame:
    """
    Join a ratings table onto a games table and form home-minus-away differences.
    Games where either team is unrated (an FCS opponent, a team with no prior
    plays) come back with NaN differences and must be dropped by the caller, not
    filled with zero — a zero difference is a real prediction of a coin flip.
    """
    r = ratings.set_index("team")
    out = games.copy()
    for col in columns:
        out[f"d_{col}"] = (
            out["home_team"].map(r[col]).to_numpy(dtype=np.float64)
            - out["away_team"].map(r[col]).to_numpy(dtype=np.float64)
        )
    return out


def tune_alpha(
    plays: pd.DataFrame,
    games: pd.DataFrame,
    alphas: Sequence[float] = (75, 125, 175, 225, 325),
    season_holdout: int | None = None,
) -> pd.DataFrame:
    """
    Grid over alpha scored on held-out GAME margin error, not on play-level fit.

    This distinction matters: play-level R^2 is dominated by the 17:1 play-to-play
    noise and will happily pick an alpha that predicts games worse. The thing we
    care about is margin MAE, so that is what selects.
    """
    rows = []
    seasons = sorted(games["season"].unique())
    holdout = season_holdout if season_holdout is not None else seasons[-1]
    train_plays = plays.loc[plays["season"] != holdout]
    test_games = games.loc[games["season"] == holdout]
    all_teams = sorted(set(plays["offense"]) | set(plays["defense"]))

    for a in alphas:
        model = BoudEff(alpha_epa=a, alpha_success=a).fit(train_plays, teams=all_teams)
        ratings = model.team_ratings()
        train_feat = game_features(
            games.loc[games["season"] != holdout], ratings
        ).dropna(subset=["d_net_epa", "margin"])
        if len(train_feat) < 10:
            continue
        mm = MarginModel().fit(train_feat)
        feat = game_features(test_games, ratings).dropna(subset=["d_net_epa", "margin"])
        if feat.empty:
            continue
        pred = mm.predict(feat)
        err = feat["margin"].to_numpy(dtype=np.float64) - pred
        rows.append(
            {
                "alpha": a,
                "holdout_season": holdout,
                "n_games": len(feat),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "b_epa": mm.b_epa,
                "b_succ": mm.b_succ,
            }
        )
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
