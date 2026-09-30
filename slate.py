"""
Week 5 pool slate: the form's number against the market, the comparators and
our model. Produces the table the three PDFs are built from.

Sign convention throughout: every margin is HOME MINUS AWAY, positive = home
favoured. The form's posted spread is converted to that frame once, here, and
never re-derived downstream.

  gap = form_margin - market_margin
  gap > 0  the form is more generous to the AWAY team -> value on the away side
  gap < 0  value on the home side
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"

# game_key: (home, away, form_home_margin, cow_multiplier, kickoff, network, liquidity)
# form_home_margin read off the Week 5 form PDF, converted to home-minus-away.
SLATE = [
    ("Virginia Tech",     "Pittsburgh",  +4.5, 1, "Fri 7:00p",  "ESPN",  "mid"),
    ("Minnesota",         "Michigan",     -5.5, 1, "Sat 12:00p", "FOX",   "liquid"),
    ("North Carolina",    "Notre Dame",  -21.5, 1, "Sat 12:00p", "ESPN",  "liquid"),
    ("Air Force",         "Navy",         +3.5, 1, "Sat 12:00p", "CBS",   "mid"),
    ("Houston",           "UCF",         +12.5, 1, "Sat 12:00p", "ESPN2", "thin"),
    ("Florida State",     "Virginia",     -3.5, 1, "Sat 3:30p",  "ESPN2", "mid"),
    ("Missouri",          "Florida",      -5.5, 1, "Sat 3:30p",  "ABC",   "liquid"),
    ("Tennessee",         "Auburn",       +7.5, 1, "Sat 3:30p",  "ESPN",  "liquid"),
    ("TCU",               "BYU",          -6.5, 1, "Sat 7:00p",  "ESPN",  "mid"),
    ("USC",               "Washington",  +10.5, 1, "Sat 7:30p",  "NBC",   "liquid"),
    ("Clemson",           "Miami",       -17.5, 1, "Sat 7:30p",  "ABC",   "liquid"),
    ("Arizona State",     "Baylor",       +4.5, 1, "Sat 10:30p", "ESPN",  "mid"),
    ("Arizona",           "Cincinnati",   +7.5, 1, "Sat 11:00p", "FOX",   "thin"),
    ("Mississippi State", "Alabama",      -6.5, 2, "Sat 12:00p", "ABC",   "liquid"),
    ("Iowa",              "Ohio State",  -13.5, 2, "Sat 3:30p",  "CBS",   "liquid"),
]

# ThePredictionTracker, read 2026-09-29. Keyed by (their home, their away).
# line_open, line_now, consensus_avg, consensus_median, sd, home_covers_prob
TPT = {
    ("Virginia Tech", "Pittsburgh"):      (5.5, 3.5, -1.90, -2.00, 5.25, 0.4082),
    ("Minnesota", "Michigan"):            (-6.5, -5.5, -4.05, -4.08, 3.26, 0.5253),
    ("North Carolina", "Notre Dame"):     (-21.5, -21.0, -23.11, -22.50, 5.78, 0.4641),
    ("Air Force", "Navy"):                (1.5, 3.0, 2.72, 3.34, 6.52, 0.4953),
    ("Houston", "UCF"):                   (11.5, 12.0, 6.88, 7.38, 4.57, 0.4125),
    ("Florida State", "Virginia"):        (-3.0, -2.5, -3.54, -2.67, 5.27, 0.4823),
    ("Missouri", "Florida"):              (-4.5, -5.0, -6.55, -6.50, 6.53, 0.4738),
    ("Tennessee", "Auburn"):              (7.5, 7.0, 8.56, 8.66, 3.24, 0.5271),
    ("TCU", "BYU"):                       (-6.5, -6.5, -6.06, -5.50, 4.60, 0.5076),
    ("USC", "Washington"):                (9.5, 9.5, 9.01, 9.09, 3.68, 0.4916),
    ("Clemson", "Miami"):                 (-17.5, -16.5, -14.71, -14.91, 6.75, 0.5302),
    ("Arizona State", "Baylor"):          (4.5, 4.0, 1.78, 3.25, 6.90, 0.4628),
    ("Arizona", "Cincinnati"):            (7.5, 7.0, 4.08, 4.83, 4.74, 0.4498),
    ("Mississippi State", "Alabama"):     (-4.5, -6.0, -7.03, -7.25, 3.26, 0.4821),
    ("Iowa", "Ohio State"):               (-14.5, -14.0, -7.26, -7.25, 4.99, 0.6142),
}

# Named systems from ThePredictionTracker's per-system table, same read.
SYSTEMS = ["Sagarin Pred.", "Pi-Ratings Bias", "ESPN FPI", "Sonny Moore",
           "Massey Ratings", "Team Rankings", "FEI Projections", "Dunkel Index",
           "Beck Elo"]
SYS = {
    ("Virginia Tech", "Pittsburgh"):  [0.52, 3.6, -2.68, 1.26, -3.09, -2.0, -5.0, 0.03, 4.99],
    ("Minnesota", "Michigan"):        [-2.20, -5.9, -9.63, 3.75, -8.75, -3.0, -7.6, -4.23, -3.89],
    ("North Carolina", "Notre Dame"): [-16.68, -32.8, -20.03, -24.78, -25.17, -24.4, -30.1, -23.20, -17.37],
    ("Air Force", "Navy"):            [3.39, -2.2, 5.31, 8.02, -1.15, 2.0, 0.3, -10.70, 5.62],
    ("Houston", "UCF"):               [7.77, 8.9, 8.14, 5.38, 5.20, 6.8, 4.0, 8.18, 9.69],
    ("Florida State", "Virginia"):    [-0.17, 1.0, -2.89, -10.63, 1.35, -6.6, 3.8, -3.25, 0.81],
    ("Missouri", "Florida"):          [-3.17, -7.4, -7.29, -6.78, -3.48, -8.7, -3.1, -6.79, -3.13],
    ("Tennessee", "Auburn"):          [7.98, 10.9, 7.62, 16.04, 10.18, 9.3, 9.7, 12.22, 12.42],
    ("TCU", "BYU"):                   [-0.67, -10.2, -7.42, -2.91, -2.85, -5.2, -8.7, -10.79, -14.52],
    ("USC", "Washington"):            [10.22, 9.3, 14.02, 1.97, 6.60, 10.8, 9.4, 1.55, 6.73],
    ("Clemson", "Miami"):             [-7.88, -15.1, -16.5, -28.47, -10.71, -18.0, -17.2, -23.60, -14.91],
    ("Arizona State", "Baylor"):      [4.70, 7.6, -0.81, 4.22, 4.78, 4.1, 1.6, 4.23, 0.77],
    ("Arizona", "Cincinnati"):        [5.69, 8.7, 2.12, -2.03, 4.83, 3.4, 4.2, 10.77, 8.53],
    ("Mississippi State", "Alabama"): [-3.65, -10.2, -8.98, -5.93, -10.38, -8.1, None, -6.94, -5.24],
    ("Iowa", "Ohio State"):           [-5.78, -11.0, -13.46, -6.91, -10.91, -13.0, -10.0, -7.25, -6.63],
}

# CFBD team naming differs in three places.
CFBD_ALIAS = {"UCF": "UCF", "Miami": "Miami"}


def main() -> int:
    pred = pd.read_csv(CACHE / "week5_predictions.csv")
    pred = pred.set_index(["home_team", "away_team"])

    rows = []
    for home, away, form, cow, kick, net, liq in SLATE:
        key = (home, away)
        open_, now, cavg, cmed, csd, hcov = TPT[key]
        p = pred.loc[(CFBD_ALIAS.get(home, home), CFBD_ALIAS.get(away, away))]

        gap = form - now
        value_side = away if gap > 0 else (home if gap < 0 else "—")
        wgap = abs(gap) * cow

        sysvals = [v for v in SYS[key] if v is not None]
        # A system "agrees" with the value side if its margin lands on that side
        # of the FORM's number — the number we actually have to beat.
        if gap > 0:
            agree = sum(1 for v in sysvals if v < form)
            model_agrees = p["model_margin"] < form
            boud_agrees = p["boudeff_only"] < form
        elif gap < 0:
            agree = sum(1 for v in sysvals if v > form)
            model_agrees = p["model_margin"] > form
            boud_agrees = p["boudeff_only"] > form
        else:
            agree = sum(1 for v in sysvals if v > form)
            model_agrees = p["model_margin"] > form
            boud_agrees = p["boudeff_only"] > form

        rows.append({
            "home": home, "away": away, "cow": cow, "kick": kick,
            "net": net, "liquidity": liq,
            "form": form, "mkt_open": open_, "mkt_now": now,
            "cfbd_now": round(float(p["market_margin_now"]), 1),
            "gap": round(gap, 2), "wgap": round(wgap, 2),
            "value_side": value_side,
            "consensus": cavg, "cons_median": cmed, "cons_sd": csd,
            "home_covers": hcov,
            "model": round(float(p["model_margin"]), 1),
            "boudeff": round(float(p["boudeff_only"]), 1),
            "elo": round(float(p["elo_only"]), 1),
            "win_prob_home": round(float(p["win_prob_home"]), 3),
            "d_net_epa": round(float(p["d_net_epa"]), 4),
            "sys_agree": agree, "sys_n": len(sysvals),
            "model_agrees": bool(model_agrees),
            "boud_agrees": bool(boud_agrees),
            "cons_vs_form": round(cavg - form, 2),
            "model_vs_form": round(float(p["model_margin"]) - form, 2),
        })

    df = pd.DataFrame(rows)
    df["market_disagree"] = (df["mkt_now"] - df["cfbd_now"]).abs()
    df = df.sort_values(["wgap", "sys_agree"], ascending=False).reset_index(drop=True)
    df.to_csv(CACHE / "week5_slate.csv", index=False)

    pd.set_option("display.width", 260)
    cols = ["home", "away", "cow", "form", "mkt_now", "gap", "wgap", "value_side",
            "consensus", "model", "boudeff", "elo", "sys_agree", "sys_n",
            "model_agrees", "liquidity"]
    print(df[cols].to_string(index=False))

    print("\n--- two-source market check (TPT vs CFBD DraftKings+Bovada) ---")
    bad = df.loc[df["market_disagree"] > 0.01, ["home", "away", "mkt_now", "cfbd_now", "market_disagree"]]
    print(bad.to_string(index=False) if len(bad) else "  identical on all 15")

    print("\n--- C.O.W. games ---")
    print(df.loc[df["cow"] == 2, cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
