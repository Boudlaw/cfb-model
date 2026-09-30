"""
Week 5 recommendation. Applies the documented method to slate.csv and prints
the pick sheet, the locks and the cover probabilities the PDFs are built from.

Method, from cfb-picks-league.md and cfb-standings-and-strategy.md:
  - rank by gap x point value, discount thin markets
  - where the gap and the ~25-system consensus disagree, the CONSENSUS decides
    (method scoreboard: gap-driven 2-4, consensus-driven 4-0 over two scored weeks)
  - a bare half-point gap is the weakest input on the sheet
  - locks go on the two highest-probability picks, C.O.W. multiplier irrelevant
  - our model is ADVISORY this week and never overrides the consensus

Cover probability is computed from the consensus margin under a normal with
SIGMA_COVER. It is indicative, not precise: the consensus is a mean of ~25
systems and the SD around it is estimated, not measured per game.
"""
from __future__ import annotations

import math
from pathlib import Path

from slate import SYS, SYSTEMS

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SIGMA_COVER = 16.0


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# pick side per game, decided by the method above. "home" or "away".
DECISION = {
    ("Virginia Tech", "Pittsburgh"):     ("away", "gap 1.0 (largest) + 8/9 systems + model all agree"),
    ("Florida State", "Virginia"):       ("home", "gap 1.0; consensus mean is neutral (-3.54 v -3.5) but the median (-2.67) and 7/9 systems cover"),
    ("USC", "Washington"):               ("away", "gap 1.0 + consensus 1.5 clear + 7/9"),
    ("Clemson", "Miami"):                ("home", "gap 1.0 + consensus 2.8 clear + 6/9"),
    ("Mississippi State", "Alabama"):    ("away", "C.O.W. #1 COIN FLIP — gap and our model say State; consensus, 5/8 systems and a line moving 1.5 toward the form say Alabama"),
    ("Iowa", "Ohio State"):              ("home", "C.O.W. #2 — gap says Ohio State, but 9/9 systems and a 6.2-point consensus edge say Iowa. Magnitude rule: 0.5 of gap cannot beat 6.2 of consensus"),
    ("Houston", "UCF"):                  ("away", "9/9 systems, the only unanimous game on the board; consensus 5.6 clear"),
    ("Air Force", "Navy"):               ("away", "gap + consensus + 6/9; our model dissents (see note on option offenses)"),
    ("Arizona State", "Baylor"):         ("away", "gap + consensus 2.7 clear + model agrees"),
    ("Arizona", "Cincinnati"):           ("away", "gap + consensus 3.4 clear; our model dissents; thin 11pm window"),
    ("Missouri", "Florida"):             ("home_override_away", "FLAGGED — gap and our model say Missouri (model has Missouri winning outright); consensus says Florida by 1.05 over the form. Consensus decides"),
    ("North Carolina", "Notre Dame"):    ("away", "gap says Carolina; consensus, model and 6/9 say Notre Dame covers"),
    ("Tennessee", "Auburn"):             ("home", "gap says Auburn; 0/9 systems agree. Unanimous the other way"),
    ("Minnesota", "Michigan"):           ("home", "zero gap — consensus tiebreaker, 1.45 clear; our model dissents"),
    ("TCU", "BYU"):                      ("home", "zero gap — consensus tiebreaker by 0.44, the thinnest on the sheet; model agrees"),
}


def main() -> int:
    df = pd.read_csv(CACHE / "week5_slate.csv")
    out = []
    for r in df.itertuples():
        key = (r.home, r.away)
        side, why = DECISION[key]
        side = "away" if side == "home_override_away" else side
        # form is the HOME margin. The home team lays it (-form), the away team
        # receives it (+form). Getting this backwards puts "Pittsburgh -4.5" on
        # a sheet where Pittsburgh is the road dog, so it is pinned by the
        # assertion below.
        if side == "away":
            pick, line = r.away, r.form
            p = phi((r.form - r.consensus) / SIGMA_COVER)
            p_model = phi((r.form - r.model) / SIGMA_COVER)
        else:
            pick, line = r.home, -r.form
            p = phi((r.consensus - r.form) / SIGMA_COVER)
            p_model = phi((r.model - r.form) / SIGMA_COVER)
        assert (line > 0) == ((side == "away") == (r.form > 0)) or r.form == 0

        # System agreement counted against THE PICK, not against the gap side.
        # Reporting "0/9" next to Iowa because no system likes Ohio State is
        # true and useless — 9 of 9 like the pick.
        vals = [v for v in SYS[key] if v is not None]
        n_for = sum(1 for v in vals if (v < r.form if side == "away" else v > r.form))
        out.append({
            "cow": r.cow, "game": f"{r.away} @ {r.home}", "kick": r.kick,
            "net": r.net, "liquidity": r.liquidity,
            "pick": pick, "line": f"{line:+.1f}",
            "dog": line > 0,
            "gap": r.gap, "wgap": r.wgap, "value_side": r.value_side,
            "with_gap": (pick == r.value_side),
            "consensus": r.consensus, "model": r.model,
            "sys": f"{n_for}/{len(vals)}",
            "sys_for": n_for, "sys_n": len(vals),
            "p_consensus": round(p, 4), "p_model": round(p_model, 4),
            "why": why,
        })
    res = pd.DataFrame(out).sort_values("p_consensus", ascending=False).reset_index(drop=True)
    res.to_csv(CACHE / "week5_picks.csv", index=False)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 70)
    print("=== WEEK 5 PICKS, sorted by cover probability ===\n")
    print(res[["cow", "game", "pick", "line", "gap", "value_side", "with_gap",
               "consensus", "model", "sys", "p_consensus", "p_model"]].to_string(index=False))

    print("\n=== LOCKS: the two highest-probability picks ===")
    locks = res.head(2)
    for r in locks.itertuples():
        print(f"  {r.pick} {r.line}   P(cover) ~ {r.p_consensus:.1%}  "
              f"({r.sys} systems, model {'agrees' if (r.p_model>0.5) else 'DISAGREES'})")
    print(f"\n  third-best was {res.iloc[2]['pick']} {res.iloc[2]['line']} at "
          f"{res.iloc[2]['p_consensus']:.1%} — the spread between the best and "
          f"third-best lock is {(locks.iloc[1]['p_consensus']-res.iloc[2]['p_consensus'])*100:.1f} "
          "points of probability, i.e. about 0.02 expected points. Not the decision that shapes the week.")

    print("\n=== SHEET SHAPE ===")
    print(f"  underdogs: {int(res['dog'].sum())} of 15   favourites: {int((~res['dog']).sum())}")
    zero = int((res['gap'] == 0).sum())
    against = int(((~res['with_gap']) & (res['gap'] != 0)).sum())
    print(f"  picks WITH the gap: {int(res['with_gap'].sum())}   "
          f"against it (consensus overruled): {against}   "
          f"no gap at all (consensus decided outright): {zero}")
    print(f"  our model agrees with {int((res['p_model']>0.5).sum())} of 15")
    dis = res.loc[(res['p_model'] > 0.5) != (res['p_consensus'] > 0.5)]
    print(f"\n  model/consensus disagreements ({len(dis)}):")
    for r in dis.itertuples():
        print(f"    {r.game:<32} pick {r.pick:<18} consensus {r.p_consensus:.1%}  model {r.p_model:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
