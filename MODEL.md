# Modeling layer — what is built, and how to run it

Phases 2–4 of the build plan. Phase 1 (the weekly archiver) is already live and
is untouched by any of this.

## Files

| File | Phase | What it is |
|---|---|---|
| `ingest.py` | 0/2a | One-time historical pull of 2022–2025 plays, games, lines. Stdlib only. Runs in Actions. |
| `features.py` | 2b | Success rate + EPA per play. Garbage-time and non-play filters. The Phase 2 correctness gate. |
| `ratings.py` | 3 | BOUD-EFF: ridge opponent adjustment, linear on EPA and **logistic on success**. Prior blending, alpha tuning, ratings→points. |
| `backtest.py` | 4 | Season-as-fold walk-forward, full metric suite, calibration, leak warnings. |
| `run_model.py` | — | Driver. Loads `data/`, runs the gate, tunes, walks forward, prints the report. |
| `test_model_offline.py` | — | 39 checks, no network. Includes four leakage guards. |

## Order of operations

The CFBD API is **not reachable** from the environment where the modeling code
runs, so the network half and the modeling half are separate:

1. **Actions → "Historical CFB ingest (one-time)" with `smoke = true`.**
   Reads the API and prints the real response keys. Writes nothing. Read the log:
   `ingest.py`'s field mapping was written without API access, so this is the
   first confirmation it is correct. Any field reported as `MAPPED TO NONE` must
   be fixed in `map_play()` before proceeding.
2. **Same workflow with `smoke = false`.** ~76 API calls, inside the free tier's
   1,000/month. Commits `data/plays_*.csv.gz`, `games_*`, `lines_*` to the repo.
3. **`python run_model.py`** wherever the repo is cloned.

## Reading the output, in order

1. **The Phase 2 gate.** Success rate must land near 40%. If it does not, stop —
   the fault is in the ingest mapping and nothing downstream means anything.
2. **`unknown_play_types`.** Non-empty means CFBD renamed something; review
   `SCRIMMAGE_PLAY_TYPES`.
3. **The WARNINGS block.** MAE under 12, winner accuracy over 76%, or ATS over
   55% are *symptoms of a leak*, not achievements. The likeliest leak in this
   project is a season-level SP+/FPI rating used to predict a game inside that
   season — which is the entire reason the weekly archiver exists.
4. **Then** the metrics, always against the benchmarks printed beside them.

## The two leakage surfaces, and why both are closed

Closing only the first is the common half-fix:

1. The **ratings** must not see the week being predicted → `BoudEff.fit_through()`
   filters to plays strictly before the cutoff and asserts it.
2. The **ratings→points mapping** must not see the season being predicted →
   `walk_forward()` fits `MarginModel` on prior seasons only.

`test_model_offline.py` proves this rather than asserting it. The decisive check,
`LEAK GUARD: corrupting late weeks does not move early-week predictions`,
corrupts the outcomes of weeks 8+ and requires week-4 predictions to be
bit-for-bit identical. Verified to actually fail: patching `fit_through` to fit
the whole season makes three guards fail, the perturbation test reporting a
3.55-point delta.

## Choices already made, and the reasoning

- **Logistic ridge on success**, not linear. Success is binary; linear ridge on a
  0/1 target can predict outside [0,1] and has heteroskedastic errors by
  construction. No published college football implementation appears to do this
  correctly, which makes it the one genuinely novel piece here.
- **Two coefficients, not an average.** EPA coefficients are points-per-play,
  success coefficients are log-odds. `MarginModel` learns the conversion instead
  of averaging incompatible units.
- **HFA fixed at 2.5**, not fitted, and not per team — team-specific home
  advantage has near-zero year-over-year correlation, so fitting it is fitting
  noise.
- **σ = 16** for win probabilities, not 14.1 (that is the SD around the *Vegas
  closing spread*, a sharper predictor — borrowing it would make every
  probability overconfident by 2–4 points) and not 13 (an NFL number).
- **The betting line is the yardstick, never a feature.** Feeding it in produces
  excellent MAE and zero ability to detect an edge, because the result is a
  line-adjuster wearing a model's clothes.
- **Weeks 1–2 are excluded**, not guessed. ~300 plays (4.3 games) are needed
  before a team estimate is even the size of the true signal. Predicting them
  needs the preseason prior — Phase 5.

## Known limits

- Numbers from `run_model.py` on the synthetic test data are **not** indicative:
  the generator uses a 13-point noise SD against reality's ~16, so it produces an
  MAE near 10.5 and trips the leak warning by construction. On real data an MAE
  under 12 should be treated as a genuine red flag.
- `ingest.py`'s field mapping is inferred, not verified. Run `--smoke` first.
- Phase 5 (the ensemble) and Phase 6 (weekly prediction output) are not built.
  Phase 5 needs point-in-time SP+/FPI/Sonny Moore/Pi-Rate, which the weekly
  archiver is accumulating now — one row per week, and it cannot be backfilled.
- The Elo→points divisor of 25 is **wrong** (measured −7.1 points mean signed
  deviation on opening weekend 2026). Fit it before Elo joins any blend.
- Sonny Moore team names need an alias map before that source can be joined.
