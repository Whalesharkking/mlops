#set page(paper: "a4", margin: 2cm)
#set text(font: "Libertinus Serif", size: 10.5pt)
#set heading(numbering: "1.")
#set par(justify: true)

#let status(s) = {
  let c = if s == "Changed" { rgb("#1a7f37") } else if s == "Partly" { rgb("#9a6700") } else if s == "Rejected" { rgb("#cf222e") } else { rgb("#57606a") }
  box(inset: (x: 3pt, y: 1pt), radius: 2pt, stroke: 0.5pt + c, text(fill: c, size: 9pt, weight: "bold")[#s])
}

// One entry per review point: label · title — status (commit), then the answer.
#let entry(label, title, st, commit, body) = block(spacing: 0.9em)[
  *#label · #title* — #status(st) #if commit != none [(#raw(commit))] \
  #body
]

#align(center)[
  #text(size: 16pt, weight: "bold")[MS2 Summary — Regional Nectar-Flow Forecast]
  #v(0.1em)
  #text(size: 10pt)[Elias Christen · I.BA_MLOPS HS26 · #link("https://github.com/Whalesharkking/mlops") · tag `ms2`]
]

= What I achieved in this milestone

// TODO: rewrite in own words once the pipeline runs; do not copy the README.

- *What works now:* TODO. The feature pipeline runs every day at TODO UTC on GitHub Actions. It polls the HiveWatch regional weight, the scale count and the 00 UTC ECMWF run from Open-Meteo, archives the raw responses, and writes features and labels to the HF Hub dataset (TODO: link).
- *Backfill:* TODO. It runs from 14.03.2024 with the same code as the daily job; result: TODO rows, dataset revision TODO.
- *Key decisions:*
  - The momentum features end at $t_0 - 2$ (see A1).
  - Each row stores both its issue date and its target date, which makes the joins point-in-time correct.
  - The data is versioned by HF Hub dataset revisions instead of DVC, because every write already is a Git commit.
  - The pipeline refuses to write an incomplete weather run and retries. On 15.10.2026 at 07:46 UTC, the same day's run held only 72 of 240 hours.
- *What it cost:*
  - Momentum ending at $t_0 - 2$ means the newest day is not used as an input.
  - The scale count exists only from 15.10.2026 on; there is no history for it.
  - TODO: anything else, such as rows lost to flags or a rewrite.
- *Look at first:* TODO workflow file, TODO feature module, TODO dataset on the HF Hub, `tests/`.

= Response to MS1 reviews

All labels are from the MS1 editorial summary.

#entry("A4", "Permission to poll the bienen.ch endpoint", "Partly", none)[
  On 15.10.2026 I asked BienenSchweiz (cc HiveWatch) in writing whether daily polling and showing derived regional values are allowed. Answer: TODO.

  I tested the fallbacks the same day:
  - *Hiveeyes* has no Swiss nodes and offers only raw single-hive weight, so it is not usable.
  - The regional `midnightvalues` cover only 8 days and come from the same provider, so they are not independent.
  - I chose *TrachtNet Bayern* (state bee institutes) instead. It is public, gives the daily corrected change per region with the scale count, and has data from 2011. Permission was asked there too: TODO.

  If access is refused, only the source adapter changes. Files: `README.md` §Data sources, TODO adapter.
]

#entry("A1", "Training/serving skew in the momentum features", "Changed", "TODO")[
  I measured the problem on 15.10.2026. TrachtNet listed about 357 scales for 13.10, but only 279 for the previous day. In HiveWatch, the previous day's midnight value was still missing at 09:00 UTC.

  The suggested comparison of as-of and final values over the backfill is not possible, because the API returns only final values for the past. I removed the skew by construction instead:
  - All momentum windows end at $t_0 - 2$, so training and serving see the same, final days.
  - The raw daily responses are archived from now on. In MS3 they will show how much the last days still change.

  Files: TODO.
]

#entry("A2", "Changing scale population", "Partly", "TODO")[
  The set of scales does change; HiveWatch shows 364 for Nordalpen today but keeps no history. The effect on the label is small, though. The regional curve is an average of per-scale weight changes (it runs around 0 kg, not around absolute hive weights of about 60 kg), so a scale that joins or leaves shifts the average only by its own share. A single faulty scale is the larger risk: a +20 kg error moves the mean of 364 scales by about 0.05 kg, which is as large as a typical winter day.

  How I handle it:
  - The current count is stored daily as `n_scales`.
  - Days with an implausible change (|Δ| > TODO kg) get a `jump_flag` and are excluded from the label and the momentum.
  - The climatology uses the median.

  In the backfill, TODO days were flagged. Files: TODO.
]

#entry("A3", "Champion gate in winter", "Not yet", none)[
  I agree; I will implement it in MS3. One model keeps forecasting all year. A challenger can become champion only from March to September, if it beats the champion by at least TODO kg/day. From October to February, the challenger is trained and logged but never promoted, because near-zero targets cannot separate models.
]

#entry("B1", "Little training data for the later horizons", "Not yet", none)[
  I agree. The training set has about 650 forecast days (≈ 6,500 correlated rows), and the test set is one season. The bottleneck is the forecast archive, which starts on 14.03.2024, not the hive data (HiveWatch has data from 2019).

  I checked an alternative: the open GFS archive on AWS reaches back to 01.01.2021, which would give about three times as many days. In MS3 I will report the MAE per horizon with the sample size, and test whether GFS lets $h = 6…10$ beat climatology.
]

#entry("B2", "Time-ordered cross-validation for tuning", "Not yet", none)[
  I will implement it in MS3 as suggested:
  - expanding-window folds over forecast days, with all ten horizons of a day in one fold;
  - a 10-day gap before each validation block;
  - the first validation block in the 2025 flow season.

  The 2026 season stays untouched until the final test.
]
