#set page(paper: "a4", margin: 2cm)
#set text(font: "Libertinus Serif", size: 10.5pt)
#set heading(numbering: "1.")
#set par(justify: true)

#let status(s) = {
  let c = if s == "Changed" { rgb("#1a7f37") } else if s == "Partly" { rgb("#9a6700") } else if s == "Rejected" { rgb("#cf222e") } else { rgb("#57606a") }
  box(inset: (x: 3pt, y: 1pt), radius: 2pt, stroke: 0.5pt + c, text(fill: c, size: 9pt, weight: "bold")[#s])
}

// One entry per review point with label, title, status and commit, then the answer.
#let entry(label, title, st, commit, body) = block(spacing: 0.9em)[
  *#label · #title* · #status(st) #if commit != none [(#raw(commit))] \
  #body
]

#align(center)[
  #text(size: 16pt, weight: "bold")[MS2 Summary · Regional Nectar-Flow Forecast]
  #v(0.1em)
  #text(size: 10pt)[Elias Christen · I.BA_MLOPS HS26 · #link("https://github.com/Whalesharkking/mlops") · tag `ms2`]
]

= What I achieved in this milestone

- *What works now:* The feature pipeline runs on its own. Every morning a GitHub Actions job fetches the latest hive weights and the new weather forecast. It builds the features for the next ten days and saves them with the labels in my feature store on Hugging Face (#link("https://huggingface.co/datasets/Whalesharkking/nectarflow-features")[`Whalesharkking/nectarflow-features`]). If the forecast is not complete yet, the job waits for it. Every push also runs tests, a linter, a Docker build and a security scan.
- *Backfill:* It runs from 14.03.2024 with the same code as the daily job (`--start`), 940 issue dates (9,400 rows, 14.03.2024 to 17.10.2026) in one GitHub Actions run of 27 minutes. The dataset revision is `2752f4c`.
- *Key decisions:*
  - The momentum features end at $t_0 - 2$ (see A1).
  - Each row stores both its issue date and its target date, which makes the joins point-in-time correct.
  - The data is versioned by HF Hub dataset revisions instead of DVC, because every write already is a Git commit.
  - The pipeline refuses to write an incomplete weather run and retries. On 15.10.2026 at 07:46 UTC, the same day's run held only 72 of 240 hours.
- *What it cost:*
  - Momentum ending at $t_0 - 2$ means the newest day is not used as an input.
  - The scale count exists only from 17.10.2026 (first pipeline run) on, because HiveWatch keeps no history of it.
  - Eight archived ECMWF runs are missing at Open-Meteo (05.-09.08.2025, 11.06., 23.06. and 19.09.2026). These issue dates have no features. The pipeline logs and skips them instead of writing a partial run.
  - 56 of 2813 days are flagged as implausible jumps, mostly snow load in winter (see A2). A momentum window that contains such a day stays empty for up to 14 days.
  - With momentum ending at $t_0 - 2$, the persistence baseline may also use only the last final day. Recomputed on the 2026 data, persistence reaches 0.31 at $h = 1$ and climatology (±7 days, median) 0.29 at every $h$, so climatology is now the stronger baseline at all horizons (the proposal had 0.24 and 0.32).
- *Look at first:* `.github/workflows/feature-daily.yml`, `src/nectarflow/features/pipeline.py` and `compute.py`, the dataset on the HF Hub (link above), `tests/`.

= Response to MS1 reviews

All labels are from the MS1 editorial summary, which covers every concern of both reviews. The Must point comes first.

#entry("A4", "Permission to poll the bienen.ch endpoint", "Partly", none)[
  On 15.10.2026 I asked BienenSchweiz (cc HiveWatch) in writing whether daily polling and showing derived regional values are allowed. No reply so far (last checked 17.10.2026). Until then the pipeline polls once a day, credits the source and publishes only regional aggregates.

  I tested the fallbacks the same day:
  - *Hiveeyes* has no Swiss nodes and offers only raw single-hive weight, so it is not usable.
  - The regional `midnightvalues` cover only 8 days and come from the same provider, so they are not independent.
  - I chose *TrachtNet Bayern* (state bee institutes) instead. It is public, gives the daily corrected change per region with the scale count, and has data from 2011. Permission was asked there too, also without reply so far.

  TrachtNet stays a documented plan. An adapter is only built if HiveWatch access is refused or breaks, and then only the ingest module changes. See the section Data sources in `README.md`.
]

#entry("A1", "Training/serving skew in the momentum features", "Changed", "97c80d1")[
  I measured the problem on 15.10.2026. TrachtNet listed about 357 scales for 13.10, but only 279 for the previous day. In HiveWatch, the previous day's midnight value was still missing at 09:00 UTC.

  The suggested comparison of as-of and final values over the backfill is not possible, because the API returns only final values for the past. I removed the skew by construction instead:
  - All momentum windows end at $t_0 - 2$, so training and serving see the same, final days.
  - From now on, every run archives the last 30 days of the weight series exactly as polled (`raw/hivewatch/`). This gives the as-of values Reviewer A asked for, and in MS3 they show how much the last days still change before they are final.

  See `src/nectarflow/features/compute.py` (`momentum`), `labels.py` (`is_final`), `pipeline.py` (as-of snapshot in `raw/hivewatch/`).
]

#entry("A2", "Changing scale population", "Partly", "97c80d1")[
  The set of scales does change. HiveWatch shows 364 for Nordalpen today but keeps no history. The effect on the label is small, though. The regional curve is an average of per-scale weight changes (it runs around 0 kg, not around absolute hive weights of about 60 kg), so a scale that joins or leaves shifts the average only by its own share. A single faulty scale is the larger risk. A +20 kg error moves the mean of 364 scales by about 0.05 kg, which is as large as a typical winter day.

  How I handle it:
  - The current count is stored daily as `n_scales`.
  - Days with an implausible change (more than 2.5 kg in April to July, more than 1.0 kg otherwise) get a `jump_flag` and are excluded from the momentum and the climatology.
  - The climatology uses the median.

  56 of 2813 days since 2019 are flagged, none in the flow season, 21 in January 2021 (snow). See `src/nectarflow/features/labels.py`, `quality/n_scales.parquet`.
]

#entry("A3", "Champion gate in winter", "Not yet", none)[
  I agree and will implement it in MS3. One model keeps forecasting all year. A challenger can become champion only from March to September, if it beats the champion by at least 0.01 kg/day MAE (about 3 % of the climatology MAE of 0.29 and below the 10 % success margin, to be checked with the CV of B2). From October to February, the challenger is trained and logged but never promoted, because near-zero targets cannot separate models.
]

#entry("B1", "Little training data for the later horizons", "Not yet", none)[
  I agree, and the backfill confirms the numbers. There are 644 training issue dates with 6,197 labelled rows (about 620 per horizon, 2,300 in the flow season April to July), and 2,805 test rows in 2026. The bottleneck is the forecast archive, which starts on 14.03.2024, not the hive data (HiveWatch has data from 2019).

  In MS3 I will report the MAE per horizon together with these sample sizes and say whether $h = 6…10$ can beat climatology at all. A longer archive (GFS on AWS, from 2021) would triple the days but needs raw GRIB downloads. I only consider it if the later horizons clearly suffer from too little data.
]

#entry("B2", "Time-ordered cross-validation for tuning", "Not yet", none)[
  I will implement it in MS3 as suggested:
  - Expanding-window folds over forecast days, with all ten horizons of a day in one fold.
  - A 10-day gap before each validation block.
  - The first validation block in the 2025 flow season.

  The 2026 season stays untouched until the final test.
]
