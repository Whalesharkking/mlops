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

- *What works now.* The feature pipeline runs on its own. Every morning a GitHub Actions job fetches the latest hive weights and the new weather forecast. It builds the features for the next ten days and saves them with the labels in my feature store on Hugging Face (#link("https://huggingface.co/datasets/Whalesharkking/nectarflow-features")[`Whalesharkking/nectarflow-features`]). If the forecast is not complete yet, the job waits for it. Every push also runs tests, a linter, a Docker build and a security scan.
- *Backfill.* It runs from 14.03.2024 with the same code as the daily job (`--start`), 940 issue dates (9,400 rows, 14.03.2024 to 17.10.2026) in one GitHub Actions run of 27 minutes. The backfill is in dataset revision `94f2481`.
- *Key decisions:*
  - The momentum features end at $t_0 - 2$ (see A1).
  - Each row stores both its issue date and its target date, which makes the joins point-in-time correct.
  - The data is versioned by HF Hub dataset revisions instead of DVC, because every write already is a Git commit.
  - The pipeline refuses to write an incomplete weather run and retries. On 15.10.2026 at 07:46 UTC, the same day's run held only 72 of 240 hours.
  - cron-job.org starts the daily run at 08:30 UTC, because GitHub started scheduled runs up to seven hours late. A GitHub cron stays as fallback and stops at once if today's features already exist.
- *What it cost:*
  - Momentum ending at $t_0 - 2$ means the newest day is not used as an input.
  - The scale count exists only from 17.10.2026 (first pipeline run) on, because HiveWatch keeps no history of it.
  - Eight archived ECMWF runs are missing at Open-Meteo (05.-09.08.2025, 11.06., 23.06. and 19.09.2026). These issue dates have no features. The pipeline logs and skips them instead of writing a partial run.
  - 56 of 2813 days are flagged as implausible jumps, mostly snow load in winter (see A2). A momentum window that contains such a day stays empty for up to 14 days.
  - The persistence baseline gets weaker, because it may also use only final days (see A1).
- *Look at first.* `.github/workflows/feature-daily.yml`, `src/nectarflow/features/pipeline.py` and `compute.py`, the dataset on the HF Hub (link above), `tests/`.

= Response to MS1 reviews

All labels are from the MS1 editorial summary, which covers every concern of both reviews. It lists no E points and no points marked as not confirmed. The Must point comes first.

#entry("A4", "Permission to poll the bienen.ch endpoint", "Changed", "27eddb4, 623a2b6")[
  On 15.10.2026 I asked BienenSchweiz and HiveWatch in writing. On 19.10.2026 HiveWatch granted permission for this non-commercial project. Daily polling and the history download from 2019 are allowed. Derived regional daily values may be shown in the repo and on the demo page, without raw data or data of single scales and with the source credited as "HiveWatch (www.hivewatch.ch)". The terms are recorded in the section Data sources in `README.md` and in the dataset card. The as-of archive keeps daily labels only, as the terms require.

  On 15.10.2026 I also tested the fallbacks:
  - *Hiveeyes* has no Swiss nodes and offers only raw single-hive weight, so it is not usable.
  - The regional `midnightvalues` cover only 8 days and come from the same provider, so they are not independent.
  - I chose *TrachtNet Bayern* (state bee institutes) instead. It is public, gives the daily corrected change per region with the scale count, and has data from 2011. Permission was asked there too, without reply so far.

  TrachtNet stays a documented plan and is only built if HiveWatch access breaks. It covers Bavaria, not the Nordalpen, so the model would then have to be retrained for the new region.
]

#entry("A1", "Training/serving skew in the momentum features", "Changed", "97c80d1, 623a2b6")[
  The problem is real in the HiveWatch data. On 15.10.2026 at 09:00 UTC, the midnight value of the previous day was still missing.

  The suggested comparison of as-of and final values over the backfill is not possible, because the API returns only final values for the past. I removed the skew by construction instead:
  - All momentum windows end at $t_0 - 2$, so training and serving see the same, final days.
  - From now on, every run archives the daily labels of the last 30 days as computed on that day (`asof/labels/`). This gives the as-of values Reviewer A asked for. Momentum is a sum of daily labels, so the momentum as known at each $t_0$ can be rebuilt from them.

  The archive already supports the two-day lag. From 17.10. to 23.10.2026, the label of the previous day was still missing at poll time on 2 of 7 days. The label of two days before was always there, and no label changed after it first appeared. Seven days are not proof yet, so the archive keeps measuring in MS3.

  This changes the success criterion. The proposal compared against persistence with MAE 0.24 at $h = 1$. For a fair comparison, persistence may now also use only final days, which gives 0.31 on 2026. Climatology has 0.29 at every horizon and is now the stronger baseline at all horizons. The rule itself stays the same. The model must beat the stronger baseline by 10 % for $h = 1…5$ and match it for $h = 6…10$.

  See `src/nectarflow/features/compute.py` (`momentum`), `labels.py` (`is_final`), `pipeline.py` (as-of snapshot in `asof/labels/`).
]

#entry("A2", "Changing scale population", "Changed", "97c80d1")[
  HiveWatch confirmed by e-mail how the regional value is built. It sums the daily increases of all scales and divides them by the number of scales. Scales below 15 kg are left out. A scale that joins or stops sending therefore does not shift the average. The set itself does change. It grew from about 50 scales in 2019 to 364 today (HiveWatch estimate). A single faulty scale is the remaining risk. A +20 kg error moves the mean of 364 scales by about 0.05 kg, which is as large as a typical winter day.

  How I handle it:
  - The current count is stored daily as `n_scales`.
  - Days with an implausible change (more than 2.5 kg in April to July, more than 1.0 kg otherwise) get a `jump_flag` and are excluded from the momentum and the climatology.
  - The climatology uses the median instead of the mean planned in the proposal, so single faulty days have less weight.

  56 of 2813 days since 2019 are flagged, none in the flow season, 21 in January 2021 (snow). Smaller winter noise below 1 kg remains. It matters little, because the model is judged mainly in the flow season and there is no promotion in winter (A3). For the years before 17.10.2026, HiveWatch suggests a linear estimate of the scale count. In MS3 I test whether it helps, for example to weight the noisier early years less. See `src/nectarflow/features/labels.py`, `quality/n_scales.parquet`.
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
