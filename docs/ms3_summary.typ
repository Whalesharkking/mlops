#set page(paper: "a4", margin: (x: 1.8cm, y: 1.6cm))
#set text(font: "Libertinus Serif", size: 10pt)
#set heading(numbering: "1.")
#set par(justify: true)
#show heading: set block(above: 0.9em, below: 0.6em)

#let status(s) = {
  let c = if s == "Changed" { rgb("#1a7f37") } else if s == "Partly" { rgb("#9a6700") } else if s == "Rejected" { rgb("#cf222e") } else { rgb("#57606a") }
  box(inset: (x: 3pt, y: 1pt), radius: 2pt, stroke: 0.5pt + c, text(fill: c, size: 8.5pt, weight: "bold")[#s])
}

// One entry per review point with label, title, status and file, then the answer.
#let entry(label, title, st, where, body) = block(spacing: 0.75em)[
  *#label · #title* · #status(st) #if where != none [(#raw(where))] \
  #body
]

#align(center)[
  #text(size: 15pt, weight: "bold")[MS3 Summary · Regional Nectar-Flow Forecast]
  #v(0.05em)
  #text(size: 9.5pt)[Elias Christen · I.BA_MLOPS HS26 · #link("https://github.com/Whalesharkking/mlops") · tag `ms3`]
]

= What I achieved in this milestone

- *What works now.* A training pipeline reads the feature store at one pinned dataset revision, trains one LightGBM model for all ten horizons, compares it per horizon with both baselines, logs the run to W&B and registers the model. A weekly GitHub Actions job retrains it from scratch, and a gate decides whether it becomes the new champion.
- *Result.* Tested once on 2026 (3155 rows, revision `cae98fa`), the model has an MAE of 0.19 to 0.23 kg/day. Climatology, the better baseline, has 0.25. The model meets the success criterion of the proposal at every horizon. In March to September it reaches 0.230 against 0.297 for climatology.
- *Key decisions:*
  - Two protocols. The evaluation protocol searches the parameters on the CV of 2024 to 2025 and tests once on 2026. The weekly retrain uses these parameters frozen in `configs/model.json`, so a weekly run is fast, deterministic and explainable.
  - Train on `label_clean`, but evaluate and gate on the raw `label`, so a flagged day still counts as an error.
  - L1 objective (the metric is MAE), fixed seed and deterministic mode. Two runs on the same revision give the same model and the same parameters.
  - Every model carries its feature columns, training period, dataset revision and Git commit, which MS4 uses to check its input.
- *What it cost:*
  - The weekly challenger is trained without the last eight weeks, because they are its holdout. The champion is always a model that has not seen the newest data.
  - The weekly gate needs 200 season rows in its holdout, which it gets only from about late March. Polling ends on 10.01.2027, so the only champion in this project is the first model from the evaluation protocol, trained up to 2025.
  - The 2026 test is not untouched in the strict sense, because I had looked at the baselines on it before.
  - Almost all of the gain comes from the weather forecast. Without the weather features the model falls back to the level of climatology (0.247), and the momentum features add nothing measurable (`--mode ablation`).
- *Look at first.* `src/nectarflow/training/train.py` and `gate.py`, `.github/workflows/train.yml`, `tests/test_training.py`, the W&B project `nectarflow`.

= Response to MS2 reviews

All labels are from the MS2 editorial summary, which covers every point of both reviews. The Must point comes first.

#entry("E1", "Training-side reader with a pinned revision", "Changed", "training/data.py, features/store.py")[
  `download()` resolves a branch to its commit and fetches only the feature files, labels and scale counts of one Hub revision and returns its commit id, which every run logs. `load_frame()` joins `labels.parquet` on `target_date` and keeps only `is_final` rows. Tests check the join and the revision.
]

#entry("B1", "A HiveWatch error stops the whole daily run (also B2)", "Changed", "features/hivewatch.py, pipeline.py")[
  Both HiveWatch calls now retry with backoff on connection errors, 429 and 5xx, like `weather.fetch_run`. A failed scale count logs a warning and leaves the day's value empty. `ingest_hive()` now runs inside the `try/finally`, so what was computed is always uploaded. The 14:23 UTC fallback no longer exists, it was removed after MS2 because a late start could fall after midnight.
]

#entry("B3", "The success criterion cannot be reproduced from the repo", "Changed", "training/evaluate.py")[
  `python -m nectarflow.training.evaluate --until <date> --revision <sha>` prints the MAE of both baselines and the row count per horizon. `train.py` uses the same function for the model, so model and baselines come from one script. All predictors are scored on the same rows. The old numbers in the text (0.29 and 0.31) came from a manual count on fewer rows and are replaced by this table. On 2026 up to 23.11. the baselines are 0.25 (climatology) and 0.27 (persistence, $h = 1$). They differ from the proposal (0.24 and 0.32) because persistence now uses only final days (MS1 A1), the climatology uses the median instead of the mean (MS1 A2), and the test now covers 2026 up to November.
]

#entry("E2", "Champion gate and time-ordered CV first", "Changed", "training/cv.py, gate.py")[
  Both exist and are tested, and the first model was logged only after them. Details under MS1 A3, B1 and B2 below.
]

#entry("A1", "The two-day label margin is thin, make the pipeline watch it (also E3)", "Changed", "pipeline.py, features/quality.py")[
  My MS2 figure "2 of 7 days" was too optimistic. Reviewer A is right. From 17.10. to 25.11.2026, yesterday's label was missing at poll time on 35 of 40 days, and on every day since 22.10. The label of $t_0 - 2$ was always there, and 0 of 68 labels changed after first appearing (largest change 0.0000 kg). The daily run now checks the label of $t_0 - 2$. If it is missing, the run still writes the features with empty momentum and uploads them, then it fails so the gap is seen. `quality.py` prints the weekly share of missing labels and runs in every weekly training job.
]

#entry("A2", "Day h = 1 is built from fewer hours", "Changed", "common/config.py")[
  `n_hours` is now in `FEATURE_COLUMNS`. It was already stored, so no feature file had to be rebuilt.
]

#entry("E4", "Decide what n_scales is", "Rejected", "training/data.py")[
  Not a feature, because it would be a linear guess in training and a real count in serving. I tested it as a sample weight, as promised in MS2. The CV got worse (MAE 0.188 with the weight, 0.184 without), so it is not used. The weighted run reuses the parameters searched without the weight, which slightly favours "no weight". The test runs again in every evaluation run. The daily count stays in the store as a monitoring signal for MS4.
]

#entry("A3", "The exact uv pin blocks newer uv versions", "Changed", "pyproject.toml, README.md")[
  The pin is now the range `>=0.13.7,<0.14`. I moved to 0.13, because uv 0.13 is the current version and the lock file is written with it. Docker uses 0.13.7. The README describes in four steps how to set up the cron-job.org trigger again.
]

#entry("E5", "Comment the cache invariant", "Changed", "features/store.py")[
  `FeatureStore.exists()` now explains why reading the file list once is safe. Daily run and backfill share the concurrency group `feature-store`.
]

*Open points from MS1*, answered "Not yet" in MS2 and done now.

#entry("MS1 A3", "Champion gate in winter", "Changed", "training/gate.py")[
  The gate compares challenger, champion and both baselines on the last eight weeks, only on rows with a target in March to September. It promotes only if the challenger beats the champion by 0.01 kg/day and both baselines, if the holdout ends in March to September and if it has at least 200 season rows. So nothing is promoted from October to February, as promised. The gate also refuses if the champion was trained on a holdout day. The first champion comes from the evaluation protocol if it beats both baselines on the 2026 season. A retrain on the data of 25.11. was not promoted (off season). The CV cannot check the margin of 0.01 kg/day directly, as promised in MS2, because the fold MAEs range from 0.08 to 0.30 with the season. The margin is about 4 % of the season MAE of the model (0.230), and I keep it.
]

#entry("MS1 B1", "Little training data for the later horizons", "Changed", "README.md (Result)")[
  The MAE is reported per horizon with its row count (311 to 320 test rows, about 614 training rows per horizon). Even $h = 6…10$ beat climatology clearly (0.203 to 0.227 against 0.25), so I do not add the GFS archive.
]

#entry("MS1 B2", "Time-ordered cross-validation for tuning", "Changed", "training/cv.py")[
  Folds are built on issue dates, one validation block per month of April to September 2025. Training rows end 10 days before a block, so no target is shared, and all ten rows of an issue date stay in one fold. The search uses only this CV, 2026 is used once for the final test.
]

