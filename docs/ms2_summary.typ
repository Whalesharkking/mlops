#set page(paper: "a4", margin: 2cm)
#set text(font: "Libertinus Serif", size: 10.5pt)
#set heading(numbering: "1.")
#set par(justify: true)

#let status(s) = {
  let c = if s == "Changed" { rgb("#1a7f37") } else if s == "Partly" { rgb("#9a6700") } else if s == "Rejected" { rgb("#cf222e") } else { rgb("#57606a") }
  box(inset: (x: 3pt, y: 1pt), radius: 2pt, stroke: 0.5pt + c, text(fill: c, size: 9pt, weight: "bold")[#s])
}

#let entry(label, title, st, body) = block(spacing: 0.9em)[
  *#label · #title* #h(0.4em) #status(st) \
  #body
]

#align(center)[
  #text(size: 16pt, weight: "bold")[MS2 Summary: Feature Pipeline]
  #v(0.1em)
  #text(size: 11pt)[Regional Nectar-Flow Forecast · I.BA_MLOPS · HS26]
  #v(0.1em)
  #text(size: 10pt)[Elias Christen · #link("https://github.com/Whalesharkking/mlops")]
]

= Summary of the milestone

// TODO: 4–6 sentences once the pipeline runs. State what works now.

*What works.* TODO: The feature pipeline runs every day at TODO UTC on GitHub Actions. It polls the HiveWatch regional weight and the 00 UTC ECMWF IFS run from the Open-Meteo Single Runs API, archives both raw responses, and writes features and labels to the feature store on the HuggingFace Hub (TODO: link). It refuses to write an incomplete weather run and retries instead: on 15.10.2026 at 07:46 UTC, the same day's run held only 72 of 240 hours. A backfill from 14.03.2024 ran with the same code (TODO: number of rows, dataset revision).

*Key decisions.*
- The momentum features end at $t_0 - 2$, so they use only final days (A1).
- Every row stores both its issue date and its target date, which makes the joins point-in-time correct.
- The feature store is versioned through HF Hub dataset revisions instead of DVC: every write is a Git commit, and that revision is the data version logged with each model in MS3.

*Where to look first.*
- TODO: workflow file
- TODO: feature module
- TODO: dataset on the HF Hub
- The README sections "Data sources & permission to use" and "Label & features"

= Response to reviews

All labels are from the MS1 editorial summary.

#entry("A4", "Permission to poll the bienen.ch endpoint", "Partly")[
  On *15.10.2026* I asked BienenSchweiz (cc HiveWatch) in writing whether daily polling and publishing derived regional values are allowed. Answer: TODO (pending / allowed / refused, date).

  The fallbacks were tested on 15.10.2026:
  - *Hiveeyes* has no Swiss nodes and offers only raw single-hive weight, so it cannot replace the label.
  - *`midnightvalues`* exists only per single scale, so I dropped it as a regional path.
  - The new fallback is *TrachtNet Bayern*, run by state bee institutes. It is public, gives the daily corrected change per region with the scale count, and has data from 2011. Permission was asked there too (TODO: status).

  If access is refused, only the source adapter changes. Files: `README.md` §"Data sources & permission to use", TODO: adapter.
]

#entry("A1", "Training/serving skew in the momentum features", "Changed")[
  Past as-of states cannot be reconstructed for the backfill, because the API only returns today's (final) values for history. A comparison of as-of and final values over the backfill, as suggested, is therefore not possible. I solved the skew by construction instead:
  - All momentum windows end at $t_0 - 2$, so they only use days whose label is already final. Training and serving see identical inputs.
  - In addition, the pipeline archives every daily raw response as it was known at $t_0$. From the first live day on, this lets me measure how much the last two days still change, and decide in MS3 whether the newest days are worth adding as features.

  Files: TODO.
]

#entry("A2", "Changing scale population", "Partly")[
  The HiveWatch response contains no scale count. TODO, pick the final approach:
  - The pipeline flags days whose change exceeds TODO kg without a matching weather signal, and stores the flag as a column.
  - The fallback source TrachtNet provides the daily scale count directly.

  TODO: what was observed in the backfill (number of flagged days). Files: TODO.
]

#block(spacing: 0.9em)[
  *A3, B1, B2 · Champion gate, training data, cross-validation* #h(0.4em) #status("Not yet") \
  All three concern training and evaluation and are planned for MS3. I accept them as suggested:
  - *A3:* the champion gate compares models only on flow-season weeks (April–July), with a minimum MAE gain of TODO kg/day.
  - *B1:* the MAE is reported per horizon together with the sample size (about 6,500 correlated rows, one test season). I will discuss whether $h = 6 … 10$ can beat climatology.
  - *B2:* expanding-window CV over forecast days, with all ten horizons of a day in the same fold, a 10-day gap before each block, and the first validation block in the 2025 flow season. The 2026 season stays untouched until the final evaluation.
]
