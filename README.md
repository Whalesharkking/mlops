# Regional Nectar-Flow Forecast

> 10-day forecast of the daily net hive-weight change for the Central-Swiss pre-Alps (HiveWatch region *Nordalpen*), built as a live FTI system for HSLU I.BA_MLOPS HS26.

<!-- TODO MS4: video pitch link goes here (first line of README) -->

| | |
|---|---|
| **Milestone** | MS2 – Feature pipeline (due 05.11.2026) |
| **Proposal** | [docs/proposal.pdf](docs/proposal.pdf) |
| **MS2 summary & review response** | [docs/ms2_summary.pdf](docs/ms2_summary.pdf) |
| **Feature store** | TODO: link to HF Hub dataset |

## What it predicts

Every morning the system forecasts the **net weight change in kg/day** of the monitored colonies in the *Nordalpen* region (roughly the cantons of Lucerne, Schwyz, Uri, Obwalden and Nidwalden), for today and the next nine days (horizon *h* = 1…10). The value is the regional average of about 372 scales, with beekeeper interventions (feeding, honey removal, adding boxes) filtered out.

A positive value means the colonies collect more nectar than they consume. A negative value means they live on their stores.

The model must beat two baselines on per-horizon MAE:

- **Persistence:** every coming day repeats the last complete day.
- **Climatology:** the usual change for that date, averaged over 2019–2023 (±7 days).

The target is at least 10 % better than the better baseline for *h* = 1…5, and at least equal for *h* = 6…10.

## Architecture

```mermaid
flowchart LR
  subgraph src[Live data]
    H[HiveWatch API<br/>regional weight, nightly]
    T[TrachtNet Bayern<br/>fallback]
    O[Open-Meteo Single Runs<br/>ECMWF IFS, 10 days]
  end
  H --> FP
  T -.->|fallback| FP
  O --> FP
  FP[Feature pipeline<br/>daily + backfill<br/>GitHub Actions] -->|features + labels| FS[(Feature store<br/>Parquet on HF Hub)]
  FS -->|read| TP[Training pipeline<br/>weekly, LightGBM]
  TP -->|challenger| MR[(Model registry<br/>W&B)]
  MR -->|@champion| IP[Inference pipeline<br/>daily batch]
  FS -.->|features at t0| IP
  IP --> UI[Gradio UI<br/>HF Spaces]
```

| Component | Status |
|---|---|
| Feature pipeline (daily + backfill) | MS2 – TODO: in progress / done |
| Feature store (Parquet on HF Hub) | MS2 – TODO |
| Training pipeline + W&B registry | MS3 – planned |
| Inference pipeline + Gradio UI | MS4 – planned |

## Data sources & permission to use

### Hive weight – HiveWatch / BienenSchweiz (primary)

- **Source:** the public scale map on [bienen.ch](https://bienen.ch/imkerei/zucht/waagvoelker/), operated by BienenSchweiz with HiveWatch.
- **Endpoint:** `GET https://app.hivewatch.net/region/11/averages?sensorType=ext_weight&startdate=<epoch ms>&enddate=<epoch ms>`. It needs no login and returns hourly, intervention-filtered weight for region 11 ("Nordalpen").
- **Permission:** the endpoint serves the public map and is not a documented API. On **15.10.2026** I asked BienenSchweiz (cc HiveWatch) in writing whether automated daily polling and publishing derived regional daily values are allowed.
  - **Status:** TODO – reply pending / answer: …
- **Polling:** once per day, plus a one-time history download, and only for the duration of the project (until about January 2027).

### Hive weight – fallback options (tested 15.10.2026)

| Source | Result | Usable as fallback? |
|---|---|---|
| HiveWatch `midnightvalues` | Exists only per single scale (`controller/<id>/midnightvalues`), not per region | No – it is not a regional fallback |
| [Hiveeyes](https://swarm.hiveeyes.org) | Open API (no key). There are 332 public dashboards, none in Switzerland, and the sampled nodes are inactive (e.g. last data May 2023). It serves raw single-hive weight that still includes interventions | No |
| [TrachtNet](https://www.bienenkunde.rlp.de) (DLR Rheinland-Pfalz & LWG Bayern) | Public web portal without login. It provides daily **corrected** weight change per region and the **number of scales per day**. Data runs from 2011 and is fresh up to the previous day. Bayern had about 127 scales in 2019, about 250 in 2025 and about 335 in 2026. Switzerland has only 1–4 scales | **Yes** – Bayern / Oberbayern |

A permission request was also sent to TrachtNet (trachtnet@dlr.rlp.de). Status: TODO.

Switzerland has no other open hive-scale API. I checked the Romandie, Ticino, Agroscope and opendata.swiss.

### Weather forecasts – Open-Meteo Single Runs API

- **Endpoint:** `https://single-runs-api.open-meteo.com/v1/forecast`. It serves every archived run of **ECMWF IFS HRES (9 km)** since 14.03.2024 and needs no key. Data: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0).
- **Run used:** the 00 UTC run, for four locations: Lucerne, Schwyz, Altdorf and Sarnen. The values are averaged and aggregated from hourly to daily.
- **Run completeness:** at 07:46 UTC the same day's 00 UTC run was only partly available (72 of 240 h). The pipeline therefore runs later and checks that the run is complete before writing (see [Schedule](#schedule)).

## Label & features

One row per forecast day *t0* and horizon *h*. Each row stores **two dates**: the day the forecast was issued and the day it is for. Training therefore always uses real archived forecasts, never observed weather.

**Label**

- The net weight change of one day: the weight at midnight after the day minus the weight at midnight before it.
- Days without a midnight value stay empty; they are not interpolated.
- A label counts as final only after 2 days, because scales can upload late.

**Features**

| Group | Features |
|---|---|
| Weather forecast for the target day | `tmax`, `tmin`, `precip_sum`, `shortwave_sum`, `wind_max` – TODO: final column names |
| Weight momentum | net change over the last 1 / 3 / 7 / 14 days, on a fixed daily grid, **ending at t0−2** (final days only, see review point A1) |
| Climatology | mean change for the target date, 2019–2023 ±7 days |
| Calendar | day of year as sin/cos, horizon *h* |
| Data quality | TODO: scale count / jump flag (review point A2) |

**Leakage controls**

- Point-in-time correct: each row carries both its issue date and its target date.
- The backfill runs the same code as the daily job.
- The raw daily API responses are archived as they were known at *t0*.
- The climatology uses only 2019–2023, so no training or test year enters it.

**Split**

- Training: forecast days from 14.03.2024 to 22.12.2025.
- Test: the unseen 2026 season.

## Feature store

TODO: HF Hub dataset name and link.

| Table | Content | Key |
|---|---|---|
| `raw/hivewatch/` | archived daily API responses (as of poll date) | poll date |
| `raw/openmeteo/` | archived forecast runs | run date |
| `features/` | one row per (*t0*, *h*) | `issue_date`, `target_date` |
| `labels/` | daily net change + `is_final` flag | `date` |

TODO: final layout. Every write creates a new dataset revision (Git commit on the HF Hub). That revision is the data version logged with each model.

## Run it

TODO: fill in once the code exists.

```bash
git clone https://github.com/Whalesharkking/mlops.git && cd mlops
cp .env.example .env          # add HF_TOKEN
uv sync --locked              # pinned dependencies from uv.lock

# daily run (today)
uv run python -m TODO.pipelines.feature_pipeline

# backfill a date range with the same code
uv run python -m TODO.pipelines.feature_pipeline --start 2024-03-14 --end 2026-10-31

# or with Docker
docker build -t nectarflow:dev .
docker run --rm --env-file .env nectarflow:dev
```

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | write access to the HF Hub feature store |

Secrets live only in `.env` (gitignored) locally and in GitHub Secrets in CI.

## Schedule

| Workflow | Trigger | What it does |
|---|---|---|
| TODO `feature-daily.yml` | cron TODO UTC, daily | Poll HiveWatch and Open-Meteo, check that the run is complete (retry if not), compute features and labels, push to HF Hub |
| TODO `backfill.yml` | manual (`workflow_dispatch`) with a date range | Same code as the daily run, for history |
| `render-pdf.yml` | push to `docs/` | Renders the Typst docs to PDF |

The workflows use `concurrency:` so that runs never overlap.

## Repository layout

TODO once the code exists.

```
docs/            proposal, milestone summaries (Typst → PDF via CI)
.github/workflows/
```

## Credits

- Hive data: HiveWatch / BienenSchweiz ([bienen.ch](https://bienen.ch)). TODO: wording as agreed in the permission reply.
- Weather forecasts: [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), ECMWF IFS HRES.
- Fallback hive data: TrachtNet Deutschland (DLR Rheinland-Pfalz, LWG Bayern).
