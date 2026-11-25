"""Project-wide constants and settings.

Feature definitions live here and in ``nectarflow.features`` only, so training and
inference use the same code (no training/serving skew).
"""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Zurich")

# --- Hive data: HiveWatch / BienenSchweiz -------------------------------------------
HIVEWATCH_BASE_URL = "https://app.hivewatch.net"
HIVEWATCH_HEADERS = {"X-Bienenschweiz": "1"}
REGION_ID = 11  # "Nordalpen"
HIVE_HISTORY_START = date(2019, 1, 1)

# A weight counts as "midnight weight" if it lies at most this far from local midnight.
MIDNIGHT_TOLERANCE_MINUTES = 45

# Labels are final only after this many days, because scales upload late (review A1).
# The label of day d is final at poll date p if d <= p - FINAL_LAG_DAYS.
FINAL_LAG_DAYS = 2

# Implausible daily changes (review A2). In the flow season real changes reach ~1.7 kg/day,
# outside it they stay well below 1 kg/day; larger values come from snow load or faulty scales.
JUMP_THRESHOLD_FLOW_SEASON_KG = 2.5
JUMP_THRESHOLD_OFF_SEASON_KG = 1.0
FLOW_SEASON_MONTHS = frozenset({4, 5, 6, 7})

# --- Weather: Open-Meteo Single Runs API -----------------------------------------------
OPEN_METEO_SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
WEATHER_MODEL = "ecmwf_ifs"
WEATHER_ARCHIVE_START = date(2024, 3, 14)  # first archived ECMWF IFS run
FORECAST_HOURS = 240
WEATHER_LOCATIONS = {
    "lucerne": (47.05, 8.31),
    "schwyz": (47.02, 8.65),
    "altdorf": (46.88, 8.64),
    "sarnen": (46.90, 8.25),
}
HOURLY_VARIABLES = ("temperature_2m", "precipitation", "shortwave_radiation", "wind_speed_10m")

# --- Features ------------------------------------------------------------------------
HORIZONS = range(1, 11)  # h = 1 is the issue day itself, h = 10 is nine days later
MOMENTUM_WINDOWS = (1, 3, 7, 14)
CLIMATOLOGY_YEARS = (2019, 2023)  # no training (2024-2025) or test (2026) year
CLIMATOLOGY_HALF_WINDOW_DAYS = 7

WEATHER_FEATURES = [
    "tmax",
    "tmin",
    "precip_sum",
    "shortwave_sum",
    "wind_max",
    "n_hours",  # h = 1 covers only 22 or 23 local hours, the run starts at 00 UTC (MS2 review A2)
]
MOMENTUM_FEATURES = [f"mom_{k}d" for k in MOMENTUM_WINDOWS]
FEATURE_COLUMNS = ["h", *WEATHER_FEATURES, *MOMENTUM_FEATURES, "clim_target", "doy_sin", "doy_cos"]

# --- Training ------------------------------------------------------------------------
# Train on the cleaned label (jump days removed), but evaluate and gate on the raw label,
# so a jump day still counts as an error and the test is not made easier.
TRAIN_TARGET = "label_clean"
EVAL_TARGET = "label"

# Evaluation protocol (proposal): train on issue dates up to 22.12.2025, so every target
# lies in 2025 or earlier, and test once on all issue dates from 01.01.2026 on.
TRAIN_LAST_ISSUE = date(2025, 12, 22)
TEST_FIRST_ISSUE = date(2026, 1, 1)

# Expanding-window CV over issue dates (MS1 review B2). One validation block per month of
# the 2025 flow season. Training rows end GAP_DAYS before a block, so the latest training
# target (t0 + 9) lies before the first validated day.
GAP_DAYS = len(HORIZONS)
CV_BLOCK_STARTS = tuple(date(2025, m, 1) for m in range(4, 10))
CV_LAST_ISSUE = date(2025, 9, 30)

# Champion gate (MS1 review A3). Holdout is the last eight weeks of issue dates with final
# labels. Only rows whose target lies in March to September count, because near-zero winter
# targets cannot separate models. Without enough such rows nothing is promoted.
GATE_HOLDOUT_DAYS = 56
GATE_MONTHS = frozenset(range(3, 10))
GATE_MIN_ROWS = 200
GATE_MIN_GAIN_KG = 0.01

# Scale count before the first pipeline run: HiveWatch estimates a linear growth from about
# 50 scales (2019) to 364 (October 2026). Used only as an optional sample weight (MS2 review E4).
N_SCALES_ESTIMATE = ((date(2019, 1, 1), 50), (date(2026, 10, 17), 364))

WANDB_PROJECT = "nectarflow"
REGISTRY_COLLECTION = "nectarflow-forecast"
CHAMPION_ALIAS = "champion"


@dataclass(frozen=True)
class Settings:
    """Runtime settings from the environment (secrets never live in code)."""

    data_dir: Path
    hf_repo: str | None
    hf_token: str | None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=Path(os.environ.get("DATA_DIR", "data")),
            hf_repo=os.environ.get("HF_DATASET_REPO") or None,
            hf_token=os.environ.get("HF_TOKEN") or None,
        )
