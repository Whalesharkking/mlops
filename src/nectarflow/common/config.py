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

FEATURE_COLUMNS = [
    "h",
    "tmax",
    "tmin",
    "precip_sum",
    "shortwave_sum",
    "wind_max",
    *(f"mom_{k}d" for k in MOMENTUM_WINDOWS),
    "clim_target",
    "doy_sin",
    "doy_cos",
]


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
