"""Ingest: regional hive weight and scale count from HiveWatch / BienenSchweiz."""

from datetime import date, datetime, time, timedelta

import pandas as pd
import requests

from nectarflow.common.config import (
    HIVE_HISTORY_START,
    HIVEWATCH_BASE_URL,
    HIVEWATCH_HEADERS,
    REGION_ID,
    TZ,
)

TIMEOUT_S = 60


def _epoch_ms(d: date) -> int:
    return int(datetime.combine(d, time(), tzinfo=TZ).timestamp() * 1000)


def fetch_weight_series(start: date, end: date, session: requests.Session | None = None) -> pd.DataFrame:
    """Hourly, intervention-filtered regional weight between two local dates (end exclusive).

    Returns columns ``ts`` (UTC) and ``weight`` (kg, relative). The series is a running sum
    of per-scale changes, so only differences between two points carry meaning.
    """
    http = session or requests.Session()
    resp = http.get(
        f"{HIVEWATCH_BASE_URL}/region/{REGION_ID}/averages",
        params={
            "sensorType": "ext_weight",
            "startdate": _epoch_ms(start),
            "enddate": _epoch_ms(end),
            "filtered": "true",
            "appendInterventions": "false",
            "appendManualInterventions": "false",
        },
        headers=HIVEWATCH_HEADERS,
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    points = resp.json()["series"]
    if not points:
        raise ValueError(f"HiveWatch returned no weight points for {start}..{end}")
    df = pd.DataFrame(points).rename(columns={"x": "ts", "y": "weight"})
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def fetch_n_scales(session: requests.Session | None = None) -> int:
    """Current number of scales in the region. HiveWatch keeps no history of it (review A2)."""
    http = session or requests.Session()
    resp = http.get(
        f"{HIVEWATCH_BASE_URL}/public/region/{REGION_ID}", headers=HIVEWATCH_HEADERS, timeout=TIMEOUT_S
    )
    resp.raise_for_status()
    body = resp.json()
    region = body[0] if isinstance(body, list) else body
    return int(region["nofScales"])


def fetch_full_history(today: date, session: requests.Session | None = None) -> pd.DataFrame:
    """The whole series from the start of the archive up to and including ``today``."""
    return fetch_weight_series(HIVE_HISTORY_START, today + timedelta(days=1), session)
