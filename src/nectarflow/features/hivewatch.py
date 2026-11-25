"""Ingest: regional hive weight and scale count from HiveWatch / BienenSchweiz."""

import time as systime
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
RETRIES = 5
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def _get(http: requests.Session, url: str, params: dict | None = None) -> requests.Response:
    """GET with retry and linear backoff on connection errors, rate limits and server errors.

    Same policy as ``weather.fetch_run`` (MS2 review B1), so a short outage of HiveWatch
    does not cost the day. A 4xx other than 429 fails at once.
    """
    for attempt in range(RETRIES):
        last = attempt == RETRIES - 1
        try:
            resp = http.get(url, params=params, headers=HIVEWATCH_HEADERS, timeout=TIMEOUT_S)
        except requests.ConnectionError, requests.Timeout:
            if last:
                raise
        else:
            if resp.status_code not in RETRY_STATUS or last:
                resp.raise_for_status()
                return resp
        systime.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


def _epoch_ms(d: date) -> int:
    return int(datetime.combine(d, time(), tzinfo=TZ).timestamp() * 1000)


def fetch_weight_series(start: date, end: date, session: requests.Session | None = None) -> pd.DataFrame:
    """Hourly, intervention-filtered regional weight between two local dates (end exclusive).

    Returns columns ``ts`` (UTC) and ``weight`` (kg, relative). The series is a running sum
    of per-scale changes, so only differences between two points carry meaning.
    """
    http = session or requests.Session()
    resp = _get(
        http,
        f"{HIVEWATCH_BASE_URL}/region/{REGION_ID}/averages",
        params={
            "sensorType": "ext_weight",
            "startdate": _epoch_ms(start),
            "enddate": _epoch_ms(end),
            "filtered": "true",
            "appendInterventions": "false",
            "appendManualInterventions": "false",
        },
    )
    points = resp.json()["series"]
    if not points:
        raise ValueError(f"HiveWatch returned no weight points for {start}..{end}")
    df = pd.DataFrame(points).rename(columns={"x": "ts", "y": "weight"})
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def fetch_n_scales(session: requests.Session | None = None) -> int:
    """Current number of scales in the region. HiveWatch keeps no history of it (review A2)."""
    http = session or requests.Session()
    resp = _get(http, f"{HIVEWATCH_BASE_URL}/public/region/{REGION_ID}")
    body = resp.json()
    region = body[0] if isinstance(body, list) else body
    return int(region["nofScales"])


def fetch_full_history(today: date, session: requests.Session | None = None) -> pd.DataFrame:
    """The whole series from the start of the archive up to and including ``today``."""
    return fetch_weight_series(HIVE_HISTORY_START, today + timedelta(days=1), session)
