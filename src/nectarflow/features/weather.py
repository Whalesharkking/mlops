"""Ingest: archived ECMWF IFS runs from the Open-Meteo Single Runs API."""

import time as systime
from datetime import date, timedelta

import pandas as pd
import requests

from nectarflow.common.config import (
    FORECAST_HOURS,
    HORIZONS,
    HOURLY_VARIABLES,
    OPEN_METEO_SINGLE_RUNS_URL,
    TZ,
    WEATHER_LOCATIONS,
    WEATHER_MODEL,
)

TIMEOUT_S = 60


class IncompleteRunError(RuntimeError):
    """The requested run is not (yet) fully published. Never write a partial run."""


def fetch_run(issue_date: date, session: requests.Session | None = None, retries: int = 5) -> pd.DataFrame:
    """Hourly forecast of the 00 UTC run of ``issue_date``, one row per location and hour.

    Columns: ``location``, ``time`` (UTC) and the hourly variables. Raises
    ``IncompleteRunError`` if fewer than ``FORECAST_HOURS`` hours are available.

    Open-Meteo counts one request with four locations as about four calls (free tier:
    600 per minute, 5000 per hour). On HTTP 429 the minute window is waited out.
    """
    http = session or requests.Session()
    params = {
        "latitude": ",".join(str(lat) for lat, _ in WEATHER_LOCATIONS.values()),
        "longitude": ",".join(str(lon) for _, lon in WEATHER_LOCATIONS.values()),
        "models": WEATHER_MODEL,
        "run": f"{issue_date.isoformat()}T00:00",
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_hours": FORECAST_HOURS,
        "timezone": "GMT",
    }
    for attempt in range(retries):
        try:
            resp = http.get(OPEN_METEO_SINGLE_RUNS_URL, params=params, timeout=TIMEOUT_S)
        except requests.ConnectionError, requests.Timeout:
            if attempt == retries - 1:
                raise
            systime.sleep(10 * (attempt + 1))
            continue
        if resp.status_code == 429:
            systime.sleep(65)
            continue
        if resp.status_code >= 500:
            systime.sleep(10 * (attempt + 1))
            continue
        break
    # A run that is not published yet, or missing from the archive (e.g. 2025-08-05),
    # comes back as HTTP 400 with a reason. Both mean: nothing to write for this date.
    if resp.status_code == 400 and "not available" in resp.text:
        raise IncompleteRunError(f"run {issue_date}: {resp.json().get('reason')}")
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, dict) and body.get("error"):
        raise IncompleteRunError(f"run {issue_date}: {body.get('reason')}")
    body = body if isinstance(body, list) else [body]

    frames = []
    for name, loc in zip(WEATHER_LOCATIONS, body, strict=True):
        hourly = pd.DataFrame(loc["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"], utc=True)
        hourly.insert(0, "location", name)
        frames.append(hourly)
    df = pd.concat(frames, ignore_index=True)
    check_complete(df, issue_date)
    return df


def check_complete(hourly: pd.DataFrame, issue_date: date) -> None:
    """A run is complete if every location has all hours with a temperature value."""
    per_loc = hourly.dropna(subset=["temperature_2m"]).groupby("location").size()
    have = int(per_loc.min()) if len(per_loc) == len(WEATHER_LOCATIONS) else 0
    if have < FORECAST_HOURS:
        raise IncompleteRunError(f"run {issue_date}: only {have} of {FORECAST_HOURS} hours available")


def daily_weather(hourly: pd.DataFrame, issue_date: date) -> pd.DataFrame:
    """Average the locations and aggregate to local calendar days for h = 1..10.

    Precipitation and radiation of the very first hour are not accumulated yet (null), so
    sums skip missing hours. ``n_hours`` records how many hours a day is built from.
    """
    mean = hourly.groupby("time")[list(HOURLY_VARIABLES)].mean().reset_index()
    mean["date"] = mean["time"].dt.tz_convert(TZ).dt.date
    daily = mean.groupby("date").agg(
        tmax=("temperature_2m", "max"),
        tmin=("temperature_2m", "min"),
        precip_sum=("precipitation", "sum"),
        shortwave_sum=("shortwave_radiation", "sum"),  # W/m² per hour -> converted below
        wind_max=("wind_speed_10m", "max"),
        n_hours=("temperature_2m", "count"),
    )
    daily["shortwave_sum"] = daily["shortwave_sum"] * 3600 / 1e6  # MJ/m²
    targets = [issue_date + timedelta(days=h - 1) for h in HORIZONS]
    daily = daily.reindex(targets)
    daily.index.name = "target_date"
    return daily.reset_index()
