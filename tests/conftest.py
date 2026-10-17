from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nectarflow.common.config import TZ, WEATHER_LOCATIONS


def make_series(start: date, days: int, daily_change, minute: int = 0) -> pd.DataFrame:
    """Hourly weight series whose midnight-to-midnight change follows ``daily_change(day)``.

    During the day the weight dips by 0.5 kg around noon (foragers out), like the real data.
    """
    rows, weight = [], 0.0
    for i in range(days):
        d = start + timedelta(days=i)
        change = daily_change(d)
        for hour in range(24):
            local = datetime(d.year, d.month, d.day, hour, minute, tzinfo=TZ)
            dip = -0.5 if 10 <= hour <= 15 else 0.0
            rows.append((local, weight + change * hour / 24 + dip))
        weight += change
    end = start + timedelta(days=days)
    rows.append((datetime(end.year, end.month, end.day, 0, minute, tzinfo=TZ), weight))
    df = pd.DataFrame(rows, columns=["ts", "weight"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def hourly_run(issue_date: date, hours: int = 240) -> pd.DataFrame:
    times = pd.date_range(pd.Timestamp(issue_date, tz="UTC"), periods=hours, freq="h")
    frames = []
    for i, name in enumerate(WEATHER_LOCATIONS):
        df = pd.DataFrame(
            {
                "location": name,
                "time": times,
                "temperature_2m": 10.0 + i,  # location means: 11.5
                "precipitation": 1.0,
                "shortwave_radiation": 100.0,
                "wind_speed_10m": 5.0,
            }
        )
        df.loc[0, ["precipitation", "shortwave_radiation"]] = np.nan  # first step not accumulated
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def series_factory():
    return make_series
